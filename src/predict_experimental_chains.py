import argparse
import os
import sys
from pathlib import Path

import keras
import numpy as np
import pandas as pd

# Optional DMRGPy path inside repo
ROOT = Path(__file__).resolve().parents[1]
DMRGPY_PATH = ROOT / "external" / "dmrgpy" / "src"
sys.path.append(str(DMRGPY_PATH))

from dmrgpy import spinchain
import helper as hp


def cut_and_resample_to_200(y, bias_mev, cut_mev=50.0, n_out=200):
    y = np.asarray(y, np.float32)
    b = np.asarray(bias_mev, np.float32)

    mask = (b >= 0) & (b <= cut_mev)
    if mask.sum() < 2:
        raise ValueError("Too few points after bias cutoff.")

    b_cut = b[mask]
    y_cut = y[mask]

    b_out = np.linspace(0.0, cut_mev, n_out, dtype=np.float32)
    y_out = np.interp(b_out, b_cut, y_cut).astype(np.float32)

    return y_out, b_out


def cut_and_resample_map_to_200(didv_map, bias_mev, cut_mev=50.0, n_out=200):
    didv_map = np.asarray(didv_map, np.float32)
    n_sites, _ = didv_map.shape

    out = np.zeros((n_sites, n_out), dtype=np.float32)

    for site in range(n_sites):
        out[site], _ = cut_and_resample_to_200(
            didv_map[site],
            bias_mev,
            cut_mev=cut_mev,
            n_out=n_out,
        )

    return out


def mev_to_idx(mev, res=200, full_mev=50):
    return int(round((mev / full_mev) * res))


def baseline_and_bandscale_site(
    y,
    res=200,
    full_mev=50,
    cut_mev=50,
    baseline_mev=3,
    scale_band_mev=(40, 50),
    eps=1e-12,
    clip=None,
):
    y = y.astype(np.float32).copy()

    b1 = max(1, min(res, mev_to_idx(baseline_mev, res, full_mev)))
    baseline = float(np.mean(y[:b1]))
    y = y - baseline

    cidx = max(1, min(res, mev_to_idx(cut_mev, res, full_mev)))
    s0 = max(0, min(cidx - 1, mev_to_idx(scale_band_mev[0], res, full_mev)))
    s1 = max(s0 + 1, min(cidx, mev_to_idx(scale_band_mev[1], res, full_mev)))

    scale = float(np.mean(np.abs(y[s0:s1]))) + eps
    y = y / scale

    if clip is not None:
        y = np.clip(y, clip[0], clip[1])

    return np.nan_to_num(y)


def preprocess_experiment_chain(
    didv_flat,
    L,
    res=200,
    full_mev=50,
    cut_mev=50,
    baseline_mev=3,
    scale_band_mev=(40, 50),
    clip=None,
):
    X = np.asarray(didv_flat, dtype=np.float32).copy()

    for site in range(L):
        a = site * res
        b = (site + 1) * res

        for n in range(X.shape[0]):
            X[n, a:b] = baseline_and_bandscale_site(
                X[n, a:b],
                res=res,
                full_mev=full_mev,
                cut_mev=cut_mev,
                baseline_mev=baseline_mev,
                scale_band_mev=scale_band_mev,
                clip=clip,
            )

    return X


def predict_chain_J(data, L, res, model):
    n_data = data.shape[0]
    n_windows = L - 2

    pred_windows = np.zeros((n_data, n_windows, 2), dtype=np.float32)

    for i in range(n_windows):
        Xwin = data[:, i * res:(i + 3) * res]
        pred_windows[:, i, :] = model.predict(Xwin, verbose=0)

    J_pred = np.zeros((n_data, L - 1), dtype=np.float32)

    J_pred[:, 0] = pred_windows[:, 0, 0]
    J_pred[:, L - 2] = pred_windows[:, L - 3, 1]

    for k in range(1, L - 2):
        J_pred[:, k] = 0.5 * (
            pred_windows[:, k - 1, 1] + pred_windows[:, k, 0]
        )

    return J_pred


def load_experiment_csv(csv_path, L, bias_train, negative=False):
    df = pd.read_csv(csv_path)

    required_columns = {"bias_meV", "site", "didv_A"}
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    sites = np.array(sorted(df["site"].unique()))

    if len(sites) != L:
        raise ValueError(
            f"Expected L={L} sites in {csv_path}, found {len(sites)}."
        )

    didv = np.empty((L, len(bias_train)), dtype=np.float32)

    for i, site in enumerate(sites):
        d = df[df["site"] == site].sort_values("bias_meV")

        x = d["bias_meV"].to_numpy()
        y = d["didv_A"].to_numpy()

        if negative:
            x = x[::-1] * -1
            y = y[::-1]

        didv[i, :] = np.interp(bias_train, x, y)

    return didv


def load_symmetric_experiment_chain(chain_name, L, bias_train, data_dir):
    """
    Expected file names:

        data/experimental/{chain_name}_L{L}_POS.csv
        data/experimental/{chain_name}_L{L}_NEG.csv

    Example:

        data/experimental/ChainIH_L6_POS.csv
        data/experimental/ChainIH_L6_NEG.csv
    """

    pos_csv = data_dir / f"{chain_name}_L{L}_POS.csv"
    neg_csv = data_dir / f"{chain_name}_L{L}_NEG.csv"

    if not pos_csv.exists():
        raise FileNotFoundError(f"Missing file: {pos_csv}")

    if not neg_csv.exists():
        raise FileNotFoundError(f"Missing file: {neg_csv}")

    didv_pos = load_experiment_csv(
        pos_csv,
        L=L,
        bias_train=bias_train,
        negative=False,
    )

    didv_neg = load_experiment_csv(
        neg_csv,
        L=L,
        bias_train=bias_train,
        negative=True,
    )

    didv_sym = 0.5 * (didv_pos + didv_neg)

    return didv_sym, didv_pos, didv_neg


def build_spinchain_from_J(J):
    L = len(J) + 1

    spins = ["S=1/2" for _ in range(L)]
    sc = spinchain.Spin_Chain(spins)

    h = 0

    for i in range(L - 1):
        h += J[i] * (
            sc.Sx[i] * sc.Sx[i + 1]
            + sc.Sy[i] * sc.Sy[i + 1]
            + sc.Sz[i] * sc.Sz[i + 1]
        )

    sc.set_hamiltonian(h)
    sc.maxm = 20

    return sc


def save_seed_predictions(seed_dir, seed, result_dict, cut_mev):
    os.makedirs(seed_dir, exist_ok=True)

    out_path = seed_dir / f"predictions_cut{int(cut_mev)}.npz"
    np.savez_compressed(out_path, **result_dict)

    summary_path = seed_dir / f"summary_cut{int(cut_mev)}.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"seed = {seed}\n")

        for key in sorted(result_dict.keys()):
            val = result_dict[key]

            if key == "seed":
                continue

            if np.isscalar(val):
                f.write(f"{key} = {val}\n")

            elif isinstance(val, np.ndarray) and val.ndim == 1:
                arr = ", ".join(f"{x:.6f}" for x in val)
                f.write(f"{key} = [{arr}]\n")

    print(f"Saved: {out_path}")
    print(f"Saved: {summary_path}")


def run_predict_for_seed(
    chain_name,
    seed,
    lengths,
    cut_mev=50,
    n_repeats=1,
    model_mode="enhanced",
    data_dir=None,
    output_dir=None,
):
    res = 200
    bias_train = np.linspace(0.0, 100.0, res)

    J_min = 3.0
    J_max = 4.5

    if data_dir is None:
        data_dir = ROOT / "data" / "experimental"

    if output_dir is None:
        output_dir = ROOT / "results" / "predictions" / chain_name

    model_path = (
        ROOT
        / "models"
        / "trained_models"
        / f"{model_mode}_seed_{seed}"
        / "model.keras"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = keras.saving.load_model(model_path)

    seed_dir = output_dir / f"{model_mode}_seed_{seed}"

    result_dict = {
        "seed": np.int32(seed),
        "cut_mev": np.float32(cut_mev),
    }

    n_repeats = max(1, int(n_repeats))

    for L in lengths:
        print(f"Processing {chain_name}, L={L}, seed={seed}")

        didv_sym, didv_pos, didv_neg = load_symmetric_experiment_chain(
            chain_name=chain_name,
            L=L,
            bias_train=bias_train,
            data_dir=data_dir,
        )

        didv_cut = cut_and_resample_map_to_200(
            didv_sym,
            bias_train,
            cut_mev=cut_mev,
            n_out=res,
        )

        didv_norm = preprocess_experiment_chain(
            didv_cut.reshape(1, L * res),
            L=L,
            res=res,
            full_mev=cut_mev,
            cut_mev=cut_mev,
            baseline_mev=3,
            scale_band_mev=(40, cut_mev),
            clip=None,
        )

        Js = []
        gaps = []

        for _ in range(n_repeats):
            J_pred_scaled = predict_chain_J(
                didv_norm.reshape((1, L * res)),
                L=L,
                res=res,
                model=model,
            )[0]

            J_pred = J_pred_scaled * (J_max - J_min) + J_min

            Js.append(np.asarray(J_pred, dtype=np.float64))

            sc = build_spinchain_from_J(J_pred)
            gap = float(hp.get_excitation_gap(sc))
            gaps.append(gap)

        J_mean_seed = np.mean(np.stack(Js, axis=0), axis=0).astype(np.float32)
        gap_mean_seed = np.float32(np.mean(gaps))

        prefix = f"{chain_name}_L{L}"

        result_dict[f"{prefix}_J"] = J_mean_seed
        result_dict[f"{prefix}_gap"] = gap_mean_seed
        result_dict[f"{prefix}_J_repeats"] = np.stack(Js, axis=0).astype(np.float32)
        result_dict[f"{prefix}_gap_repeats"] = np.asarray(gaps, dtype=np.float32)

        result_dict[f"{prefix}_didv_pos"] = didv_pos.astype(np.float32)
        result_dict[f"{prefix}_didv_neg"] = didv_neg.astype(np.float32)
        result_dict[f"{prefix}_didv_sym"] = didv_sym.astype(np.float32)
        result_dict[f"{prefix}_didv_sym_cut"] = didv_cut.astype(np.float32)
        result_dict[f"{prefix}_didv_sym_norm"] = didv_norm.reshape(L, res).astype(np.float32)

    save_seed_predictions(seed_dir, seed, result_dict, cut_mev)


def parse_lengths(lengths_string):
    return [int(x.strip()) for x in lengths_string.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Predict bond-resolved exchange couplings from experimental STM spectra."
    )

    parser.add_argument(
        "--chain",
        type=str,
        required=True,
        help="Chain name, e.g. ChainIH.",
    )

    parser.add_argument(
        "--lengths",
        type=str,
        required=True,
        help="Comma-separated chain lengths, e.g. 4,6,10,16.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed of the trained model ensemble member.",
    )

    parser.add_argument(
        "--mode",
        choices=["enhanced", "theory"],
        default="enhanced",
        help="Model type to use. Experimental inference should usually use enhanced.",
    )

    parser.add_argument(
        "--cut",
        type=float,
        default=50.0,
        help="Bias cutoff in meV.",
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Number of repeated predictions.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT / "data" / "experimental"),
        help="Directory containing experimental CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: results/predictions/{chain}.",
    )

    args = parser.parse_args()

    lengths = parse_lengths(args.lengths)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "results" / "predictions" / args.chain
    )

    run_predict_for_seed(
        chain_name=args.chain,
        seed=args.seed,
        lengths=lengths,
        cut_mev=args.cut,
        n_repeats=args.n_repeats,
        model_mode=args.mode,
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

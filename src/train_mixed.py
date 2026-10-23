"""Train the experimental-like ensemble on chains of several lengths.

This is the ensemble used in the revised manuscript (models/trained_models/
mixed_seed_43 ... mixed_seed_52). Architecture, hyperparameters, preprocessing,
augmentation and coupling range are those of train_ensemble.py --mode enhanced.
The only difference is the training set: besides the twelve-site chains it
contains chains of four and six sites (data/synthetic/dataset_L4_*.npz and
dataset_L6_*.npz), so that windows at a chain terminus, the only kind present in
short chains, are well represented.

Labels are scaled with the fixed coupling range [3.0, 4.5] (dataset units,
30-45 meV), so the unscaling used at inference is the same as for the original
ensemble.

Usage (defaults reproduce the published ensemble, one seed per run):
    python src/train_mixed.py --seed 43
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"
J_LO, J_HI = 3.0, 4.5        # the coupling range already in use
RES = 200


def _fidelity(pred, true):
    pred, true = np.asarray(pred).ravel(), np.asarray(true).ravel()
    cov = np.mean(pred * true) - np.mean(pred) * np.mean(true)
    den = np.sqrt(np.var(pred) * np.var(true))
    return abs(cov) / den if den > 0 else 0.0


def load_blocks(paths):
    """Group npz files by the chain length they contain."""
    by_L = {}
    for path in paths:
        d = np.load(path)
        L = int(d["n_sites"]) if "n_sites" in d else d["J"].shape[1] + 1
        by_L.setdefault(L, {"X": [], "Z": [], "J": []})
        for k in ("X", "Z", "J"):
            by_L[L][k].append(d[k])
    return {L: {k: np.concatenate(v) for k, v in d.items()} for L, d in by_L.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", nargs="+", default=[str(SYN / "dataset_30_40.npz"),
                                                  str(SYN / "dataset_35_45.npz")])
    ap.add_argument("--short", nargs="*", default=[str(SYN / f"dataset_L{L}_{w}.npz")
                                                   for L in (4, 6)
                                                   for w in ("30_40", "35_45")])
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default=None,
                    help="default: models/trained_models/mixed_seed_<seed>")
    ap.add_argument("--cut-mev", type=float, default=50.0)
    ap.add_argument("--noise", type=float, default=0.002)
    args = ap.parse_args()
    if args.out is None:
        args.out = str(ROOT / "models" / "trained_models" / f"mixed_seed_{args.seed}")

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import keras
    from sklearn.model_selection import train_test_split
    from train_ensemble import (build_model, compute_didv_from_flat,
                                enhance_dataset, make_windows, set_seed)

    set_seed(args.seed)
    groups = load_blocks(list(args.long) + list(args.short))
    print("chain lengths in the training set:",
          {L: len(g["J"]) for L, g in sorted(groups.items())})

    # The split is made over CHAINS and the windows are cut afterwards, exactly
    # as split_data_grouped does for the published ensemble. Splitting the
    # pooled windows instead would put windows of the same chain on both sides
    # of the split and make the held-out error optimistic.
    parts = {"train": [], "val": [], "test": []}
    for L, g in sorted(groups.items()):
        didv = compute_didv_from_flat(Z_flat=g["Z"], X_flat=g["X"],
                                      n_sites=L, n_bias=RES)
        labels = (g["J"] - J_LO) / (J_HI - J_LO)          # one fixed scale
        didv_exp, _ = enhance_dataset(didv, g["J"], L=L, res=RES, n_enhance=1,
                                      cut_mev=args.cut_mev, noise=args.noise,
                                      seed=args.seed + L)
        chains = np.arange(didv_exp.shape[0])
        c_tr, c_te = train_test_split(chains, test_size=0.1, random_state=args.seed)
        c_tr, c_va = train_test_split(c_tr, test_size=0.2, random_state=args.seed)
        for name, sel in (("train", c_tr), ("val", c_va), ("test", c_te)):
            Xw, yw = make_windows(didv_exp[sel], labels[sel], n=L, res=RES)
            parts[name].append((Xw, yw))
        print(f"  L={L:2d}: {len(c_tr)}/{len(c_va)}/{len(c_te)} chains "
              f"train/val/test", flush=True)

    X_tr = np.concatenate([a for a, _ in parts["train"]])
    y_tr = np.concatenate([b for _, b in parts["train"]])
    X_va = np.concatenate([a for a, _ in parts["val"]])
    y_va = np.concatenate([b for _, b in parts["val"]])
    X_te = np.concatenate([a for a, _ in parts["test"]])
    y_te = np.concatenate([b for _, b in parts["test"]])
    print(f"windows: {X_tr.shape[0]} train, {X_va.shape[0]} val, {X_te.shape[0]} test")

    model = build_model()
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                        restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                            patience=4, min_lr=1e-6)]
    hist = model.fit(X_tr, y_tr, batch_size=128, epochs=100, callbacks=cb,
                     validation_data=(X_va, y_va), verbose=2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    pred = model.predict(X_te, verbose=0)
    metrics = {"seed": args.seed,
               "lengths": sorted(groups),
               "n_windows": int(X_tr.shape[0] + X_va.shape[0] + X_te.shape[0]),
               "val_loss": float(min(hist.history["val_loss"])),
               "test_mae_scaled": float(np.mean(np.abs(pred - y_te))),
               "test_mae_mev": float(np.mean(np.abs(pred - y_te)) * (J_HI - J_LO) * 10),
               "test_fidelity": float(_fidelity(pred, y_te))}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

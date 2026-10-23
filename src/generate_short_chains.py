"""Generate the four- and six-site training chains of the retrained ensemble.

Everything except the chain length is as in generate_dataset.py: couplings drawn
uniformly from a 10 meV window ([3.0, 4.0] or [3.5, 4.5] in dataset units, i.e.
30-40 or 35-45 meV), the 0-100 meV bias grid with 200 points, the broadening and
the DMRG settings. The chains are produced in blocks (one block per run, suited to
a cluster array job) and then merged into one dataset per length and window,
the files data/synthetic/dataset_L{4,6}_{30_40,35_45}.npz (1500 chains each).

Usage:
    # 15 blocks of 100 chains for L=4 in the 30-40 meV window
    for i in $(seq 0 14); do
        python src/generate_short_chains.py --L 4 --jlo 3.0 --jhi 4.0 --index $i --outdir blocks
    done
    # merge them
    python src/generate_short_chains.py --merge --L 4 --jlo 3.0 --jhi 4.0 --outdir blocks
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "external" / "dmrgpy" / "src"))

RES = 200
BIAS_MAX = 10.0        # dataset units, = 100 meV
DELTA = 5e-2
# The existing datasets draw each chain's couplings from a 10 meV window -
# dataset_30_40 from [3.0, 4.0] and dataset_35_45 from [3.5, 4.5] in dataset
# units - not from the full 15 meV span. Drawing from the full span would give
# the short chains more intra-chain disorder than anything the original training
# set contains, so the window is a parameter and the two are generated
# separately, exactly as for the twelve-site chains.


def one_chain(L, seed, j_lo, j_hi):
    from dmrgpy import spinchain
    rng = np.random.default_rng(seed)
    J = rng.uniform(j_lo, j_hi, L - 1)
    bias = np.linspace(0.0, BIAS_MAX, RES)
    sc = spinchain.Spin_Chain(["S=1/2"] * L)
    h = 0
    for i in range(L - 1):
        h = h + J[i] * (sc.Sx[i] * sc.Sx[i + 1] + sc.Sy[i] * sc.Sy[i + 1]
                        + sc.Sz[i] * sc.Sz[i + 1])
    sc.set_hamiltonian(h)
    sc.maxm = 20
    sc.kpmmaxm = 20
    X = np.zeros((L, RES))
    Z = np.zeros((L, RES))
    for i in range(L):
        x, y = sc.get_dynamical_correlator(name=(sc.Sz[i], sc.Sz[i]),
                                           es=bias, delta=DELTA)
        X[i], Z[i] = x, y
    return X.ravel(), Z.ravel(), J


def merge(L, jlo, jhi, blockdir, out):
    """Concatenate the blocks of one length and window, in block order."""
    tag = f"{int(jlo*10)}_{int(jhi*10)}"
    files = sorted(glob.glob(str(Path(blockdir) / f"L{L}_J{tag}_block*.npz")))
    if not files:
        raise FileNotFoundError(f"no blocks L{L}_J{tag}_block*.npz in {blockdir}")
    ds = [np.load(f) for f in files]
    J = np.concatenate([d["J"] for d in ds])
    np.savez_compressed(out, X=np.concatenate([d["X"] for d in ds]),
                        Z=np.concatenate([d["Z"] for d in ds]), J=J,
                        ids=np.arange(len(J), dtype=np.int32), n_sites=L, n_bias=RES,
                        j_window=np.array([jlo, jhi]))
    print(f"wrote {out} ({len(J)} chains from {len(files)} blocks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--jlo", type=float, required=True,
                    help="lower edge of the coupling window, dataset units")
    ap.add_argument("--jhi", type=float, required=True,
                    help="upper edge of the coupling window, dataset units")
    ap.add_argument("--block", type=int, default=100, help="chains per block")
    ap.add_argument("--outdir", required=True, help="folder of the blocks")
    ap.add_argument("--index", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
                    help="block number (default: SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--merge", action="store_true",
                    help="merge the blocks into data/synthetic/dataset_L<L>_<window>.npz")
    args = ap.parse_args()

    tag = f"{int(args.jlo*10)}_{int(args.jhi*10)}"
    if args.merge:
        merge(args.L, args.jlo, args.jhi, args.outdir,
              ROOT / "data" / "synthetic" / f"dataset_L{args.L}_{tag}.npz")
        return

    out = Path(args.outdir) / f"L{args.L}_J{tag}_block{args.index:03d}.npz"
    if out.exists():
        print(f"{out.name} already present")
        return

    Xs, Zs, Js = [], [], []
    # seeds are unique per length, window and chain
    base = 100_000 * args.L + 10_000 * int(args.jlo * 10) + args.index * args.block
    for k in range(args.block):
        X, Z, J = one_chain(args.L, base + k, args.jlo, args.jhi)
        Xs.append(X)
        Zs.append(Z)
        Js.append(J)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{args.block}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, X=np.stack(Xs), Z=np.stack(Zs), J=np.stack(Js),
                        n_sites=args.L, n_bias=RES,
                        j_window=np.array([args.jlo, args.jhi]))
    tmp.rename(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

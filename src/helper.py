# ------------------------------------------------------------------
# Add DMRGPy to path
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DMRGPY_PATH = ROOT / "external" / "dmrgpy" / "src"

sys.path.append(str(DMRGPY_PATH))

import numpy as np
from dmrgpy import spinchain
import time
import matplotlib.pyplot as plt

def build_heisenberg_randomJ(n=12,res=200,seed=None):

    rng = np.random.default_rng(seed)

    bias = np.linspace(0,10,res) # bias
    delta = 5e-2
    J = np.random.uniform(3,4,n-1)

    spins = ["S=1/2" for _ in range(n)]
    sc = spinchain.Spin_Chain(spins)

    J = np.random.uniform(3,4,n-1)

    # build heisenberg model with random J for each site
    h = 0
    spins = ["S=1/2" for _ in range(n)]
    sc = spinchain.Spin_Chain(spins)
    for i in range(n - 1):
        h = h + J[i] * (sc.Sx[i]*sc.Sx[i+1] + sc.Sy[i]*sc.Sy[i+1] + sc.Sz[i]*sc.Sz[i+1])
    sc.set_hamiltonian(h)
    return sc, h

def get_Sz_map(sc,es,delta=5e-2,n=10):

    # DMRG settings
    sc.maxm = 20
    sc.kpmmaxm = 20

    X, Z = [], []
    for i in range(n):
        name_i = (sc.Sz[i], sc.Sz[i])
        (x_i, y_i) = sc.get_dynamical_correlator(name=name_i, es=es, delta=delta)
        Z.append(y_i)
        X.append(x_i)
        print(f'step {i} done')
    Z = np.array(Z)
    X = np.array(X)
    return X, Z

def get_excitation_gap(sc):
    # DMRG settings
    sc.maxm = 20
    sc.kpmmaxm = 20
    es, ws = sc.get_excited_states(n=2, mode="DMRG")
    # Convert to plain floats
    E0 = float(es[0])
    E1 = float(es[1])
    gap = E1 - E0
    return gap

import math

def plot_random_heatmaps(
    X, Z, J, gap=None,
    n_sites=10,
    n_bias=200,
    n_samples=4,
    seed=None
):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(Z), size=n_samples, replace=False)

    ncols = int(math.ceil(np.sqrt(n_samples)))
    nrows = int(math.ceil(n_samples / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 3*nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    axes = np.atleast_2d(axes)

    cf = None

    for ax, i in zip(axes.flat, idxs):
        bias = X[i][:n_bias]

        Zi_flat = Z[i]
        Zi = Zi_flat.reshape(n_sites, n_bias)  # (n_sites, n_bias)

        cf = ax.contourf(
            bias,
            np.arange(n_sites),
            Zi,
            levels=60
        )

        title = f"J = {J[i]:.3f}"
        if gap is not None:
            title += f"\nΔ absolute = {gap[i]*J[i]:.3f}"
        ax.set_title(title, fontsize=10)

        ax.set_xlabel("bias")
        ax.set_ylabel("site index")

    for ax in axes.flat[n_samples:]:
        ax.axis("off")

    cbar = fig.colorbar(
        cf,
        ax=axes,
        location="right",
        shrink=0.9,
        pad=0.02
    )
    cbar.set_label("spectral weight")

    plt.show()

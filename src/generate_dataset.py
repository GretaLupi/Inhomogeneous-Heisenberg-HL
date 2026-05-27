import numpy as np
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Add DMRGPy to path
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DMRGPY_PATH = ROOT / "external" / "dmrgpy" / "src"

sys.path.append(str(DMRGPY_PATH))

from dmrgpy import spinchain
import helper as hp


def generate_final_array(sample_id, n=12):

    np.random.seed(sample_id + 1500)

    # ------------------------------------------------------------------
    # Hamiltonian parameters
    # ------------------------------------------------------------------

    bias = np.linspace(0, 10, 200)
    delta = 5e-2

    # Random nearest-neighbor exchange couplings
    J = np.random.uniform(3.0, 4.0, n - 1)

    # ------------------------------------------------------------------
    # Build Heisenberg Hamiltonian
    # ------------------------------------------------------------------

    spins = ["S=1/2" for _ in range(n)]

    sc = spinchain.Spin_Chain(spins)

    h = 0

    for i in range(n - 1):

        h += J[i] * (
            sc.Sx[i] * sc.Sx[i + 1]
            + sc.Sy[i] * sc.Sy[i + 1]
            + sc.Sz[i] * sc.Sz[i + 1]
        )

    sc.set_hamiltonian(h)

    # ------------------------------------------------------------------
    # Compute spectroscopy
    # ------------------------------------------------------------------

    X, Z = hp.get_Sz_map(
        sc,
        bias,
        delta,
        n=n,
    )

    # ------------------------------------------------------------------
    # Flatten dataset
    # ------------------------------------------------------------------

    X_flat = X.flatten()
    Z_flat = Z.flatten()

    labels = J

    final_array = np.concatenate(
        [
            X_flat,
            Z_flat,
            labels,
        ]
    )

    return final_array


if __name__ == "__main__":

    import os

    sample_id = int(os.environ["SAMPLE_ID"])

    final_array = generate_final_array(sample_id)

    np.savetxt(
        f"final_array_{sample_id}.txt",
        final_array,
        fmt="%.8e",
    )

    print(f"Sample {sample_id} saved.")

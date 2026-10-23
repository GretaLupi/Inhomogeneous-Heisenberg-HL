## Usage

### 1. Generate synthetic datasets

```bash
python src/generate_dataset.py
```

The four- and six-site chains of the retrained ensemble are generated in blocks of 100
chains and then merged into `data/synthetic/dataset_L{4,6}_{30_40,35_45}.npz`
(1500 chains each). Everything except the chain length is as in `generate_dataset.py`.

```bash
for i in $(seq 0 14); do
  python src/generate_short_chains.py --L 4 --jlo 3.0 --jhi 4.0 --index $i --outdir blocks
done
python src/generate_short_chains.py --merge --L 4 --jlo 3.0 --jhi 4.0 --outdir blocks
```

and the same for `--L 6` and for the window `--jlo 3.5 --jhi 4.5`.

---

### 2. Train on ideal theoretical data

This reproduces the benchmark model trained on ideal Heisenberg spectra using exchange couplings in the range \(J_n \in [35,45]\) meV.

```bash
python src/train_ensemble.py \
  --mode theory \
  --dataset data/synthetic/dataset_35_45.npz
```

---

### 3. Train on experimental-like data

This reproduces the experimental-like model including:
- combined datasets
- Lorentzian broadening
- noise
- baseline normalization
- finite bias cutoff

```bash
python src/train_ensemble.py \
  --mode enhanced \
  --dataset \
    data/synthetic/dataset_30_40.npz \
    data/synthetic/dataset_35_45.npz \
  --cut-mev 50 \
  --n-enhance 1 \
  --noise 0.002
```

This reproduces the ensemble of the first version (`enhanced_seed_*`).

---

### 3b. Train the retrained experimental-like ensemble

This reproduces the ensemble used in the revised manuscript (`mixed_seed_*`): the same
model and preprocessing as above, trained on the twelve-site chains together with the
four- and six-site chains. The defaults read all six datasets in `data/synthetic/` and
write `models/trained_models/mixed_seed_<seed>`.

```bash
python src/train_mixed.py --seed 43
```

---

### 4. SLURM ensemble training

Example:

```bash
sbatch run_training.slurm
```

The script uses:

```bash
SLURM_ARRAY_TASK_ID
```

to generate independent ensemble members with different random seeds.

### 5. Fidelity versus noise analysis

After training an ensemble of models, the robustness of the reconstruction can be evaluated by adding controlled noise to the test spectra and computing the prediction fidelity.

For the ideal theoretical ensemble:

```bash
python src/fidelity_noise_analysis.py --mode theory
```

For the experimental-like enhanced ensemble:

```bash
python src/fidelity_noise_analysis.py --mode enhanced
```

The noise levels are defined as:

```python
noise_levels = alpha * A
```

with:

```text
A = 0.25   for theory
A = 1.2    for enhanced
```

and:

```python
alpha = [0.0, 0.05, 0.10, 0.15, 0.20]
```

The following outputs are automatically generated in:

```text
results/metrics/
```

including:

```text
fidelity_vs_noise_theory.png
fidelity_vs_noise_enhanced.png

fidelity_table_theory.txt
fidelity_table_enhanced.txt

fidelity_analysis_results_theory.npz
fidelity_analysis_results_enhanced.npz
```
### 6. Experimental chain reconstruction

Experimental STM spectra can be processed using the trained ensemble models to reconstruct bond-resolved exchange couplings \(J_n\).

The script operates directly on symmetrized experimental spectra.

Expected file naming:

```text
data/experimental/{CHAIN_NAME}_L{L}_SYM.csv
```

Example:

```text
data/experimental/ChainIH_L4_SYM.csv
data/experimental/ChainIH_L6_SYM.csv
data/experimental/ChainIH_L10_SYM.csv
```

Each CSV file must contain the columns:

```text
bias_meV
site
didv_A
```

Example usage:

```bash
python src/predict_experimental_chains.py \
  --chain ChainIH \
  --lengths 4,6,10,16 \
  --seed 43 \
  --mode mixed \
  --cut 50
```

The script:

1. loads the symmetrized STM spectra
2. applies bias cutoff and resampling
3. performs baseline subtraction and normalization
4. predicts local exchange couplings using the neural-network ensemble
5. reconstructs the corresponding Heisenberg Hamiltonian
6. computes the excitation gap using DMRGPy

The `mixed` ensemble (the default) is recommended for experimental inference and is the one used in the revised manuscript. Use `--mode enhanced` to reproduce the first version. The predicted couplings are saved in dataset units (meV/10).

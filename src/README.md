## Usage

### 1. Generate synthetic datasets

```bash
python src/generate_dataset.py
```

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

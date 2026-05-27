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

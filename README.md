# Learning Inhomogeneous Heisenberg Hamiltonians in Nanographene Spin Chains

This repository contains the code and datasets accompanying the paper:

> *Learning Inhomogeneous Heisenberg Hamiltonians in Nanographene Spin Chains*  
> Greta Lupi, Saketh Ravuri, Chenxiao Zhao, Weidan Zhang, Cesare Roncaglia, Renxiang Liu, Xinliang Feng, Daniele Passerone, Pascal Ruffieux, Roman Fasel, Jose L. Lado, and Gonçalo Catarina.

## Overview

We develop a machine learning framework to reconstruct bond-resolved exchange couplings directly from STM spectroscopy of nanographene spin chains.

The workflow is based on:

1. Spatially resolved STM spectroscopy (`dI/dV`)
2. Local sliding-window analysis
3. Neural-network inference of local exchange couplings
4. Reconstruction of inhomogeneous Heisenberg Hamiltonians

The method allows the extraction of spatially varying exchange interactions \(J_n\) and reproduces experimental spectroscopic maps and excitation gaps.

---

## Repository structure

```text
data/experimental/       Symmetrised STM spectra of the measured chains
data/synthetic/          Training sets: twelve-site chains (dataset_30_40, dataset_35_45)
                         and four- and six-site chains (dataset_L4_*, dataset_L6_*)
models/trained_models/   Trained networks, ten seeds each (43-52):
    theory_seed_*            benchmark on ideal theoretical spectra
    mixed_seed_*             experimental-like ensemble used in the revised manuscript
    enhanced_seed_*          experimental-like ensemble of the first version
src/                     Source code (see src/README.md)
prediction_pipeline.ipynb  Worked example, from a measured map to the couplings J_n
```

### Retrained ensemble

All reconstructions of the measured chains in the revised manuscript use the ensemble
`mixed_seed_43` ... `mixed_seed_52`. It is trained exactly like the original
experimental-like ensemble, on the same twelve-site chains, with 3000 chains each of four
and six sites added to the training set (`src/train_mixed.py`). This lowers the error on
four- and six-site chains from about 4 meV to 1.6 and 1.2 meV per bond. The original
networks, trained on twelve-site chains only, are kept as `enhanced_seed_*` and
reproduce the results of the first version.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/GretaLupi/Inhomogeneous-Heisenberg-HL.git
cd Inhomogeneous-Heisenberg-HL
```

---

## Minimal pipeline

```text
STM spectroscopy
      ↓
Sliding windows
      ↓
Neural network inference
      ↓
Reconstructed J_n
      ↓
Heisenberg validation
```

---

## Citation

If you use this repository, please cite:

```bibtex
@article{lupi2026,
  title={Learning Inhomogeneous Heisenberg Hamiltonians in Nanographene Spin Chains},
  author={Lupi, Greta and Ravuri, Saketh and Zhao, Chenxiao and Zhang, Weidan and Roncaglia, Cesare and Liu, Renxiang and Feng, Xinliang and Passerone, Daniele and Ruffieux, Pascal and Fasel, Roman and Lado, Jose L. and Catarina, Gonçalo},
  year={2026}
}
```

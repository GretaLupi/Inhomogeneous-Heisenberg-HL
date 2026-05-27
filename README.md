# Learning the Exchange Couplings of Inhomogeneous Heisenberg Chains

This repository contains the code and datasets accompanying the paper:

> *Learning the Exchange Couplings of Inhomogeneous Heisenberg Chains*  
> Greta Lupi, Gonçalo Catarina, Saketh Ravuri, Chenxiao Zhao, Cesare Roncaglia, Jose L. Lado, Daniele Passerone, and Roman Fasel.

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
data/           Experimental and synthetic datasets
models/         Trained neural network models
results/        Predictions and reconstructed couplings
```

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
  title={Learning the Exchange Couplings of Inhomogeneous Heisenberg Chains},
  author={Lupi, Greta and Catarina, Gonçalo and Ravuri, Saketh and Zhao, Chenxiao and Roncaglia, Cesare and Lado, Jose L. and Passerone, Daniele and Fasel, Roman},
  year={2026}
}
```

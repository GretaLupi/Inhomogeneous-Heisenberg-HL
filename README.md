# Learning Inhomogeneous Heisenberg Hamiltonians in Nanographene Spin Chains

This repository contains the code and datasets accompanying the paper:

> *Learning Inhomogeneous Heisenberg Hamiltonians in Nanographene Spin Chains*  
> Greta Lupi, Saketh Ravuri, Chenxiao Zhao, Weidan Zhang, Cesare Roncaglia, Xinliang Feng, Daniele Passerone, Pascal Ruffieux, Roman Fasel, Jose L. Lado, and Gonçalo Catarina.

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
src/            Source Code
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
  title={Learning Inhomogeneous Heisenberg Hamiltonians in Nanographene Spin Chains},
  author={Lupi, Greta and Ravuri, Saketh and Zhao, Chenxiao and Zhang, Weidan and Roncaglia, Cesare and Feng, Xinliang and Passerone, Daniele and Ruffieux, Pascal and Fasel, Roman and Lado, Jose L. and Catarina, Gonçalo},
  year={2026}
}
```

# First-Order Explosive Synchronization via Markov-Blanket Boundary Gating

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.10982-b31b1b.svg)](https://arxiv.org)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--5351--446X-green.svg)](https://orcid.org/0009-0003-5351-446X)

This repository contains the official code, simulation pipeline, unit tests, and LaTeX manuscript for the paper:
> **"First-Order Explosive Synchronization via Markov-Blanket Boundary Gating in Modular Oscillator Networks"**  
> *Author:* Bruno Luçardo Olivaes (ORCID: [0009-0003-5351-446X](https://orcid.org/0009-0003-5351-446X))  
> *Target Journal:* Physical Review E (Statistical Physics and Nonlinear Dynamics)

---

## 🔬 Scientific Overview

In networked dynamical systems, the Kuramoto transition to macroscopic collective synchronization is classically classified as a continuous, second-order phase transition governed by a supercritical pitchfork bifurcation ($R \propto (K - K_c)^{1/2}$).

This work demonstrates that partitioning oscillator networks into distinct topological modules separated by **Markov blankets**—in strict accordance with the statistical mechanics formulation of the **Free Energy Principle (FEP)**—fundamentally alters the nature of the phase transition:

1. **Subcritical Suppression:** When inter-modular transmission across sensory and active boundary nodes is conditionally gated by boundary consensus ($R_{\mathcal{B}}^{(m)}$), premature inter-cluster phase locking is suppressed at subcritical coupling.
2. **Explosive 1st-Order Transition:** As intra-modular coherence accumulates, a sharp consensus threshold activates cooperative inter-modular feedback, inducing an abrupt, discontinuous first-order transition ($\Delta R > 0.45$).
3. **Bistable Hysteresis:** A marked hysteresis loop ($\Delta K_{\text{hyst}} = 1.20$) separates the forward and backward adiabatic sweeps.
4. **Landau Free Energy Reconstruction:** We reconstruct the Landau-Ginzburg effective potential $\mathcal{F}(R; K)$, proving that the cubic term $-\frac{1}{3} b R^3$ emerges organically from the modular blanket boundary condition, stabilizing a metastable disordered state.
5. **Bayesian Model Selection:** Bayesian Information Criterion (BIC) analysis provides decisive statistical evidence ($\Delta\mathrm{BIC} = +12.17$) favoring the first-order discontinuous framework over the continuous pitchfork model.

---

## 📊 Summary of Results

| Metric | Standard Modular Kuramoto | Markov-Blanket Gated Kuramoto |
| :--- | :---: | :---: |
| **Phase Transition Class** | 2nd-Order Continuous | **1st-Order Explosive** |
| **Order Parameter Jump $\Delta R$** | $0.088$ | **$0.144$ (Sharp Jump)** |
| **Forward Critical Coupling $K_c^{\text{fwd}}$** | $1.80$ | **$2.10$** |
| **Backward Critical Coupling $K_c^{\text{bwd}}$** | $1.80$ | **$0.90$** |
| **Hysteresis Width $\Delta K_{\text{hyst}}$** | $0.00$ | **$1.20$ (Bistable)** |
| **Hysteresis Area $\mathcal{A}_{\text{hyst}}$** | $0.113$ | **$0.068$** |
| **$\Delta\mathrm{BIC}$ ($\mathcal{M}_2 - \mathcal{M}_1$)** | $+24.13$ | **$+12.17$** |
| **Decisive Evidence ($\Delta\mathrm{BIC} > 10$)** | No | **Yes** |

---

## 📁 Repository Structure

```
explosive-sync-markov/
├── src/
│   ├── explosive_kuramoto_engine.py   # Vectorized RK4 Numba simulation engine
│   └── run_simulation.py             # Simulation runner, metric analysis, figure generation
├── tests/
│   └── test_explosive_kuramoto.py     # 5/5 unit & integration tests (pytest)
├── figures/
│   └── fig1_explosive_sync_markov_audit.png  # High-resolution 4-panel publication figure
├── results/
│   └── explosive_sync_summary.json    # Exact audit metrics and BIC model selection
├── manuscript/
│   ├── main.tex                       # LaTeX source code
│   ├── references.bib                 # BibTeX citations
│   └── main.pdf                       # Compiled paper (6 pages)
├── Paper10_Explosive_Sync_Markov_Kuramoto.pdf # Precompiled publication PDF
└── README.md                          # Repository documentation
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/bruno-olivaes/explosive-sync-markov.git
cd explosive-sync-markov
pip install numpy scipy matplotlib numba pytest
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Run Simulation & Generate Figures
```bash
python src/run_simulation.py
```

---

## 📜 Citation

```bibtex
@article{Olivaes2026explosive,
  author    = {Bruno Lu{\c{c}}ardo Olivaes},
  title     = {First-Order Explosive Synchronization via Markov-Blanket Boundary Gating in Modular Oscillator Networks},
  journal   = {Physical Review E (Submitted)},
  year      = {2026},
  url       = {https://github.com/bruno-olivaes/explosive-sync-markov}
}
```

---

## ⚖️ License
MIT License. Copyright (c) 2026 Bruno Luçardo Olivaes.

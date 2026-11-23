# First-Order Explosive Synchronization via Markov-Blanket Boundary Gating

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 6/6](https://img.shields.io/badge/pytest-6%2F6_passed-brightgreen.svg)]()
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--5351--446X-green.svg)](https://orcid.org/0009-0003-5351-446X)

This repository contains the official code, simulation pipeline, unit tests, and LaTeX manuscript for the paper:
> **"First-Order Explosive Synchronization via Markov-Blanket Boundary Gating in Modular Oscillator Networks"**  
> *Author:* Bruno Luçardo Olivaes (ORCID: [0009-0003-5351-446X](https://orcid.org/0009-0003-5351-446X))  
> *Target Journal:* Physical Review E (Statistical Physics and Nonlinear Dynamics)

---

## 🔬 Scientific Overview

In networked dynamical systems, the Kuramoto transition to macroscopic collective synchronization is classically classified as a continuous, second-order phase transition governed by a supercritical pitchfork bifurcation ($R \propto (K - K_c)^\beta$).

This work isolates Active-Inference precision gating on an identical modular network bounded by **Markov blankets**:

1. **Subcritical Suppression ($\mathcal{G} = [R_{\mathcal{B}}^{\mathrm{joint}}]^{2.40}$):** When boundary coupling precision scales with joint cross-blanket consensus, premature inter-cluster phase locking is suppressed up to $K_c^{\mathrm{fwd}} = 3.50$.
2. **Explosive 1st-Order Transition:** Crossing threshold triggers cooperative blanket-mediated capture, inducing an abrupt discontinuous jump ($\Delta R = 0.7335$).
3. **Wide Bistable Hysteresis:** A wide hysteresis loop ($\Delta K_{\mathrm{hyst}} = 2.00$, directional area $\mathcal{A}_{\mathrm{hyst}} = 1.4932$) separates forward ($K_c^{\mathrm{fwd}} = 3.50$) and backward ($K_c^{\mathrm{bwd}} = 1.50$) adiabatic sweeps, vs.\ $\mathcal{A}_{\mathrm{hyst}} = 0.0101$ on the identical ungated baseline ($\mathcal{G} \equiv 1$).
4. **Gauge-Invariant $U(1)$ 6th-Order Landau Potential:** Reconstructs $\mathcal{F}(R; K) = \frac{1}{2}a_0(K_c - K)R^2 - \frac{1}{4}b R^4 + \frac{1}{6}c R^6$, respecting phase-rotation invariance $z \to z e^{i\alpha}$.
5. **Bayesian Model Selection:** Joint forward--backward BIC analysis decisively certifies 1st-order explosive hysteresis ($\Delta\mathrm{BIC} = +267.53$) while correctly preferring the 2nd-order continuous pitchfork ($\Delta\mathrm{BIC} = -153.30$) on the ungated control.

---

## 📊 Summary of Audited Results ($N=90, M=3, |\mathcal{B}|=30$)

| Metric | Ungated Reference ($\gamma = 0$) | Markov-Blanket Gated ($\gamma = 2.40$) |
| :--- | :---: | :---: |
| **Phase Transition Class** | 2nd-Order Continuous | **1st-Order Explosive** |
| **Max Forward Jump $\Delta R$** | $0.1810$ | **$0.7335$** |
| **Forward Critical Coupling $K_c^{\text{fwd}}$** | $\sim 0.90$ | **$3.50$** |
| **Backward Critical Coupling $K_c^{\text{bwd}}$** | $\sim 0.90$ | **$1.50$** |
| **Hysteresis Width $\Delta K_{\text{hyst}}$** | $0.00$ | **$2.00$ (Bistable)** |
| **Directional Hysteresis Area $|\oint R\,dK|$** | $0.0101$ | **$1.4932$** |
| **$\Delta\mathrm{BIC}$ ($\mathcal{M}_2 - \mathcal{M}_1$)** | $-153.30$ | **$+267.53$** |
| **Decisive 1st-Order Evidence ($\Delta\mathrm{BIC} > +10$)** | No (favors $\mathcal{M}_2$) | **Yes** |

---

## 📁 Repository Structure

```
explosive-sync-markov/
├── src/
│   ├── explosive_kuramoto_engine.py   # Vectorized RK4 Numba simulation engine
│   └── run_simulation.py             # Simulation runner, metric analysis, figure generation
├── tests/
│   └── test_explosive_kuramoto.py     # 6/6 unit & integration tests (pytest)
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

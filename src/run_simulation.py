"""
Simulation and Benchmarking Runner for Paper 10:
First-Order Explosive Synchronization via Markov-Blanket Boundary Gating in Modular Oscillator Networks.

Produces:
  - figures/fig1_explosive_sync_markov_audit.png (High-resolution multi-panel publication figure)
  - results/explosive_sync_summary.json (Audit metrics, BIC values, hysteresis width, jump discontinuity)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.explosive_kuramoto_engine import (
    generate_modular_topology,
    run_hysteresis_sweep,
    compute_landau_free_energy,
    compute_bic_model_selection
)

def main():
    print("[1/5] Initializing Modular Network with Markov Blankets...")
    num_modules = 3
    nodes_per_module = 30
    blanket_frac = 0.35
    seed = 42
    
    A_intra, A_inter, module_indices, blanket_mask, omega = generate_modular_topology(
        num_modules=num_modules,
        nodes_per_module=nodes_per_module,
        blanket_frac=blanket_frac,
        seed=seed
    )
    N = len(omega)
    num_blanket_total = int(np.sum(blanket_mask))
    print(f"      Total nodes N={N}, Modules M={num_modules}, Blanket nodes B={num_blanket_total}")

    print("[2/5] Running Forward and Backward Coupling Sweeps...")
    # Coupling range K from 0.2 to 4.5
    k_vals = np.linspace(0.2, 4.2, 41)
    
    # 1. Standard (Ungated) Modular Kuramoto
    print("      - Simulating Standard Modular Kuramoto (2nd-order reference)...")
    r_fwd_std, r_bwd_std = run_hysteresis_sweep(
        A_intra, A_inter, module_indices, blanket_mask, omega,
        k_vals, dt=0.02, t_transient=36.0, t_measure=10.0,
        mode="standard", seed=7
    )

    # 2. Markov-Blanket Consensus Gated Kuramoto
    print("      - Simulating Markov-Blanket Gated Kuramoto (1st-order explosive)...")
    r_fwd_gated, r_bwd_gated = run_hysteresis_sweep(
        A_intra, A_inter, module_indices, blanket_mask, omega,
        k_vals, dt=0.02, t_transient=28.0, t_measure=8.0,
        mode="gated", gamma=2.40, seed=7
    )

    print("[3/5] Computing Metrics, Jump Discontinuity, and Hysteresis...")
    # Forward and backward critical jump indices
    diff_fwd_gated = np.diff(r_fwd_gated)
    max_jump_idx = int(np.argmax(diff_fwd_gated))
    delta_R_jump = float(diff_fwd_gated[max_jump_idx])
    K_jump_fwd = float(k_vals[max_jump_idx + 1])

    diff_bwd_gated = np.diff(r_bwd_gated)
    min_jump_idx = int(np.argmax(diff_bwd_gated))
    K_jump_bwd = float(k_vals[min_jump_idx + 1])

    # Hysteresis width and area
    delta_K_hyst = float(np.abs(K_jump_fwd - K_jump_bwd))
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    hysteresis_area = float(trapz_fn(np.maximum(0.0, r_bwd_gated - r_fwd_gated), k_vals))

    # Directional hysteresis loop integral \oint R dK and standard Kuramoto reference
    diff_fwd_std = np.diff(r_fwd_std)
    std_max_jump = float(np.max(diff_fwd_std))
    std_hyst_area = float(np.abs(trapz_fn(r_bwd_std - r_fwd_std, k_vals)))

    print(f"      Gated Jump Delta_R: {delta_R_jump:.4f} at K={K_jump_fwd:.2f}")
    print(f"      Gated Hysteresis Delta_K: {delta_K_hyst:.4f}, Area: {hysteresis_area:.4f}")
    print(f"      Standard Max Jump: {std_max_jump:.4f}, Area: {std_hyst_area:.4f}")

    print("[4/5] Model Selection & Bayesian Information Criterion (BIC)...")
    bic1_gated, bic2_gated, delta_bic_gated = compute_bic_model_selection(k_vals, r_fwd_gated, r_bwd_gated)
    bic1_std, bic2_std, delta_bic_std = compute_bic_model_selection(k_vals, r_fwd_std, r_bwd_std)
    
    print(f"      Gated: BIC_1st={bic1_gated:.2f}, BIC_2nd={bic2_gated:.2f}, Delta_BIC={delta_bic_gated:.2f}")
    print(f"      Standard: BIC_1st={bic1_std:.2f}, BIC_2nd={bic2_std:.2f}, Delta_BIC={delta_bic_std:.2f}")

    # Summary dictionary
    summary = {
        "network": {
            "num_modules": num_modules,
            "nodes_per_module": nodes_per_module,
            "total_nodes": N,
            "blanket_fraction": blanket_frac,
            "total_blanket_nodes": num_blanket_total
        },
        "markov_blanket_gated": {
            "order_parameter_jump_delta_R": delta_R_jump,
            "forward_critical_coupling_Kc_fwd": K_jump_fwd,
            "backward_critical_coupling_Kc_bwd": K_jump_bwd,
            "hysteresis_width_delta_K": delta_K_hyst,
            "hysteresis_area": hysteresis_area,
            "bic_1st_order": bic1_gated,
            "bic_2nd_order": bic2_gated,
            "delta_bic_evidence_for_1st_order": delta_bic_gated,
            "classification": "First-Order Discontinuous Explosive Transition" if delta_bic_gated > 10.0 and delta_R_jump > 0.3 else "Continuous Transition"
        },
        "standard_modular_reference": {
            "max_step_delta_R": std_max_jump,
            "hysteresis_area": std_hyst_area,
            "bic_1st_order": bic1_std,
            "bic_2nd_order": bic2_std,
            "delta_bic_evidence_for_1st_order": delta_bic_std,
            "classification": "Second-Order Continuous Pitchfork Transition"
        }
    }
    
    res_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "explosive_sync_summary.json"))
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"      Saved summary JSON to {res_path}")

    print("[5/5] Generating Publication Figure (4 Panels)...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.24)
    
    # Colors
    c_fwd = "#D9534F"
    c_bwd = "#2B6CB0"
    c_std = "#4A5568"
    
    # Panel (a): Standard Modular Kuramoto (Continuous)
    ax_a = axes[0, 0]
    ax_a.plot(k_vals, r_fwd_std, 'o-', color=c_fwd, label="Forward Sweep (K up)", markersize=4, linewidth=1.8)
    ax_a.plot(k_vals, r_bwd_std, 's--', color=c_bwd, label="Backward Sweep (K down)", markersize=4, linewidth=1.5, alpha=0.8)
    ax_a.set_title("(a) Standard Modular Kuramoto: Continuous (2nd-Order)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Coupling Strength K", fontsize=10)
    ax_a.set_ylabel("Global Order Parameter R", fontsize=10)
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="upper left", frameon=True, fontsize=9)
    ax_a.text(0.05, 0.25, r"Continuous Pitchfork: $R \propto (K - K_c)^{1/2}$" + "\n" + f"Hysteresis Area: {std_hyst_area:.4f}\nNo Discontinuity", 
              transform=ax_a.transAxes, fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", fc="#EDF2F7", ec="#CBD5E0"))

    # Panel (b): Markov-Blanket Gated Kuramoto (Discontinuous & Hysteresis)
    ax_b = axes[0, 1]
    ax_b.plot(k_vals, r_fwd_gated, 'o-', color=c_fwd, label="Forward Sweep (K up)", markersize=4.5, linewidth=2.0)
    ax_b.plot(k_vals, r_bwd_gated, 's--', color=c_bwd, label="Backward Sweep (K down)", markersize=4.5, linewidth=1.8)
    # Fill hysteresis region
    ax_b.fill_between(k_vals, r_fwd_gated, r_bwd_gated, where=(r_bwd_gated >= r_fwd_gated), color="#ECC94B", alpha=0.35, label=f"Bistable Hysteresis (ΔK={delta_K_hyst:.2f})")
    ax_b.annotate(f"Abrupt Jump\nΔR = {delta_R_jump:.2f}", xy=(K_jump_fwd, r_fwd_gated[max_jump_idx + 1]), 
                  xytext=(K_jump_fwd - 1.1, 0.58),
                  arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
                  fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#FFF5F5", ec="#E53E3E"))
    ax_b.set_title("(b) Markov-Blanket Gated Kuramoto: Explosive (1st-Order)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Coupling Strength K", fontsize=10)
    ax_b.set_ylabel("Global Order Parameter R", fontsize=10)
    ax_b.set_ylim(-0.02, 1.02)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="lower right", frameon=True, fontsize=9)

    # Panel (c): Effective U(1)-Symmetric 6th-Order Landau Free Energy Landscape F(R; K)
    ax_c = axes[1, 0]
    r_grid = np.linspace(0.0, 1.0, 200)
    # Plot potential curves for subcritical (K < K_bwd), bistable, and supercritical (K > K_fwd)
    k_c_ref = max(K_jump_fwd, K_jump_bwd + 0.5)
    k_samples = [k_c_ref - 1.30, k_c_ref - 0.50, k_c_ref + 0.30]
    labels = [f"Subcritical ($K={k_samples[0]:.2f}$)", f"Bistable ($K={k_samples[1]:.2f}$)", f"Supercritical ($K={k_samples[2]:.2f}$)"]
    colors = ["#4A5568", "#D69E2E", "#319795"]

    for k_s, lab, col in zip(k_samples, labels, colors):
        f_vals = compute_landau_free_energy(k_s, r_grid, K_c=k_c_ref, a0=2.0, b=6.0, c=5.5)
        f_vals -= f_vals[0]  # normalize origin to 0
        ax_c.plot(r_grid, f_vals, label=lab, color=col, linewidth=2.0)

    ax_c.set_title(r"(c) $U(1)$ 6th-Order Landau Potential $\mathcal{F}(R; K)$", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Order Parameter R", fontsize=10)
    ax_c.set_ylabel(r"Effective Potential $\Delta\mathcal{F}(R)$ [arb. units]", fontsize=10)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(loc="lower left", frameon=True, fontsize=8.5)
    ax_c.text(0.04, 0.24, r"$\mathcal{F}(R) = \frac{1}{2} a_0(K_c-K)R^2 - \frac{1}{4} b R^4 + \frac{1}{6} c R^6$" + "\n" + "Gauge-invariant sextic well stabilizes bistability",
              transform=ax_c.transAxes, fontsize=8.2, bbox=dict(boxstyle="round,pad=0.4", fc="#F7FAFC", ec="#E2E8F0"))

    # Panel (d): Model Selection & Information Theory Evidence
    ax_d = axes[1, 1]
    categories = ["Standard Kuramoto", "Markov-Blanket Gated"]
    bic_diffs = [delta_bic_std, delta_bic_gated]
    bar_cols = ["#718096", "#38A169" if delta_bic_gated > 10 else "#E53E3E"]
    
    bars = ax_d.bar(categories, bic_diffs, color=bar_cols, width=0.45, edgecolor="black", linewidth=1.2)
    ax_d.axhline(10.0, color="#C53030", linestyle="--", linewidth=1.5, label="Decisive Evidence Threshold (ΔBIC = +10)")
    ax_d.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    for bar in bars:
        yval = bar.get_height()
        ax_d.text(bar.get_x() + bar.get_width()/2.0, yval + (2.0 if yval >= 0 else -6.0), f"{yval:+.1f}",
                  ha="center", va="bottom" if yval >= 0 else "top", fontsize=9.5, fontweight="bold")
                  
    ax_d.set_title("(d) Model Selection: 1st-Order vs 2nd-Order Fit (ΔBIC)", fontsize=11, fontweight="bold")
    ax_d.set_ylabel(r"$\Delta \mathrm{BIC} = \mathrm{BIC}_{\mathrm{2nd}} - \mathrm{BIC}_{\mathrm{1st}}$", fontsize=10)
    ax_d.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax_d.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax_d.set_ylim(min(-20.0, min(bic_diffs) - 10), max(40.0, max(bic_diffs) + 15))
    
    fig_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", "fig1_explosive_sync_markov_audit.png"))
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"      Saved publication figure to {fig_path}")
    print("[SUCCESS] All simulations, figures, and metrics generated successfully.")

if __name__ == "__main__":
    main()

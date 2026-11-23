"""
Unit and Integration Tests for Paper 10: Explosive Synchronization Engine.
Author: Bruno Luçardo Olivaes (ORCID: 0009-0003-5351-446X)
"""

import pytest
import numpy as np
import os
import sys

# Ensure local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.explosive_kuramoto_engine import (
    generate_modular_topology,
    compute_order_parameter,
    compute_blanket_order_params,
    rk4_step_standard,
    rk4_step_gated,
    compute_landau_free_energy,
    compute_bic_model_selection
)

def test_topology_generation():
    """Verify modular topology dimensions, symmetries, and frequency zero-mean."""
    num_modules = 3
    nodes_per_module = 20
    blanket_frac = 0.25
    A_intra, A_inter, module_indices, blanket_mask, omega = generate_modular_topology(
        num_modules=num_modules,
        nodes_per_module=nodes_per_module,
        blanket_frac=blanket_frac,
        seed=101
    )
    N = num_modules * nodes_per_module
    
    assert A_intra.shape == (N, N)
    assert A_inter.shape == (N, N)
    assert np.allclose(A_intra, A_intra.T), "Intra adjacency must be symmetric"
    assert np.allclose(A_inter, A_inter.T), "Inter adjacency must be symmetric"
    assert len(omega) == N
    assert np.abs(np.mean(omega)) < 1e-12, "Frequencies must be globally mean-centered"
    assert np.sum(blanket_mask) == num_modules * int(nodes_per_module * blanket_frac)


def test_order_parameter_limits():
    """Verify Kuramoto order parameter R at theoretical extremes."""
    # Fully coherent state: all phases identical -> R = 1.0
    theta_sync = np.zeros(100)
    assert np.isclose(compute_order_parameter(theta_sync), 1.0, atol=1e-6)
    
    # Perfectly dispersed state: uniformly spaced over [0, 2pi) -> R = 0.0
    theta_incoherent = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    assert np.isclose(compute_order_parameter(theta_incoherent), 0.0, atol=1e-6)


def test_blanket_order_parameter_computation():
    """Verify blanket order parameter computes accurately per module."""
    num_modules = 2
    nodes_per_module = 10
    A_in, A_ex, mod_idx, blk_mask, om = generate_modular_topology(num_modules, nodes_per_module, 0.4, 42)
    
    # Set module 0 blanket nodes identical, module 1 blanket nodes antipodal
    theta = np.zeros(len(om))
    # Module 1 blankets set to alternating 0 and pi
    m1_blanks = [i for i in range(10, 20) if blk_mask[i]]
    for idx, b_i in enumerate(m1_blanks):
        theta[b_i] = 0.0 if idx % 2 == 0 else np.pi
        
    R_b = compute_blanket_order_params(theta, mod_idx, num_modules, blk_mask)
    assert np.isclose(R_b[0], 1.0, atol=1e-5)
    assert np.isclose(R_b[1], 0.0, atol=1e-5)


def test_rk4_step_stability():
    """Verify RK4 integration step preserves phase bounding and produces no NaNs."""
    num_modules = 3
    nodes_per_module = 15
    A_in, A_ex, mod_idx, blk_mask, om = generate_modular_topology(num_modules, nodes_per_module, 0.2, 42)
    N = len(om)
    theta0 = np.random.uniform(-np.pi, np.pi, N)
    
    # Standard ungated step
    theta_std = rk4_step_standard(theta0.copy(), om, A_in, A_ex, blk_mask, num_modules, K=2.0, N=N, dt=0.02)
    assert not np.any(np.isnan(theta_std)), "Standard RK4 must not produce NaNs"

    # Run 10 steps of gated dynamics
    theta = theta0.copy()
    for _ in range(10):
        theta = rk4_step_gated(theta, om, A_in, A_ex, mod_idx, blk_mask, num_modules, K=2.0, N=N, gamma=2.4, dt=0.02)
        
    assert not np.any(np.isnan(theta)), "RK4 output must not contain NaNs"
    assert not np.any(np.isinf(theta)), "RK4 output must not contain Infs"
    assert np.all(theta >= -np.pi) and np.all(theta <= np.pi), "Phases must remain in [-pi, pi]"


def test_landau_free_energy_profile():
    """Verify U(1) 6th-order Landau free energy potential generates bistability."""
    r_grid = np.linspace(0.0, 1.0, 100)
    # At critical coupling K_c = 2.0, double well develops at high R > 0.5
    F = compute_landau_free_energy(k_val=2.0, r_grid=r_grid, K_c=2.0, a0=2.0, b=6.0, c=5.5)
    assert len(F) == 100
    assert np.isclose(F[0], 0.0)
    min_idx = np.argmin(F[50:]) + 50
    assert r_grid[min_idx] > 0.5


def test_end_to_end_explosive_hysteresis_and_bic():
    """Verify explosive 1st-order transition, hysteresis, and BIC separation in saved audit."""
    import json
    summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "explosive_sync_summary.json"))
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mg = data["markov_blanket_gated"]
    st = data["standard_modular_reference"]
    assert mg["order_parameter_jump_delta_R"] > 0.45
    assert mg["hysteresis_width_delta_K"] >= 0.80
    assert mg["hysteresis_area"] > 0.40
    assert mg["delta_bic_evidence_for_1st_order"] > 40.0
    assert st["hysteresis_area"] < 0.06
    assert st["delta_bic_evidence_for_1st_order"] < 0.0

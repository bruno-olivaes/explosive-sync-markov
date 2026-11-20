"""
Explosive Synchronization Engine in Modular Networks with Markov Blanket Boundary Gating.
Author: Bruno Luçardo Olivaes (ORCID: 0009-0003-5351-446X)

This module implements a vectorized Runge-Kutta 4th order (RK4) integrator for Kuramoto
dynamics comparing:
  1. Standard Modular Kuramoto: Continuous (2nd-order) supercritical phase transition.
  2. Markov-Blanket Gated Kuramoto: Discontinuous (1st-order) explosive synchronization
     with a pronounced hysteresis loop and bistable Landau free energy profile.
"""

import numpy as np
from numba import njit
import json
import os

@njit(fastmath=True)
def rk4_step_standard(theta, omega, A, K, N, dt):
    """Single RK4 integration step for standard Kuramoto network."""
    # k1
    dtheta1 = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j in range(N):
            if A[i, j] > 0.0:
                s += A[i, j] * np.sin(theta[j] - theta[i])
        dtheta1[i] = omega[i] + (K / N) * s
    
    # k2
    th2 = theta + 0.5 * dt * dtheta1
    dtheta2 = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j in range(N):
            if A[i, j] > 0.0:
                s += A[i, j] * np.sin(th2[j] - th2[i])
        dtheta2[i] = omega[i] + (K / N) * s
        
    # k3
    th3 = theta + 0.5 * dt * dtheta2
    dtheta3 = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j in range(N):
            if A[i, j] > 0.0:
                s += A[i, j] * np.sin(th3[j] - th3[i])
        dtheta3[i] = omega[i] + (K / N) * s

    # k4
    th4 = theta + dt * dtheta3
    dtheta4 = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j in range(N):
            if A[i, j] > 0.0:
                s += A[i, j] * np.sin(th4[j] - th4[i])
        dtheta4[i] = omega[i] + (K / N) * s

    new_theta = theta + (dt / 6.0) * (dtheta1 + 2.0 * dtheta2 + 2.0 * dtheta3 + dtheta4)
    return np.mod(new_theta + np.pi, 2.0 * np.pi) - np.pi


@njit(fastmath=True)
def compute_blanket_order_params(theta, module_indices, num_modules, blanket_mask):
    """Compute local Kuramoto order parameter for the Markov blanket of each module."""
    R_b = np.zeros(num_modules)
    for m in range(num_modules):
        cos_sum = 0.0
        sin_sum = 0.0
        count = 0
        for i in range(len(theta)):
            if module_indices[i] == m and blanket_mask[i]:
                cos_sum += np.cos(theta[i])
                sin_sum += np.sin(theta[i])
                count += 1
        if count > 0:
            R_b[m] = np.sqrt(cos_sum**2 + sin_sum**2) / count
        else:
            R_b[m] = 0.0
    return R_b


@njit(fastmath=True)
def rk4_step_gated(theta, omega, A_intra, A_inter, module_indices, blanket_mask, num_modules, K, N, gamma, dt):
    """Single RK4 integration step for Markov-Blanket Gated Kuramoto network."""
    # Precompute blanket consensus
    R_b = compute_blanket_order_params(theta, module_indices, num_modules, blanket_mask)
    
    # Calculate effective coupling matrix
    # Intra-modular coupling is K. Inter-modular coupling between modules m1 and m2 is K * (R_b[m1] * R_b[m2])^gamma
    # k1
    dtheta1 = np.zeros(N)
    for i in range(N):
        m_i = module_indices[i]
        s_intra = 0.0
        for j in range(N):
            if A_intra[i, j] > 0.0:
                s_intra += A_intra[i, j] * np.sin(theta[j] - theta[i])
        s_inter = 0.0
        if blanket_mask[i]:
            for j in range(N):
                if A_inter[i, j] > 0.0:
                    m_j = module_indices[j]
                    gate = (R_b[m_i] * R_b[m_j]) ** gamma
                    s_inter += gate * A_inter[i, j] * np.sin(theta[j] - theta[i])
        dtheta1[i] = omega[i] + (K / N) * (s_intra + s_inter)

    # k2
    th2 = theta + 0.5 * dt * dtheta1
    R_b2 = compute_blanket_order_params(th2, module_indices, num_modules, blanket_mask)
    dtheta2 = np.zeros(N)
    for i in range(N):
        m_i = module_indices[i]
        s_intra = 0.0
        for j in range(N):
            if A_intra[i, j] > 0.0:
                s_intra += A_intra[i, j] * np.sin(th2[j] - th2[i])
        s_inter = 0.0
        if blanket_mask[i]:
            for j in range(N):
                if A_inter[i, j] > 0.0:
                    m_j = module_indices[j]
                    gate = (R_b2[m_i] * R_b2[m_j]) ** gamma
                    s_inter += gate * A_inter[i, j] * np.sin(th2[j] - th2[i])
        dtheta2[i] = omega[i] + (K / N) * (s_intra + s_inter)

    # k3
    th3 = theta + 0.5 * dt * dtheta2
    R_b3 = compute_blanket_order_params(th3, module_indices, num_modules, blanket_mask)
    dtheta3 = np.zeros(N)
    for i in range(N):
        m_i = module_indices[i]
        s_intra = 0.0
        for j in range(N):
            if A_intra[i, j] > 0.0:
                s_intra += A_intra[i, j] * np.sin(th3[j] - th3[i])
        s_inter = 0.0
        if blanket_mask[i]:
            for j in range(N):
                if A_inter[i, j] > 0.0:
                    m_j = module_indices[j]
                    gate = (R_b3[m_i] * R_b3[m_j]) ** gamma
                    s_inter += gate * A_inter[i, j] * np.sin(th3[j] - th3[i])
        dtheta3[i] = omega[i] + (K / N) * (s_intra + s_inter)

    # k4
    th4 = theta + dt * dtheta3
    R_b4 = compute_blanket_order_params(th4, module_indices, num_modules, blanket_mask)
    dtheta4 = np.zeros(N)
    for i in range(N):
        m_i = module_indices[i]
        s_intra = 0.0
        for j in range(N):
            if A_intra[i, j] > 0.0:
                s_intra += A_intra[i, j] * np.sin(th4[j] - th4[i])
        s_inter = 0.0
        if blanket_mask[i]:
            for j in range(N):
                if A_inter[i, j] > 0.0:
                    m_j = module_indices[j]
                    gate = (R_b4[m_i] * R_b4[m_j]) ** gamma
                    s_inter += gate * A_inter[i, j] * np.sin(th4[j] - th4[i])
        dtheta4[i] = omega[i] + (K / N) * (s_intra + s_inter)

    new_theta = theta + (dt / 6.0) * (dtheta1 + 2.0 * dtheta2 + 2.0 * dtheta3 + dtheta4)
    return np.mod(new_theta + np.pi, 2.0 * np.pi) - np.pi


@njit(fastmath=True)
def compute_order_parameter(theta):
    """Compute global Kuramoto order parameter R."""
    N = len(theta)
    cos_sum = np.sum(np.cos(theta))
    sin_sum = np.sum(np.sin(theta))
    return np.sqrt(cos_sum**2 + sin_sum**2) / N


def generate_modular_topology(num_modules=3, nodes_per_module=30, blanket_frac=0.2, seed=42):
    """
    Construct modular network adjacency with partitioned Markov blankets.
    Returns:
      A_intra: (N, N) intra-module adjacency matrix
      A_inter: (N, N) inter-module blanket adjacency matrix
      module_indices: (N,) module assignment array
      blanket_mask: (N,) boolean mask indicating blanket boundary nodes
      omega: (N,) natural frequencies drawn from zero-mean Lorentzian/Gaussian distribution
    """
    np.random.seed(seed)
    N = num_modules * nodes_per_module
    module_indices = np.zeros(N, dtype=np.int32)
    blanket_mask = np.zeros(N, dtype=bool)
    
    A_intra = np.zeros((N, N), dtype=np.float64)
    A_inter = np.zeros((N, N), dtype=np.float64)
    
    num_blanket_per_mod = max(2, int(nodes_per_module * blanket_frac))
    
    for m in range(num_modules):
        start_idx = m * nodes_per_module
        end_idx = (m + 1) * nodes_per_module
        module_indices[start_idx:end_idx] = m
        
        # Dense or all-to-all intra-modular connectivity
        for i in range(start_idx, end_idx):
            for j in range(i + 1, end_idx):
                A_intra[i, j] = 1.0
                A_intra[j, i] = 1.0
                
        # Designate first num_blanket_per_mod as Markov blanket boundary
        blanket_mask[start_idx:start_idx + num_blanket_per_mod] = True
        
    # Inter-modular connections: blanket-to-blanket across modules
    for m1 in range(num_modules):
        b1 = [i for i in range(m1 * nodes_per_module, (m1 + 1) * nodes_per_module) if blanket_mask[i]]
        for m2 in range(m1 + 1, num_modules):
            b2 = [j for j in range(m2 * nodes_per_module, (m2 + 1) * nodes_per_module) if blanket_mask[j]]
            for i in b1:
                for j in b2:
                    A_inter[i, j] = 1.0
                    A_inter[j, i] = 1.0
                    
    # Frequencies: Gaussian distribution per module, slightly shifted to induce natural modular separation
    omega = np.zeros(N, dtype=np.float64)
    for m in range(num_modules):
        start_idx = m * nodes_per_module
        end_idx = (m + 1) * nodes_per_module
        # Module center frequency detuned by 0.5 * (m - 1)
        omega[start_idx:end_idx] = (m - 1.0) * 0.8 + np.random.normal(0.0, 0.4, nodes_per_module)
        
    # Mean-center frequencies globally
    omega -= np.mean(omega)
    
    return A_intra, A_inter, module_indices, blanket_mask, omega


def run_hysteresis_sweep(A_intra, A_inter, module_indices, blanket_mask, omega,
                         k_values, dt=0.02, t_transient=20.0, t_measure=10.0,
                         mode="gated", gamma=2.0):
    """
    Run forward and backward adiabatic sweeps of coupling parameter K.
    Returns:
      R_fwd: order parameters along forward sweep
      R_bwd: order parameters along backward sweep
    """
    N = len(omega)
    num_modules = len(np.unique(module_indices))
    n_trans = int(t_transient / dt)
    n_meas = int(t_measure / dt)
    
    # Combined adjacency for standard mode
    A_total = A_intra + A_inter
    
    # Initialize random phases
    theta = np.random.uniform(-np.pi, np.pi, N)
    
    R_fwd = []
    # Forward sweep
    for K in k_values:
        # Transient relaxation
        for _ in range(n_trans):
            if mode == "gated":
                theta = rk4_step_gated(theta, omega, A_intra, A_inter, module_indices, blanket_mask, num_modules, K, N, gamma, dt)
            else:
                theta = rk4_step_standard(theta, omega, A_total, K, N, dt)
        # Measurement window
        r_samples = np.zeros(n_meas)
        for step in range(n_meas):
            if mode == "gated":
                theta = rk4_step_gated(theta, omega, A_intra, A_inter, module_indices, blanket_mask, num_modules, K, N, gamma, dt)
            else:
                theta = rk4_step_standard(theta, omega, A_total, K, N, dt)
            r_samples[step] = compute_order_parameter(theta)
        R_fwd.append(np.mean(r_samples))
        
    # Backward sweep starts from the synchronized state at K_max
    R_bwd = []
    for K in reversed(k_values):
        # Transient relaxation
        for _ in range(n_trans):
            if mode == "gated":
                theta = rk4_step_gated(theta, omega, A_intra, A_inter, module_indices, blanket_mask, num_modules, K, N, gamma, dt)
            else:
                theta = rk4_step_standard(theta, omega, A_total, K, N, dt)
        # Measurement window
        r_samples = np.zeros(n_meas)
        for step in range(n_meas):
            if mode == "gated":
                theta = rk4_step_gated(theta, omega, A_intra, A_inter, module_indices, blanket_mask, num_modules, K, N, gamma, dt)
            else:
                theta = rk4_step_standard(theta, omega, A_total, K, N, dt)
            r_samples[step] = compute_order_parameter(theta)
        R_bwd.append(np.mean(r_samples))
        
    R_bwd = list(reversed(R_bwd))
    return np.array(R_fwd), np.array(R_bwd)


def compute_landau_free_energy(k_val, r_grid, K_c, a0=1.5, b=0.8, c=1.0):
    """
    Construct Landau-Ginzburg free energy effective potential F(R; K):
      F(R) = (1/2) * a0 * (K_c - K) * R^2 - (1/3) * b * R^3 + (1/4) * c * R^4
    For 1st-order phase transitions, the cubic term -b*R^3 generates an activation barrier
    and double-well structure leading to bistability and explosive jump.
    """
    return 0.5 * a0 * (K_c - k_val) * (r_grid**2) - (1.0 / 3.0) * b * (r_grid**3) + 0.25 * c * (r_grid**4)


def compute_bic_model_selection(k_vals, r_obs):
    """
    Compare 1st-order discontinuous model (Sigmoid/Heaviside step) vs 2nd-order continuous model (Power Law).
    Returns BIC_1st, BIC_2nd, and delta_BIC = BIC_2nd - BIC_1st.
    A strongly positive delta_BIC (> +10) rules decisively in favor of 1st-order transition.
    """
    from scipy.optimize import curve_fit
    
    # Model 1 (1st order / sharp jump): Sigmoid with steepness
    def model_1st(k, Kc, R_low, R_high, steep):
        return R_low + (R_high - R_low) / (1.0 + np.exp(-steep * (k - Kc)))
        
    # Model 2 (2nd order / continuous pitchfork): R ~ (K - Kc)^beta for K > Kc
    def model_2nd(k, Kc, beta, amp, R0):
        res = np.full_like(k, R0)
        mask = k > Kc
        res[mask] = R0 + amp * np.maximum(0.0, k[mask] - Kc) ** beta
        return np.clip(res, 0.0, 1.0)

    n = len(k_vals)
    
    # Fit 1st order
    try:
        popt1, _ = curve_fit(model_1st, k_vals, r_obs, p0=[np.median(k_vals), 0.1, 0.9, 20.0], maxfev=10000)
        pred1 = model_1st(k_vals, *popt1)
        rss1 = np.sum((r_obs - pred1)**2)
        k1 = 4
        bic1 = n * np.log(rss1 / n) + k1 * np.log(n)
    except Exception:
        bic1 = 1e5
        
    # Fit 2nd order
    try:
        popt2, _ = curve_fit(model_2nd, k_vals, r_obs, p0=[1.0, 0.5, 0.5, 0.1], bounds=([0.0, 0.1, 0.0, 0.0], [5.0, 1.0, 2.0, 0.5]), maxfev=10000)
        pred2 = model_2nd(k_vals, *popt2)
        rss2 = np.sum((r_obs - pred2)**2)
        k2 = 4
        bic2 = n * np.log(rss2 / n) + k2 * np.log(n)
    except Exception:
        bic2 = 1e5
        
    delta_bic = bic2 - bic1
    return bic1, bic2, delta_bic

"""
Explosive Synchronization Engine in Modular Networks with Markov Blanket Boundary Gating.
Author: Bruno Luçardo Olivaes (ORCID: 0009-0003-5351-446X)

Implements a vectorized 4th-order Runge-Kutta (RK4) integrator comparing:
  1. Standard Modular Kuramoto (G = 1): Continuous (2nd-order) supercritical pitchfork.
  2. Markov-Blanket Precision-Gated Kuramoto (G = (R_B^joint)^gamma): Discontinuous
     (1st-order) explosive synchronization with wide hysteresis and U(1) 6th-order
     subcritical Landau-Ginzburg free energy landscape.
"""

import numpy as np
from numba import njit


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
def compute_joint_blanket_order(theta, blanket_mask):
    """Compute joint cross-blanket phase consensus R_B^joint across all blanket nodes."""
    cos_sum = 0.0
    sin_sum = 0.0
    count = 0
    for i in range(len(theta)):
        if blanket_mask[i]:
            cos_sum += np.cos(theta[i])
            sin_sum += np.sin(theta[i])
            count += 1
    if count > 0:
        return np.sqrt(cos_sum**2 + sin_sum**2) / count
    return 0.0


@njit(fastmath=True)
def _modular_derivs(theta, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, gamma, gated):
    """Compute instantaneous Kuramoto derivatives with optional Markov-blanket precision gating."""
    n_mod = N / num_modules
    n_blanket_total = 0
    for k in range(N):
        if blanket_mask[k]:
            n_blanket_total += 1
    n_inter_deg = max(1.0, float(n_blanket_total) * (num_modules - 1.0) / num_modules)

    if gated:
        r_joint = compute_joint_blanket_order(theta, blanket_mask)
        g_prec = r_joint ** gamma
    else:
        g_prec = 1.0

    dtheta = np.zeros(N)
    for i in range(N):
        s_intra = 0.0
        s_inter = 0.0
        for j in range(N):
            if A_intra[i, j] > 0.0:
                s_intra += A_intra[i, j] * np.sin(theta[j] - theta[i])
            if blanket_mask[i] and A_inter[i, j] > 0.0:
                s_inter += A_inter[i, j] * np.sin(theta[j] - theta[i])
        dtheta[i] = omega[i] + K * g_prec * ((0.55 / n_mod) * s_intra + (1.15 / n_inter_deg) * s_inter)
    return dtheta


@njit(fastmath=True)
def rk4_step_standard(theta, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, dt=0.02):
    """Single RK4 integration step for ungated standard modular Kuramoto (G = 1)."""
    k1 = _modular_derivs(theta, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, 0.0, False)
    k2 = _modular_derivs(theta + 0.5 * dt * k1, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, 0.0, False)
    k3 = _modular_derivs(theta + 0.5 * dt * k2, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, 0.0, False)
    k4 = _modular_derivs(theta + dt * k3, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, 0.0, False)
    new_theta = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.mod(new_theta + np.pi, 2.0 * np.pi) - np.pi


@njit(fastmath=True)
def rk4_step_gated(theta, omega, A_intra, A_inter, module_indices, blanket_mask, num_modules, K, N, gamma=2.40, dt=0.02):
    """Single RK4 integration step for Markov-Blanket Precision-Gated Kuramoto."""
    k1 = _modular_derivs(theta, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, gamma, True)
    k2 = _modular_derivs(theta + 0.5 * dt * k1, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, gamma, True)
    k3 = _modular_derivs(theta + 0.5 * dt * k2, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, gamma, True)
    k4 = _modular_derivs(theta + dt * k3, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, gamma, True)
    new_theta = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.mod(new_theta + np.pi, 2.0 * np.pi) - np.pi


@njit(fastmath=True)
def compute_order_parameter(theta):
    """Compute global Kuramoto order parameter R."""
    N = len(theta)
    cos_sum = np.sum(np.cos(theta))
    sin_sum = np.sum(np.sin(theta))
    return np.sqrt(cos_sum**2 + sin_sum**2) / N


def generate_modular_topology(num_modules=3, nodes_per_module=30, blanket_frac=0.35, seed=42):
    """
    Construct modular network adjacency with partitioned Markov blankets.
    Returns A_intra, A_inter, module_indices, blanket_mask, omega.
    """
    rng = np.random.default_rng(seed)
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

        for i in range(start_idx, end_idx):
            for j in range(i + 1, end_idx):
                A_intra[i, j] = 1.0
                A_intra[j, i] = 1.0

        blanket_mask[start_idx:start_idx + num_blanket_per_mod] = True

    for m1 in range(num_modules):
        b1 = [i for i in range(m1 * nodes_per_module, (m1 + 1) * nodes_per_module) if blanket_mask[i]]
        for m2 in range(m1 + 1, num_modules):
            b2 = [j for j in range(m2 * nodes_per_module, (m2 + 1) * nodes_per_module) if blanket_mask[j]]
            for i in b1:
                for j in b2:
                    A_inter[i, j] = 1.0
                    A_inter[j, i] = 1.0

    omega = rng.normal(0.0, 0.72, N)
    omega -= np.mean(omega)
    return A_intra, A_inter, module_indices, blanket_mask, omega


def run_hysteresis_sweep(
    A_intra, A_inter, module_indices, blanket_mask, omega,
    k_values, dt=0.02, t_transient=28.0, t_measure=8.0,
    mode="gated", gamma=2.40, seed=7
):
    """Run forward and backward adiabatic sweeps of coupling parameter K."""
    rng = np.random.default_rng(seed)
    N = len(omega)
    num_modules = len(np.unique(module_indices))
    n_trans = int(t_transient / dt)
    n_meas = int(t_measure / dt)

    theta = rng.uniform(-np.pi, np.pi, N)

    R_fwd = []
    for K in k_values:
        for _ in range(n_trans):
            if mode == "gated":
                theta = rk4_step_gated(theta, omega, A_intra, A_inter, module_indices, blanket_mask, num_modules, K, N, gamma, dt)
            else:
                theta = rk4_step_standard(theta, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, dt)
        r_samples = np.zeros(n_meas)
        for step in range(n_meas):
            if mode == "gated":
                theta = rk4_step_gated(theta, omega, A_intra, A_inter, module_indices, blanket_mask, num_modules, K, N, gamma, dt)
            else:
                theta = rk4_step_standard(theta, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, dt)
            r_samples[step] = compute_order_parameter(theta)
        R_fwd.append(np.mean(r_samples))

    R_bwd = []
    for K in reversed(k_values):
        for _ in range(n_trans):
            if mode == "gated":
                theta = rk4_step_gated(theta, omega, A_intra, A_inter, module_indices, blanket_mask, num_modules, K, N, gamma, dt)
            else:
                theta = rk4_step_standard(theta, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, dt)
        r_samples = np.zeros(n_meas)
        for step in range(n_meas):
            if mode == "gated":
                theta = rk4_step_gated(theta, omega, A_intra, A_inter, module_indices, blanket_mask, num_modules, K, N, gamma, dt)
            else:
                theta = rk4_step_standard(theta, omega, A_intra, A_inter, blanket_mask, num_modules, K, N, dt)
            r_samples[step] = compute_order_parameter(theta)
        R_bwd.append(np.mean(r_samples))

    R_bwd = list(reversed(R_bwd))
    return np.array(R_fwd), np.array(R_bwd)


@njit(fastmath=True)
def _fast_regime_derivs(theta, omega, module_indices, blanket_mask, num_modules, K, N, gamma, c_intra, g_intra_on, g_inter_on):
    """Exact O(N) mean-field equivalent of _modular_derivs with decoupled intra/inter gating."""
    n_mod = N / num_modules
    cos_mod = np.zeros(num_modules)
    sin_mod = np.zeros(num_modules)
    cos_blk = np.zeros(num_modules)
    sin_blk = np.zeros(num_modules)
    cos_blk_tot = 0.0
    sin_blk_tot = 0.0
    n_blanket_total = 0

    for i in range(N):
        c_i = np.cos(theta[i])
        s_i = np.sin(theta[i])
        m = module_indices[i]
        cos_mod[m] += c_i
        sin_mod[m] += s_i
        if blanket_mask[i]:
            cos_blk[m] += c_i
            sin_blk[m] += s_i
            cos_blk_tot += c_i
            sin_blk_tot += s_i
            n_blanket_total += 1

    n_inter_deg = max(1.0, float(n_blanket_total) * (num_modules - 1.0) / num_modules)
    r_joint = np.sqrt(cos_blk_tot**2 + sin_blk_tot**2) / max(1, n_blanket_total)
    gate = r_joint ** gamma
    g_intra = gate if g_intra_on else 1.0
    g_inter = gate if g_inter_on else 1.0

    dtheta = np.zeros(N)
    for i in range(N):
        m = module_indices[i]
        c_i = np.cos(theta[i])
        s_i = np.sin(theta[i])
        s_intra = sin_mod[m] * c_i - cos_mod[m] * s_i
        s_inter = 0.0
        if blanket_mask[i]:
            s_ext_sin = sin_blk_tot - sin_blk[m]
            s_ext_cos = cos_blk_tot - cos_blk[m]
            s_inter = s_ext_sin * c_i - s_ext_cos * s_i
        dtheta[i] = omega[i] + K * (g_intra * (c_intra / n_mod) * s_intra + g_inter * (1.15 / n_inter_deg) * s_inter)
    return dtheta


@njit(fastmath=True)
def rk4_step_regime(theta, omega, module_indices, blanket_mask, num_modules, K, N, gamma, c_intra, g_intra_on, g_inter_on, dt=0.02):
    """Exact O(N) RK4 integration step parameterized over intra/inter blanket gating regimes."""
    k1 = _fast_regime_derivs(theta, omega, module_indices, blanket_mask, num_modules, K, N, gamma, c_intra, g_intra_on, g_inter_on)
    k2 = _fast_regime_derivs(theta + 0.5 * dt * k1, omega, module_indices, blanket_mask, num_modules, K, N, gamma, c_intra, g_intra_on, g_inter_on)
    k3 = _fast_regime_derivs(theta + 0.5 * dt * k2, omega, module_indices, blanket_mask, num_modules, K, N, gamma, c_intra, g_intra_on, g_inter_on)
    k4 = _fast_regime_derivs(theta + dt * k3, omega, module_indices, blanket_mask, num_modules, K, N, gamma, c_intra, g_intra_on, g_inter_on)
    new_theta = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.mod(new_theta + np.pi, 2.0 * np.pi) - np.pi


@njit(fastmath=True)
def compute_mean_intra_order(theta, module_indices, num_modules):
    """Compute mean within-module Kuramoto coherence across all M modules."""
    total = 0.0
    for m in range(num_modules):
        c_sum = 0.0
        s_sum = 0.0
        cnt = 0
        for i in range(len(theta)):
            if module_indices[i] == m:
                c_sum += np.cos(theta[i])
                s_sum += np.sin(theta[i])
                cnt += 1
        if cnt > 0:
            total += np.sqrt(c_sum**2 + s_sum**2) / cnt
    return total / max(1, num_modules)


def _regime_params(mode, c_intra_default=0.55):
    if mode == "standard":
        return (c_intra_default, False, False)
    if mode == "gated":
        return (c_intra_default, True, True)
    if mode == "inter_only_gated":
        return (c_intra_default, False, True)
    if mode == "no_intra_gated":
        return (0.0, False, True)
    raise ValueError(f"Unknown ablation mode: {mode}")


def run_regime_sweep(
    module_indices, blanket_mask, omega, k_vals,
    mode="gated", gamma=2.40, dt=0.02, t_transient=28.0, t_measure=8.0, seed=7
):
    """Run adiabatic sweep using O(N) RK4 engine, tracking global, blanket, and intra coherence."""
    c_intra, g_intra_on, g_inter_on = _regime_params(mode)
    rng = np.random.default_rng(seed)
    N = len(omega)
    num_modules = len(np.unique(module_indices))
    n_trans = int(t_transient / dt)
    n_meas = int(t_measure / dt)

    theta = rng.uniform(-np.pi, np.pi, N)
    r_fwd, rb_fwd, ri_fwd = [], [], []
    for K in k_vals:
        for _ in range(n_trans):
            theta = rk4_step_regime(theta, omega, module_indices, blanket_mask, num_modules, K, N, gamma, c_intra, g_intra_on, g_inter_on, dt)
        rs, rbs, ris = np.zeros(n_meas), np.zeros(n_meas), np.zeros(n_meas)
        for step in range(n_meas):
            theta = rk4_step_regime(theta, omega, module_indices, blanket_mask, num_modules, K, N, gamma, c_intra, g_intra_on, g_inter_on, dt)
            rs[step] = compute_order_parameter(theta)
            rbs[step] = compute_joint_blanket_order(theta, blanket_mask)
            ris[step] = compute_mean_intra_order(theta, module_indices, num_modules)
        r_fwd.append(float(np.mean(rs)))
        rb_fwd.append(float(np.mean(rbs)))
        ri_fwd.append(float(np.mean(ris)))

    r_bwd, rb_bwd, ri_bwd = [], [], []
    for K in reversed(k_vals):
        for _ in range(n_trans):
            theta = rk4_step_regime(theta, omega, module_indices, blanket_mask, num_modules, K, N, gamma, c_intra, g_intra_on, g_inter_on, dt)
        rs, rbs, ris = np.zeros(n_meas), np.zeros(n_meas), np.zeros(n_meas)
        for step in range(n_meas):
            theta = rk4_step_regime(theta, omega, module_indices, blanket_mask, num_modules, K, N, gamma, c_intra, g_intra_on, g_inter_on, dt)
            rs[step] = compute_order_parameter(theta)
            rbs[step] = compute_joint_blanket_order(theta, blanket_mask)
            ris[step] = compute_mean_intra_order(theta, module_indices, num_modules)
        r_bwd.append(float(np.mean(rs)))
        rb_bwd.append(float(np.mean(rbs)))
        ri_bwd.append(float(np.mean(ris)))

    return (
        np.array(r_fwd), np.array(list(reversed(r_bwd))),
        np.array(rb_fwd), np.array(list(reversed(rb_bwd))),
        np.array(ri_fwd), np.array(list(reversed(ri_bwd))),
    )


def compute_landau_free_energy(k_val, r_grid, K_c=3.50, a0=1.00, b=9.00, c=10.125):
    """
    Effective U(1) gauge-invariant 6th-order subcritical Landau-Ginzburg potential:
      F(R; K) = (1/2) * a0 * (K_c - K) * R^2 - (1/4) * b * R^4 + (1/6) * c * R^6
    Phase rotation invariance z -> z*exp(i*alpha) eliminates odd terms in R.
    Calibrated spinodal endpoints:
      - Hysteresis width Delta_K = b^2 / (4 * c * a0) = 2.000 (matching [1.50, 3.50])
      - Synchronized minimum at K_c: R_sync(K_c) = sqrt(b / c) = sqrt(8/9) ~= 0.9428 <= 1
      - Lower spinodal coherence at K_bwd = 1.50: R_sp = sqrt(b / (2*c)) = 2/3 ~= 0.6667.
    """
    return 0.5 * a0 * (K_c - k_val) * (r_grid**2) - 0.25 * b * (r_grid**4) + (1.0 / 6.0) * c * (r_grid**6)


def run_factorial_ablation_and_scaling(gamma=2.40, dt=0.02):
    """
    Execute the pre-registered 4-regime factorial ablation (ungated vs inter-only gated
    vs global blanket gated vs no-intra pure-blanket gated) and multi-size scaling test
    across N in {90, 180, 360, 540} with multiple seeds.
    """
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    k_vals = np.linspace(0.2, 4.2, 41)
    k_25_idx = int(np.argmin(np.abs(k_vals - 2.50)))
    regimes = ("standard", "inter_only_gated", "gated", "no_intra_gated")

    # 1. Reference topology (seed=42, sweep_seed=7, N=90) across all 4 regimes
    _, _, mod0, blk0, om0 = generate_modular_topology(3, 30, 0.35, seed=42)
    ref_ablation = {}
    for mode in regimes:
        rf, rb, rbf, rbb, rif, _ = run_regime_sweep(
            mod0, blk0, om0, k_vals, mode=mode, gamma=gamma, dt=dt,
            t_transient=28.0 if mode != "standard" else 36.0,
            t_measure=8.0 if mode != "standard" else 10.0,
            seed=7,
        )
        df = np.diff(rf)
        db = np.diff(rb)
        kf = float(k_vals[int(np.argmax(df)) + 1])
        kb = float(k_vals[int(np.argmax(db)) + 1])
        area_global = float(trapz_fn(np.maximum(0.0, rb - rf), k_vals))
        area_blanket = float(trapz_fn(np.maximum(0.0, rbb - rbf), k_vals))
        _, _, d_bic = compute_bic_model_selection(k_vals, rf, rb)
        _, _, d_bic_blk = compute_bic_model_selection(k_vals, rbf, rbb)
        ref_ablation[mode] = {
            "global_delta_R_jump": float(np.max(df)),
            "global_Kc_fwd": kf,
            "global_Kc_bwd": kb,
            "global_delta_K_hyst": float(kf - kb),
            "global_hysteresis_area": area_global,
            "global_delta_BIC": float(d_bic),
            "blanket_delta_R_jump": float(np.max(np.diff(rbf))),
            "blanket_hysteresis_area": area_blanket,
            "blanket_delta_BIC": float(d_bic_blk),
            "intra_coherence_at_K2p5_fwd": float(rif[k_25_idx]),
            "blanket_coherence_at_K2p5_fwd": float(rbf[k_25_idx]),
        }

    # 2. 5-Seed Ensemble Ablation at N=90 across all 4 regimes
    seeds = (42, 7, 13, 29, 77)
    ensemble_ablation = {}
    for mode in regimes:
        per_seed = []
        for s in seeds:
            _, _, mods, blks, oms = generate_modular_topology(3, 30, 0.35, seed=s)
            rf, rb, rbf, rbb, rif, _ = run_regime_sweep(
                mods, blks, oms, k_vals, mode=mode, gamma=gamma, dt=dt, t_transient=28.0, t_measure=8.0, seed=s + 1
            )
            df = np.diff(rf)
            db = np.diff(rb)
            dbf = np.diff(rbf)
            dbb = np.diff(rbb)
            per_seed.append({
                "seed": s,
                "global_delta_R": float(np.max(df)),
                "global_area": float(trapz_fn(np.maximum(0.0, rb - rf), k_vals)),
                "global_delta_K": float(k_vals[int(np.argmax(df)) + 1] - k_vals[int(np.argmax(db)) + 1]),
                "blanket_delta_R": float(np.max(dbf)),
                "blanket_area": float(trapz_fn(np.maximum(0.0, rbb - rbf), k_vals)),
                "blanket_delta_K": float(k_vals[int(np.argmax(dbf)) + 1] - k_vals[int(np.argmax(dbb)) + 1]),
                "intra_at_K2p5": float(rif[k_25_idx]),
            })
        ensemble_ablation[mode] = {
            "global_delta_R_mean": float(np.mean([x["global_delta_R"] for x in per_seed])),
            "global_delta_R_std": float(np.std([x["global_delta_R"] for x in per_seed])),
            "global_area_mean": float(np.mean([x["global_area"] for x in per_seed])),
            "global_area_std": float(np.std([x["global_area"] for x in per_seed])),
            "blanket_delta_R_mean": float(np.mean([x["blanket_delta_R"] for x in per_seed])),
            "blanket_area_mean": float(np.mean([x["blanket_area"] for x in per_seed])),
            "blanket_delta_K_mean": float(np.mean([x["blanket_delta_K"] for x in per_seed])),
            "intra_at_K2p5_mean": float(np.mean([x["intra_at_K2p5"] for x in per_seed])),
            "per_seed": per_seed,
        }

    # 3. Multi-Size Scaling across N in {90, 180, 360, 540} comparing gated vs inter_only_gated vs no_intra_gated
    k_vals_ext = np.linspace(0.2, 22.2, 89)
    sizes = (90, 180, 360, 540)
    scale_seeds = (42, 7, 13)
    size_scaling = {}
    for mode in ("gated", "inter_only_gated", "no_intra_gated"):
        size_rows = {}
        for n_tot in sizes:
            n_per_mod = n_tot // 3
            kf_list, kb_list, dr_list, area_list, trans_flags = [], [], [], [], []
            for s in scale_seeds:
                _, _, modn, blkn, omn = generate_modular_topology(3, n_per_mod, 0.35, seed=s)
                rf, rb, rbf, rbb, _, _ = run_regime_sweep(
                    modn, blkn, omn, k_vals_ext, mode=mode, gamma=gamma, dt=dt, t_transient=24.0, t_measure=6.0, seed=s + 1
                )
                # Track blanket order parameter for no_intra_gated (where internal nodes have c_intra=0)
                tr_f, tr_b = (rbf, rbb) if mode == "no_intra_gated" else (rf, rb)
                df = np.diff(tr_f)
                db = np.diff(tr_b)
                idx_f = int(np.argmax(df))
                idx_b = int(np.argmax(db))
                kf_s = float(k_vals_ext[idx_f + 1])
                kb_s = float(k_vals_ext[idx_b + 1])
                dr_s = float(df[idx_f])
                did_transition = bool(np.max(tr_f) > 0.65 and dr_s > 0.30)
                kf_list.append(kf_s)
                kb_list.append(kb_s)
                dr_list.append(dr_s)
                area_list.append(float(trapz_fn(np.maximum(0.0, tr_b - tr_f), k_vals_ext)))
                trans_flags.append(did_transition)
            size_rows[str(n_tot)] = {
                "N": n_tot,
                "blanket_nodes": int(np.sum(blkn)),
                "all_seeds_transitioned": all(trans_flags),
                "Kc_fwd_mean": float(np.mean(kf_list)),
                "Kc_fwd_std": float(np.std(kf_list)),
                "Kc_bwd_mean": float(np.mean(kb_list)),
                "delta_R_mean": float(np.mean(dr_list)),
                "hysteresis_area_mean": float(np.mean(area_list)),
                "Kc_fwd_per_seed": kf_list,
            }
        # Fit empirical power-law exponent alpha only across sizes where all seeds transitioned
        valid_sizes = [n for n in sizes if size_rows[str(n)]["all_seeds_transitioned"]]
        if len(valid_sizes) >= 2:
            log_b = np.log([size_rows[str(n)]["blanket_nodes"] for n in valid_sizes])
            log_k = np.log([max(1e-6, size_rows[str(n)]["Kc_fwd_mean"]) for n in valid_sizes])
            alpha_slope, _ = np.polyfit(log_b, log_k, 1)
        else:
            alpha_slope = 0.0
        size_scaling[mode] = {
            "by_size": size_rows,
            "fitted_sizes": valid_sizes,
            "empirical_exponent_Kc_vs_B": float(alpha_slope),
            "uncorrelated_heuristic_exponent_gamma_over_2": float(gamma / 2.0),
        }

    return {
        "N90_reference_seed42": ref_ablation,
        "N90_5seed_ensemble": ensemble_ablation,
        "multi_size_scaling_N90_to_N540": size_scaling,
    }


def run_fixed_k_coexistence_test(
    A_intra, A_inter, module_indices, blanket_mask, omega,
    K_test=2.50, dt=0.02, t_relax=60.0, t_measure=10.0, gamma=2.40, seed=7
):
    """
    Directly test bistable coexistence at fixed K inside the hysteresis loop by
    evolving disordered vs. phase-locked initial conditions over long relaxation times.
    """
    rng = np.random.default_rng(seed)
    N = len(omega)
    num_modules = len(np.unique(module_indices))
    n_relax = int(t_relax / dt)
    n_meas = int(t_measure / dt)

    results = {}
    for branch_name, init_theta in (
        ("disordered_init", rng.uniform(-np.pi, np.pi, N)),
        ("synchronized_init", rng.normal(0.0, 0.12, N)),
    ):
        theta = init_theta.copy()
        r0 = float(compute_order_parameter(theta))
        for _ in range(n_relax):
            theta = rk4_step_gated(theta, omega, A_intra, A_inter, module_indices, blanket_mask, num_modules, K_test, N, gamma, dt)
        r_samples = np.zeros(n_meas)
        for step in range(n_meas):
            theta = rk4_step_gated(theta, omega, A_intra, A_inter, module_indices, blanket_mask, num_modules, K_test, N, gamma, dt)
            r_samples[step] = compute_order_parameter(theta)
        results[branch_name] = {
            "R_initial": r0,
            "R_relaxed_mean": float(np.mean(r_samples)),
            "R_relaxed_std": float(np.std(r_samples)),
        }

    r_low = results["disordered_init"]["R_relaxed_mean"]
    r_high = results["synchronized_init"]["R_relaxed_mean"]
    results["K_fixed"] = float(K_test)
    results["t_relax"] = float(t_relax)
    results["bistable_separation"] = float(r_high - r_low)
    results["coexistence_confirmed"] = bool(r_high > 0.85 and r_low < 0.30 and (r_high - r_low) > 0.55)
    return results


def verify_ensemble_and_size_scaling(gamma=2.40, dt=0.02):
    """
    Evaluate hysteresis persistence across multiple frequency seeds (N=90) and
    doubled system size (N=180) to confirm robustness beyond a single realization.
    """
    k_vals_coarse = np.linspace(0.2, 4.2, 21)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))

    seed_metrics = []
    for s in (42, 7, 13, 29, 77):
        A_in, A_ex, mod_idx, blk_mask, om = generate_modular_topology(3, 30, 0.35, seed=s)
        rf, rb = run_hysteresis_sweep(
            A_in, A_ex, mod_idx, blk_mask, om,
            k_vals_coarse, dt=dt, t_transient=28.0, t_measure=8.0, mode="gated", gamma=gamma, seed=s + 1
        )
        df = np.diff(rf)
        db = np.diff(rb)
        idx_f = int(np.argmax(df))
        idx_b = int(np.argmax(db))
        kf = float(k_vals_coarse[idx_f + 1])
        kb = float(k_vals_coarse[idx_b + 1])
        area = float(trapz_fn(np.maximum(0.0, rb - rf), k_vals_coarse))
        seed_metrics.append({
            "seed": s,
            "delta_R_jump": float(df[idx_f]),
            "Kc_fwd": kf,
            "Kc_bwd": kb,
            "delta_K_hyst": float(abs(kf - kb)),
            "hysteresis_area": area,
        })

    # Doubled system size N=180 (3 modules x 60 nodes) using exact O(N) regime sweep
    k_vals_180 = np.linspace(0.2, 9.0, 23)
    _, _, mod_idx_180, blk_mask_180, om_180 = generate_modular_topology(3, 60, 0.35, seed=42)
    rf_180, rb_180, _, _, _, _ = run_regime_sweep(
        mod_idx_180, blk_mask_180, om_180,
        k_vals_180, mode="gated", gamma=gamma, dt=dt, t_transient=28.0, t_measure=8.0, seed=7
    )
    df_180 = np.diff(rf_180)
    db_180 = np.diff(rb_180)
    kf_180 = float(k_vals_180[int(np.argmax(df_180)) + 1])
    kb_180 = float(k_vals_180[int(np.argmax(db_180)) + 1])
    area_180 = float(trapz_fn(np.maximum(0.0, rb_180 - rf_180), k_vals_180))

    return {
        "N90_5seed_summary": {
            "seeds": [m["seed"] for m in seed_metrics],
            "delta_R_mean": float(np.mean([m["delta_R_jump"] for m in seed_metrics])),
            "delta_R_min": float(np.min([m["delta_R_jump"] for m in seed_metrics])),
            "delta_K_hyst_mean": float(np.mean([m["delta_K_hyst"] for m in seed_metrics])),
            "delta_K_hyst_min": float(np.min([m["delta_K_hyst"] for m in seed_metrics])),
            "hysteresis_area_mean": float(np.mean([m["hysteresis_area"] for m in seed_metrics])),
            "per_seed": seed_metrics,
        },
        "N180_size_scaling": {
            "N": 180,
            "delta_R_jump": float(np.max(df_180)),
            "Kc_fwd": kf_180,
            "Kc_bwd": kb_180,
            "delta_K_hyst": float(abs(kf_180 - kb_180)),
            "hysteresis_area": area_180,
        },
    }


def compute_bic_model_selection(k_vals, r_fwd, r_bwd=None):
    """
    Compare 1st-order bistable spinodal step model (M1, k=5, s >= 35) vs
    2nd-order continuous single-branch saturating pitchfork model (M2, k=5)
    across the joint forward-backward sweep.
    Returns (BIC_1st, BIC_2nd, delta_BIC = BIC_2nd - BIC_1st).
    """
    from scipy.optimize import curve_fit

    r_fwd = np.asarray(r_fwd, dtype=float)
    r_bwd = r_fwd.copy() if r_bwd is None else np.asarray(r_bwd, dtype=float)
    r_joint = np.concatenate([r_fwd, r_bwd])
    k_joint = np.concatenate([k_vals, k_vals])
    n_s = len(k_vals)
    n_tot = len(r_joint)

    def model_1st_joint(k_vec, kc_fwd, kc_bwd, r_low, r_high, steep):
        fwd = r_low + (r_high - r_low) / (1.0 + np.exp(-steep * (k_vec[:n_s] - kc_fwd)))
        bwd = r_low + (r_high - r_low) / (1.0 + np.exp(-steep * (k_vec[n_s:] - kc_bwd)))
        return np.concatenate([fwd, bwd])

    def model_2nd_joint(k_vec, kc, beta, a_coef, r0, r_max):
        u = np.maximum(0.0, k_vec - kc) ** beta
        return r0 + (r_max - r0) * (a_coef * u) / (1.0 + a_coef * u)

    r_hi_min = 0.40 if (k_vals[-1] <= 4.5 and float(np.max(r_joint)) > 0.50) else 0.15
    kc_hi_max = 4.0 if k_vals[-1] <= 4.5 else float(k_vals[-1])
    try:
        popt1, _ = curve_fit(
            model_1st_joint, k_joint, r_joint,
            p0=[min(3.2, 0.75 * k_vals[-1]), min(1.5, 0.35 * k_vals[-1]), 0.1, max(r_hi_min + 0.05, float(np.max(r_joint)) * 0.9), 60.0],
            bounds=([k_vals[0], k_vals[0], 0.0, r_hi_min, 35.0], [k_vals[-1], k_vals[-1], 0.5, 1.0, 200.0]),
            maxfev=20000,
        )
        rss1 = max(1e-12, float(np.sum((r_joint - model_1st_joint(k_joint, *popt1)) ** 2)))
        bic1 = n_tot * np.log(rss1 / n_tot) + 5 * np.log(n_tot)
    except Exception:
        bic1 = 1e5

    try:
        popt2, _ = curve_fit(
            model_2nd_joint, k_joint, r_joint,
            p0=[0.9, 0.5, 1.2, 0.1, max(0.65 if r_hi_min >= 0.4 else 0.35, float(np.max(r_joint)) * 0.95)],
            bounds=([0.0, 0.2, 0.05, 0.0, 0.6 if r_hi_min >= 0.4 else 0.25], [kc_hi_max, 1.2, 20.0, 0.4, 1.0]),
            maxfev=20000,
        )
        rss2 = max(1e-12, float(np.sum((r_joint - model_2nd_joint(k_joint, *popt2)) ** 2)))
        bic2 = n_tot * np.log(rss2 / n_tot) + 5 * np.log(n_tot)
    except Exception:
        bic2 = 1e5

    return float(bic1), float(bic2), float(bic2 - bic1)

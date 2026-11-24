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

    # Doubled system size N=180 (3 modules x 60 nodes): incoherent R_B^joint ~ |B|^(-1/2)
    # shifts unnormalized G=[R_B^joint]^gamma downward by 2^(-gamma/2) ~= 0.435x, raising K_c^fwd by ~2.3x.
    k_vals_180 = np.linspace(0.2, 9.0, 23)
    A_in_180, A_ex_180, mod_idx_180, blk_mask_180, om_180 = generate_modular_topology(3, 60, 0.35, seed=42)
    rf_180, rb_180 = run_hysteresis_sweep(
        A_in_180, A_ex_180, mod_idx_180, blk_mask_180, om_180,
        k_vals_180, dt=dt, t_transient=28.0, t_measure=8.0, mode="gated", gamma=gamma, seed=7
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

    try:
        popt1, _ = curve_fit(
            model_1st_joint, k_joint, r_joint,
            p0=[3.2, 1.5, 0.1, 0.92, 60.0],
            bounds=([k_vals[0], k_vals[0], 0.0, 0.4, 35.0], [k_vals[-1], k_vals[-1], 0.5, 1.0, 200.0]),
            maxfev=20000,
        )
        rss1 = max(1e-12, float(np.sum((r_joint - model_1st_joint(k_joint, *popt1)) ** 2)))
        bic1 = n_tot * np.log(rss1 / n_tot) + 5 * np.log(n_tot)
    except Exception:
        bic1 = 1e5

    try:
        popt2, _ = curve_fit(
            model_2nd_joint, k_joint, r_joint,
            p0=[0.9, 0.5, 1.2, 0.1, 0.98],
            bounds=([0.0, 0.2, 0.05, 0.0, 0.6], [4.0, 1.2, 20.0, 0.4, 1.0]),
            maxfev=20000,
        )
        rss2 = max(1e-12, float(np.sum((r_joint - model_2nd_joint(k_joint, *popt2)) ** 2)))
        bic2 = n_tot * np.log(rss2 / n_tot) + 5 * np.log(n_tot)
    except Exception:
        bic2 = 1e5

    return float(bic1), float(bic2), float(bic2 - bic1)

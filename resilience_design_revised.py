from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erfc

from computation import predictive_transition_detection

ArrayLike = Union[np.ndarray, list, tuple]
ParamsLike = Union[Mapping[str, Any], Any]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _get_param(params: ParamsLike, name: str, default: Any = None) -> Any:
    if isinstance(params, Mapping):
        return params.get(name, default)
    return getattr(params, name, default)


def _as_array(x: ArrayLike, *, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _as_column(x: ArrayLike, *, dtype=float) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr
    raise ValueError(f"Expected a column vector or 1-D array, got shape {arr.shape}.")


def _scalar(x: ArrayLike) -> float:
    arr = np.asarray(x, dtype=float)
    return float(arr.reshape(-1)[0])


def _info_rate_nats(l: int, n: int, params: Optional[ParamsLike] = None) -> float:
    """Return coding rate in nats/channel-use."""
    if params is None:
        unit = "bits"
    else:
        unit = str(_get_param(params, "INFO_BITS_UNIT", "bits")).lower()

    if unit == "bits":
        return float(l) * math.log(2.0) / float(n)
    if unit == "nats":
        return float(l) / float(n)
    raise ValueError("INFO_BITS_UNIT must be either 'bits' or 'nats'.")


# -----------------------------------------------------------------------------
# Finite-blocklength communication model
# -----------------------------------------------------------------------------

def gaussian_q(x: float) -> float:
    """Gaussian Q-function."""
    return 0.5 * float(erfc(x / math.sqrt(2.0)))


def compute_instantaneous_packet_error(
    gamma: float,
    n: int,
    l: int,
    params: Optional[ParamsLike] = None,
) -> float:
    """
    Instantaneous packet error rate:
        epsilon_k ~= Q( sqrt(n / V(gamma)) * ( C(gamma) - R ) )
    with C(gamma)=ln(1+gamma), V(gamma)=1-(1+gamma)^(-2).
    """
    gamma = float(max(gamma, 0.0))
    n = int(n)
    l = int(l)

    if gamma <= 1e-14:
        return 1.0

    R_nats = _info_rate_nats(l=l, n=n, params=params)
    C_gamma = math.log(1.0 + gamma)
    V_gamma = 1.0 - (1.0 + gamma) ** (-2.0)

    if V_gamma <= 1e-14:
        return 1.0

    arg = math.sqrt(n / V_gamma) * (C_gamma - R_nats)
    eps = gaussian_q(arg)
    return float(np.clip(eps, 0.0, 1.0))


def compute_average_packet_error_closed_form(
    pt: float,
    noise_var: float,
    n: int,
    l: int,
    params: Optional[ParamsLike] = None,
) -> Dict[str, float]:
    """Average packet error rate using the segmented-linear FBL approximation."""
    pt = float(pt)
    noise_var = float(noise_var)
    n = int(n)
    l = int(l)

    if pt <= 0.0 or noise_var <= 0.0:
        raise ValueError("pt and noise_var must be positive.")

    gbar = pt / noise_var
    R_nats = _info_rate_nats(l=l, n=n, params=params)
    phi = math.exp(R_nats) - 1.0
    beta = -math.sqrt(n / (2.0 * math.pi * (math.exp(2.0 * R_nats) - 1.0)))
    v = math.exp(-phi / gbar)

    eps_bar = 1.0 + (beta * gbar - beta * gbar * math.exp(1.0 / (2.0 * beta * gbar)) - 0.5) * v
    eps_bar = float(np.clip(eps_bar, 0.0, 1.0))

    return {
        "pt": pt,
        "noise_var": noise_var,
        "gbar": gbar,
        "R_nats_per_cu": R_nats,
        "phi": phi,
        "beta": beta,
        "v": v,
        "eps_bar": eps_bar,
    }


def compute_average_packet_error_monte_carlo(
    pt: float,
    noise_var: float,
    n: int,
    l: int,
    params: Optional[ParamsLike] = None,
    num_samples: int = 50000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Monte Carlo cross-check of average packet error over Rayleigh fading."""
    pt = float(pt)
    noise_var = float(noise_var)
    n = int(n)
    l = int(l)
    num_samples = int(num_samples)

    rng = np.random.default_rng(seed)
    gbar = pt / noise_var
    h2 = rng.exponential(scale=1.0, size=num_samples)
    gamma = h2 * gbar
    eps = np.array(
        [compute_instantaneous_packet_error(g, n, l, params=params) for g in gamma],
        dtype=float,
    )

    return {
        "pt": pt,
        "noise_var": noise_var,
        "gbar": gbar,
        "num_samples": num_samples,
        "eps_bar": float(np.mean(eps)),
    }


def compute_average_packet_error(pt: float, params: ParamsLike, method: Optional[str] = None) -> Dict[str, float]:
    """Wrapper for average packet error rate."""
    if method is None:
        method = str(_get_param(params, "AVERAGE_EPSILON_METHOD", "closed_form"))

    noise_var = float(_get_param(params, "CHANNEL_NOISE_VAR", 1.0))
    n = int(_get_param(params, "BLOCKLENGTH_N", 128))
    l = int(_get_param(params, "INFO_BITS_L", 64))

    if method == "closed_form":
        out = compute_average_packet_error_closed_form(
            pt=pt,
            noise_var=noise_var,
            n=n,
            l=l,
            params=params,
        )
        out["method"] = "closed_form"
        return out

    if method == "monte_carlo":
        num_samples = int(_get_param(params, "AVERAGE_EPSILON_MC_SAMPLES", 50000))
        seed = _get_param(params, "AVERAGE_EPSILON_MC_SEED", None)
        out = compute_average_packet_error_monte_carlo(
            pt=pt,
            noise_var=noise_var,
            n=n,
            l=l,
            params=params,
            num_samples=num_samples,
            seed=seed,
        )
        out["method"] = "monte_carlo"
        return out

    raise ValueError(f"Unsupported method='{method}'. Use 'closed_form' or 'monte_carlo'.")


# -----------------------------------------------------------------------------
# AoI, recovery, and transition-conditioned blackout formulas
# -----------------------------------------------------------------------------

def compute_pu_star(theta_s: int, eps_r: float, eps_bar: float) -> Dict[str, float]:
    """
    Common-epsilon_r design:
        P(A_s > theta_s) = eps_r.

    Let rho_s = 1 - p_u,s (1 - eps_bar). Then rho_s^theta_s = eps_r, so
        p_u,s^* = (1 - eps_r^(1/theta_s)) / (1 - eps_bar).

    Here eps_r is interpreted as the maximum false-recovery/AoI-tail tolerance,
    not as part of the lead-time reliability product.
    """
    theta_s = int(theta_s)
    eps_r = float(eps_r)
    eps_bar = float(eps_bar)

    if theta_s <= 0:
        raise ValueError("theta_s must be positive.")
    if not (0.0 < eps_r < 1.0):
        return {"pu_star": float("inf"), "rho_s": float("nan"), "eta_s": float("nan"), "feasible": False}

    denom = 1.0 - eps_bar
    if denom <= 0.0:
        return {"pu_star": float("inf"), "rho_s": float("nan"), "eta_s": float("nan"), "feasible": False}

    rho_s = eps_r ** (1.0 / theta_s)
    eta_s = 1.0 - rho_s
    pu_star = eta_s / denom
    feasible = bool(0.0 <= pu_star <= 1.0)

    return {
        "pu_star": float(pu_star),
        "rho_s": float(rho_s),
        "eta_s": float(eta_s),
        "feasible": feasible,
    }


def compute_expected_D_theta(theta_s: int, eta_s: float) -> float:
    """
    E[D_theta,s] with geometric AoI:
        P(A=a) = eta_s (1-eta_s)^a,
        D_theta = (theta_s - A)^+.
    """
    theta_s = int(theta_s)
    eta_s = float(np.clip(eta_s, 0.0, 1.0))

    if theta_s <= 0:
        raise ValueError("theta_s must be positive.")

    q = 1.0 - eta_s
    if eta_s <= 0.0:
        return float(theta_s)
    if eta_s >= 1.0:
        return float(theta_s)

    value = theta_s - q * (1.0 - q ** theta_s) / (1.0 - q)
    return float(value)


def dtheta_pmf_from_geometric_aoi(theta_s: int, eta_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distribution of D_theta = (theta_s - A)^+ under stationary geometric AoI.
    """
    theta_s = int(theta_s)
    eta_s = float(np.clip(eta_s, 0.0, 1.0))

    if theta_s <= 0:
        raise ValueError("theta_s must be positive.")

    if eta_s <= 0.0:
        return np.array([float(theta_s)], dtype=float), np.array([1.0], dtype=float)

    if eta_s >= 1.0:
        return np.array([float(theta_s), 0.0], dtype=float), np.array([1.0, 0.0], dtype=float)

    q = 1.0 - eta_s
    d_vals = []
    probs = []
    for a in range(theta_s):
        d_vals.append(float(theta_s - a))
        probs.append(eta_s * (q ** a))

    d_vals.append(0.0)
    probs.append(q ** theta_s)

    d_vals_arr = np.asarray(d_vals, dtype=float)
    probs_arr = np.asarray(probs, dtype=float)
    probs_arr /= np.sum(probs_arr)
    return d_vals_arr, probs_arr


def discrete_weibull_pmf(
    lam: float,
    kappa: float,
    tail_tol: float = 1e-10,
    max_tau: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    P(t_h=tau)=exp(-(tau/lambda)^kappa)-exp(-((tau+1)/lambda)^kappa), tau=0,1,...
    """
    lam = float(lam)
    kappa = float(kappa)

    if lam <= 0.0 or kappa <= 0.0:
        raise ValueError("lam and kappa must be positive.")

    taus = []
    probs = []
    for tau in range(max_tau + 1):
        p = math.exp(-((tau / lam) ** kappa)) - math.exp(-(((tau + 1) / lam) ** kappa))
        taus.append(tau)
        probs.append(p)
        tail = math.exp(-(((tau + 1) / lam) ** kappa))
        if tail < tail_tol:
            break

    taus_arr = np.asarray(taus, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)
    probs_arr /= np.sum(probs_arr)
    return taus_arr, probs_arr


def discrete_weibull_mean(
    lam: float,
    kappa: float,
    tail_tol: float = 1e-12,
    max_tau: int = 100000,
) -> float:
    """Mean recovery time E[t_h] using the survival identity."""
    lam = float(lam)
    kappa = float(kappa)

    if lam <= 0.0 or kappa <= 0.0:
        raise ValueError("lam and kappa must be positive.")

    total = 0.0
    for tau in range(1, int(max_tau) + 1):
        tail = math.exp(-((float(tau) / lam) ** kappa))
        total += tail
        if tail < tail_tol:
            break
    return float(total)


def geometric_pmf_from_qss(
    q_ss: float,
    tail_tol: float = 1e-10,
    max_t: int = 100000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sojourn time T_s ~ Geom(1-q_ss), support {1,2,...}."""
    q_ss = float(q_ss)
    if not (0.0 <= q_ss < 1.0):
        raise ValueError("q_ss must lie in [0, 1).")
    if q_ss == 0.0:
        return np.array([1], dtype=int), np.array([1.0], dtype=float)

    ts = []
    probs = []
    for n in range(1, max_t + 1):
        p = (q_ss ** (n - 1)) * (1.0 - q_ss)
        ts.append(n)
        probs.append(p)
        if q_ss ** n < tail_tol:
            break

    ts_arr = np.asarray(ts, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)
    probs_arr /= np.sum(probs_arr)
    return ts_arr, probs_arr


def compute_blackout_duration_pmf_from_eps_r(
    theta_s: int,
    eps_r: float,
    weibull_lambda: float,
    weibull_kappa: float,
    tail_tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Build the PMF of B_s = D_theta,s + t_h using common eps_r.
    """
    theta_s = int(theta_s)
    eps_r = float(eps_r)
    rho_s = eps_r ** (1.0 / theta_s)
    eta_s = 1.0 - rho_s

    D_vals, D_pmf = dtheta_pmf_from_geometric_aoi(theta_s=theta_s, eta_s=eta_s)
    th_vals, th_pmf = discrete_weibull_pmf(
        lam=weibull_lambda,
        kappa=weibull_kappa,
        tail_tol=tail_tol,
    )

    mass: Dict[int, float] = {}
    for d, pd in zip(D_vals, D_pmf):
        d_int = int(round(float(d)))
        for th, pth in zip(th_vals, th_pmf):
            b = d_int + int(th)
            mass[b] = mass.get(b, 0.0) + float(pd) * float(pth)

    B_vals = np.asarray(sorted(mass.keys()), dtype=float)
    B_pmf = np.asarray([mass[int(b)] for b in B_vals], dtype=float)
    B_pmf /= np.sum(B_pmf)

    return {
        "B_vals": B_vals,
        "B_pmf": B_pmf,
        "D_vals": D_vals,
        "D_pmf": D_pmf,
        "th_vals": th_vals.astype(float),
        "th_pmf": th_pmf,
        "rho_s": float(rho_s),
        "eta_s": float(eta_s),
        "E_D_theta_s": float(np.sum(D_vals * D_pmf)),
        "E_th": float(np.sum(th_vals * th_pmf)),
        "E_B_s": float(np.sum(B_vals * B_pmf)),
    }


def _paired_samples(T_samples: ArrayLike, I_samples: ArrayLike, *, sample_cap: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Clean and optionally subsample paired (T,I) samples."""
    T = np.asarray(T_samples, dtype=float).reshape(-1)
    I = np.asarray(I_samples, dtype=float).reshape(-1)
    m = min(len(T), len(I))
    T = T[:m]
    I = I[:m]
    mask = np.isfinite(T) & np.isfinite(I) & (T > 0.0)
    T = T[mask]
    I = I[mask]
    I = np.maximum(I, 0.0)

    if len(T) == 0:
        raise ValueError("No valid paired T/I samples are available.")

    sample_cap = int(sample_cap)
    if sample_cap > 0 and len(T) > sample_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(T), size=sample_cap, replace=False)
        T = T[idx]
        I = I[idx]
    return T.astype(float), I.astype(float)


def _mean_min_one_x_over_T(x_values: np.ndarray, T_current: np.ndarray) -> np.ndarray:
    """
    For each x, compute empirical E_T[min(1, x^+/T)].

    This avoids an O(N*M) outer product in Lambda_s.
    """
    x = np.asarray(x_values, dtype=float)
    T = np.sort(np.asarray(T_current, dtype=float).reshape(-1))
    T = T[T > 0.0]
    if len(T) == 0:
        raise ValueError("T_current must contain positive samples.")

    invT = 1.0 / T
    prefix_inv = np.concatenate(([0.0], np.cumsum(invT)))
    total_inv = float(prefix_inv[-1])
    idx = np.searchsorted(T, x, side="right")
    idx = np.clip(idx, 0, len(T))
    sum_inv_gt = total_inv - prefix_inv[idx]
    out = (idx.astype(float) + np.maximum(x, 0.0) * sum_inv_gt) / float(len(T))
    out = np.where(x > 0.0, out, 0.0)
    return np.clip(out, 0.0, 1.0)


def compute_gamma_from_samples(
    T_samples: ArrayLike,
    I_samples: ArrayLike,
    B_vals: np.ndarray,
    B_pmf: np.ndarray,
    *,
    sample_cap: int = 3000,
    seed: int = 123,
) -> Dict[str, float]:
    """
    Gamma_s = E[min{B_s, (T_s-I_s)^+}/T_s]
    using paired empirical (T_s,I_s) samples and analytical B_s PMF.
    """
    T, I = _paired_samples(T_samples, I_samples, sample_cap=sample_cap, seed=seed)
    residual = np.maximum(T - I, 0.0)

    term = np.zeros_like(T, dtype=float)
    for b, pb in zip(B_vals, B_pmf):
        term += float(pb) * np.minimum(float(b), residual) / T

    return {
        "Gamma_s": float(np.clip(np.mean(term), 0.0, 1.0)),
        "num_current_samples": int(len(T)),
        "E_residual_current": float(np.mean(residual)),
    }


def compute_lambda_next_transition_from_samples(
    T_current_samples: ArrayLike,
    T_next_samples: ArrayLike,
    I_next_samples: ArrayLike,
    B_vals: np.ndarray,
    B_pmf: np.ndarray,
    *,
    sample_cap: int = 3000,
    seed: int = 123,
) -> Dict[str, float]:
    """
    Lambda_s = E[min{1, [B_s-(T_bar-I_bar)]^+/T_s}].

    This is the probability that a disruption starting in the current state-s
    sojourn also blocks the next transition update in the opposite state.
    """
    Tcur = np.asarray(T_current_samples, dtype=float).reshape(-1)
    Tcur = Tcur[np.isfinite(Tcur) & (Tcur > 0.0)]
    if len(Tcur) == 0:
        raise ValueError("No valid current T samples are available.")

    sample_cap = int(sample_cap)
    if sample_cap > 0 and len(Tcur) > sample_cap:
        rng = np.random.default_rng(seed)
        Tcur = Tcur[rng.choice(len(Tcur), size=sample_cap, replace=False)]

    Tnext, Inext = _paired_samples(T_next_samples, I_next_samples, sample_cap=sample_cap, seed=seed + 17)
    residual_next = np.maximum(Tnext - Inext, 0.0)

    lambda_value = 0.0
    for b, pb in zip(B_vals, B_pmf):
        x = float(b) - residual_next
        lambda_value += float(pb) * float(np.mean(_mean_min_one_x_over_T(x, Tcur)))

    return {
        "Lambda_s": float(np.clip(lambda_value, 0.0, 1.0)),
        "num_current_samples": int(len(Tcur)),
        "num_next_samples": int(len(Tnext)),
        "E_residual_next": float(np.mean(residual_next)),
    }


def compute_transition_blackout_hit_probability(
    theta0: int,
    theta1: int,
    eps_r: float,
    pr: float,
    predictive_stats: Dict[str, Any],
    weibull_lambda: float,
    weibull_kappa: float,
    *,
    sample_cap: int = 3000,
    sample_seed: int = 123,
    tail_tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Transition-conditioned blackout-hit probability for one-shot P updates.

    Disruption model:
      - one disruption opportunity in each state sojourn;
      - it occurs with probability pr;
      - if it occurs, its start time is uniform in that sojourn.

    Terms:
      Gamma_s: disruption in state s blocks the transition out of state s.
      Lambda_s: disruption in state s lasts into the next state and blocks the
                next transition update.

    For transitions out of state s:
      p_hit,s = 1 - (1-pr*Gamma_s)(1-pr*Lambda_bar_s).
    """
    eps_r = float(eps_r)
    pr = float(pr)
    if not (0.0 <= pr <= 1.0):
        raise ValueError("pr must lie in [0,1].")
    if not (0.0 < eps_r < 1.0):
        raise ValueError("eps_r must lie in (0,1).")

    T0 = np.asarray(predictive_stats["T_0_samples"], dtype=float)
    I0 = np.asarray(predictive_stats["I_0_samples"], dtype=float)
    T1 = np.asarray(predictive_stats["T_1_samples"], dtype=float)
    I1 = np.asarray(predictive_stats["I_1_samples"], dtype=float)

    B0_info = compute_blackout_duration_pmf_from_eps_r(
        theta_s=theta0,
        eps_r=eps_r,
        weibull_lambda=weibull_lambda,
        weibull_kappa=weibull_kappa,
        tail_tol=tail_tol,
    )
    B1_info = compute_blackout_duration_pmf_from_eps_r(
        theta_s=theta1,
        eps_r=eps_r,
        weibull_lambda=weibull_lambda,
        weibull_kappa=weibull_kappa,
        tail_tol=tail_tol,
    )

    gamma0 = compute_gamma_from_samples(
        T0, I0, B0_info["B_vals"], B0_info["B_pmf"], sample_cap=sample_cap, seed=sample_seed
    )
    gamma1 = compute_gamma_from_samples(
        T1, I1, B1_info["B_vals"], B1_info["B_pmf"], sample_cap=sample_cap, seed=sample_seed + 1
    )

    # Lambda_s uses B_s from disruption in current state s and the next transition
    # samples from the opposite state.
    lambda0 = compute_lambda_next_transition_from_samples(
        T_current_samples=T0,
        T_next_samples=T1,
        I_next_samples=I1,
        B_vals=B0_info["B_vals"],
        B_pmf=B0_info["B_pmf"],
        sample_cap=sample_cap,
        seed=sample_seed + 2,
    )
    lambda1 = compute_lambda_next_transition_from_samples(
        T_current_samples=T1,
        T_next_samples=T0,
        I_next_samples=I0,
        B_vals=B1_info["B_vals"],
        B_pmf=B1_info["B_pmf"],
        sample_cap=sample_cap,
        seed=sample_seed + 3,
    )

    Gamma_0 = float(gamma0["Gamma_s"])
    Gamma_1 = float(gamma1["Gamma_s"])
    Lambda_0 = float(lambda0["Lambda_s"])
    Lambda_1 = float(lambda1["Lambda_s"])

    p_hit_0 = 1.0 - (1.0 - pr * Gamma_0) * (1.0 - pr * Lambda_1)
    p_hit_1 = 1.0 - (1.0 - pr * Gamma_1) * (1.0 - pr * Lambda_0)
    p_hit = 0.5 * (p_hit_0 + p_hit_1)
    p_hit_linear = 0.5 * pr * (Gamma_0 + Gamma_1 + Lambda_0 + Lambda_1)

    return {
        "p_hit_P": float(np.clip(p_hit, 0.0, 1.0)),
        "p_hit_P_linear": float(np.clip(p_hit_linear, 0.0, 1.0)),
        "p_hit_state0": float(np.clip(p_hit_0, 0.0, 1.0)),
        "p_hit_state1": float(np.clip(p_hit_1, 0.0, 1.0)),
        "Gamma_0": Gamma_0,
        "Gamma_1": Gamma_1,
        "Lambda_0": Lambda_0,
        "Lambda_1": Lambda_1,
        "B0": float(B0_info["E_B_s"]),
        "B1": float(B1_info["E_B_s"]),
        "E_D_theta_0": float(B0_info["E_D_theta_s"]),
        "E_D_theta_1": float(B1_info["E_D_theta_s"]),
        "E_th": float(B0_info["E_th"]),
        "rho0": float(B0_info["rho_s"]),
        "rho1": float(B1_info["rho_s"]),
        "eta0_design": float(B0_info["eta_s"]),
        "eta1_design": float(B1_info["eta_s"]),
        "gamma0_info": gamma0,
        "gamma1_info": gamma1,
        "lambda0_info": lambda0,
        "lambda1_info": lambda1,
        "model": "transition_conditioned_two_transition_per_sojourn",
    }


def compute_transition_miss_probability(p_hit_P: float, eps_bar: float) -> float:
    """epsilon_P = p_hit_P + (1-p_hit_P) eps_bar."""
    p_hit_P = float(np.clip(p_hit_P, 0.0, 1.0))
    eps_bar = float(np.clip(eps_bar, 0.0, 1.0))
    return float(np.clip(p_hit_P + (1.0 - p_hit_P) * eps_bar, 0.0, 1.0))


def find_min_pt_for_packet_error(
    eps_target: float,
    params: ParamsLike,
    pt_min: float,
    pt_max: float,
    method: Optional[str] = None,
    *,
    max_iter: int = 80,
) -> Optional[float]:
    """
    Find the smallest pt in [pt_min, pt_max] such that eps_bar(pt) <= eps_target.
    Uses monotonic bisection; returns None if no feasible pt exists in the interval.
    """
    eps_target = float(eps_target)
    pt_min = float(pt_min)
    pt_max = float(pt_max)

    if eps_target <= 0.0:
        return None
    if eps_target >= 1.0:
        return pt_min

    e_lo = float(compute_average_packet_error(pt_min, params=params, method=method)["eps_bar"])
    if e_lo <= eps_target:
        return pt_min

    e_hi = float(compute_average_packet_error(pt_max, params=params, method=method)["eps_bar"])
    if e_hi > eps_target:
        return None

    lo = pt_min
    hi = pt_max
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        e_mid = float(compute_average_packet_error(mid, params=params, method=method)["eps_bar"])
        if e_mid <= eps_target:
            hi = mid
        else:
            lo = mid
    return float(hi)


# -----------------------------------------------------------------------------
# Sensor-only Monte Carlo for predictive horizon moments
# -----------------------------------------------------------------------------

def _sample_true_process_step(
    x_true: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    w = rng.multivariate_normal(mean=np.zeros(A.shape[0]), cov=Q).reshape(-1, 1)
    return A @ x_true + w


def _sample_measurement(
    x_true: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    m = C.shape[0]
    v = rng.multivariate_normal(mean=np.zeros(m), cov=R).reshape(-1, 1)
    return C @ x_true + v


def _sensor_kf_update(
    x_hat: np.ndarray,
    P: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x_pred = A @ x_hat
    P_pred = A @ P @ A.T + Q

    S = C @ P_pred @ C.T + R
    K = P_pred @ C.T @ np.linalg.inv(S)

    innovation = y - C @ x_pred
    x_new = x_pred + K @ innovation

    I = np.eye(A.shape[0])
    P_new = (I - K @ C) @ P_pred @ (I - K @ C).T + K @ R @ K.T
    return x_new, P_new


def estimate_predictive_horizon_moments(
    params: ParamsLike,
    derived: Dict[str, Any],
    num_transitions_per_state: Optional[int] = None,
    burn_in: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate predictive-horizon statistics using sensor-side simulation only.
    Transitions are defined by the sensor-side operational decision.
    """
    A = _as_array(_get_param(params, "A"))
    C = _as_array(_get_param(params, "C"))
    Q = _as_array(_get_param(params, "Q"))
    R = _as_array(_get_param(params, "R"))
    c = _as_column(_get_param(params, "c"))
    Delta = float(_get_param(params, "Delta"))
    alpha_fp = float(_get_param(params, "ALPHA_FP", 0.05))
    alpha_fn = float(_get_param(params, "ALPHA_FN", 0.05))
    ell = int(_get_param(params, "LOOKAHEAD_ELL", 5))
    xi = float(derived["steady_state_benchmark"]["xi"])

    n_state = A.shape[0]
    seed = int(_get_param(params, "I_MONTE_CARLO_SEED", 42))
    rng = np.random.default_rng(seed)

    if num_transitions_per_state is None:
        num_transitions_per_state = int(_get_param(params, "NUM_I_MONTE_CARLO", 3000))
    else:
        num_transitions_per_state = int(num_transitions_per_state)

    if burn_in is None:
        burn_in = int(_get_param(params, "I_MONTE_CARLO_BURN_IN", 1000))
    else:
        burn_in = int(burn_in)

    if max_steps is None:
        max_steps = int(_get_param(params, "I_MONTE_CARLO_MAX_STEPS", 500000))
    else:
        max_steps = int(max_steps)

    x_true0 = _get_param(params, "X_TRUE0", np.zeros((n_state, 1)))
    x_s0 = _get_param(params, "X_S0", np.zeros((n_state, 1)))
    P_s0 = _get_param(params, "P_S0", np.eye(n_state))

    x_true = _as_column(x_true0).copy()
    x_hat = _as_column(x_s0).copy()
    P = _as_array(P_s0).copy()

    for _ in range(burn_in):
        x_true = _sample_true_process_step(x_true, A, Q, rng)
        y = _sample_measurement(x_true, C, R, rng)
        x_hat, P = _sensor_kf_update(x_hat, P, y, A, C, Q, R)

    def sensor_decision(xh: np.ndarray) -> int:
        return 1 if _scalar(c.T @ xh) >= xi else 0

    current_state = sensor_decision(x_hat)
    sojourn_start = 0
    first_prediction_time: Optional[int] = None

    I_0 = []
    I_1 = []
    T_0 = []
    T_1 = []

    trans_00 = 0
    trans_01 = 0
    trans_10 = 0
    trans_11 = 0
    count_state_0 = 0
    count_state_1 = 0

    step_index = 0

    while (len(I_0) < num_transitions_per_state or len(I_1) < num_transitions_per_state) and step_index < max_steps:
        previous_state = current_state

        if previous_state == 0:
            count_state_0 += 1
        else:
            count_state_1 += 1

        if first_prediction_time is None:
            pred = predictive_transition_detection(
                x_hat_sensor=x_hat,
                P_sensor=P,
                A=A,
                Q=Q,
                c=c,
                Delta=Delta,
                alpha_fp=alpha_fp,
                alpha_fn=alpha_fn,
                previous_decision=previous_state,
                ell=ell,
                xi=xi,
            )
            if pred.get("found_transition") and pred.get("predicted_horizon") is not None:
                first_prediction_time = step_index

        x_true = _sample_true_process_step(x_true, A, Q, rng)
        y = _sample_measurement(x_true, C, R, rng)
        x_hat, P = _sensor_kf_update(x_hat, P, y, A, C, Q, R)
        step_index += 1
        current_state = sensor_decision(x_hat)

        if previous_state == 0 and current_state == 0:
            trans_00 += 1
        elif previous_state == 0 and current_state == 1:
            trans_01 += 1
        elif previous_state == 1 and current_state == 0:
            trans_10 += 1
        elif previous_state == 1 and current_state == 1:
            trans_11 += 1

        if current_state != previous_state:
            sojourn_length = step_index - sojourn_start
            if first_prediction_time is None:
                I_val = 0
            else:
                I_val = max(step_index - first_prediction_time, 0)

            if previous_state == 0:
                if len(I_0) < num_transitions_per_state:
                    I_0.append(I_val)
                    T_0.append(sojourn_length)
            else:
                if len(I_1) < num_transitions_per_state:
                    I_1.append(I_val)
                    T_1.append(sojourn_length)

            sojourn_start = step_index
            first_prediction_time = None

    if len(I_0) == 0 and len(I_1) == 0:
        raise RuntimeError(
            "Failed to collect predictive-horizon samples for both states. "
            "Increase NUM_I_MONTE_CARLO and/or I_MONTE_CARLO_MAX_STEPS."
        )

    denom0 = trans_00 + trans_01
    denom1 = trans_10 + trans_11
    q00_hat = trans_00 / denom0 if denom0 > 0 else float(derived["markov_surrogate"]["q00"])
    q01_hat = trans_01 / denom0 if denom0 > 0 else float(derived["markov_surrogate"]["q01"])
    q11_hat = trans_11 / denom1 if denom1 > 0 else float(derived["markov_surrogate"]["q11"])
    q10_hat = trans_10 / denom1 if denom1 > 0 else float(derived["markov_surrogate"]["q10"])

    total_count = count_state_0 + count_state_1
    pi0_hat = count_state_0 / total_count if total_count > 0 else float(derived["markov_chain_statistics"]["pi0"])
    pi1_hat = count_state_1 / total_count if total_count > 0 else float(derived["markov_chain_statistics"]["pi1"])

    I_0_arr = np.asarray(I_0, dtype=float)
    I_1_arr = np.asarray(I_1, dtype=float)
    T_0_arr = np.asarray(T_0, dtype=float)
    T_1_arr = np.asarray(T_1, dtype=float)

    return {
        "I_0_samples": I_0_arr,
        "I_1_samples": I_1_arr,
        "T_0_samples": T_0_arr,
        "T_1_samples": T_1_arr,
        "E_I_0": float(np.mean(I_0_arr)) if len(I_0_arr) else 0.0,
        "E_I_1": float(np.mean(I_1_arr)) if len(I_1_arr) else 0.0,
        "E_T_0_sensor": float(np.mean(T_0_arr)) if len(T_0_arr) else float("nan"),
        "E_T_1_sensor": float(np.mean(T_1_arr)) if len(T_1_arr) else float("nan"),
        "num_samples_0": len(I_0_arr),
        "num_samples_1": len(I_1_arr),
        "burn_in": burn_in,
        "seed": seed,
        "sensor_markov": {
            "q00": float(q00_hat),
            "q01": float(q01_hat),
            "q10": float(q10_hat),
            "q11": float(q11_hat),
            "pi0": float(pi0_hat),
            "pi1": float(pi1_hat),
            "trans_00": int(trans_00),
            "trans_01": int(trans_01),
            "trans_10": int(trans_10),
            "trans_11": int(trans_11),
        },
    }


def add_q_power_moments_to_predictive_horizon_stats(
    predictive_stats: Dict[str, Any],
    q00: float,
    q11: float,
) -> Dict[str, Any]:
    """Add E[q00^I0] and E[q11^I1] once q00 and q11 are known."""
    I_0 = np.asarray(predictive_stats["I_0_samples"], dtype=float)
    I_1 = np.asarray(predictive_stats["I_1_samples"], dtype=float)

    out = dict(predictive_stats)
    out["E_q00_pow_I0"] = float(np.mean(q00 ** I_0)) if len(I_0) else 1.0
    out["E_q11_pow_I1"] = float(np.mean(q11 ** I_1)) if len(I_1) else 1.0
    return out


# -----------------------------------------------------------------------------
# Average transmission rate and objective
# -----------------------------------------------------------------------------

def compute_average_rate(
    pu0: float,
    pu1: float,
    E_q00_pow_I0: float,
    E_q11_pow_I1: float,
    q00: float,
    q11: float,
    q01: float,
    q10: float,
) -> float:
    """
    Average transmission rate:
        r = [2 + pu0 E[q00^I0] q00/q01 + pu1 E[q11^I1] q11/q10]
            / [q01^{-1} + q10^{-1}].
    """
    numerator = 2.0 + pu0 * E_q00_pow_I0 * (q00 / q01) + pu1 * E_q11_pow_I1 * (q11 / q10)
    denominator = (1.0 / q01) + (1.0 / q10)
    return float(numerator / denominator)


def compute_energy_objective(pt: float, n_factor: int, average_rate: float) -> float:
    """Objective: n_factor * pt * average_rate."""
    return float(n_factor * pt * average_rate)


def _select_design_markov_stats(
    derived: Dict[str, Any],
    predictive_stats: Dict[str, Any],
    params: ParamsLike,
) -> Dict[str, float]:
    """Select Markov statistics used in design."""
    source = str(_get_param(params, "DESIGN_MARKOV_SOURCE", "sensor_decision")).lower()

    if source == "sensor_decision":
        sm = predictive_stats.get("sensor_markov", {})
        return {
            "source": "sensor_decision",
            "q00": float(sm.get("q00", derived["markov_surrogate"]["q00"])),
            "q01": float(sm.get("q01", derived["markov_surrogate"]["q01"])),
            "q10": float(sm.get("q10", derived["markov_surrogate"]["q10"])),
            "q11": float(sm.get("q11", derived["markov_surrogate"]["q11"])),
            "pi0": float(sm.get("pi0", derived["markov_chain_statistics"]["pi0"])),
            "pi1": float(sm.get("pi1", derived["markov_chain_statistics"]["pi1"])),
        }

    if source == "derived":
        return {
            "source": "derived",
            "q00": float(derived["markov_surrogate"]["q00"]),
            "q01": float(derived["markov_surrogate"]["q01"]),
            "q10": float(derived["markov_surrogate"]["q10"]),
            "q11": float(derived["markov_surrogate"]["q11"]),
            "pi0": float(derived["markov_chain_statistics"]["pi0"]),
            "pi1": float(derived["markov_chain_statistics"]["pi1"]),
        }

    raise ValueError("DESIGN_MARKOV_SOURCE must be 'sensor_decision' or 'derived'.")


# -----------------------------------------------------------------------------
# Revised design evaluation and solver
# -----------------------------------------------------------------------------

def evaluate_resilience_design(
    pt: float,
    theta0: int,
    theta1: int,
    derived: Dict[str, Any],
    params: ParamsLike,
    predictive_stats: Dict[str, Any],
    *,
    precomputed_hit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate one candidate (pt, theta0, theta1) under the revised problem.

    Revised reliability model:
        eps_P = p_hit^P + (1-p_hit^P) eps_bar(pt) <= eps_l.

    eps_r is read from PARAMETERS.py and used as a common false-recovery/AoI-tail
    tolerance for both states:
        P(A_s > theta_s) <= eps_r,  s=0,1.

    No fixed point is used. eps_r is not calculated from eps_l.
    """
    theta0 = int(theta0)
    theta1 = int(theta1)
    pt = float(pt)

    markov = _select_design_markov_stats(derived, predictive_stats, params)
    pi0 = float(markov["pi0"])
    pi1 = float(markov["pi1"])
    q00 = float(markov["q00"])
    q11 = float(markov["q11"])
    q01 = float(markov["q01"])
    q10 = float(markov["q10"])

    pr = float(_get_param(params, "P_R", 0.05))
    eps_l = float(_get_param(params, "EPSILON_L", 0.05))
    eps_r = float(_get_param(params, "EPSILON_R", 0.05))
    n_block = int(_get_param(params, "BLOCKLENGTH_N", 128))
    objective_factor = int(_get_param(params, "OBJECTIVE_BLOCKLENGTH_FACTOR", 1))
    lam = float(_get_param(params, "TH_RECOVERY_LAMBDA", 4.0))
    kappa = float(_get_param(params, "TH_RECOVERY_KAPPA", 2.0))
    average_eps_method = str(_get_param(params, "AVERAGE_EPSILON_METHOD", "closed_form"))
    hit_sample_cap = int(_get_param(params, "PHIT_SAMPLE_CAP", 3000))
    hit_sample_seed = int(_get_param(params, "PHIT_SAMPLE_SEED", 123))
    tail_tol = float(_get_param(params, "DESIGN_TAIL_TOL", 1e-10))

    if theta0 <= 0 or theta1 <= 0:
        return {"feasible": False, "reason": "theta_not_positive", "pt": pt, "theta0": theta0, "theta1": theta1}

    if not (0.0 < eps_r < 1.0):
        return {"feasible": False, "reason": "eps_r_must_be_in_0_1", "pt": pt, "theta0": theta0, "theta1": theta1, "eps_r": eps_r}

    if not (0.0 < eps_l < 1.0):
        return {"feasible": False, "reason": "eps_l_must_be_in_0_1", "pt": pt, "theta0": theta0, "theta1": theta1, "eps_l": eps_l}

    avg_eps_info = compute_average_packet_error(pt=pt, params=params, method=average_eps_method)
    eps_bar = float(avg_eps_info["eps_bar"])

    pu0_info = compute_pu_star(theta_s=theta0, eps_r=eps_r, eps_bar=eps_bar)
    pu1_info = compute_pu_star(theta_s=theta1, eps_r=eps_r, eps_bar=eps_bar)
    if not pu0_info["feasible"] or not pu1_info["feasible"]:
        return {
            "feasible": False,
            "reason": "pu_star_infeasible",
            "pt": pt,
            "theta0": theta0,
            "theta1": theta1,
            "eps_bar": eps_bar,
            "eps_r": eps_r,
            "pu0_info": pu0_info,
            "pu1_info": pu1_info,
        }

    if precomputed_hit is None:
        hit_info = compute_transition_blackout_hit_probability(
            theta0=theta0,
            theta1=theta1,
            eps_r=eps_r,
            pr=pr,
            predictive_stats=predictive_stats,
            weibull_lambda=lam,
            weibull_kappa=kappa,
            sample_cap=hit_sample_cap,
            sample_seed=hit_sample_seed,
            tail_tol=tail_tol,
        )
        hit_info = apply_p_hit_safety_guard(hit_info, params)
    else:
        hit_info = precomputed_hit

    p_hit_P = float(hit_info["p_hit_P"])
    eps_P = compute_transition_miss_probability(p_hit_P=p_hit_P, eps_bar=eps_bar)

    eps_P_state0 = compute_transition_miss_probability(
        p_hit_P=hit_info["p_hit_state0"],
        eps_bar=eps_bar,
    )

    eps_P_state1 = compute_transition_miss_probability(
        p_hit_P=hit_info["p_hit_state1"],
        eps_bar=eps_bar,
    )

    require_statewise = bool(_get_param(params, "REQUIRE_STATEWISE_LEAD_TIME", True))

    eps_P_state0 = compute_transition_miss_probability(
        p_hit_P=float(hit_info["p_hit_state0"]),
        eps_bar=eps_bar,
    )
    eps_P_state1 = compute_transition_miss_probability(
        p_hit_P=float(hit_info["p_hit_state1"]),
        eps_bar=eps_bar,
    )

    if eps_P > eps_l + 1e-12:
        return {
            "feasible": False,
            "reason": "overall_transition_miss_constraint_violated",
            "pt": pt,
            "theta0": theta0,
            "theta1": theta1,
            "eps_bar": eps_bar,
            "eps_l": eps_l,
            "eps_r": eps_r,
            "eps_P": eps_P,
            "eps_P_state0": eps_P_state0,
            "eps_P_state1": eps_P_state1,
            "p_hit_P": p_hit_P,
            "hit_info": hit_info,
        }

    if require_statewise and max(eps_P_state0, eps_P_state1) > eps_l + 1e-12:
        return {
            "feasible": False,
            "reason": "statewise_transition_miss_constraint_violated",
            "pt": pt,
            "theta0": theta0,
            "theta1": theta1,
            "eps_bar": eps_bar,
            "eps_l": eps_l,
            "eps_r": eps_r,
            "eps_P": eps_P,
            "eps_P_state0": eps_P_state0,
            "eps_P_state1": eps_P_state1,
            "p_hit_P": p_hit_P,
            "p_hit_state0": float(hit_info["p_hit_state0"]),
            "p_hit_state1": float(hit_info["p_hit_state1"]),
            "hit_info": hit_info,
        }

    pu0 = float(pu0_info["pu_star"])
    pu1 = float(pu1_info["pu_star"])
    eta0 = pu0 * (1.0 - eps_bar)
    eta1 = pu1 * (1.0 - eps_bar)

    avg_rate = compute_average_rate(
        pu0=pu0,
        pu1=pu1,
        E_q00_pow_I0=float(predictive_stats["E_q00_pow_I0"]),
        E_q11_pow_I1=float(predictive_stats["E_q11_pow_I1"]),
        q00=q00,
        q11=q11,
        q01=q01,
        q10=q10,
    )
    J = compute_energy_objective(pt=pt, n_factor=objective_factor, average_rate=avg_rate)

    if p_hit_P < 1.0:
        eps_bar_max_from_lead = (eps_l - p_hit_P) / (1.0 - p_hit_P)
    else:
        eps_bar_max_from_lead = -float("inf")

    return {
        "feasible": True,
        "reason": "ok",
        "pt": pt,
        "theta0": theta0,
        "theta1": theta1,
        "eps_bar": eps_bar,
        "eps_l": eps_l,
        "eps_r": eps_r,
        "alpha_FR": eps_r,
        "eps_P": eps_P,
        "eps_transition_miss": eps_P,
        # Backward-compatible aliases; eps_d is now the real transition miss prob.
        "eps_d": eps_P,
        "p_hit_P": p_hit_P,
        "p_blk": p_hit_P,
        "p_hit_state0": float(hit_info["p_hit_state0"]),
        "p_hit_state1": float(hit_info["p_hit_state1"]),
        "eps_bar_max_from_lead": float(eps_bar_max_from_lead),
        "pu0_star": pu0,
        "pu1_star": pu1,
        "rho0": float(pu0_info["rho_s"]),
        "rho1": float(pu1_info["rho_s"]),
        "eta0": eta0,
        "eta1": eta1,
        "E_D_theta_0": float(hit_info["E_D_theta_0"]),
        "E_D_theta_1": float(hit_info["E_D_theta_1"]),
        "E_th": float(hit_info["E_th"]),
        "B0": float(hit_info["B0"]),
        "B1": float(hit_info["B1"]),
        "Gamma_0": float(hit_info["Gamma_0"]),
        "Gamma_1": float(hit_info["Gamma_1"]),
        "Lambda_0": float(hit_info["Lambda_0"]),
        "Lambda_1": float(hit_info["Lambda_1"]),
        "E_I_0": float(predictive_stats["E_I_0"]),
        "E_I_1": float(predictive_stats["E_I_1"]),
        "E_T_0_sensor": float(predictive_stats.get("E_T_0_sensor", float("nan"))),
        "E_T_1_sensor": float(predictive_stats.get("E_T_1_sensor", float("nan"))),
        "E_q00_pow_I0": float(predictive_stats["E_q00_pow_I0"]),
        "E_q11_pow_I1": float(predictive_stats["E_q11_pow_I1"]),
        "average_rate": avg_rate,
        "objective": J,
        "objective_factor": objective_factor,
        "blocklength_n": n_block,
        "average_packet_error_info": avg_eps_info,
        "markov_source": markov["source"],
        "q00": q00,
        "q01": q01,
        "q10": q10,
        "q11": q11,
        "pi0": pi0,
        "pi1": pi1,
        "outage_constraint_method": "transition_conditioned_two_transition",
        "hit_info": hit_info,
        "blackout_info": hit_info,
        "fp_converged": True,
        "fp_iterations": 0,
        "fp_residual": 0.0,
        "fp_final_residual": 0.0,
        "eps_P_state0": eps_P_state0,
        "eps_P_state1": eps_P_state1,
    }


def _objective_for_fixed_theta(
    pt_value: float,
    theta0: int,
    theta1: int,
    derived: Dict[str, Any],
    params: ParamsLike,
    predictive_stats: Dict[str, Any],
    hit_info: Dict[str, Any],
) -> float:
    result = evaluate_resilience_design(
        pt=pt_value,
        theta0=theta0,
        theta1=theta1,
        derived=derived,
        params=params,
        predictive_stats=predictive_stats,
        precomputed_hit=hit_info,
    )
    return float(result["objective"]) if result["feasible"] else 1e18


def solve_resilience_design(
    derived: Dict[str, Any],
    params: ParamsLike,
) -> Dict[str, Any]:
    """
    Semi-closed numerical solution of the revised design problem.

    For each (theta0,theta1):
      1. Use EPSILON_R as the common false-recovery tolerance.
      2. Compute p_hit^P from Gamma/Lambda transition-conditioned terms.
      3. Convert the lead-time constraint into eps_bar(pt) <= e_max.
      4. Find the smallest feasible pt by bisection.
      5. Minimize the one-dimensional objective over [pt_lower, pt_max].
      6. Keep the best feasible design.
    """
    theta0_candidates = list(_get_param(params, "THETA0_CANDIDATES", list(range(1, 9))))
    theta1_candidates = list(_get_param(params, "THETA1_CANDIDATES", list(range(1, 9))))
    pt_min = float(_get_param(params, "PT_MIN", 0.05))
    pt_max = float(_get_param(params, "PT_MAX", _get_param(params, "RHO", 5.0)))

    pr = float(_get_param(params, "P_R", 0.05))
    eps_l = float(_get_param(params, "EPSILON_L", 0.05))
    eps_r = float(_get_param(params, "EPSILON_R", 0.05))
    lam = float(_get_param(params, "TH_RECOVERY_LAMBDA", 4.0))
    kappa = float(_get_param(params, "TH_RECOVERY_KAPPA", 2.0))
    average_eps_method = str(_get_param(params, "AVERAGE_EPSILON_METHOD", "closed_form"))
    hit_sample_cap = int(_get_param(params, "PHIT_SAMPLE_CAP", 3000))
    hit_sample_seed = int(_get_param(params, "PHIT_SAMPLE_SEED", 123))
    tail_tol = float(_get_param(params, "DESIGN_TAIL_TOL", 1e-10))

    predictive_stats = estimate_predictive_horizon_moments(params=params, derived=derived)

    markov_for_q = _select_design_markov_stats(derived, predictive_stats, params)
    q00 = float(markov_for_q["q00"])
    q11 = float(markov_for_q["q11"])

    predictive_stats = add_q_power_moments_to_predictive_horizon_stats(
        predictive_stats=predictive_stats,
        q00=q00,
        q11=q11,
    )

    best = None
    best_obj = float("inf")
    candidates = []

    for theta0 in theta0_candidates:
        for theta1 in theta1_candidates:
            theta0 = int(theta0)
            theta1 = int(theta1)

            candidate_base: Dict[str, Any] = {
                "theta0": theta0,
                "theta1": theta1,
                "eps_l": eps_l,
                "eps_r": eps_r,
                "alpha_FR": eps_r,
            }

            if theta0 <= 0 or theta1 <= 0 or not (0.0 < eps_r < 1.0):
                candidate_base.update({"feasible": False, "reason": "invalid_theta_or_eps_r"})
                candidates.append(candidate_base)
                continue

            try:
                hit_info = compute_transition_blackout_hit_probability(
                    theta0=theta0,
                    theta1=theta1,
                    eps_r=eps_r,
                    pr=pr,
                    predictive_stats=predictive_stats,
                    weibull_lambda=lam,
                    weibull_kappa=kappa,
                    sample_cap=hit_sample_cap,
                    sample_seed=hit_sample_seed,
                    tail_tol=tail_tol,
                )
                hit_info = apply_p_hit_safety_guard(hit_info, params)
            except Exception as exc:  # keep search robust
                candidate_base.update({"feasible": False, "reason": f"p_hit_failed: {exc}"})
                candidates.append(candidate_base)
                continue

            p_hit_P = float(hit_info["p_hit_P"])
            if not (p_hit_P < eps_l):
                candidate_base.update({
                    "feasible": False,
                    "reason": "p_hit_exceeds_epsilon_l_infeasible_even_with_zero_packet_error",
                    "p_hit_P": p_hit_P,
                    "hit_info": hit_info,
                    "Gamma_0": float(hit_info["Gamma_0"]),
                    "Gamma_1": float(hit_info["Gamma_1"]),
                    "Lambda_0": float(hit_info["Lambda_0"]),
                    "Lambda_1": float(hit_info["Lambda_1"]),
                })
                candidates.append(candidate_base)
                continue

            require_statewise = bool(_get_param(params, "REQUIRE_STATEWISE_LEAD_TIME", True))

            p_hit_P = float(hit_info["p_hit_P"])
            p_hit_state0 = float(hit_info["p_hit_state0"])
            p_hit_state1 = float(hit_info["p_hit_state1"])

            lead_hit_values = [p_hit_P]

            if require_statewise:
                lead_hit_values.extend([p_hit_state0, p_hit_state1])

            p_hit_worst = max(lead_hit_values)

            if not (p_hit_worst < eps_l):
                candidate_base.update({
                    "feasible": False,
                    "reason": "p_hit_exceeds_epsilon_l_infeasible_even_with_zero_packet_error",
                    "p_hit_P": p_hit_P,
                    "p_hit_state0": p_hit_state0,
                    "p_hit_state1": p_hit_state1,
                    "p_hit_worst": p_hit_worst,
                    "hit_info": hit_info,
                    "Gamma_0": float(hit_info["Gamma_0"]),
                    "Gamma_1": float(hit_info["Gamma_1"]),
                    "Lambda_0": float(hit_info["Lambda_0"]),
                    "Lambda_1": float(hit_info["Lambda_1"]),
                })
                candidates.append(candidate_base)
                continue

            eps_bar_max_from_lead = min(
                (eps_l - p_hit) / (1.0 - p_hit)
                for p_hit in lead_hit_values
            )
            eps_bar_max_from_pu = min(eps_r ** (1.0 / theta0), eps_r ** (1.0 / theta1))
            eps_bar_max = min(eps_bar_max_from_lead, eps_bar_max_from_pu)

            if eps_bar_max <= 0.0:
                candidate_base.update({
                    "feasible": False,
                    "reason": "nonpositive_eps_bar_max",
                    "p_hit_P": p_hit_P,
                    "eps_bar_max_from_lead": eps_bar_max_from_lead,
                    "eps_bar_max_from_pu": eps_bar_max_from_pu,
                    "hit_info": hit_info,
                })
                candidates.append(candidate_base)
                continue

            pt_lower = find_min_pt_for_packet_error(
                eps_target=eps_bar_max,
                params=params,
                pt_min=pt_min,
                pt_max=pt_max,
                method=average_eps_method,
            )

            if pt_lower is None:
                candidate_base.update({
                    "feasible": False,
                    "reason": "pt_range_cannot_meet_required_packet_error",
                    "p_hit_P": p_hit_P,
                    "eps_bar_max": eps_bar_max,
                    "eps_bar_max_from_lead": eps_bar_max_from_lead,
                    "eps_bar_max_from_pu": eps_bar_max_from_pu,
                    "hit_info": hit_info,
                })
                candidates.append(candidate_base)
                continue

            def objective_pt(pt_value: float) -> float:
                return _objective_for_fixed_theta(
                    pt_value=pt_value,
                    theta0=theta0,
                    theta1=theta1,
                    derived=derived,
                    params=params,
                    predictive_stats=predictive_stats,
                    hit_info=hit_info,
                )

            # The objective can have either a boundary optimum or an interior optimum.
            # We check both the bounded minimizer and the lower feasible boundary.
            search = minimize_scalar(objective_pt, bounds=(pt_lower, pt_max), method="bounded")
            boundary = evaluate_resilience_design(
                pt=pt_lower,
                theta0=theta0,
                theta1=theta1,
                derived=derived,
                params=params,
                predictive_stats=predictive_stats,
                precomputed_hit=hit_info,
            )
            interior = evaluate_resilience_design(
                pt=float(search.x),
                theta0=theta0,
                theta1=theta1,
                derived=derived,
                params=params,
                predictive_stats=predictive_stats,
                precomputed_hit=hit_info,
            )
            interior["line_search_success"] = bool(search.success)
            interior["line_search_message"] = str(search.message)
            interior["pt_lower"] = float(pt_lower)
            interior["eps_bar_max"] = float(eps_bar_max)
            interior["eps_bar_max_from_lead"] = float(eps_bar_max_from_lead)
            interior["eps_bar_max_from_pu"] = float(eps_bar_max_from_pu)

            boundary["line_search_success"] = bool(search.success)
            boundary["line_search_message"] = "boundary_checked; " + str(search.message)
            boundary["pt_lower"] = float(pt_lower)
            boundary["eps_bar_max"] = float(eps_bar_max)
            boundary["eps_bar_max_from_lead"] = float(eps_bar_max_from_lead)
            boundary["eps_bar_max_from_pu"] = float(eps_bar_max_from_pu)

            feasible_results = [r for r in (boundary, interior) if r.get("feasible", False)]
            if not feasible_results:
                candidate_base.update({
                    "feasible": False,
                    "reason": "evaluation_failed_after_pt_lower",
                    "boundary_result": boundary,
                    "interior_result": interior,
                    "p_hit_P": p_hit_P,
                    "eps_bar_max": eps_bar_max,
                })
                candidates.append(candidate_base)
                continue

            result = min(feasible_results, key=lambda r: float(r["objective"]))
            candidates.append(result)

            if result["feasible"] and result["objective"] < best_obj:
                best_obj = float(result["objective"])
                best = result

    if best is None:
        return {
            "feasible": False,
            "reason": "no_feasible_design_found",
            "predictive_stats": predictive_stats,
            "candidates": candidates,
        }

    return {
        "feasible": True,
        "best_design": best,
        "predictive_stats": predictive_stats,
        "candidates": candidates,
    }

def apply_p_hit_safety_guard(
    hit_info: Dict[str, Any],
    params: ParamsLike,
) -> Dict[str, Any]:
    """
    Inflate transition-conditioned blackout-hit probabilities to compensate
    for modeling mismatch between the semi-closed Gamma/Lambda approximation
    and the full simulator.

    Set PHIT_SAFETY_FACTOR=1 and PHIT_SAFETY_MARGIN=0 to disable.
    """
    if "p_hit_P_raw" in hit_info:
        # Already guarded.
        return hit_info

    factor = float(_get_param(params, "PHIT_SAFETY_FACTOR", 1.0))
    margin = float(_get_param(params, "PHIT_SAFETY_MARGIN", 0.0))

    out = dict(hit_info)

    for key in ["p_hit_P", "p_hit_state0", "p_hit_state1", "p_hit_P_linear"]:
        if key in out:
            raw = float(out[key])
            out[key + "_raw"] = raw
            out[key] = float(np.clip(factor * raw + margin, 0.0, 1.0))

    out["p_hit_safety_factor"] = factor
    out["p_hit_safety_margin"] = margin
    return out


if __name__ == "__main__":
    import PARAMETERS as P
    from computation import precompute_all

    derived = precompute_all(P)
    solution = solve_resilience_design(derived=derived, params=P)

    print("=== Revised resilience design summary ===")
    if not solution["feasible"]:
        print("No feasible design found.")
        print(solution["reason"])
    else:
        best = solution["best_design"]
        print(f"theta0             : {best['theta0']}")
        print(f"theta1             : {best['theta1']}")
        print(f"pt*                : {best['pt']:.6f}")
        print(f"pt lower           : {best.get('pt_lower', float('nan')):.6f}")
        print(f"eps_bar            : {best['eps_bar']:.6f}")
        print(f"eps_l              : {best['eps_l']:.6f}")
        print(f"eps_r / alpha_FR   : {best['eps_r']:.6f}")
        print(f"eps_P=P(L<0)       : {best['eps_P']:.6f}")
        print(f"P(L>=0) predicted  : {1.0 - best['eps_P']:.6f}")
        print(f"p_hit_P            : {best['p_hit_P']:.6f}")
        print(f"p_hit state 0      : {best['p_hit_state0']:.6f}")
        print(f"p_hit state 1      : {best['p_hit_state1']:.6f}")
        print(f"eps_bar max        : {best.get('eps_bar_max', float('nan')):.6f}")
        print(f"pu0*               : {best['pu0_star']:.6f}")
        print(f"pu1*               : {best['pu1_star']:.6f}")
        print(f"rho0               : {best['rho0']:.6f}")
        print(f"rho1               : {best['rho1']:.6f}")
        print(f"E[D0]              : {best['E_D_theta_0']:.6f}")
        print(f"E[D1]              : {best['E_D_theta_1']:.6f}")
        print(f"B0                 : {best['B0']:.6f}")
        print(f"B1                 : {best['B1']:.6f}")
        print(f"Gamma0             : {best['Gamma_0']:.6f}")
        print(f"Gamma1             : {best['Gamma_1']:.6f}")
        print(f"Lambda0            : {best['Lambda_0']:.6f}")
        print(f"Lambda1            : {best['Lambda_1']:.6f}")
        print(f"E[I0]              : {best['E_I_0']:.6f}")
        print(f"E[I1]              : {best['E_I_1']:.6f}")
        print(f"E[q00^I0]          : {best['E_q00_pow_I0']:.6f}")
        print(f"E[q11^I1]          : {best['E_q11_pow_I1']:.6f}")
        print(f"average rate       : {best['average_rate']:.6f}")
        print(f"objective          : {best['objective']:.6f}")
        print(f"markov source      : {best['markov_source']}")
        print(f"outage method      : {best['outage_constraint_method']}")
        print(f"line search        : {best.get('line_search_success', False)}")
        print(f"eps_P state 0       : {best.get('eps_P_state0', float('nan')):.6f}")
        print(f"eps_P state 1       : {best.get('eps_P_state1', float('nan')):.6f}")

        print(f"p_hit_P raw         : {best.get('p_hit_P_raw', best.get('p_hit_P', float('nan'))):.6f}")
        print(f"p_hit_P guarded     : {best.get('p_hit_P', float('nan')):.6f}")
        print(f"p_hit state 0 raw   : {best.get('p_hit_state0_raw', best.get('p_hit_state0', float('nan'))):.6f}")
        print(f"p_hit state 0 guard : {best.get('p_hit_state0', float('nan')):.6f}")
        print(f"p_hit state 1 raw   : {best.get('p_hit_state1_raw', best.get('p_hit_state1', float('nan'))):.6f}")
        print(f"p_hit state 1 guard : {best.get('p_hit_state1', float('nan')):.6f}")
        print(f"p_hit safety factor : {best.get('p_hit_safety_factor', 1.0):.3f}")

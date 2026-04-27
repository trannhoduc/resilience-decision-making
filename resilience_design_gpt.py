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


def _coding_rate_nats_per_channel_use(n: int, l: int, info_unit: str = "bits") -> float:
    """
    Convert the packet information size to nats/channel-use.

    The finite-blocklength formulas below use natural logarithms. Therefore,
    if INFO_BITS_L is measured in bits, the coding rate is l*ln(2)/n.

    Set INFO_BITS_UNIT = "nats" in PARAMETERS.py if l is already measured in
    nats rather than bits.
    """
    n = int(n)
    l = int(l)
    if n <= 0 or l < 0:
        raise ValueError("n must be positive and l must be non-negative.")

    unit = str(info_unit).strip().lower()
    if unit in {"bit", "bits", "bpcu", "bit/channel-use", "bits/channel-use"}:
        return float(l * math.log(2.0) / n)
    if unit in {"nat", "nats", "npcu", "nat/channel-use", "nats/channel-use"}:
        return float(l / n)
    raise ValueError("INFO_BITS_UNIT must be either 'bits' or 'nats'.")


# -----------------------------------------------------------------------------
# Finite-blocklength communication model (Section II-C)
# -----------------------------------------------------------------------------

def gaussian_q(x: float) -> float:
    """Gaussian Q-function."""
    return 0.5 * float(erfc(x / math.sqrt(2.0)))


def compute_instantaneous_packet_error(
    gamma: float,
    n: int,
    l: int,
    info_unit: str = "bits",
) -> float:
    """
    Instantaneous packet error rate:
        epsilon_k ~= Q( sqrt(n / V(gamma)) * ( C(gamma) - R ) )

    where natural logarithms are used:
        C(gamma) = ln(1 + gamma),
        V(gamma) = 1 - (1 + gamma)^(-2),
        R = l*ln(2)/n if l is in bits, or R = l/n if l is in nats.
    """
    gamma = float(max(gamma, 0.0))
    n = int(n)
    l = int(l)

    if gamma <= 1e-14:
        return 1.0

    R_nats = _coding_rate_nats_per_channel_use(n=n, l=l, info_unit=info_unit)
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
    info_unit: str = "bits",
) -> Dict[str, float]:
    """
    Average packet error rate closed-form approximation.

    The formula is written using natural logarithms. If l is specified in bits,
    the code first converts l/n to nats/channel-use.
    """
    pt = float(pt)
    noise_var = float(noise_var)
    n = int(n)
    l = int(l)

    if pt <= 0.0 or noise_var <= 0.0:
        raise ValueError("pt and noise_var must be positive.")

    gbar = pt / noise_var
    R_nats = _coding_rate_nats_per_channel_use(n=n, l=l, info_unit=info_unit)
    exp_R = math.exp(R_nats)
    exp_2R = math.exp(2.0 * R_nats)

    phi = exp_R - 1.0
    denom = exp_2R - 1.0
    if denom <= 1e-14:
        raise ValueError("The coding rate is too close to zero for this approximation.")

    beta = -math.sqrt(n / (2.0 * math.pi * denom))
    v = math.exp(-phi / gbar)

    eps_bar = 1.0 + (
        beta * gbar
        - beta * gbar * math.exp(1.0 / (2.0 * beta * gbar))
        - 0.5
    ) * v
    eps_bar = float(np.clip(eps_bar, 0.0, 1.0))

    return {
        "pt": pt,
        "noise_var": noise_var,
        "gbar": float(gbar),
        "R_nats_per_channel_use": float(R_nats),
        "info_unit": str(info_unit),
        "phi": float(phi),
        "beta": float(beta),
        "v": float(v),
        "eps_bar": eps_bar,
    }


def compute_average_packet_error_monte_carlo(
    pt: float,
    noise_var: float,
    n: int,
    l: int,
    num_samples: int = 50000,
    seed: Optional[int] = None,
    info_unit: str = "bits",
) -> Dict[str, float]:
    """
    Optional Monte Carlo cross-check of average packet error rate by averaging
    the instantaneous finite-blocklength approximation over Rayleigh fading.
    """
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
        [compute_instantaneous_packet_error(g, n, l, info_unit=info_unit) for g in gamma],
        dtype=float,
    )

    return {
        "pt": pt,
        "noise_var": noise_var,
        "gbar": float(gbar),
        "num_samples": num_samples,
        "R_nats_per_channel_use": float(
            _coding_rate_nats_per_channel_use(n=n, l=l, info_unit=info_unit)
        ),
        "info_unit": str(info_unit),
        "eps_bar": float(np.mean(eps)),
    }


def compute_average_packet_error(
    pt: float,
    params: ParamsLike,
    method: Optional[str] = None,
) -> Dict[str, float]:
    """Wrapper for average packet error rate."""
    if method is None:
        method = str(_get_param(params, "AVERAGE_EPSILON_METHOD", "closed_form"))

    noise_var = float(_get_param(params, "CHANNEL_NOISE_VAR", 1.0))
    n = int(_get_param(params, "BLOCKLENGTH_N", 128))
    l = int(_get_param(params, "INFO_BITS_L", 64))
    info_unit = str(_get_param(params, "INFO_BITS_UNIT", "bits"))

    if method == "closed_form":
        out = compute_average_packet_error_closed_form(
            pt=pt,
            noise_var=noise_var,
            n=n,
            l=l,
            info_unit=info_unit,
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
            num_samples=num_samples,
            seed=seed,
            info_unit=info_unit,
        )
        out["method"] = "monte_carlo"
        return out

    raise ValueError(f"Unsupported method='{method}'. Use 'closed_form' or 'monte_carlo'.")


# -----------------------------------------------------------------------------
# Reliability allocation and AoI formulas (Section V-B)
# -----------------------------------------------------------------------------

def compute_eps_d(
    pr: float,
    pi0: float,
    pi1: float,
    outage_term_0: float,
    outage_term_1: float,
    eps_bar: float,
) -> float:
    """
    Active version of constraint (19):
        eps_d = pr * sum_s pi_s * Gamma_s(theta_s) + (1-pr) * eps_bar
    """
    pr = float(pr)
    pi0 = float(pi0)
    pi1 = float(pi1)
    outage_term_0 = float(outage_term_0)
    outage_term_1 = float(outage_term_1)
    eps_bar = float(eps_bar)

    eps_d = pr * (pi0 * outage_term_0 + pi1 * outage_term_1) + (1.0 - pr) * eps_bar
    return float(np.clip(eps_d, 0.0, 1.0))


def compute_eps_r_from_eps_d(eps_l: float, eps_d: float) -> float:
    """
    Active version of constraint (20):
        (1 - eps_r)(1 - eps_d) = 1 - eps_l
    so
        eps_r = 1 - (1 - eps_l)/(1 - eps_d).
    """
    eps_l = float(eps_l)
    eps_d = float(eps_d)

    if eps_d >= 1.0:
        return 1.0

    eps_r = 1.0 - (1.0 - eps_l) / (1.0 - eps_d)
    return float(np.clip(eps_r, 0.0, 1.0))


def compute_pu_star(theta_s: int, eps_r: float, eps_bar: float) -> Dict[str, float]:
    """
    Equation (16):
        p_u,s^* = (1 - eps_r^(1/theta_s)) / (1 - eps_bar)
    """
    theta_s = int(theta_s)
    eps_r = float(eps_r)
    eps_bar = float(eps_bar)

    if theta_s <= 0:
        raise ValueError("theta_s must be positive.")

    denom = 1.0 - eps_bar
    if denom <= 0.0:
        return {"pu_star": float("inf"), "feasible": False}

    pu_star = (1.0 - eps_r ** (1.0 / theta_s)) / denom
    feasible = bool(0.0 <= pu_star <= 1.0)

    return {
        "pu_star": float(pu_star),
        "feasible": feasible,
    }


def compute_expected_D_theta(theta_s: int, eta_s: float) -> float:
    """
    Equation (17):
        E[D_theta,s] ~= sum_{a=0}^{theta_s-1} (theta_s - a) eta_s (1-eta_s)^a
    """
    theta_s = int(theta_s)
    eta_s = float(np.clip(eta_s, 0.0, 1.0))

    if theta_s <= 0:
        raise ValueError("theta_s must be positive.")

    value = 0.0
    for a in range(theta_s):
        value += (theta_s - a) * eta_s * ((1.0 - eta_s) ** a)
    return float(value)


# -----------------------------------------------------------------------------
# Discrete Weibull, geometric, and outage helper functions
# -----------------------------------------------------------------------------

def discrete_weibull_pmf(
    lam: float,
    kappa: float,
    tail_tol: float = 1e-10,
    max_tau: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    P(t_h = tau) = exp(-(tau/lambda)^kappa)
                 - exp(-((tau+1)/lambda)^kappa), tau = 0, 1, 2, ...
    """
    lam = float(lam)
    kappa = float(kappa)

    if lam <= 0.0 or kappa <= 0.0:
        raise ValueError("lam and kappa must be positive.")

    taus = []
    probs = []

    for tau in range(int(max_tau) + 1):
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


def geometric_pmf_from_qss(
    q_ss: float,
    tail_tol: float = 1e-10,
    max_t: int = 100000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sojourn time T_s under the two-state Markov surrogate:
        P(T_s = n) = q_ss^(n-1) * (1 - q_ss), n = 1, 2, ...
    """
    q_ss = float(q_ss)

    if not (0.0 <= q_ss < 1.0):
        raise ValueError("q_ss must lie in [0, 1).")

    if q_ss == 0.0:
        return np.array([1], dtype=int), np.array([1.0], dtype=float)

    ts = []
    probs = []

    for n in range(1, int(max_t) + 1):
        p = (q_ss ** (n - 1)) * (1.0 - q_ss)
        ts.append(n)
        probs.append(p)
        tail = q_ss ** n
        if tail < tail_tol:
            break

    ts_arr = np.asarray(ts, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)
    probs_arr /= np.sum(probs_arr)
    return ts_arr, probs_arr


def _expected_min_with_recovery_distribution(
    E_D_theta_s: float,
    residuals: np.ndarray,
    th_vals: np.ndarray,
    th_pmf: np.ndarray,
    chunk_size: int = 50000,
) -> np.ndarray:
    """
    For each residual r, compute E_t_h[min(E_D_theta_s + t_h, r)].
    Chunking avoids a large temporary array when residuals is large.
    """
    residuals_arr = np.maximum(np.asarray(residuals, dtype=float), 0.0)
    shape = residuals_arr.shape
    flat = residuals_arr.reshape(-1)
    out = np.empty_like(flat, dtype=float)

    th_vals = np.asarray(th_vals, dtype=float).reshape(1, -1)
    th_pmf = np.asarray(th_pmf, dtype=float).reshape(-1)
    recovery_values = float(E_D_theta_s) + th_vals

    for start in range(0, flat.size, int(chunk_size)):
        stop = min(start + int(chunk_size), flat.size)
        r = flat[start:stop].reshape(-1, 1)
        out[start:stop] = np.minimum(recovery_values, r) @ th_pmf

    return out.reshape(shape)


def compute_outage_term_state(
    theta_s: int,
    q_ss: float,
    E_I_s: float,
    E_D_theta_s: float,
    weibull_lambda: float,
    weibull_kappa: float,
    tail_tol: float = 1e-10,
    I_samples: Optional[ArrayLike] = None,
    i_method: str = "mean",
    max_i_samples: int = 2000,
    i_sample_seed: int = 12345,
) -> Dict[str, float]:
    """
    Approximate the state-s contribution in constraint (19):
        Gamma_s(theta_s) = E[min{D_theta,s + t_h, T_s - I_s}/T_s].

    Supported approximations:
    - i_method = "mean": replace I_s by E[I_s]. This is the mean-field version.
    - i_method = "empirical": average over empirical I_s samples, while still
      using E[D_theta,s] for the AoI residual term.

    In both cases, the geometric T_s distribution and discrete Weibull recovery
    distribution are retained explicitly.
    """
    del theta_s  # kept only for interface readability

    T_vals, T_pmf = geometric_pmf_from_qss(q_ss=q_ss, tail_tol=tail_tol)
    th_vals, th_pmf = discrete_weibull_pmf(
        lam=weibull_lambda,
        kappa=weibull_kappa,
        tail_tol=tail_tol,
    )

    E_I_s = float(max(E_I_s, 0.0))
    E_D_theta_s = float(max(E_D_theta_s, 0.0))
    method = str(i_method).strip().lower()

    if method in {"empirical", "samples", "sample"} and I_samples is not None:
        I_arr = np.asarray(I_samples, dtype=float).reshape(-1)
        I_arr = I_arr[np.isfinite(I_arr)]
        I_arr = np.maximum(I_arr, 0.0)

        if I_arr.size == 0:
            method = "mean"
        else:
            max_i_samples = int(max_i_samples)
            if max_i_samples > 0 and I_arr.size > max_i_samples:
                rng = np.random.default_rng(int(i_sample_seed))
                idx = rng.choice(I_arr.size, size=max_i_samples, replace=False)
                I_arr = I_arr[idx]

            residuals = np.maximum(
                T_vals.astype(float).reshape(-1, 1) - I_arr.reshape(1, -1),
                0.0,
            )
            expected_min = _expected_min_with_recovery_distribution(
                E_D_theta_s=E_D_theta_s,
                residuals=residuals,
                th_vals=th_vals,
                th_pmf=th_pmf,
            )
            gamma_s = np.sum(
                T_pmf.reshape(-1, 1) * expected_min / T_vals.astype(float).reshape(-1, 1)
            ) / float(I_arr.size)

            return {
                "Gamma_s": float(np.clip(gamma_s, 0.0, 1.0)),
                "I_method": "empirical",
                "I_samples_used": int(I_arr.size),
                "E_I_s": E_I_s,
                "E_D_theta_s": E_D_theta_s,
                "E_th": float(np.sum(th_vals * th_pmf)),
                "E_T_s": float(np.sum(T_vals * T_pmf)),
            }

    residuals = np.maximum(T_vals.astype(float) - E_I_s, 0.0)
    expected_min = _expected_min_with_recovery_distribution(
        E_D_theta_s=E_D_theta_s,
        residuals=residuals,
        th_vals=th_vals,
        th_pmf=th_pmf,
    )
    gamma_s = np.sum(T_pmf * expected_min / T_vals.astype(float))

    return {
        "Gamma_s": float(np.clip(gamma_s, 0.0, 1.0)),
        "I_method": "mean",
        "I_samples_used": 0,
        "E_I_s": E_I_s,
        "E_D_theta_s": E_D_theta_s,
        "E_th": float(np.sum(th_vals * th_pmf)),
        "E_T_s": float(np.sum(T_vals * T_pmf)),
    }


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


def _sensor_operational_decision(x_hat: np.ndarray, c: np.ndarray, threshold: float) -> int:
    return 1 if _scalar(c.T @ x_hat) >= float(threshold) else 0


def _safe_ratio(num: float, den: float, default: float = float("nan")) -> float:
    den = float(den)
    if den <= 0.0:
        return float(default)
    return float(num) / den


def estimate_predictive_horizon_moments(
    params: ParamsLike,
    derived: Dict[str, Any],
    num_transitions_per_state: Optional[int] = None,
    burn_in: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate predictive-horizon moments using sensor-side simulation only.

    Important modeling choice used here:
    - No estimator is involved.
    - No packet drops or communication outage are involved.
    - The operational state is the sensor-side decision based on x_hat_sensor.

    For each operational sojourn in state s, the code records the first valid
    predictive trigger issued during that sojourn. The predictive horizon sample
    is the realized lead time until the next sensor-side decision transition:
        I_s = transition_time - trigger_time.

    If no predictive trigger is issued before the transition, I_s = 0.

    This is different from simply appending pred["predicted_horizon"]. The latter
    would force the prediction to be correct by construction.
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

    decision_threshold = _get_param(params, "OPERATIONAL_DECISION_THRESHOLD", None)
    if decision_threshold is None:
        decision_threshold = xi
    decision_threshold = float(decision_threshold)

    n = A.shape[0]
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

    clear_stale_predictions = bool(_get_param(params, "CLEAR_STALE_PREDICTIONS", True))

    x_true0 = _get_param(params, "X_TRUE0", np.zeros((n, 1)))
    x_s0 = _get_param(params, "X_S0", np.zeros((n, 1)))
    P_s0 = _get_param(params, "P_S0", np.eye(n))

    x_true = _as_column(x_true0).copy()
    x_hat = _as_column(x_s0).copy()
    P = _as_array(P_s0).copy()

    for _ in range(burn_in):
        x_true = _sample_true_process_step(x_true, A, Q, rng)
        y = _sample_measurement(x_true, C, R, rng)
        x_hat, P = _sensor_kf_update(x_hat, P, y, A, C, Q, R)

    I_samples = {0: [], 1: []}
    sojourn_lengths = {0: [], 1: []}
    transition_counts = np.zeros((2, 2), dtype=int)
    state_step_counts = np.zeros(2, dtype=int)

    current_state = _sensor_operational_decision(x_hat, c, decision_threshold)
    sojourn_start_step = 0
    step_index = 0

    pending_issue_step: Optional[int] = None
    pending_expire_step: Optional[int] = None
    pending_predicted_horizon: Optional[int] = None
    pending_predicted_decision: Optional[int] = None

    predictions_issued = 0
    stale_predictions_cleared = 0

    while (
        len(I_samples[0]) < num_transitions_per_state
        or len(I_samples[1]) < num_transitions_per_state
    ) and step_index < max_steps:
        # Run Algorithm 1 at the current sensor state if no prediction is pending.
        if pending_issue_step is None:
            pred = predictive_transition_detection(
                x_hat_sensor=x_hat,
                P_sensor=P,
                A=A,
                Q=Q,
                c=c,
                Delta=Delta,
                alpha_fp=alpha_fp,
                alpha_fn=alpha_fn,
                previous_decision=current_state,
                ell=ell,
                xi=xi,
            )

            if pred.get("found_transition") and pred.get("predicted_horizon") is not None:
                horizon = int(pred["predicted_horizon"])
                if horizon > 0:
                    pending_issue_step = step_index
                    pending_expire_step = step_index + horizon
                    pending_predicted_horizon = horizon
                    pending_predicted_decision = 1 - current_state
                    predictions_issued += 1

        # Advance the sensor-side process by one step.
        x_true = _sample_true_process_step(x_true, A, Q, rng)
        y = _sample_measurement(x_true, C, R, rng)
        x_hat, P = _sensor_kf_update(x_hat, P, y, A, C, Q, R)

        next_step = step_index + 1
        next_state = _sensor_operational_decision(x_hat, c, decision_threshold)

        state_step_counts[current_state] += 1
        transition_counts[current_state, next_state] += 1

        if next_state != current_state:
            sojourn_length = next_step - sojourn_start_step
            if pending_issue_step is None:
                lead_time = 0
            else:
                lead_time = max(next_step - pending_issue_step, 0)

            I_samples[current_state].append(float(lead_time))
            sojourn_lengths[current_state].append(float(sojourn_length))

            current_state = next_state
            sojourn_start_step = next_step

            pending_issue_step = None
            pending_expire_step = None
            pending_predicted_horizon = None
            pending_predicted_decision = None
        else:
            if (
                pending_issue_step is not None
                and clear_stale_predictions
                and pending_expire_step is not None
                and next_step >= pending_expire_step
            ):
                stale_predictions_cleared += 1
                pending_issue_step = None
                pending_expire_step = None
                pending_predicted_horizon = None
                pending_predicted_decision = None

        step_index = next_step

    I_0 = np.asarray(I_samples[0], dtype=float)
    I_1 = np.asarray(I_samples[1], dtype=float)
    T_0 = np.asarray(sojourn_lengths[0], dtype=float)
    T_1 = np.asarray(sojourn_lengths[1], dtype=float)

    if I_0.size == 0 or I_1.size == 0:
        raise RuntimeError(
            "Failed to collect predictive-horizon samples for both operational states. "
            "Increase NUM_I_MONTE_CARLO and/or I_MONTE_CARLO_MAX_STEPS."
        )

    visits0 = float(state_step_counts[0])
    visits1 = float(state_step_counts[1])
    total_visits = visits0 + visits1

    q00 = _safe_ratio(transition_counts[0, 0], visits0)
    q01 = _safe_ratio(transition_counts[0, 1], visits0)
    q10 = _safe_ratio(transition_counts[1, 0], visits1)
    q11 = _safe_ratio(transition_counts[1, 1], visits1)
    pi0 = _safe_ratio(visits0, total_visits)
    pi1 = _safe_ratio(visits1, total_visits)

    return {
        "I_0_samples": I_0,
        "I_1_samples": I_1,
        "E_I_0": float(np.mean(I_0)),
        "E_I_1": float(np.mean(I_1)),
        "num_samples_0": int(I_0.size),
        "num_samples_1": int(I_1.size),
        "T_0_samples": T_0,
        "T_1_samples": T_1,
        "E_T_0_sensor": float(np.mean(T_0)) if T_0.size else float("nan"),
        "E_T_1_sensor": float(np.mean(T_1)) if T_1.size else float("nan"),
        "burn_in": int(burn_in),
        "seed": int(seed),
        "steps_simulated": int(step_index),
        "decision_threshold": float(decision_threshold),
        "decision_threshold_source": "OPERATIONAL_DECISION_THRESHOLD" if _get_param(params, "OPERATIONAL_DECISION_THRESHOLD", None) is not None else "xi",
        "predictions_issued": int(predictions_issued),
        "stale_predictions_cleared": int(stale_predictions_cleared),
        "clear_stale_predictions": bool(clear_stale_predictions),
        "sensor_transition_counts": transition_counts,
        "sensor_state_step_counts": state_step_counts,
        "sensor_markov_surrogate": {
            "q00": float(q00),
            "q01": float(q01),
            "q10": float(q10),
            "q11": float(q11),
        },
        "sensor_markov_chain_statistics": {
            "pi0": float(pi0),
            "pi1": float(pi1),
            "E_T_0": float(np.mean(T_0)) if T_0.size else float("nan"),
            "E_T_1": float(np.mean(T_1)) if T_1.size else float("nan"),
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
    out["E_q00_pow_I0"] = float(np.mean(float(q00) ** I_0))
    out["E_q11_pow_I1"] = float(np.mean(float(q11) ** I_1))
    return out


# -----------------------------------------------------------------------------
# Average transmission rate and objective (Section V-B)
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
    Average transmission rate from Section V-B:
        r = [2 + pu0 E[q00^I0] q00/q01
               + pu1 E[q11^I1] q11/q10]
            / [q01^{-1} + q10^{-1}]
    """
    q01 = float(q01)
    q10 = float(q10)
    if q01 <= 0.0 or q10 <= 0.0:
        return float("inf")

    numerator = (
        2.0
        + float(pu0) * float(E_q00_pow_I0) * (float(q00) / q01)
        + float(pu1) * float(E_q11_pow_I1) * (float(q11) / q10)
    )
    denominator = (1.0 / q01) + (1.0 / q10)
    return float(numerator / denominator)


def compute_energy_objective(pt: float, n: int, average_rate: float) -> float:
    """Objective scale: J = n * pt * r(...)."""
    return float(int(n) * float(pt) * float(average_rate))


# -----------------------------------------------------------------------------
# Design evaluation and solver (Algorithm 2)
# -----------------------------------------------------------------------------

def get_design_markov_statistics(
    derived: Dict[str, Any],
    predictive_stats: Optional[Dict[str, Any]],
    params: ParamsLike,
) -> Dict[str, Any]:
    """
    Select the Markov surrogate used by the design equations.

    Default is sensor_decision, because I_s is estimated from the sensor-side
    operational decision process. Set DESIGN_MARKOV_SOURCE = "derived" to use
    the precomputed Markov statistics from computation.py instead.
    """
    source = str(_get_param(params, "DESIGN_MARKOV_SOURCE", "sensor_decision")).strip().lower()

    if source in {"sensor", "sensor_decision", "operational", "sensor_operational"}:
        if predictive_stats is None:
            raise ValueError("predictive_stats is required when DESIGN_MARKOV_SOURCE='sensor_decision'.")
        q = predictive_stats.get("sensor_markov_surrogate")
        pi = predictive_stats.get("sensor_markov_chain_statistics")
        if q is None or pi is None:
            raise ValueError("predictive_stats does not contain sensor-side Markov statistics.")
        selected_source = "sensor_decision"
    elif source in {"derived", "true", "physical", "precomputed"}:
        q = derived["markov_surrogate"]
        pi = derived["markov_chain_statistics"]
        selected_source = "derived"
    else:
        raise ValueError("DESIGN_MARKOV_SOURCE must be 'sensor_decision' or 'derived'.")

    q00 = float(q["q00"])
    q01 = float(q["q01"])
    q10 = float(q["q10"])
    q11 = float(q["q11"])
    pi0 = float(pi["pi0"])
    pi1 = float(pi["pi1"])

    if not np.all(np.isfinite([q00, q01, q10, q11, pi0, pi1])):
        raise ValueError("Selected Markov statistics contain non-finite values.")
    if q01 <= 0.0 or q10 <= 0.0:
        raise ValueError("Selected Markov surrogate must have positive q01 and q10.")
    if not (0.0 <= q00 < 1.0 and 0.0 <= q11 < 1.0):
        raise ValueError("Selected q00 and q11 must lie in [0, 1).")

    return {
        "source": selected_source,
        "pi0": pi0,
        "pi1": pi1,
        "q00": q00,
        "q01": q01,
        "q10": q10,
        "q11": q11,
    }


def evaluate_resilience_design(
    pt: float,
    theta0: int,
    theta1: int,
    derived: Dict[str, Any],
    params: ParamsLike,
    predictive_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate one candidate design tuple (pt, theta0, theta1).

    Main correction relative to the previous version:
    the coupled variables eps_d -> eps_r -> p_u -> D_theta -> Gamma -> eps_d
    are solved by a fixed-point loop. The returned eps_d, eps_r, p_u, D_theta,
    and Gamma values therefore come from the same consistent iteration.
    """
    theta0 = int(theta0)
    theta1 = int(theta1)

    try:
        markov = get_design_markov_statistics(derived, predictive_stats, params)
    except Exception as exc:
        return {
            "feasible": False,
            "reason": f"markov_statistics_invalid: {exc}",
            "pt": float(pt),
            "theta0": theta0,
            "theta1": theta1,
        }

    pi0 = float(markov["pi0"])
    pi1 = float(markov["pi1"])
    q00 = float(markov["q00"])
    q11 = float(markov["q11"])
    q01 = float(markov["q01"])
    q10 = float(markov["q10"])

    pr = float(_get_param(params, "P_R", 0.05))
    eps_l = float(_get_param(params, "EPSILON_L", 0.05))
    n = int(_get_param(params, "BLOCKLENGTH_N", 128))
    lam = float(_get_param(params, "TH_RECOVERY_LAMBDA", 4.0))
    kappa = float(_get_param(params, "TH_RECOVERY_KAPPA", 2.0))
    average_eps_method = str(_get_param(params, "AVERAGE_EPSILON_METHOD", "closed_form"))
    outage_i_method = str(_get_param(params, "OUTAGE_I_METHOD", "mean"))
    outage_i_sample_cap = int(_get_param(params, "OUTAGE_I_SAMPLE_CAP", 2000))
    outage_i_sample_seed = int(_get_param(params, "OUTAGE_I_SAMPLE_SEED", 12345))

    fp_max_iter = int(_get_param(params, "EPS_FIXED_POINT_MAX_ITER", 30))
    fp_tol = float(_get_param(params, "EPS_FIXED_POINT_TOL", 1e-8))
    fp_damping = float(_get_param(params, "EPS_FIXED_POINT_DAMPING", 1.0))
    fp_damping = float(np.clip(fp_damping, 1e-6, 1.0))
    require_fp_convergence = bool(_get_param(params, "REQUIRE_FIXED_POINT_CONVERGENCE", True))

    avg_eps_info = compute_average_packet_error(pt=pt, params=params, method=average_eps_method)
    eps_bar = float(avg_eps_info["eps_bar"])

    def evaluate_from_eps_d(eps_d_current: float) -> Dict[str, Any]:
        eps_d_current = float(eps_d_current)
        eps_r_current = compute_eps_r_from_eps_d(eps_l=eps_l, eps_d=eps_d_current)

        if not (0.0 < eps_r_current < 1.0):
            return {
                "feasible": False,
                "reason": "eps_r_out_of_bounds_in_fixed_point",
                "eps_d_current": eps_d_current,
                "eps_r": eps_r_current,
            }

        pu0_info = compute_pu_star(theta_s=theta0, eps_r=eps_r_current, eps_bar=eps_bar)
        pu1_info = compute_pu_star(theta_s=theta1, eps_r=eps_r_current, eps_bar=eps_bar)

        if not pu0_info["feasible"] or not pu1_info["feasible"]:
            return {
                "feasible": False,
                "reason": "pu_star_infeasible_in_fixed_point",
                "eps_d_current": eps_d_current,
                "eps_r": eps_r_current,
                "pu0_star": float(pu0_info["pu_star"]),
                "pu1_star": float(pu1_info["pu_star"]),
            }

        pu0 = float(pu0_info["pu_star"])
        pu1 = float(pu1_info["pu_star"])
        eta0 = pu0 * (1.0 - eps_bar)
        eta1 = pu1 * (1.0 - eps_bar)

        E_D_theta_0 = compute_expected_D_theta(theta_s=theta0, eta_s=eta0)
        E_D_theta_1 = compute_expected_D_theta(theta_s=theta1, eta_s=eta1)

        outage0 = compute_outage_term_state(
            theta_s=theta0,
            q_ss=q00,
            E_I_s=float(predictive_stats["E_I_0"]),
            E_D_theta_s=E_D_theta_0,
            weibull_lambda=lam,
            weibull_kappa=kappa,
            I_samples=predictive_stats.get("I_0_samples"),
            i_method=outage_i_method,
            max_i_samples=outage_i_sample_cap,
            i_sample_seed=outage_i_sample_seed,
        )
        outage1 = compute_outage_term_state(
            theta_s=theta1,
            q_ss=q11,
            E_I_s=float(predictive_stats["E_I_1"]),
            E_D_theta_s=E_D_theta_1,
            weibull_lambda=lam,
            weibull_kappa=kappa,
            I_samples=predictive_stats.get("I_1_samples"),
            i_method=outage_i_method,
            max_i_samples=outage_i_sample_cap,
            i_sample_seed=outage_i_sample_seed + 1,
        )

        eps_d_new = compute_eps_d(
            pr=pr,
            pi0=pi0,
            pi1=pi1,
            outage_term_0=outage0["Gamma_s"],
            outage_term_1=outage1["Gamma_s"],
            eps_bar=eps_bar,
        )

        return {
            "feasible": True,
            "eps_d_current": eps_d_current,
            "eps_d_new": float(eps_d_new),
            "eps_r": float(eps_r_current),
            "pu0_star": pu0,
            "pu1_star": pu1,
            "eta0": float(eta0),
            "eta1": float(eta1),
            "E_D_theta_0": float(E_D_theta_0),
            "E_D_theta_1": float(E_D_theta_1),
            "Gamma_0": float(outage0["Gamma_s"]),
            "Gamma_1": float(outage1["Gamma_s"]),
            "outage0_info": outage0,
            "outage1_info": outage1,
        }

    eps_d = (1.0 - pr) * eps_bar
    fp_converged = False
    fp_iterations = 0
    fp_residual = float("inf")
    last_info: Optional[Dict[str, Any]] = None

    for fp_iter in range(fp_max_iter):
        fp_iterations = fp_iter + 1
        info = evaluate_from_eps_d(eps_d)
        last_info = info

        if not info.get("feasible", False):
            return {
                "feasible": False,
                "reason": info.get("reason", "fixed_point_step_infeasible"),
                "pt": float(pt),
                "theta0": theta0,
                "theta1": theta1,
                "eps_bar": eps_bar,
                "eps_d": float(eps_d),
                "eps_r": float(info.get("eps_r", float("nan"))),
                "fp_iterations": fp_iterations,
                "fp_converged": False,
                "markov_source": markov["source"],
            }

        eps_d_new = float(info["eps_d_new"])
        fp_residual = abs(eps_d_new - eps_d)

        if fp_residual < fp_tol:
            eps_d = eps_d_new
            fp_converged = True
            break

        eps_d = (1.0 - fp_damping) * eps_d + fp_damping * eps_d_new

    final_info = evaluate_from_eps_d(eps_d)
    if not final_info.get("feasible", False):
        return {
            "feasible": False,
            "reason": final_info.get("reason", "fixed_point_final_infeasible"),
            "pt": float(pt),
            "theta0": theta0,
            "theta1": theta1,
            "eps_bar": eps_bar,
            "eps_d": float(eps_d),
            "eps_r": float(final_info.get("eps_r", float("nan"))),
            "fp_iterations": fp_iterations,
            "fp_converged": fp_converged,
            "markov_source": markov["source"],
        }

    fp_residual = abs(float(final_info["eps_d_new"]) - float(eps_d))
    if fp_residual < fp_tol:
        fp_converged = True

    if require_fp_convergence and not fp_converged:
        return {
            "feasible": False,
            "reason": "fixed_point_not_converged",
            "pt": float(pt),
            "theta0": theta0,
            "theta1": theta1,
            "eps_bar": eps_bar,
            "eps_d": float(eps_d),
            "eps_r": float(final_info["eps_r"]),
            "fp_iterations": fp_iterations,
            "fp_residual": float(fp_residual),
            "fp_converged": False,
            "markov_source": markov["source"],
            "last_fixed_point_info": last_info,
        }

    eps_d_final = float(final_info["eps_d_new"])
    eps_r = float(final_info["eps_r"])

    if not (0.0 < eps_d_final < eps_l):
        return {
            "feasible": False,
            "reason": "eps_d_out_of_bounds",
            "pt": float(pt),
            "theta0": theta0,
            "theta1": theta1,
            "eps_bar": eps_bar,
            "eps_d": eps_d_final,
            "eps_l": eps_l,
            "eps_r": eps_r,
            "fp_iterations": fp_iterations,
            "fp_residual": float(fp_residual),
            "fp_converged": fp_converged,
            "markov_source": markov["source"],
        }

    if not (0.0 < eps_r < 1.0):
        return {
            "feasible": False,
            "reason": "eps_r_out_of_bounds_final",
            "pt": float(pt),
            "theta0": theta0,
            "theta1": theta1,
            "eps_bar": eps_bar,
            "eps_d": eps_d_final,
            "eps_r": eps_r,
            "fp_iterations": fp_iterations,
            "fp_residual": float(fp_residual),
            "fp_converged": fp_converged,
            "markov_source": markov["source"],
        }

    pu0 = float(final_info["pu0_star"])
    pu1 = float(final_info["pu1_star"])
    eta0 = float(final_info["eta0"])
    eta1 = float(final_info["eta1"])

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

    if not np.isfinite(avg_rate):
        return {
            "feasible": False,
            "reason": "average_rate_nonfinite",
            "pt": float(pt),
            "theta0": theta0,
            "theta1": theta1,
            "eps_bar": eps_bar,
            "eps_d": eps_d_final,
            "eps_r": eps_r,
            "markov_source": markov["source"],
        }

    # The mathematically exact paper objective has factor n. You said you want
    # to keep the baseline normalization easy, so the default is 1. Set
    # OBJECTIVE_BLOCKLENGTH_FACTOR = BLOCKLENGTH_N in PARAMETERS.py if desired.
    objective_blocklength = int(_get_param(params, "OBJECTIVE_BLOCKLENGTH_FACTOR", 1))
    J = compute_energy_objective(pt=pt, n=objective_blocklength, average_rate=avg_rate)

    return {
        "feasible": True,
        "reason": "ok",
        "pt": float(pt),
        "theta0": theta0,
        "theta1": theta1,
        "eps_bar": eps_bar,
        "eps_d": eps_d_final,
        "eps_r": eps_r,
        "pu0_star": pu0,
        "pu1_star": pu1,
        "eta0": eta0,
        "eta1": eta1,
        "E_D_theta_0": float(final_info["E_D_theta_0"]),
        "E_D_theta_1": float(final_info["E_D_theta_1"]),
        "Gamma_0": float(final_info["Gamma_0"]),
        "Gamma_1": float(final_info["Gamma_1"]),
        "E_I_0": float(predictive_stats["E_I_0"]),
        "E_I_1": float(predictive_stats["E_I_1"]),
        "E_q00_pow_I0": float(predictive_stats["E_q00_pow_I0"]),
        "E_q11_pow_I1": float(predictive_stats["E_q11_pow_I1"]),
        "average_rate": float(avg_rate),
        "objective": float(J),
        "objective_blocklength_factor": int(objective_blocklength),
        "average_packet_error_info": avg_eps_info,
        "fp_converged": bool(fp_converged),
        "fp_iterations": int(fp_iterations),
        "fp_residual": float(fp_residual),
        "markov_source": markov["source"],
        "markov_statistics_used": markov,
        "outage_i_method": str(outage_i_method),
        "outage0_info": final_info["outage0_info"],
        "outage1_info": final_info["outage1_info"],
    }


def solve_resilience_design(
    derived: Dict[str, Any],
    params: ParamsLike,
) -> Dict[str, Any]:
    """
    Numerical solution of Problem 1, following Algorithm 2:
    - estimate predictive-horizon moments once via sensor-only Monte Carlo,
    - select Markov statistics consistently with the operational state model,
    - loop over theta0, theta1,
    - for each pair solve a 1-D bounded search over pt,
    - keep the best feasible design.
    """
    theta0_candidates = list(_get_param(params, "THETA0_CANDIDATES", list(range(1, 9))))
    theta1_candidates = list(_get_param(params, "THETA1_CANDIDATES", list(range(1, 9))))
    pt_min = float(_get_param(params, "PT_MIN", 0.05))
    pt_max = float(_get_param(params, "PT_MAX", _get_param(params, "RHO", 5.0)))

    predictive_stats = estimate_predictive_horizon_moments(params=params, derived=derived)

    markov = get_design_markov_statistics(
        derived=derived,
        predictive_stats=predictive_stats,
        params=params,
    )

    predictive_stats = add_q_power_moments_to_predictive_horizon_stats(
        predictive_stats=predictive_stats,
        q00=float(markov["q00"]),
        q11=float(markov["q11"]),
    )
    predictive_stats["design_markov_statistics"] = markov

    best = None
    best_obj = float("inf")
    candidates = []

    for theta0 in theta0_candidates:
        for theta1 in theta1_candidates:
            def objective_pt(pt_value: float) -> float:
                result = evaluate_resilience_design(
                    pt=pt_value,
                    theta0=int(theta0),
                    theta1=int(theta1),
                    derived=derived,
                    params=params,
                    predictive_stats=predictive_stats,
                )
                return float(result["objective"]) if result["feasible"] else 1e18

            search = minimize_scalar(objective_pt, bounds=(pt_min, pt_max), method="bounded")
            result = evaluate_resilience_design(
                pt=float(search.x),
                theta0=int(theta0),
                theta1=int(theta1),
                derived=derived,
                params=params,
                predictive_stats=predictive_stats,
            )
            result["line_search_success"] = bool(search.success)
            result["line_search_message"] = str(search.message)
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


if __name__ == "__main__":
    import PARAMETERS as P
    from computation import precompute_all

    derived = precompute_all(P)
    solution = solve_resilience_design(derived=derived, params=P)

    print("=== Resilience design summary ===")
    if not solution["feasible"]:
        print("No feasible design found.")
        print(solution["reason"])
        from collections import Counter
        reasons = Counter(c.get("reason", "unknown") for c in solution.get("candidates", []))
        print("=== Candidate failure reasons ===")
        for k, v in reasons.items():
            print(k, v)
    else:
        best = solution["best_design"]
        print(f"theta0             : {best['theta0']}")
        print(f"theta1             : {best['theta1']}")
        print(f"pt*                : {best['pt']:.6f}")
        print(f"eps_bar            : {best['eps_bar']:.6f}")
        print(f"eps_d              : {best['eps_d']:.6f}")
        print(f"eps_r              : {best['eps_r']:.6f}")
        print(f"pu0*               : {best['pu0_star']:.6f}")
        print(f"pu1*               : {best['pu1_star']:.6f}")
        print(f"E[I0]              : {best['E_I_0']:.6f}")
        print(f"E[I1]              : {best['E_I_1']:.6f}")
        print(f"E[q00^I0]          : {best['E_q00_pow_I0']:.6f}")
        print(f"E[q11^I1]          : {best['E_q11_pow_I1']:.6f}")
        print(f"Gamma0             : {best['Gamma_0']:.6f}")
        print(f"Gamma1             : {best['Gamma_1']:.6f}")
        print(f"average rate       : {best['average_rate']:.6f}")
        print(f"objective          : {best['objective']:.6f}")
        print(f"markov source      : {best['markov_source']}")
        print(f"outage I method    : {best['outage_i_method']}")
        print(f"FP converged       : {best['fp_converged']}")
        print(f"FP iterations      : {best['fp_iterations']}")
        print(f"FP residual        : {best['fp_residual']:.3e}")

    # print("\n=== Candidate table ===")
    # rows = []
    # for c in solution["candidates"]:
    #     rows.append({
    #         "theta0": c.get("theta0"),
    #         "theta1": c.get("theta1"),
    #         "feasible": c.get("feasible"),
    #         "reason": c.get("reason"),
    #         "objective": c.get("objective", None),
    #         "pt": c.get("pt", None),
    #         "eps_bar": c.get("eps_bar", None),
    #         "eps_d": c.get("eps_d", None),
    #         "eps_r": c.get("eps_r", None),
    #         "pu0": c.get("pu0_star", None),
    #         "pu1": c.get("pu1_star", None),
    #         "Gamma0": c.get("Gamma_0", None),
    #         "Gamma1": c.get("Gamma_1", None),
    #         "rate": c.get("average_rate", None),
    #     })

    # rows = sorted(rows, key=lambda r: (
    #     0 if r["feasible"] else 1,
    #     r["objective"] if r["objective"] is not None else 1e99
    # ))

    # for r in rows[:30]:
    #     print(r)


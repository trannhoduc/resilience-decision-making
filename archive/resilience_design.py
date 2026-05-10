from __future__ import annotations

#from ctypes import GetLastError
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import math
import numpy as np
from fontTools.ttLib.tables.otTables import DeltaSetIndexMap
from scipy.fftpack import ifft2
from scipy.optimize import minimize_scalar
from scipy.special import erfc
import matplotlib.pyplot as plt

from computation import predictive_transition_detection, evaluate_decision

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


# -----------------------------------------------------------------------------
# Finite-blocklength communication model (Section II-C)
# -----------------------------------------------------------------------------

def gaussian_q(x: float) -> float:
    """Gaussian Q-function."""
    return 0.5 * float(erfc(x / math.sqrt(2.0)))


def compute_instantaneous_packet_error(gamma: float, n: int, l: int) -> float:
    """
    Instantaneous packet error rate from equation (7):
        epsilon_k ≈ Q( sqrt(n / V(gamma)) * ( C(gamma) - l/n ) )
    where
        C(gamma) = log(1 + gamma),
        V(gamma) = 1 - (1 + gamma)^(-2).
    """
    gamma = float(max(gamma, 0.0))
    n = int(n)
    l = int(l)

    if gamma <= 1e-14:
        return 1.0

    #C_gamma = math.log2(1.0 + gamma)
    C_gamma = math.log(1.0 + gamma)
    V_gamma = 1.0 - (1.0 + gamma) ** (-2.0)

    if V_gamma <= 1e-14:
        return 1.0

    arg = math.sqrt(n / V_gamma) * (C_gamma - l / n)
    eps = gaussian_q(arg)
    return float(np.clip(eps, 0.0, 1.0))



def compute_average_packet_error_closed_form(
    pt: float,
    noise_var: float,
    n: int,
    l: int,
) -> Dict[str, float]:
    """
    Average packet error rate from equation (8):
        eps_bar ≈ 1 + ( beta * gbar - beta * gbar * exp(1 / (2 beta gbar)) - 1/2 ) * v
    where
        gbar = pt / sigma^2,
        phi  = exp(l / n),
        v    = exp(-phi / gbar) - 1,
        beta = -sqrt( n / (2*pi*(exp(2l/n)-1)) ).

    This closed-form expression is used for resilience design / optimization.
    """
    pt = float(pt)
    noise_var = float(noise_var)
    n = int(n)
    l = int(l)

    if pt <= 0.0 or noise_var <= 0.0:
        raise ValueError("pt and noise_var must be positive.")

    gbar = pt / noise_var
    phi = math.exp(l / n) - 1.0
    #print(f"phi: {phi}")
    beta = -math.sqrt(n / (2.0 * math.pi * (math.exp(2.0 * l / n) - 1.0)))
    v = math.exp(-phi / gbar)

    eps_bar = 1.0 + (beta * gbar - beta * gbar * math.exp(1.0 / (2.0 * beta * gbar)) - 0.5) * v
    #print(f"esp bar: {eps_bar}")
    eps_bar = float(np.clip(eps_bar, 0.0, 1.0))

    #print(f"beta: {beta}")
    #print(f"gamma bar: {gbar}")
    #print(f"v: {v}")
    #print(f"esp bar: {eps_bar}")

    return {
        "pt": pt,
        "noise_var": noise_var,
        "gbar": gbar,
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
    num_samples: int = 50000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Optional Monte Carlo cross-check of the average packet error rate by
    averaging equation (7) over Rayleigh fading, i.e. |h|^2 ~ Exp(1).

    This is NOT the default used in design. The default design uses equation (8).
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
    eps = np.array([compute_instantaneous_packet_error(g, n, l) for g in gamma], dtype=float)

    return {
        "pt": pt,
        "noise_var": noise_var,
        "gbar": gbar,
        "num_samples": num_samples,
        "eps_bar": float(np.mean(eps)),
    }


def compute_average_packet_error(pt: float, params: ParamsLike, method: Optional[str] = None) -> Dict[str, float]:
    """
    Wrapper for average packet error rate.

    Default behavior for design/optimization: use equation (8) in closed form.
    """
    if method is None:
        method = str(_get_param(params, "AVERAGE_EPSILON_METHOD", "closed_form"))

    noise_var = float(_get_param(params, "CHANNEL_NOISE_VAR", 1.0))
    n = int(_get_param(params, "BLOCKLENGTH_N", 128))
    l = int(_get_param(params, "INFO_BITS_L", 64))

    if method == "closed_form":
        out = compute_average_packet_error_closed_form(pt=pt, noise_var=noise_var, n=n, l=l)
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
    feasible = bool(pu_star <= 1.0 and pu_star >= 0.0)

    return {
        "pu_star": float(pu_star),
        "feasible": feasible,
    }



def compute_expected_D_theta(theta_s: int, eta_s: float) -> float:
    """
    Equation (17):
        E[D_theta,s] ≈ sum_{a=0}^{theta_s-1} (theta_s - a) * eta_s * (1-eta_s)^a
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
# Discrete Weibull and geometric helpers for outage term in constraint (19)
# -----------------------------------------------------------------------------

def discrete_weibull_pmf(
    lam: float,
    kappa: float,
    tail_tol: float = 1e-10,
    max_tau: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    P(t_h = tau) = exp(-(tau/lambda)^kappa) - exp(-((tau+1)/lambda)^kappa), tau = 0,1,2,...
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



def geometric_pmf_from_qss(
    q_ss: float,
    tail_tol: float = 1e-10,
    max_t: int = 100000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sojourn time T_s under the two-state Markov surrogate:
        P(T_s = n) = (q_ss)^(n-1) * (1 - q_ss), n=1,2,...
    """
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
        tail = q_ss ** n
        if tail < tail_tol:
            break

    ts_arr = np.asarray(ts, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)
    probs_arr /= np.sum(probs_arr)
    return ts_arr, probs_arr



def compute_outage_term_state(
    theta_s: int,
    q_ss: float,
    E_I_s: float,
    E_D_theta_s: float,
    weibull_lambda: float,
    weibull_kappa: float,
    tail_tol: float = 1e-10,
) -> Dict[str, float]:
    """
    Approximate the state-s contribution in constraint (19):
        Gamma_s(theta_s) = E[ min{ D_theta,s + t_h, T_s - I_s } / T_s ]

    Design choice used here:
    - keep the formula structure for all known terms,
    - replace I_s by its Monte Carlo estimate E[I_s],
    - use E[D_theta,s] from equation (17),
    - use the exact discrete Weibull pmf and geometric sojourn pmf in the outer expectation.
    """
    del theta_s  # kept only for interface readability

    T_vals, T_pmf = geometric_pmf_from_qss(q_ss=q_ss, tail_tol=tail_tol)
    th_vals, th_pmf = discrete_weibull_pmf(lam=weibull_lambda, kappa=weibull_kappa, tail_tol=tail_tol)

    E_I_s = float(max(E_I_s, 0.0))
    E_D_theta_s = float(max(E_D_theta_s, 0.0))

    gamma_s = 0.0
    for T, pT in zip(T_vals, T_pmf):
        residual = max(float(T) - E_I_s, 0.0)
        inner = 0.0
        for th, pth in zip(th_vals, th_pmf):
            inner += min(E_D_theta_s + float(th), residual) / float(T) * pth
        gamma_s += pT * inner

    return {
        "Gamma_s": float(np.clip(gamma_s, 0.0, 1.0)),
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



def _true_binary_state(x_true: np.ndarray, c: np.ndarray, Delta: float) -> int:
    value = _scalar(c.T @ x_true)
    return 1 if value >= Delta else 0



def estimate_predictive_horizon_moments(
    params: ParamsLike,
    derived: Dict[str, Any],
    num_transitions_per_state: Optional[int] = None,
    burn_in: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate predictive-horizon moments using sensor-side simulation only.

    IMPORTANT:
    - No estimator is involved.
    - No packet drops / no channel are involved.
    - No xi is involved.
    - The sensor runs its local KF and Algorithm 1 only.

    For each true sojourn in state s, we record the first time the sensor's
    predictive procedure identifies the upcoming transition. The predictive
    horizon is then measured as the ACTUAL time-to-transition:
        I_s = T_s - k_detect,
    with I_s = 0 if the transition is not identified in advance.
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

    x_true0 = _get_param(params, "X_TRUE0", np.zeros((n, 1)))
    x_s0 = _get_param(params, "X_S0", np.zeros((n, 1)))
    P_s0 = _get_param(params, "P_S0", np.eye(n))

    x_true = _as_column(x_true0).copy()
    x_hat = _as_column(x_s0).copy()
    P = _as_array(P_s0).copy()

    prev_reliable_decision = 1 if _scalar(c.T @ x_hat) >= xi else 0

    # burn in
    for _ in range(burn_in):
        x_true = _sample_true_process_step(x_true, A, Q, rng)
        y = _sample_measurement(x_true, C, R, rng)
        x_hat, P = _sensor_kf_update(x_hat, P, y, A, C, Q, R)

    # --------------------------------------------------------
    # State / decision control
    # --------------------------------------------------------
    I_0 = []
    I_1 = []

    # sensor current decision using xi
    prev_current_pi = 1 if _scalar(c.T @ x_hat) >= xi else 0

    # predictive update control
    prediction_active = False
    predicted_transition_step = None
    predicted_decision = None

    step_index = 0

    while (len(I_0) < num_transitions_per_state or len(I_1) < num_transitions_per_state) and step_index < max_steps:
        # ----------------------------------------------------
        # 1) advance sensor one step
        # ----------------------------------------------------
        x_true = _sample_true_process_step(x_true, A, Q, rng)
        y = _sample_measurement(x_true, C, R, rng)
        x_hat, P = _sensor_kf_update(x_hat, P, y, A, C, Q, R)

        # current decision based on xi
        current_pi = 1 if _scalar(c.T @ x_hat) >= xi else 0

        # ----------------------------------------------------
        # 2) if a prediction is active:
        #    - do nothing until predicted transition time
        #    - at predicted time, flip the decision
        #    - then allow new predictive detection again
        # ----------------------------------------------------
        if prediction_active:
            if step_index < predicted_transition_step:
                step_index += 1
                continue

            if step_index == predicted_transition_step:
                # auto-switch at predicted time
                prev_current_pi = predicted_decision
                current_pi = predicted_decision

                prediction_active = False
                predicted_transition_step = None
                predicted_decision = None

                # after switching, try to detect a new transition immediately
                pred = predictive_transition_detection(
                    x_hat_sensor=x_hat,
                    P_sensor=P,
                    A=A,
                    Q=Q,
                    c=c,
                    Delta=Delta,
                    alpha_fp=alpha_fp,
                    alpha_fn=alpha_fn,
                    previous_decision=prev_current_pi,
                    ell=ell,
                    xi=xi,
                )

                if pred.get("found_transition") and pred.get("predicted_horizon") is not None:
                    i = int(pred["predicted_horizon"])
                    if i > 0:
                        if prev_current_pi == 0:
                            I_0.append(i)
                        else:
                            I_1.append(i)

                        prediction_active = True
                        predicted_transition_step = step_index + i
                        predicted_decision = 1 - prev_current_pi

                step_index += 1
                continue

        # ----------------------------------------------------
        # 3) no active prediction:
        #    - if current decision changed, record I = 0 for old decision
        #    - update prev_current_pi
        #    - then still run predictive_transition_detection
        # ----------------------------------------------------
        if current_pi != prev_current_pi:
            if prev_current_pi == 0:
                I_0.append(0)
            else:
                I_1.append(0)

            prev_current_pi = current_pi

        pred = predictive_transition_detection(
            x_hat_sensor=x_hat,
            P_sensor=P,
            A=A,
            Q=Q,
            c=c,
            Delta=Delta,
            alpha_fp=alpha_fp,
            alpha_fn=alpha_fn,
            previous_decision=prev_current_pi,
            ell=ell,
            xi=xi,
        )

        if pred.get("found_transition") and pred.get("predicted_horizon") is not None:
            i = int(pred["predicted_horizon"])
            if i > 0:
                if prev_current_pi == 0:
                    I_0.append(i)
                else:
                    I_1.append(i)

                prediction_active = True
                predicted_transition_step = step_index + i
                predicted_decision = 1 - prev_current_pi

        step_index += 1

    if len(I_0) == 0 and len(I_1) == 0:
        raise RuntimeError(
            "Failed to collect predictive-horizon samples for both states. "
            "Increase NUM_I_MONTE_CARLO and/or I_MONTE_CARLO_MAX_STEPS."
        )

    # q values from Markov surrogate are needed for E[q_ss^{I_s}]
    # We estimate only the I_s samples here; q values are injected later.
    return {
        "I_0_samples": np.asarray(I_0, dtype=float),
        "I_1_samples": np.asarray(I_1, dtype=float),
        "E_I_0": float(np.mean(I_0)),
        "E_I_1": float(np.mean(I_1)),
        "num_samples_0": len(I_0),
        "num_samples_1": len(I_1),
        "burn_in": burn_in,
        "seed": seed,
    }



def add_q_power_moments_to_predictive_horizon_stats(
    predictive_stats: Dict[str, Any],
    q00: float,
    q11: float,
) -> Dict[str, Any]:
    """
    Add E[q00^{I0}] and E[q11^{I1}] once q00 and q11 are known.
    """
    I_0 = np.asarray(predictive_stats["I_0_samples"], dtype=float)
    I_1 = np.asarray(predictive_stats["I_1_samples"], dtype=float)

    out = dict(predictive_stats)
    out["E_q00_pow_I0"] = float(np.mean(q00 ** I_0))
    out["E_q11_pow_I1"] = float(np.mean(q11 ** I_1))
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
        r = [2 + pu0 E[q00^I0] q00/q01 + pu1 E[q11^I1] q11/q10] / [q01^{-1} + q10^{-1}]
    """
    numerator = 2.0 + pu0 * E_q00_pow_I0 * (q00 / q01) + pu1 * E_q11_pow_I1 * (q11 / q10)
    denominator = (1.0 / q01) + (1.0 / q10)
    return float(numerator / denominator)



def compute_energy_objective(pt: float, n: int, average_rate: float) -> float:
    """Objective (18): J = n * pt * r(...)."""
    return float(n * pt * average_rate)


# -----------------------------------------------------------------------------
# Design evaluation and solver (Algorithm 2)
# -----------------------------------------------------------------------------

def evaluate_resilience_design(
    pt: float,
    theta0: int,
    theta1: int,
    derived: Dict[str, Any],
    params: ParamsLike,
    predictive_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate one candidate design tuple (pt, theta0, theta1) using:
    - equation (8) for eps_bar,
    - exact formulas for E[D_theta,s], p_u,s^*, rate,
    - semi-analytic outage term in (19), with only E[I_s] coming from Monte Carlo.
    """
    theta0 = int(theta0)
    theta1 = int(theta1)

    pi0 = float(derived["markov_chain_statistics"]["pi0"])
    pi1 = float(derived["markov_chain_statistics"]["pi1"])
    q00 = float(derived["markov_surrogate"]["q00"])
    q11 = float(derived["markov_surrogate"]["q11"])
    q01 = float(derived["markov_surrogate"]["q01"])
    q10 = float(derived["markov_surrogate"]["q10"])

    pr = float(_get_param(params, "P_R", 0.05))
    eps_l = float(_get_param(params, "EPSILON_L", 0.05))
    n = int(_get_param(params, "BLOCKLENGTH_N", 128))
    lam = float(_get_param(params, "TH_RECOVERY_LAMBDA", 4.0))
    kappa = float(_get_param(params, "TH_RECOVERY_KAPPA", 2.0))
    average_eps_method = str(_get_param(params, "AVERAGE_EPSILON_METHOD", "closed_form"))

    avg_eps_info = compute_average_packet_error(pt=pt, params=params, method=average_eps_method)
    eps_bar = float(avg_eps_info["eps_bar"])

    # -----------------------------------------------------------------
    # Iterative fixed-point: eps_d -> eps_r -> pu -> eta -> D_theta
    #                        -> Gamma -> eps_d (repeat until stable)
    #
    # The coupling: D_theta depends on eta, which depends on eps_r,
    # which depends on eps_d, which depends on D_theta. A single pass
    # uses D_theta from an optimistic eps_r and never corrects it.
    # The fix: iterate until eps_d converges.
    # -----------------------------------------------------------------
    MAX_FP_ITER = int(_get_param(params, "MAX_FP_ITER", 15))
    FP_TOL = 1e-8

    # Optimistic initialisation: ignore outage contribution entirely
    eps_d = (1.0 - pr) * eps_bar

    # These will be set by the loop
    pu0 = pu1 = eta0 = eta1 = 0.0
    E_D_theta_0 = E_D_theta_1 = 0.0
    outage0 = outage1 = {"Gamma_s": 0.0}
    converged = False
    _fp_iter = 0

    for _fp_iter in range(MAX_FP_ITER):
        # --- eps_r from current eps_d ---
        eps_r = compute_eps_r_from_eps_d(eps_l=eps_l, eps_d=eps_d)
        if not (0.0 < eps_r < 1.0):
            return {
                "feasible": False,
                "reason": f"eps_r_out_of_bounds_iter{_fp_iter}",
                "pt": float(pt), "theta0": theta0, "theta1": theta1,
                "eps_bar": eps_bar, "eps_d": eps_d, "eps_r": eps_r,
            }

        # --- pu_star from eps_r ---
        pu0_info = compute_pu_star(theta_s=theta0, eps_r=eps_r, eps_bar=eps_bar)
        pu1_info = compute_pu_star(theta_s=theta1, eps_r=eps_r, eps_bar=eps_bar)
        if not pu0_info["feasible"] or not pu1_info["feasible"]:
            return {
                "feasible": False,
                "reason": f"pu_star_infeasible_iter{_fp_iter}",
                "pt": float(pt), "theta0": theta0, "theta1": theta1,
                "eps_bar": eps_bar, "eps_d": eps_d, "eps_r": eps_r,
            }

        pu0 = float(pu0_info["pu_star"])
        pu1 = float(pu1_info["pu_star"])
        eta0 = pu0 * (1.0 - eps_bar)
        eta1 = pu1 * (1.0 - eps_bar)

        # --- D_theta from eta ---
        E_D_theta_0 = compute_expected_D_theta(theta_s=theta0, eta_s=eta0)
        E_D_theta_1 = compute_expected_D_theta(theta_s=theta1, eta_s=eta1)

        # --- Gamma (outage term) from D_theta ---
        outage0 = compute_outage_term_state(
            theta_s=theta0, q_ss=q00,
            E_I_s=float(predictive_stats["E_I_0"]),
            E_D_theta_s=E_D_theta_0,
            weibull_lambda=lam, weibull_kappa=kappa,
        )
        outage1 = compute_outage_term_state(
            theta_s=theta1, q_ss=q11,
            E_I_s=float(predictive_stats["E_I_1"]),
            E_D_theta_s=E_D_theta_1,
            weibull_lambda=lam, weibull_kappa=kappa,
        )

        # --- eps_d from Gamma ---
        eps_d_new = compute_eps_d(
            pr=pr, pi0=pi0, pi1=pi1,
            outage_term_0=float(outage0["Gamma_s"]),
            outage_term_1=float(outage1["Gamma_s"]),
            eps_bar=eps_bar,
        )

        # --- Convergence check ---
        if abs(eps_d_new - eps_d) < FP_TOL:
            eps_d = eps_d_new
            converged = True
            break
        eps_d = eps_d_new

    # After loop: check feasibility with converged values
    if not (0.0 < eps_d < eps_l):
        return {
            "feasible": False,
            "reason": "eps_d_out_of_bounds",
            "pt": float(pt), "theta0": theta0, "theta1": theta1,
            "eps_bar": eps_bar, "eps_d": eps_d, "eps_l": eps_l,
            "fp_converged": converged,
        }

    # Final eps_r from converged eps_d
    eps_r = compute_eps_r_from_eps_d(eps_l=eps_l, eps_d=eps_d)
    if not (0.0 < eps_r < 1.0):
        return {
            "feasible": False,
            "reason": "eps_r_out_of_bounds_final",
            "pt": float(pt), "theta0": theta0, "theta1": theta1,
            "eps_bar": eps_bar, "eps_d": eps_d, "eps_r": eps_r,
        }

    # pu0, pu1, eta0, eta1 are already consistent from the last iteration.
    # Continue to the avg_rate computation as before.
    pu0_info = compute_pu_star(theta_s=theta0, eps_r=eps_r, eps_bar=eps_bar)
    pu1_info = compute_pu_star(theta_s=theta1, eps_r=eps_r, eps_bar=eps_bar)

    if not pu0_info["feasible"] or not pu1_info["feasible"]:
        return {
            "feasible": False,
            "reason": "pu_star_infeasible",
            "pt": float(pt),
            "theta0": theta0,
            "theta1": theta1,
            "eps_bar": eps_bar,
            "eps_d": eps_d,
            "eps_r": eps_r,
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
    J = compute_energy_objective(pt=pt, n=1, average_rate=avg_rate)

    return {
        "feasible": True,
        "reason": "ok",
        "pt": float(pt),
        "theta0": theta0,
        "theta1": theta1,
        "eps_bar": eps_bar,
        "eps_d": eps_d,
        "eps_r": eps_r,
        "pu0_star": pu0,
        "pu1_star": pu1,
        "eta0": eta0,
        "eta1": eta1,
        "E_D_theta_0": E_D_theta_0,
        "E_D_theta_1": E_D_theta_1,
        "Gamma_0": float(outage0["Gamma_s"]),
        "Gamma_1": float(outage1["Gamma_s"]),
        "E_I_0": float(predictive_stats["E_I_0"]),
        "E_I_1": float(predictive_stats["E_I_1"]),
        "E_q00_pow_I0": float(predictive_stats["E_q00_pow_I0"]),
        "E_q11_pow_I1": float(predictive_stats["E_q11_pow_I1"]),
        "average_rate": avg_rate,
        "objective": J,
        "average_packet_error_info": avg_eps_info,
        "fp_converged": converged,
        "fp_iterations": _fp_iter + 1,
    }



def solve_resilience_design(
    derived: Dict[str, Any],
    params: ParamsLike,
) -> Dict[str, Any]:
    """
    Numerical solution of Problem 1, following the spirit of Algorithm 2:
    - estimate predictive-horizon moments once via sensor-only Monte Carlo,
    - loop over theta0, theta1,
    - for each pair solve a 1-D bounded search over pt,
    - keep the best feasible design.
    """
    theta0_candidates = list(_get_param(params, "THETA0_CANDIDATES", list(range(1, 9))))
    theta1_candidates = list(_get_param(params, "THETA1_CANDIDATES", list(range(1, 9))))
    pt_min = float(_get_param(params, "PT_MIN", 0.05))
    pt_max = float(_get_param(params, "PT_MAX", _get_param(params, "RHO", 5.0)))

    q00 = float(derived["markov_surrogate"]["q00"])
    q11 = float(derived["markov_surrogate"]["q11"])

    xi = float(derived["steady_state_benchmark"]["xi"])

    predictive_stats = estimate_predictive_horizon_moments(params=params, derived=derived)
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
            result["line_search_message"] = search.message
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

    # ----------------------------------------------------------------
    # Before/after comparison: run evaluate_resilience_design with
    # MAX_FP_ITER=1 (old single-pass behaviour) and MAX_FP_ITER=15
    # (converged behaviour) using the same Monte Carlo predictive stats.
    # ----------------------------------------------------------------
    q00 = float(derived["markov_surrogate"]["q00"])
    q11 = float(derived["markov_surrogate"]["q11"])
    predictive_stats = estimate_predictive_horizon_moments(params=P, derived=derived)
    predictive_stats = add_q_power_moments_to_predictive_horizon_stats(
        predictive_stats=predictive_stats, q00=q00, q11=q11,
    )

    solution = solve_resilience_design(derived=derived, params=P)

    print("=== Resilience design summary ===")
    if not solution["feasible"]:
        print("No feasible design found.")
        print(solution["reason"])
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
        print(f"fp_converged       : {best['fp_converged']}")
        print(f"fp_iterations      : {best['fp_iterations']}")

        # --- Before/after at the best (theta0, theta1, pt) ---
        t0, t1, pt_best = best["theta0"], best["theta1"], best["pt"]
        P.MAX_FP_ITER = 1
        old_r = evaluate_resilience_design(
            pt=pt_best, theta0=t0, theta1=t1,
            derived=derived, params=P, predictive_stats=predictive_stats,
        )
        P.MAX_FP_ITER = 15
        new_r = evaluate_resilience_design(
            pt=pt_best, theta0=t0, theta1=t1,
            derived=derived, params=P, predictive_stats=predictive_stats,
        )

        print(f"\n=== Before/After at best (theta0={t0}, theta1={t1}, pt={pt_best:.4f}) ===")
        print(f"{'':20s} {'OLD (1-pass)':>18s} {'NEW (converged)':>18s}")
        for key in ("eps_d", "eps_r", "pu0_star", "pu1_star", "objective"):
            ov = old_r.get(key, float("nan")) if old_r.get("feasible") else float("nan")
            nv = new_r.get(key, float("nan")) if new_r.get("feasible") else float("nan")
            print(f"{key:20s} {ov:>18.6f} {nv:>18.6f}")
        old_iters = old_r.get("fp_iterations", 1) if old_r.get("feasible") else 1
        new_iters = new_r.get("fp_iterations", "?") if new_r.get("feasible") else "?"
        print(f"{'fp_iterations':20s} {old_iters:>18} {new_iters!s:>18}")
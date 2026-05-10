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
    """
    Return coding rate in nats/channel-use.

    The finite-blocklength formula in this file uses C(gamma)=ln(1+gamma),
    so the rate subtracted from C(gamma) must be in nats/channel-use.

    Parameters
    ----------
    INFO_BITS_UNIT = "bits":
        l is information bits. Use R = l ln(2) / n. This is dimensionally
        correct if the parameter table says l is bits.

    INFO_BITS_UNIT = "nats":
        l is already measured in nats. Use R = l / n. This reproduces the
        common printed formula phi=exp(l/n)-1 when the paper silently treats
        l as nats.
    """
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
    where
        C(gamma) = ln(1 + gamma),
        V(gamma) = 1 - (1 + gamma)^(-2),
        R is in nats/channel-use.

    If INFO_BITS_UNIT='bits', R = l ln(2)/n.
    If INFO_BITS_UNIT='nats', R = l/n.
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
    """
    Average packet error rate using the segmented-linear FBL approximation.

    With C(gamma)=ln(1+gamma), the rate must be in nats/channel-use.
    Therefore:
        R = l ln(2)/n if l is in bits,
        R = l/n       if l is in nats.

    Then:
        phi = exp(R) - 1,
        beta = -sqrt(n / (2*pi*(exp(2R)-1))).
    """
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
    """
    Monte Carlo cross-check of average packet error by averaging the
    instantaneous FBL PER over Rayleigh fading |h|^2 ~ Exp(1).
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
# Reliability allocation and AoI formulas
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
    Active version of the communication/disruption miss budget:
        eps_d = pr * sum_s pi_s * Gamma_s(theta_s) + (1-pr) * eps_bar.
    """
    eps_d = float(pr) * (float(pi0) * float(outage_term_0) + float(pi1) * float(outage_term_1))
    eps_d += (1.0 - float(pr)) * float(eps_bar)
    return float(np.clip(eps_d, 0.0, 1.0))


def compute_eps_r_from_eps_d(eps_l: float, eps_d: float) -> float:
    """
    Active version of:
        (1 - eps_r)(1 - eps_d) = 1 - eps_l.
    Therefore:
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
    p_u,s^* = (1 - eps_r^(1/theta_s)) / (1 - eps_bar).
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
    E[D_theta,s] under the geometric AoI approximation:
        E[D_theta,s] ~= sum_{a=0}^{theta_s-1} (theta_s - a) eta_s (1-eta_s)^a.
    """
    theta_s = int(theta_s)
    eta_s = float(np.clip(eta_s, 0.0, 1.0))

    if theta_s <= 0:
        raise ValueError("theta_s must be positive.")

    value = 0.0
    for a in range(theta_s):
        value += (theta_s - a) * eta_s * ((1.0 - eta_s) ** a)
    return float(value)


def dtheta_pmf_from_geometric_aoi(theta_s: int, eta_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distribution of D_theta = (theta_s - A)^+ under approximate stationary AoI:
        P(A = a) = eta_s (1 - eta_s)^a, a = 0,1,...

    Thus:
        D = theta_s - a, for a = 0,...,theta_s-1,
        D = 0, with probability P(A >= theta_s) = (1 - eta_s)^theta_s.
    """
    theta_s = int(theta_s)
    eta_s = float(np.clip(eta_s, 0.0, 1.0))

    if theta_s <= 0:
        raise ValueError("theta_s must be positive.")

    if eta_s <= 0.0:
        return np.array([float(theta_s)], dtype=float), np.array([1.0], dtype=float)

    if eta_s >= 1.0:
        return np.array([float(theta_s), 0.0], dtype=float), np.array([1.0, 0.0], dtype=float)

    d_vals = []
    probs = []
    for a in range(theta_s):
        d_vals.append(float(theta_s - a))
        probs.append(eta_s * ((1.0 - eta_s) ** a))

    d_vals.append(0.0)
    probs.append((1.0 - eta_s) ** theta_s)

    d_vals_arr = np.asarray(d_vals, dtype=float)
    probs_arr = np.asarray(probs, dtype=float)
    probs_arr /= np.sum(probs_arr)
    return d_vals_arr, probs_arr


# -----------------------------------------------------------------------------
# Discrete Weibull and geometric helpers
# -----------------------------------------------------------------------------

def discrete_weibull_pmf(
    lam: float,
    kappa: float,
    tail_tol: float = 1e-10,
    max_tau: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    P(t_h = tau) = exp(-(tau/lambda)^kappa) - exp(-((tau+1)/lambda)^kappa), tau=0,1,...
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
        P(T_s = n) = q_ss^(n-1) * (1 - q_ss), n=1,2,...
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


def geometric_inverse_time_moment(q_ss: float) -> float:
    """
    Compute psi_s = E[1 / T_s] for T_s ~ Geom(1 - q_ss), support {1,2,...}:

        P(T_s=t) = (1 - q_ss) q_ss^(t-1).

    Closed form for 0 < q_ss < 1:

        E[1/T_s] = ((1 - q_ss) / q_ss) * [-log(1 - q_ss)].

    This term is used in the conservative outage bound

        Gamma_s <= (E[D_theta,s] + E[t_h]) E[1/T_s].
    """
    q_ss = float(q_ss)

    if not (0.0 <= q_ss < 1.0):
        raise ValueError("q_ss must lie in [0, 1).")

    if q_ss <= 1e-14:
        # q_ss = 0 means T_s = 1 almost surely.
        return 1.0

    p_leave = 1.0 - q_ss
    value = (p_leave / q_ss) * (-math.log(p_leave))
    return float(max(value, 0.0))


def discrete_weibull_mean(
    lam: float,
    kappa: float,
    tail_tol: float = 1e-12,
    max_tau: int = 100000,
) -> float:
    """
    Mean recovery time for the discrete Weibull model used in the paper:

        P(t_h = tau) = exp(-(tau/lambda)^kappa)
                     - exp(-((tau+1)/lambda)^kappa), tau = 0,1,...

    We use the survival identity for a nonnegative integer random variable:

        E[t_h] = sum_{tau=1}^infty P(t_h >= tau)
               = sum_{tau=1}^infty exp(-(tau/lambda)^kappa).

    This is fast because the tail decays quickly for the parameter regimes used here.
    """
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


def compute_outage_term_state_upper_bound_simple(
    theta_s: int,
    q_ss: float,
    eta_s: float,
    weibull_lambda: float,
    weibull_kappa: float,
) -> Dict[str, float]:
    """
    Fast conservative outage-miss surrogate for state s.

    The original Lemma-1 term is

        Gamma_s = E[ min{D_theta,s + t_h, T_s - I_s} / T_s ].

    For tractability, we use the sufficient upper bound

        min{D_theta,s + t_h, T_s - I_s} <= D_theta,s + t_h,

    hence

        Gamma_s <= E[(D_theta,s + t_h)/T_s].

    Under the scheduling-level independence approximation between the AoI/recovery
    process and the Markov sojourn length,

        Gamma_s <= (E[D_theta,s] + E[t_h]) E[1/T_s].

    This avoids the expensive nonlinear expectation over the joint distribution of
    D_theta,s, t_h, T_s, and I_s. It also removes the optimistic mean-inside-min
    approximation.

    Notes
    -----
    * This bound intentionally ignores the helpful effect of predictive lead time I_s
      in the outage constraint. Prediction still affects the average-rate expression
      through E[q_ss^I_s].
    * The returned Gamma_s is clipped to [0,1] because it is a probability bound.
    """
    theta_s = int(theta_s)
    eta_s = float(np.clip(eta_s, 0.0, 1.0))

    E_D_theta_s = compute_expected_D_theta(theta_s=theta_s, eta_s=eta_s)
    E_th = discrete_weibull_mean(lam=weibull_lambda, kappa=weibull_kappa)
    E_inv_T = geometric_inverse_time_moment(q_ss=q_ss)

    gamma_ub_raw = (E_D_theta_s + E_th) * E_inv_T
    gamma_ub = float(np.clip(gamma_ub_raw, 0.0, 1.0))

    return {
        "Gamma_s": gamma_ub,
        "Gamma_raw": float(gamma_ub_raw),
        "Gamma_method": "upper_bound_simple",
        "E_D_theta_s": float(E_D_theta_s),
        "E_th": float(E_th),
        "E_inv_T_s": float(E_inv_T),
        "eta_s": float(eta_s),
    }




def compute_blackout_probability_per_sojourn(
    pr: float,
    q01: float,
    q10: float,
    B0: float,
    B1: float,
) -> Dict[str, float]:
    """
    Long-run blackout probability for the disruption model with one outage
    opportunity per sojourn/stage.

    In each state-s sojourn, an outage starts with probability pr and the start
    time is uniform in that sojourn. If the mean blackout duration is B_s, then
    the expected blackout time per 0->1->0 cycle is pr*B0 + pr*B1. The expected
    cycle length is E[T0] + E[T1] = 1/q01 + 1/q10. Thus

        p_blk = pr * (B0 + B1) / (1/q01 + 1/q10),

    clipped to [0,1]. This is a first-order non-overlap approximation and is
    intended to capture that one disruption can suppress multiple updates.
    """
    pr = float(pr)
    q01 = float(q01)
    q10 = float(q10)
    B0 = float(max(B0, 0.0))
    B1 = float(max(B1, 0.0))

    if not (0.0 < q01 <= 1.0 and 0.0 < q10 <= 1.0):
        raise ValueError("q01 and q10 must lie in (0, 1].")

    E_T0 = 1.0 / q01
    E_T1 = 1.0 / q10
    cycle_length = E_T0 + E_T1
    raw = pr * (B0 + B1) / cycle_length

    return {
        "p_blk": float(np.clip(raw, 0.0, 1.0)),
        "p_blk_raw": float(raw),
        "outage_start_rate_total": float(2.0 * pr / cycle_length),
        "outage_start_rate_state0": float(pr / cycle_length),
        "outage_start_rate_state1": float(pr / cycle_length),
        "E_T0": float(E_T0),
        "E_T1": float(E_T1),
        "cycle_length": float(cycle_length),
        "B0": float(B0),
        "B1": float(B1),
        "model": "per_sojourn_uniform_start",
    }


def compute_blackout_probability_per_slot(
    pr: float,
    pi0: float,
    pi1: float,
    B0: float,
    B1: float,
) -> Dict[str, float]:
    """
    Alternative model: outage starts independently in each slot with probability pr.
    This is NOT the default for the current paper model, but is kept for comparison.
    """
    raw = float(pr) * (float(pi0) * float(max(B0, 0.0)) + float(pi1) * float(max(B1, 0.0)))
    return {
        "p_blk": float(np.clip(raw, 0.0, 1.0)),
        "p_blk_raw": float(raw),
        "outage_start_rate_total": float(pr),
        "outage_start_rate_state0": float(pr) * float(pi0),
        "outage_start_rate_state1": float(pr) * float(pi1),
        "B0": float(max(B0, 0.0)),
        "B1": float(max(B1, 0.0)),
        "model": "per_slot_bernoulli",
    }


def compute_eps_d_from_blackout(p_blk: float, eps_bar: float) -> float:
    """
    Effective delivery-failure probability under the blackout model:

        eps_d = p_blk + (1 - p_blk) eps_bar
              = 1 - (1 - p_blk)(1 - eps_bar).
    """
    p_blk = float(np.clip(p_blk, 0.0, 1.0))
    eps_bar = float(np.clip(eps_bar, 0.0, 1.0))
    return float(np.clip(p_blk + (1.0 - p_blk) * eps_bar, 0.0, 1.0))

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
    Legacy mean-field approximation:
        Gamma_s ~= E_{T,t_h}[ min{E[D_theta] + t_h, T - E[I]} / T ].

    Kept for comparison. Prefer compute_outage_term_state_semianalytic().
    """
    del theta_s

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
        "outage_i_method": "mean",
        "outage_d_method": "mean",
    }


def compute_outage_term_state_semianalytic(
    theta_s: int,
    q_ss: float,
    E_I_s: float,
    E_D_theta_s: float,
    weibull_lambda: float,
    weibull_kappa: float,
    eta_s: Optional[float] = None,
    I_samples: Optional[np.ndarray] = None,
    outage_i_method: str = "mean",
    outage_d_method: str = "mean",
    sample_cap: int = 2000,
    sample_seed: int = 123,
    tail_tol: float = 1e-10,
) -> Dict[str, float]:
    """
    Semi-analytic evaluation of:
        Gamma_s = E[ min{D_theta,s + t_h, T_s - I_s} / T_s ].

    This preserves the nonlinear min structure better than the legacy
    mean-field approximation.

    outage_i_method:
        "mean"      -> use I_s = E[I_s]
        "empirical" -> use empirical I_s samples from the sensor-only Monte Carlo

    outage_d_method:
        "mean"         -> use D_theta = E[D_theta]
        "distribution" -> use geometric-AoI-induced distribution of D_theta
    """
    theta_s = int(theta_s)
    outage_i_method = str(outage_i_method).lower()
    outage_d_method = str(outage_d_method).lower()

    T_vals, T_pmf = geometric_pmf_from_qss(q_ss=q_ss, tail_tol=tail_tol)
    th_vals, th_pmf = discrete_weibull_pmf(lam=weibull_lambda, kappa=weibull_kappa, tail_tol=tail_tol)

    # Representation of I_s.
    if outage_i_method == "mean":
        I_vals = np.array([float(max(E_I_s, 0.0))], dtype=float)
        I_pmf = np.array([1.0], dtype=float)
    elif outage_i_method == "empirical":
        if I_samples is None or len(I_samples) == 0:
            I_vals = np.array([float(max(E_I_s, 0.0))], dtype=float)
            I_pmf = np.array([1.0], dtype=float)
        else:
            I_arr = np.asarray(I_samples, dtype=float)
            I_arr = I_arr[np.isfinite(I_arr)]
            I_arr = np.maximum(I_arr, 0.0)

            if len(I_arr) == 0:
                I_vals = np.array([float(max(E_I_s, 0.0))], dtype=float)
                I_pmf = np.array([1.0], dtype=float)
            else:
                sample_cap = int(sample_cap)
                if sample_cap > 0 and len(I_arr) > sample_cap:
                    rng = np.random.default_rng(sample_seed)
                    idx = rng.choice(len(I_arr), size=sample_cap, replace=False)
                    I_arr = I_arr[idx]

                unique, counts = np.unique(I_arr.astype(int), return_counts=True)
                I_vals = unique.astype(float)
                I_pmf = counts.astype(float) / float(np.sum(counts))
    else:
        raise ValueError("outage_i_method must be 'mean' or 'empirical'.")

    # Representation of D_theta.
    if outage_d_method == "mean":
        D_vals = np.array([float(max(E_D_theta_s, 0.0))], dtype=float)
        D_pmf = np.array([1.0], dtype=float)
    elif outage_d_method == "distribution":
        if eta_s is None:
            raise ValueError("eta_s must be provided when outage_d_method='distribution'.")
        D_vals, D_pmf = dtheta_pmf_from_geometric_aoi(theta_s=theta_s, eta_s=float(eta_s))
    else:
        raise ValueError("outage_d_method must be 'mean' or 'distribution'.")

    gamma_s = 0.0
    for T, pT in zip(T_vals, T_pmf):
        T_float = float(T)
        for I_val, pI in zip(I_vals, I_pmf):
            residual = max(T_float - float(I_val), 0.0)
            if residual <= 0.0:
                continue

            inner = 0.0
            for th, pth in zip(th_vals, th_pmf):
                th_float = float(th)
                for D_val, pD in zip(D_vals, D_pmf):
                    inner += min(float(D_val) + th_float, residual) / T_float * pth * pD

            gamma_s += pT * pI * inner

    return {
        "Gamma_s": float(np.clip(gamma_s, 0.0, 1.0)),
        "E_I_s": float(np.sum(I_vals * I_pmf)),
        "E_D_theta_s": float(np.sum(D_vals * D_pmf)),
        "E_th": float(np.sum(th_vals * th_pmf)),
        "E_T_s": float(np.sum(T_vals * T_pmf)),
        "num_I_support": int(len(I_vals)),
        "num_D_support": int(len(D_vals)),
        "outage_i_method": outage_i_method,
        "outage_d_method": outage_d_method,
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


def estimate_predictive_horizon_moments(
    params: ParamsLike,
    derived: Dict[str, Any],
    num_transitions_per_state: Optional[int] = None,
    burn_in: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate predictive-horizon statistics using sensor-side simulation only.

    This function follows the operational model:
      - No estimator is involved.
      - No packet drops/channel/outage are involved.
      - Transitions are defined by the sensor-side operational decision.
      - I_s is the realized lead time from first predictive/current P issue
        to the next sensor-side decision transition.

    This is different from using the latent true physical threshold state.
    FPR/FNR should still evaluate the relation to the true physical state in
    the full simulation.
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

    del Delta  # Algorithm 1 still receives Delta through predictive_transition_detection.
    Delta = float(_get_param(params, "Delta"))

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

    # Burn in the sensor-side KF.
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

        # Record state occupancy before advancing.
        if previous_state == 0:
            count_state_0 += 1
        else:
            count_state_1 += 1

        # If no predictive packet has been issued in the current sojourn,
        # run Algorithm 1. If it predicts a current/future transition, record
        # the issue time. We do not force the transition to occur at the
        # predicted time; we wait for the realized sensor-side transition.
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

        # Advance sensor one step.
        x_true = _sample_true_process_step(x_true, A, Q, rng)
        y = _sample_measurement(x_true, C, R, rng)
        x_hat, P = _sensor_kf_update(x_hat, P, y, A, C, Q, R)
        step_index += 1
        current_state = sensor_decision(x_hat)

        # Markov transition counts for the sensor-side operational state.
        if previous_state == 0 and current_state == 0:
            trans_00 += 1
        elif previous_state == 0 and current_state == 1:
            trans_01 += 1
        elif previous_state == 1 and current_state == 0:
            trans_10 += 1
        elif previous_state == 1 and current_state == 1:
            trans_11 += 1

        # Realized sensor-side transition.
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
    """
    Objective scaling:
        paper objective: n * pt * r
        simulation-energy matching often uses pt * r.

    Use OBJECTIVE_BLOCKLENGTH_FACTOR in PARAMETERS.py to choose.
    """
    return float(n_factor * pt * average_rate)


def _select_design_markov_stats(
    derived: Dict[str, Any],
    predictive_stats: Dict[str, Any],
    params: ParamsLike,
) -> Dict[str, float]:
    """
    Select which Markov statistics are used in the design.

    DESIGN_MARKOV_SOURCE = "sensor_decision" uses the operational sensor-side
    process estimated in estimate_predictive_horizon_moments().

    DESIGN_MARKOV_SOURCE = "derived" uses derived['markov_surrogate'] and
    derived['markov_chain_statistics'].
    """
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
# Design evaluation and solver
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
    Evaluate one candidate design tuple (pt, theta0, theta1).

    Default disruption model in this revised file:
      - one outage opportunity exists in each sojourn/stage;
      - outage starts with probability P_R in that sojourn;
      - if it starts, its start time is uniform in the sojourn;
      - one blackout can suppress several transition updates.

    The fixed-point coupling is

        eps_d -> eps_r -> p_u -> eta -> D_theta -> p_blk -> eps_d,

    with

        B_s = E[D_theta,s] + E[t_h],
        p_blk = P_R * (B_0 + B_1) / (E[T_0] + E[T_1]),
        eps_d = p_blk + (1 - p_blk) eps_bar(pt).

    The rate objective remains unchanged and still uses E[q_ss^{I_s}].
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
    n_block = int(_get_param(params, "BLOCKLENGTH_N", 128))
    objective_factor = int(_get_param(params, "OBJECTIVE_BLOCKLENGTH_FACTOR", 1))
    lam = float(_get_param(params, "TH_RECOVERY_LAMBDA", 4.0))
    kappa = float(_get_param(params, "TH_RECOVERY_KAPPA", 2.0))
    average_eps_method = str(_get_param(params, "AVERAGE_EPSILON_METHOD", "closed_form"))

    outage_constraint_method = str(
        _get_param(params, "OUTAGE_CONSTRAINT_METHOD", "blackout_per_sojourn")
    ).lower()
    allowed_methods = {
        "blackout_per_sojourn",
        "blackout_stage",
        "blackout_per_stage",
        "blackout_per_slot",
        "upper_bound_simple",
        "option_a",
    }
    if outage_constraint_method not in allowed_methods:
        raise ValueError(
            "OUTAGE_CONSTRAINT_METHOD must be one of: " + ", ".join(sorted(allowed_methods))
        )

    max_fp_iter = int(_get_param(params, "RESILIENCE_FP_MAX_ITER", 50))
    fp_tol = float(_get_param(params, "RESILIENCE_FP_TOL", 1e-8))
    require_fp_convergence = bool(_get_param(params, "REQUIRE_FP_CONVERGENCE", True))

    avg_eps_info = compute_average_packet_error(pt=pt, params=params, method=average_eps_method)
    eps_bar = float(avg_eps_info["eps_bar"])
    E_th = discrete_weibull_mean(lam=lam, kappa=kappa)

    eps_d = eps_bar  # first iterate ignores blackout

    pu0 = pu1 = eta0 = eta1 = 0.0
    E_D_theta_0 = E_D_theta_1 = 0.0
    B0 = B1 = E_th
    p_blk = 0.0
    blackout_info: Dict[str, float] = {"p_blk": 0.0, "p_blk_raw": 0.0, "model": "init"}
    eps_r = 0.0
    fp_residual = float("inf")
    fp_converged = False
    fp_iterations = 0

    for fp_iter in range(max_fp_iter):
        fp_iterations = fp_iter + 1

        if not (0.0 <= eps_d < 1.0):
            return {"feasible": False, "reason": "eps_d_invalid_in_fixed_point", "pt": pt,
                    "theta0": theta0, "theta1": theta1, "eps_bar": eps_bar, "eps_d": eps_d}

        eps_r = compute_eps_r_from_eps_d(eps_l=eps_l, eps_d=eps_d)
        if not (0.0 < eps_r < 1.0):
            return {"feasible": False, "reason": "eps_r_out_of_bounds_in_fixed_point", "pt": pt,
                    "theta0": theta0, "theta1": theta1, "eps_bar": eps_bar, "eps_d": eps_d,
                    "eps_r": eps_r, "p_blk": p_blk}

        pu0_info = compute_pu_star(theta_s=theta0, eps_r=eps_r, eps_bar=eps_bar)
        pu1_info = compute_pu_star(theta_s=theta1, eps_r=eps_r, eps_bar=eps_bar)
        if not pu0_info["feasible"] or not pu1_info["feasible"]:
            return {"feasible": False, "reason": "pu_star_infeasible_in_fixed_point", "pt": pt,
                    "theta0": theta0, "theta1": theta1, "eps_bar": eps_bar, "eps_d": eps_d,
                    "eps_r": eps_r, "p_blk": p_blk}

        pu0 = float(pu0_info["pu_star"])
        pu1 = float(pu1_info["pu_star"])
        eta0 = pu0 * (1.0 - eps_bar)
        eta1 = pu1 * (1.0 - eps_bar)

        E_D_theta_0 = compute_expected_D_theta(theta_s=theta0, eta_s=eta0)
        E_D_theta_1 = compute_expected_D_theta(theta_s=theta1, eta_s=eta1)
        B0 = E_D_theta_0 + E_th
        B1 = E_D_theta_1 + E_th

        if outage_constraint_method in {"blackout_per_sojourn", "blackout_stage", "blackout_per_stage"}:
            blackout_info = compute_blackout_probability_per_sojourn(pr=pr, q01=q01, q10=q10, B0=B0, B1=B1)
            p_blk = float(blackout_info["p_blk"])
            eps_d_new = compute_eps_d_from_blackout(p_blk=p_blk, eps_bar=eps_bar)
        elif outage_constraint_method == "blackout_per_slot":
            blackout_info = compute_blackout_probability_per_slot(pr=pr, pi0=pi0, pi1=pi1, B0=B0, B1=B1)
            p_blk = float(blackout_info["p_blk"])
            eps_d_new = compute_eps_d_from_blackout(p_blk=p_blk, eps_bar=eps_bar)
        else:
            # Legacy Option-A method kept for comparison.
            outage0 = compute_outage_term_state_upper_bound_simple(
                theta_s=theta0, q_ss=q00, eta_s=eta0, weibull_lambda=lam, weibull_kappa=kappa
            )
            outage1 = compute_outage_term_state_upper_bound_simple(
                theta_s=theta1, q_ss=q11, eta_s=eta1, weibull_lambda=lam, weibull_kappa=kappa
            )
            eps_d_new = compute_eps_d(
                pr=pr, pi0=pi0, pi1=pi1,
                outage_term_0=float(outage0["Gamma_s"]),
                outage_term_1=float(outage1["Gamma_s"]),
                eps_bar=eps_bar,
            )
            p_blk = float(pr * (pi0 * outage0["Gamma_s"] + pi1 * outage1["Gamma_s"]))
            blackout_info = {"p_blk": p_blk, "p_blk_raw": p_blk, "model": "legacy_upper_bound_simple",
                             "Gamma_0": float(outage0["Gamma_s"]), "Gamma_1": float(outage1["Gamma_s"])}

        fp_residual = abs(eps_d_new - eps_d)
        eps_d = eps_d_new
        if fp_residual < fp_tol:
            fp_converged = True
            break

    if require_fp_convergence and not fp_converged:
        return {"feasible": False, "reason": "fixed_point_not_converged", "pt": pt,
                "theta0": theta0, "theta1": theta1, "eps_bar": eps_bar, "eps_d": eps_d,
                "eps_r": eps_r, "p_blk": p_blk, "blackout_info": blackout_info,
                "fp_iterations": fp_iterations, "fp_residual": fp_residual}

    if not (0.0 < eps_d < eps_l):
        return {"feasible": False, "reason": "eps_d_out_of_bounds", "pt": pt,
                "theta0": theta0, "theta1": theta1, "eps_bar": eps_bar, "eps_d": eps_d,
                "eps_l": eps_l, "eps_r": eps_r, "p_blk": p_blk, "blackout_info": blackout_info,
                "fp_converged": fp_converged, "fp_iterations": fp_iterations, "fp_residual": fp_residual}

    # Final consistency pass.
    eps_r = compute_eps_r_from_eps_d(eps_l=eps_l, eps_d=eps_d)
    if not (0.0 < eps_r < 1.0):
        return {"feasible": False, "reason": "eps_r_out_of_bounds_final", "pt": pt,
                "theta0": theta0, "theta1": theta1, "eps_bar": eps_bar, "eps_d": eps_d,
                "eps_r": eps_r, "p_blk": p_blk}

    pu0_info = compute_pu_star(theta_s=theta0, eps_r=eps_r, eps_bar=eps_bar)
    pu1_info = compute_pu_star(theta_s=theta1, eps_r=eps_r, eps_bar=eps_bar)
    if not pu0_info["feasible"] or not pu1_info["feasible"]:
        return {"feasible": False, "reason": "pu_star_infeasible_final", "pt": pt,
                "theta0": theta0, "theta1": theta1, "eps_bar": eps_bar, "eps_d": eps_d,
                "eps_r": eps_r, "p_blk": p_blk}

    pu0 = float(pu0_info["pu_star"])
    pu1 = float(pu1_info["pu_star"])
    eta0 = pu0 * (1.0 - eps_bar)
    eta1 = pu1 * (1.0 - eps_bar)
    E_D_theta_0 = compute_expected_D_theta(theta_s=theta0, eta_s=eta0)
    E_D_theta_1 = compute_expected_D_theta(theta_s=theta1, eta_s=eta1)
    B0 = E_D_theta_0 + E_th
    B1 = E_D_theta_1 + E_th

    if outage_constraint_method in {"blackout_per_sojourn", "blackout_stage", "blackout_per_stage"}:
        blackout_info = compute_blackout_probability_per_sojourn(pr=pr, q01=q01, q10=q10, B0=B0, B1=B1)
        p_blk = float(blackout_info["p_blk"])
        eps_d_recomputed = compute_eps_d_from_blackout(p_blk=p_blk, eps_bar=eps_bar)
    elif outage_constraint_method == "blackout_per_slot":
        blackout_info = compute_blackout_probability_per_slot(pr=pr, pi0=pi0, pi1=pi1, B0=B0, B1=B1)
        p_blk = float(blackout_info["p_blk"])
        eps_d_recomputed = compute_eps_d_from_blackout(p_blk=p_blk, eps_bar=eps_bar)
    else:
        outage0 = compute_outage_term_state_upper_bound_simple(
            theta_s=theta0, q_ss=q00, eta_s=eta0, weibull_lambda=lam, weibull_kappa=kappa
        )
        outage1 = compute_outage_term_state_upper_bound_simple(
            theta_s=theta1, q_ss=q11, eta_s=eta1, weibull_lambda=lam, weibull_kappa=kappa
        )
        eps_d_recomputed = compute_eps_d(
            pr=pr, pi0=pi0, pi1=pi1,
            outage_term_0=float(outage0["Gamma_s"]),
            outage_term_1=float(outage1["Gamma_s"]),
            eps_bar=eps_bar,
        )
        p_blk = float(pr * (pi0 * outage0["Gamma_s"] + pi1 * outage1["Gamma_s"]))
        blackout_info = {"p_blk": p_blk, "p_blk_raw": p_blk, "model": "legacy_upper_bound_simple",
                         "Gamma_0": float(outage0["Gamma_s"]), "Gamma_1": float(outage1["Gamma_s"])}

    fp_final_residual = abs(eps_d_recomputed - eps_d)
    eps_d = eps_d_recomputed
    eps_r = compute_eps_r_from_eps_d(eps_l=eps_l, eps_d=eps_d)

    if not (0.0 < eps_d < eps_l):
        return {"feasible": False, "reason": "eps_d_out_of_bounds_final", "pt": pt,
                "theta0": theta0, "theta1": theta1, "eps_bar": eps_bar, "eps_d": eps_d,
                "eps_l": eps_l, "eps_r": eps_r, "p_blk": p_blk, "blackout_info": blackout_info,
                "fp_final_residual": fp_final_residual}

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

    return {
        "feasible": True,
        "reason": "ok",
        "pt": pt,
        "theta0": theta0,
        "theta1": theta1,
        "eps_bar": eps_bar,
        "eps_d": eps_d,
        "eps_r": eps_r,
        "p_blk": p_blk,
        "p_blk_raw": float(blackout_info.get("p_blk_raw", p_blk)),
        "pu0_star": pu0,
        "pu1_star": pu1,
        "eta0": eta0,
        "eta1": eta1,
        "E_D_theta_0": E_D_theta_0,
        "E_D_theta_1": E_D_theta_1,
        "E_th": E_th,
        "B0": B0,
        "B1": B1,
        "Gamma_0": float(blackout_info.get("Gamma_0", np.nan)),
        "Gamma_1": float(blackout_info.get("Gamma_1", np.nan)),
        "E_I_0": float(predictive_stats["E_I_0"]),
        "E_I_1": float(predictive_stats["E_I_1"]),
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
        "outage_constraint_method": outage_constraint_method,
        "blackout_info": blackout_info,
        "fp_converged": fp_converged,
        "fp_iterations": fp_iterations,
        "fp_residual": fp_residual,
        "fp_final_residual": fp_final_residual,
    }

def solve_resilience_design(
    derived: Dict[str, Any],
    params: ParamsLike,
) -> Dict[str, Any]:
    """
    Numerical solution of Problem 1:
      - estimate predictive-horizon moments once via sensor-only Monte Carlo,
      - select Markov statistics according to DESIGN_MARKOV_SOURCE,
      - loop over theta0, theta1,
      - for each pair solve a 1-D bounded search over pt,
      - keep the best feasible design.
    """
    theta0_candidates = list(_get_param(params, "THETA0_CANDIDATES", list(range(1, 9))))
    theta1_candidates = list(_get_param(params, "THETA1_CANDIDATES", list(range(1, 9))))
    pt_min = float(_get_param(params, "PT_MIN", 0.05))
    pt_max = float(_get_param(params, "PT_MAX", _get_param(params, "RHO", 5.0)))

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
    else:
        best = solution["best_design"]
        print(f"theta0             : {best['theta0']}")
        print(f"theta1             : {best['theta1']}")
        print(f"pt*                : {best['pt']:.6f}")
        print(f"eps_bar            : {best['eps_bar']:.6f}")
        print(f"eps_d              : {best['eps_d']:.6f}")
        print(f"eps_r              : {best['eps_r']:.6f}")
        if 'p_blk' in best:
            print(f"p_blk              : {best['p_blk']:.6f}")
            print(f"B0                 : {best.get('B0', float('nan')):.6f}")
            print(f"B1                 : {best.get('B1', float('nan')):.6f}")
        print(f"pu0*               : {best['pu0_star']:.6f}")
        print(f"pu1*               : {best['pu1_star']:.6f}")
        print(f"E[I0]              : {best['E_I_0']:.6f}")
        print(f"E[I1]              : {best['E_I_1']:.6f}")
        print(f"E[q00^I0]          : {best['E_q00_pow_I0']:.6f}")
        print(f"E[q11^I1]          : {best['E_q11_pow_I1']:.6f}")
        if 'Gamma_0' in best and np.isfinite(best.get('Gamma_0', float('nan'))):
            print(f"Gamma0             : {best['Gamma_0']:.6f}")
            print(f"Gamma1             : {best['Gamma_1']:.6f}")
        print(f"average rate       : {best['average_rate']:.6f}")
        print(f"objective          : {best['objective']:.6f}")
        print(f"markov source      : {best['markov_source']}")
        print(f"outage method      : {best['outage_constraint_method']}")
        print(f"FP converged       : {best['fp_converged']}")
        print(f"FP iterations      : {best['fp_iterations']}")
        print(f"FP residual        : {best['fp_residual']:.3e}")
        print(f"FP final residual  : {best['fp_final_residual']:.3e}")

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
from scipy.linalg import solve, solve_discrete_lyapunov, solve_discrete_are
from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize_scalar

ArrayLike = Union[np.ndarray, list, tuple]
ParamsLike = Union[Mapping[str, Any], Any]


DEFAULT_ALPHA_FN = 0.0005
DEFAULT_ALPHA_FP = 0.0005
DEFAULT_LOOKAHEAD_ELL = 5


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _get_param(params: ParamsLike, name: str, default: Any = None) -> Any:
    """Read a parameter from either a module-like object or a dictionary."""
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


def _clip_probability(x: float, eps: float = 1e-12) -> float:
    return float(np.clip(x, eps, 1.0 - eps))


def spectral_radius(A: np.ndarray) -> float:
    """Return the spectral radius rho(A)."""
    eigvals = np.linalg.eigvals(A)
    return float(np.max(np.abs(eigvals)))

def compute_steady_state_kf_posterior_covariance(
    A: ArrayLike,
    C: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    P0: Optional[ArrayLike] = None,
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> Dict[str, Any]:
    """
    Compute the steady-state posterior covariance of the sensor-side Kalman filter.

    This is the covariance of estimation error after measurement update:
        P_inf = lim P_{k|k}

    It is NOT the same as Sigma = var(x_k).
    """
    A = _as_array(A)
    C = _as_array(C)
    Q = _as_array(Q)
    R = _as_array(R)

    n = A.shape[0]
    I = np.eye(n)

    if P0 is None:
        P = np.eye(n, dtype=float)
    else:
        P = _as_array(P0).copy()

    last_diff = None

    for it in range(max_iter):
        P_pred = A @ P @ A.T + Q
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)

        # Joseph form
        P_next = (I - K @ C) @ P_pred @ (I - K @ C).T + K @ R @ K.T

        diff = np.linalg.norm(P_next - P, ord="fro")
        P = P_next

        if diff < tol:
            last_diff = diff
            return {
                "P_inf": P,
                "num_iterations": it + 1,
                "converged": True,
                "residual_norm": diff,
            }

        last_diff = diff

    return {
        "P_inf": P,
        "num_iterations": max_iter,
        "converged": False,
        "residual_norm": last_diff,
    }


def _phi2(x: float, y: float, rho: float) -> float:
    """
    Bivariate standard normal CDF Phi_2(x, y; rho).
    """
    rho = float(np.clip(rho, -1.0, 1.0))
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    return float(multivariate_normal(mean=[0.0, 0.0], cov=cov).cdf([x, y]))


def compute_xi_interval(
    Delta: float,
    sigma_p: float,
    alpha_fp: float,
    alpha_fn: float,
) -> Dict[str, float]:
    """
    Compute the admissible interval of xi inside the confusion region
    in the hat{s}-domain.

    gamma_1 = Delta - z_minus * sigma_p
    gamma_0 = Delta - z_plus  * sigma_p
    """
    bounds = compute_decision_bounds(alpha_fp=alpha_fp, alpha_fn=alpha_fn)
    z_minus = bounds["z_minus"]
    z_plus = bounds["z_plus"]

    gamma_1 = float(Delta - z_minus * sigma_p)
    gamma_0 = float(Delta - z_plus * sigma_p)

    if gamma_0 > gamma_1:
        raise ValueError(
            f"Invalid xi interval: gamma_0={gamma_0} > gamma_1={gamma_1}."
        )

    return {
        "z_minus": z_minus,
        "z_plus": z_plus,
        "gamma_0": gamma_0,
        "gamma_1": gamma_1,
        "xi_lower": gamma_0,
        "xi_upper": gamma_1,
    }


def compute_benchmark_rates_from_xi(
        s_bar: float,
        sigma_s2: float,
        sigma_p2: float,
        Delta: float,
        xi: float,
    ) -> Dict[str, float]:
    """
    Exact steady-state FPR/FNR of the full current decision rule
    when the hybrid rule collapses to the single-threshold rule:
        pi_cur = 1{ hat{s} >= xi }

    This is valid when xi lies inside the admissible interval [gamma_0, gamma_1].
    """
    sigma_s2 = float(max(sigma_s2, 0.0))
    sigma_p2 = float(max(sigma_p2, 0.0))

    if sigma_p2 > sigma_s2 + 1e-10:
        raise ValueError(
            f"sigma_p^2 must not exceed sigma_s^2, got sigma_p2={sigma_p2}, sigma_s2={sigma_s2}."
        )

    sigma_hat2 = max(sigma_s2 - sigma_p2, 0.0)
    sigma_s = float(np.sqrt(sigma_s2))
    sigma_hat = float(np.sqrt(sigma_hat2))

    if sigma_s <= 0.0:
        raise ValueError("sigma_s must be positive.")
    if sigma_hat <= 0.0:
        raise ValueError("sigma_hat must be positive.")

    a = float((Delta - s_bar) / sigma_s)
    b = float((xi - s_bar) / sigma_hat)
    rho = float(np.clip(sigma_hat / sigma_s, -1.0, 1.0))

    Phi_a = float(norm.cdf(a))
    Phi_b = float(norm.cdf(b))
    Phi2_ab = _phi2(a, b, rho)

    denom_neg = Phi_a
    denom_pos = 1.0 - Phi_a

    FPR = 0.0 if denom_neg <= 0.0 else float((Phi_a - Phi2_ab) / denom_neg)
    FNR = 0.0 if denom_pos <= 0.0 else float((Phi_b - Phi2_ab) / denom_pos)

    return {
        "FPR_xi": FPR,
        "FNR_xi": FNR,
        "sigma_hat2": sigma_hat2,
        "sigma_hat": sigma_hat,
        "sigma_s": sigma_s,
        "rho": rho,
        "a": a,
        "b_xi": b,
        "Phi_a": Phi_a,
        "Phi_b_xi": Phi_b,
        "Phi2_a_bxi": Phi2_ab,
    }


def compute_optimal_xi(
    s_bar: float,
    sigma_s2: float,
    sigma_p2: float,
    Delta: float,
    alpha_fp: float,
    alpha_fn: float,
    weight_fp: float = 1.0,
    weight_fn: float = 1.0,
) -> Dict[str, Any]:
    """
    Optimize xi inside [gamma_0, gamma_1] for the current sensor-side decision.

    Objective:
        J(xi) = weight_fp * FPR(xi) + weight_fn * FNR(xi)
    """
    sigma_p = float(np.sqrt(max(sigma_p2, 0.0)))
    xi_info = compute_xi_interval(
        Delta=Delta,
        sigma_p=sigma_p,
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
    )

    xi_lower = xi_info["xi_lower"]
    xi_upper = xi_info["xi_upper"]

    def objective(xi_value: float) -> float:
        rates = compute_benchmark_rates_from_xi(
            s_bar=s_bar,
            sigma_s2=sigma_s2,
            sigma_p2=sigma_p2,
            Delta=Delta,
            xi=xi_value,
        )
        return float(weight_fp * rates["FPR_xi"] + weight_fn * rates["FNR_xi"])

    result = minimize_scalar(
        objective,
        bounds=(xi_lower, xi_upper),
        method="bounded",
    )

    xi_star = float(result.x)
    rates_star = compute_benchmark_rates_from_xi(
        s_bar=s_bar,
        sigma_s2=sigma_s2,
        sigma_p2=sigma_p2,
        Delta=Delta,
        xi=xi_star,
    )

    return {
        **xi_info,
        "xi_star": xi_star,
        "objective_value": float(result.fun),
        "optimization_success": bool(result.success),
        "optimization_message": result.message,
        "weight_fp": float(weight_fp),
        "weight_fn": float(weight_fn),
        **rates_star,
    }


def compute_steady_state_benchmark(
    A: ArrayLike,
    C: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    c: ArrayLike,
    Delta: float,
    alpha_fp: float,
    alpha_fn: float,
    mu_w: Optional[ArrayLike] = None,
    xi_mode: str = "optimal",
    xi_value: Optional[float] = None,
    weight_fp: float = 1.0,
    weight_fn: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute the steady-state benchmark for the CURRENT sensor-side decision
    using the hybrid rule resolved by xi inside the confusion region.

    IMPORTANT:
    - This benchmark is for the current decision only.
    - Predictive update logic should still use the reliable thresholds z_minus, z_plus.
    """
    stationary = compute_stationary_statistics(
        A=A,
        Q=Q,
        c=c,
        Delta=Delta,
        mu_w=mu_w,
    )

    kf_ss = compute_steady_state_kf_posterior_covariance(
        A=A,
        C=C,
        Q=Q,
        R=R,
    )
    P_inf = kf_ss["P_inf"]

    c_col = _as_column(c)
    sigma_p2 = _scalar(c_col.T @ P_inf @ c_col)
    sigma_p2 = max(sigma_p2, 0.0)
    sigma_p = float(np.sqrt(sigma_p2))

    xi_interval = compute_xi_interval(
        Delta=Delta,
        sigma_p=sigma_p,
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
    )

    if xi_mode == "optimal":
        xi_result = compute_optimal_xi(
            s_bar=stationary["s_bar"],
            sigma_s2=stationary["sigma_s2"],
            sigma_p2=sigma_p2,
            Delta=Delta,
            alpha_fp=alpha_fp,
            alpha_fn=alpha_fn,
            weight_fp=weight_fp,
            weight_fn=weight_fn,
        )
        xi = xi_result["xi_star"]

    elif xi_mode == "delta":
        xi = float(Delta)
        if not (xi_interval["xi_lower"] <= xi <= xi_interval["xi_upper"]):
            xi = float(np.clip(xi, xi_interval["xi_lower"], xi_interval["xi_upper"]))
        xi_result = {
            **xi_interval,
            "xi_star": xi,
            "objective_value": None,
            "optimization_success": True,
            "optimization_message": "xi fixed at Delta, clipped into admissible interval if needed.",
            "weight_fp": float(weight_fp),
            "weight_fn": float(weight_fn),
        }

    elif xi_mode == "manual":
        if xi_value is None:
            raise ValueError("xi_value must be provided when xi_mode='manual'.")
        xi = float(xi_value)
        xi_result = {
            **xi_interval,
            "xi_star": xi,
            "objective_value": None,
            "optimization_success": True,
            "optimization_message": "xi fixed manually by user.",
            "weight_fp": float(weight_fp),
            "weight_fn": float(weight_fn),
        }

    else:
        raise ValueError(
            f"Unsupported xi_mode='{xi_mode}'. Use 'optimal', 'delta', or 'manual'."
        )

    xi_in_interval = bool(xi_interval["xi_lower"] <= xi <= xi_interval["xi_upper"])

    rates = compute_benchmark_rates_from_xi(
        s_bar=stationary["s_bar"],
        sigma_s2=stationary["sigma_s2"],
        sigma_p2=sigma_p2,
        Delta=Delta,
        xi=xi,
    )

    return {
        "s_bar": stationary["s_bar"],
        "sigma_s2": stationary["sigma_s2"],
        "Sigma": stationary["Sigma"],
        "P_inf": P_inf,
        "sigma_p2": sigma_p2,
        "sigma_p": sigma_p,
        "xi": xi,
        "xi_mode": xi_mode,
        "xi_in_admissible_interval": xi_in_interval,
        **xi_interval,
        **xi_result,
        **rates,
        "kf_steady_state": kf_ss,
    }


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_parameters(params: ParamsLike) -> None:
    """
    Validate the core system matrices used throughout the paper.

    Required entries:
        A, C, Q, R, c, Delta
    Optional but recommended:
        MU_W, ALPHA_FN, ALPHA_FP, LOOKAHEAD_ELL
    """
    A = _as_array(_get_param(params, "A"))
    C = _as_array(_get_param(params, "C"))
    Q = _as_array(_get_param(params, "Q"))
    R = _as_array(_get_param(params, "R"))
    c = _as_column(_get_param(params, "c"))
    Delta = _get_param(params, "Delta")

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}.")

    n = A.shape[0]

    if C.ndim != 2 or C.shape[1] != n:
        raise ValueError(f"C must have shape (m, {n}), got {C.shape}.")

    if Q.shape != (n, n):
        raise ValueError(f"Q must have shape ({n}, {n}), got {Q.shape}.")

    if c.shape != (n, 1):
        raise ValueError(f"c must have shape ({n}, 1), got {c.shape}.")

    m = C.shape[0]
    if R.shape != (m, m):
        raise ValueError(f"R must have shape ({m}, {m}), got {R.shape}.")

    if not np.allclose(Q, Q.T, atol=1e-12):
        raise ValueError("Q must be symmetric.")

    if not np.allclose(R, R.T, atol=1e-12):
        raise ValueError("R must be symmetric.")

    if Delta is None:
        raise ValueError("Delta is required.")

    mu_w = _get_param(params, "MU_W", None)
    if mu_w is not None:
        mu_w = _as_column(mu_w)
        if mu_w.shape != (n, 1):
            raise ValueError(f"MU_W must have shape ({n}, 1), got {mu_w.shape}.")

    alpha_fn = _get_param(params, "ALPHA_FN", DEFAULT_ALPHA_FN)
    alpha_fp = _get_param(params, "ALPHA_FP", DEFAULT_ALPHA_FP)
    if not (0.0 < alpha_fn < 1.0):
        raise ValueError("ALPHA_FN must lie in (0, 1).")
    if not (0.0 < alpha_fp < 1.0):
        raise ValueError("ALPHA_FP must lie in (0, 1).")

    ell = _get_param(params, "LOOKAHEAD_ELL", DEFAULT_LOOKAHEAD_ELL)
    if int(ell) < 0:
        raise ValueError("LOOKAHEAD_ELL must be nonnegative.")


# -----------------------------------------------------------------------------
# Stationary system quantities
# -----------------------------------------------------------------------------

def compute_stationary_statistics(
    A: ArrayLike,
    Q: ArrayLike,
    c: ArrayLike,
    Delta: float,
    mu_w: Optional[ArrayLike] = None,
) -> Dict[str, Any]:
    """
    Compute stationary Gaussian quantities for the stable LTI model.

    Returns:
        dict containing rho(A), x_bar, Sigma, s_bar, sigma_s2, sigma_s, a, rho_s
    """
    A = _as_array(A)
    Q = _as_array(Q)
    c = _as_column(c)

    n = A.shape[0]
    if mu_w is None:
        mu_w = np.zeros((n, 1), dtype=float)
    else:
        mu_w = _as_column(mu_w)

    I = np.eye(n)
    rho_A = spectral_radius(A)

    x_bar = solve(I - A, mu_w, assume_a="gen")
    Sigma = solve_discrete_lyapunov(A, Q)

    s_bar = _scalar(c.T @ x_bar)
    sigma_s2 = _scalar(c.T @ Sigma @ c)
    if sigma_s2 < 0:
        raise ValueError(f"Computed negative variance sigma_s^2 = {sigma_s2}.")

    sigma_s2 = max(sigma_s2, 0.0)
    sigma_s = float(np.sqrt(sigma_s2))

    if sigma_s == 0.0:
        a = np.inf if Delta > s_bar else (-np.inf if Delta < s_bar else 0.0)
        rho_s = 0.0
    else:
        a = float((Delta - s_bar) / sigma_s)
        rho_s = _scalar(c.T @ A @ Sigma @ c) / sigma_s2
        rho_s = float(np.clip(rho_s, -1.0, 1.0))

    return {
        "spectral_radius": rho_A,
        "is_asymptotically_stable": rho_A < 1.0,
        "x_bar": x_bar,
        "Sigma": Sigma,
        "s_bar": s_bar,
        "sigma_s2": sigma_s2,
        "sigma_s": sigma_s,
        "a": a,
        "rho_s": rho_s,
    }


# -----------------------------------------------------------------------------
# Decision-feasibility quantities
# -----------------------------------------------------------------------------

def compute_decision_bounds(alpha_fp: float, alpha_fn: float) -> Dict[str, float]:
    """
    Compute the feasible decision boundaries:
        z_minus = Phi^{-1}(alpha_FP)
        z_plus  = Phi^{-1}(1 - alpha_FN)
    """
    alpha_fp = _clip_probability(alpha_fp)
    alpha_fn = _clip_probability(alpha_fn)

    z_minus = float(norm.ppf(alpha_fp))
    z_plus = float(norm.ppf(1.0 - alpha_fn))

    return {
        "alpha_fp": alpha_fp,
        "alpha_fn": alpha_fn,
        "z_minus": z_minus,
        "z_plus": z_plus,
        "confusion_region": (z_minus, z_plus),
    }


def compute_decision_statistics(
    x_hat: ArrayLike,
    P: ArrayLike,
    c: ArrayLike,
    Delta: float,
) -> Dict[str, float]:
    """
    Compute s_hat, sigma^2, sigma, and z = (Delta - s_hat)/sigma.
    """
    x_hat = _as_column(x_hat)
    P = _as_array(P)
    c = _as_column(c)

    s_hat = _scalar(c.T @ x_hat)
    sigma2 = _scalar(c.T @ P @ c)
    if sigma2 < -1e-12:
        raise ValueError(f"Conditional variance is negative: {sigma2}")

    sigma2 = max(sigma2, 0.0)
    sigma = float(np.sqrt(sigma2))

    if sigma == 0.0:
        if Delta > s_hat:
            z = np.inf
        elif Delta < s_hat:
            z = -np.inf
        else:
            z = 0.0
    else:
        z = float((Delta - s_hat) / sigma)

    return {
        "s_hat": s_hat,
        "sigma2": sigma2,
        "sigma": sigma,
        "z": z,
    }


def infer_decision_region(
    z: float,
    z_minus: float,
    z_plus: float,
    previous_decision: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Apply the decision-feasibility rule from Section III-A.
    """
    if z <= z_minus:
        return {"pi": 1, "region": "state_1_feasible"}
    if z >= z_plus:
        return {"pi": 0, "region": "state_0_feasible"}
    return {"pi": previous_decision, "region": "confusion"}


def evaluate_decision(
    x_hat: ArrayLike,
    P: ArrayLike,
    c: ArrayLike,
    Delta: float,
    alpha_fp: float,
    alpha_fn: float,
    previous_decision: Optional[int] = None,
) -> Dict[str, Any]:
    """
    One-stop helper for the decision rule:
    compute (s_hat, sigma^2, z) and the resulting feasible decision.
    """
    stats = compute_decision_statistics(x_hat, P, c, Delta)
    bounds = compute_decision_bounds(alpha_fp, alpha_fn)
    region = infer_decision_region(
        z=stats["z"],
        z_minus=bounds["z_minus"],
        z_plus=bounds["z_plus"],
        previous_decision=previous_decision,
    )

    return {**stats, **bounds, **region}


# -----------------------------------------------------------------------------
# Prediction helpers and Algorithm 1
# -----------------------------------------------------------------------------

def predict_one_step(
    x_hat: ArrayLike,
    P: ArrayLike,
    A: ArrayLike,
    Q: ArrayLike,
) -> Dict[str, np.ndarray]:
    """
    One-step state and covariance prediction:
        x^+ = A x
        P^+ = A P A^T + Q
    """
    x_hat = _as_column(x_hat)
    P = _as_array(P)
    A = _as_array(A)
    Q = _as_array(Q)

    x_next = A @ x_hat
    P_next = A @ P @ A.T + Q
    return {"x_hat": x_next, "P": P_next}



def predictive_transition_detection(
    x_hat_sensor: ArrayLike,
    P_sensor: ArrayLike,
    previous_decision: Optional[int],
    ell: int,
    A: ArrayLike,
    Q: ArrayLike,
    c: ArrayLike,
    Delta: float,
    alpha_fp: float,
    alpha_fn: float,
) -> Dict[str, Any]:
    """
    Implementation of Algorithm 1 in horizon form.

    Notes:
    - This function returns a predicted horizon offset, not the absolute time k+i.
    - offset = 0 means a reliable decision change is already detected now.
    - offset = None means no certified transition is found within ell steps.
    """
    ell = int(ell)
    if ell < 0:
        raise ValueError("ell must be nonnegative.")

    current = evaluate_decision(
        x_hat=x_hat_sensor,
        P=P_sensor,
        c=c,
        Delta=Delta,
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
        previous_decision=previous_decision,
    )

    current_pi = current["pi"]
    history = [
        {
            "offset": 0,
            "x_hat": _as_column(x_hat_sensor).copy(),
            "P": _as_array(P_sensor).copy(),
            "decision": current,
        }
    ]

    if current["region"] != "confusion" and current_pi != previous_decision:
        return {
            "found_transition": True,
            "predicted_horizon": 0,
            "predicted_decision": current_pi,
            "path": history,
        }

    x_pred = _as_column(x_hat_sensor).copy()
    P_pred = _as_array(P_sensor).copy()

    for i in range(1, ell + 1):
        pred = predict_one_step(x_pred, P_pred, A, Q)
        x_pred = pred["x_hat"]
        P_pred = pred["P"]

        decision_i = evaluate_decision(
            x_hat=x_pred,
            P=P_pred,
            c=c,
            Delta=Delta,
            alpha_fp=alpha_fp,
            alpha_fn=alpha_fn,
            previous_decision=current_pi,
        )

        history.append(
            {
                "offset": i,
                "x_hat": x_pred.copy(),
                "P": P_pred.copy(),
                "decision": decision_i,
            }
        )

        if decision_i["region"] != "confusion" and decision_i["pi"] != current_pi:
            return {
                "found_transition": True,
                "predicted_horizon": i,
                "predicted_decision": decision_i["pi"],
                "path": history,
            }

    return {
        "found_transition": False,
        "predicted_horizon": None,
        "predicted_decision": None,
        "path": history,
    }



# -----------------------------------------------------------------------------
# Markov-surrogate quantities from Theorem 1
# -----------------------------------------------------------------------------

def compute_markov_surrogate(
    A: ArrayLike,
    Sigma: ArrayLike,
    c: ArrayLike,
    Delta: float,
    s_bar: float,
    sigma_s: float,
) -> Dict[str, Any]:
    """
    Compute the one-step threshold-crossing probabilities q01 and q10,
    then derive q00 and q11.
    """
    A = _as_array(A)
    Sigma = _as_array(Sigma)
    c = _as_column(c)

    if sigma_s <= 0.0:
        raise ValueError("sigma_s must be positive to compute threshold-crossing probabilities.")

    sigma_s2 = float(sigma_s ** 2)
    rho_s = _scalar(c.T @ A @ Sigma @ c) / sigma_s2
    rho_s = float(np.clip(rho_s, -1.0, 1.0))

    a = float((Delta - s_bar) / sigma_s)
    phi_a = float(norm.cdf(a))
    phi_a = _clip_probability(phi_a)

    cov = np.array([[1.0, rho_s], [rho_s, 1.0]], dtype=float)
    phi2_aa = float(multivariate_normal(mean=[0.0, 0.0], cov=cov).cdf([a, a]))

    q01 = 1.0 - phi2_aa / phi_a
    q10 = (phi_a - phi2_aa) / (1.0 - phi_a)

    q01 = float(np.clip(q01, 0.0, 1.0))
    q10 = float(np.clip(q10, 0.0, 1.0))
    q00 = float(1.0 - q01)
    q11 = float(1.0 - q10)

    return {
        "rho_s": rho_s,
        "a": a,
        "phi_a": phi_a,
        "phi2_aa": phi2_aa,
        "q00": q00,
        "q01": q01,
        "q10": q10,
        "q11": q11,
        "transition_matrix": np.array([[q00, q01], [q10, q11]], dtype=float),
    }



def compute_markov_chain_statistics(q00: float, q01: float, q10: float, q11: float) -> Dict[str, Any]:
    """
    Compute stationary probabilities and mean sojourn times of the two-state chain.
    """
    for name, value in {"q00": q00, "q01": q01, "q10": q10, "q11": q11}.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}.")

    if not np.isclose(q00 + q01, 1.0, atol=1e-10):
        raise ValueError("Row 0 of the transition matrix must sum to 1.")
    if not np.isclose(q10 + q11, 1.0, atol=1e-10):
        raise ValueError("Row 1 of the transition matrix must sum to 1.")

    denom = q01 + q10
    if denom <= 0.0:
        raise ValueError("q01 + q10 must be positive to compute stationary probabilities.")

    pi0 = float(q10 / denom)
    pi1 = float(q01 / denom)

    mean_sojourn_state_0 = float(np.inf if q01 == 0.0 else 1.0 / q01)
    mean_sojourn_state_1 = float(np.inf if q10 == 0.0 else 1.0 / q10)

    return {
        "pi0": pi0,
        "pi1": pi1,
        "stationary_distribution": np.array([pi0, pi1], dtype=float),
        "mean_sojourn_state_0": mean_sojourn_state_0,
        "mean_sojourn_state_1": mean_sojourn_state_1,
    }


# -----------------------------------------------------------------------------
# High-level wrapper
# -----------------------------------------------------------------------------

def precompute_all(params: ParamsLike) -> Dict[str, Any]:
    """
    Main entry point for deterministic quantities that are already fixed by the
    current parameters file.

    This function intentionally does NOT compute optimization-stage variables
    such as pt, theta_0, theta_1, epsilon_d, epsilon_r, or Monte-Carlo-based E[q_ss^I].
    """
    validate_parameters(params)

    A = _as_array(_get_param(params, "A"))
    C = _as_array(_get_param(params, "C"))
    Q = _as_array(_get_param(params, "Q"))
    R = _as_array(_get_param(params, "R"))
    c = _as_column(_get_param(params, "c"))
    Delta = float(_get_param(params, "Delta"))

    mu_w = _get_param(params, "MU_W", np.zeros((A.shape[0], 1), dtype=float))
    mu_w = _as_column(mu_w)

    alpha_fn = float(_get_param(params, "ALPHA_FN", DEFAULT_ALPHA_FN))
    alpha_fp = float(_get_param(params, "ALPHA_FP", DEFAULT_ALPHA_FP))
    ell = int(_get_param(params, "LOOKAHEAD_ELL", DEFAULT_LOOKAHEAD_ELL))

    xi_mode = _get_param(params, "XI_MODE", "optimal")
    xi_value = _get_param(params, "XI_VALUE", None)
    weight_fp = float(_get_param(params, "XI_WEIGHT_FP", 1.0))
    weight_fn = float(_get_param(params, "XI_WEIGHT_FN", 1.0))

    stationary = compute_stationary_statistics(
        A=A,
        Q=Q,
        c=c,
        Delta=Delta,
        mu_w=mu_w,
    )

    decision_bounds = compute_decision_bounds(
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
    )

    markov = compute_markov_surrogate(
        A=A,
        Sigma=stationary["Sigma"],
        c=c,
        Delta=Delta,
        s_bar=stationary["s_bar"],
        sigma_s=stationary["sigma_s"],
    )

    chain_stats = compute_markov_chain_statistics(
        q00=markov["q00"],
        q01=markov["q01"],
        q10=markov["q10"],
        q11=markov["q11"],
    )

    benchmark = compute_steady_state_benchmark(
        A=A,
        C=C,
        Q=Q,
        R=R,
        c=c,
        Delta=Delta,
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
        mu_w=mu_w,
        xi_mode=xi_mode,
        xi_value=xi_value,
        weight_fp=weight_fp,
        weight_fn=weight_fn,
    )

    return {
        "used_defaults": {
            "MU_W_defaulted": _get_param(params, "MU_W", None) is None,
            "ALPHA_FN_defaulted": _get_param(params, "ALPHA_FN", None) is None,
            "ALPHA_FP_defaulted": _get_param(params, "ALPHA_FP", None) is None,
            "LOOKAHEAD_ELL_defaulted": _get_param(params, "LOOKAHEAD_ELL", None) is None,
            "XI_MODE_defaulted": _get_param(params, "XI_MODE", None) is None,
        },
        "basic_inputs": {
            "Delta": Delta,
            "MU_W": mu_w,
            "ALPHA_FN": alpha_fn,
            "ALPHA_FP": alpha_fp,
            "LOOKAHEAD_ELL": ell,
            "XI_MODE": xi_mode,
            "XI_VALUE": xi_value,
            "XI_WEIGHT_FP": weight_fp,
            "XI_WEIGHT_FN": weight_fn,
        },
        "stationary": stationary,
        "decision_bounds": decision_bounds,
        "markov_surrogate": markov,
        "markov_chain_statistics": chain_stats,
        "steady_state_benchmark": benchmark,
    }


if __name__ == "__main__":
    import PARAMETERS as P

    derived = precompute_all(P)

    print("=== Deterministic precomputation summary ===")
    print(f"spectral_radius(A) : {derived['stationary']['spectral_radius']:.6f}")
    print(f"stable             : {derived['stationary']['is_asymptotically_stable']}")
    print(f"s_bar              : {derived['stationary']['s_bar']:.6f}")
    print(f"sigma_s^2          : {derived['stationary']['sigma_s2']:.6f}")
    print(f"a                  : {derived['stationary']['a']:.6f}")
    print(f"q00                : {derived['markov_surrogate']['q00']:.6f}")
    print(f"q01                : {derived['markov_surrogate']['q01']:.6f}")
    print(f"q10                : {derived['markov_surrogate']['q10']:.6f}")
    print(f"q11                : {derived['markov_surrogate']['q11']:.6f}")
    print(f"pi0                : {derived['markov_chain_statistics']['pi0']:.6f}")
    print(f"pi1                : {derived['markov_chain_statistics']['pi1']:.6f}")

    print("\n=== Steady-state benchmark for current decision ===")
    print(f"xi mode            : {derived['steady_state_benchmark']['xi_mode']}")
    print(f"xi*                : {derived['steady_state_benchmark']['xi']:.6f}")
    print(f"xi lower bound     : {derived['steady_state_benchmark']['xi_lower']:.6f}")
    print(f"xi upper bound     : {derived['steady_state_benchmark']['xi_upper']:.6f}")
    print(f"xi admissible      : {derived['steady_state_benchmark']['xi_in_admissible_interval']}")
    print(f"FPR(xi*)           : {derived['steady_state_benchmark']['FPR_xi']:.6f}")
    print(f"FNR(xi*)           : {derived['steady_state_benchmark']['FNR_xi']:.6f}")
    print(f"objective value    : {derived['steady_state_benchmark']['objective_value']}")
    print(f"sigma_p^2          : {derived['steady_state_benchmark']['sigma_p2']:.6f}")
    print(f"sigma_hat^2        : {derived['steady_state_benchmark']['sigma_hat2']:.6f}")
    print(f"rho                : {derived['steady_state_benchmark']['rho']:.6f}")

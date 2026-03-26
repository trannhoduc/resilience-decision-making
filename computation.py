from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
from scipy.linalg import solve, solve_discrete_lyapunov
from scipy.stats import multivariate_normal, norm

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
    Q = _as_array(_get_param(params, "Q"))
    c = _as_column(_get_param(params, "c"))
    Delta = float(_get_param(params, "Delta"))

    mu_w = _get_param(params, "MU_W", np.zeros((A.shape[0], 1), dtype=float))
    mu_w = _as_column(mu_w)

    alpha_fn = float(_get_param(params, "ALPHA_FN", DEFAULT_ALPHA_FN))
    alpha_fp = float(_get_param(params, "ALPHA_FP", DEFAULT_ALPHA_FP))
    ell = int(_get_param(params, "LOOKAHEAD_ELL", DEFAULT_LOOKAHEAD_ELL))

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

    return {
        "used_defaults": {
            "MU_W_defaulted": _get_param(params, "MU_W", None) is None,
            "ALPHA_FN_defaulted": _get_param(params, "ALPHA_FN", None) is None,
            "ALPHA_FP_defaulted": _get_param(params, "ALPHA_FP", None) is None,
            "LOOKAHEAD_ELL_defaulted": _get_param(params, "LOOKAHEAD_ELL", None) is None,
        },
        "basic_inputs": {
            "Delta": Delta,
            "MU_W": mu_w,
            "ALPHA_FN": alpha_fn,
            "ALPHA_FP": alpha_fp,
            "LOOKAHEAD_ELL": ell,
        },
        "stationary": stationary,
        "decision_bounds": decision_bounds,
        "markov_surrogate": markov,
        "markov_chain_statistics": chain_stats,
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

"""
Baseline calibration for the resilience-decision-making simulation.

Given the proposed policy's optimal transmit power p_t_star and its average
transmission rate r_prop, this module computes transmit powers and scheduling
parameters for baselines so that every baseline matches the same average
energy budget

    E = p_t_star * r_prop

The baselines are:
  1. Predictive-only
  2. Event-triggered
  3. Probability-based
  4. AoII-based

Important:
AoI detection thresholds are computed from each baseline's effective successful
update rate, not directly from the Markov sojourn probability. This is more
consistent with the AoI-tail approximation

    P(AoI > theta) ≈ (1 - eta)^theta,

where eta is the per-slot probability of receiving a useful update under
normal operation.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np
from scipy.optimize import minimize_scalar


def _clip_probability(x: float, eps: float = 1e-12) -> float:
    """
    Keep a probability strictly inside (0, 1) to avoid log(0)
    and division problems.
    """
    return min(max(float(x), eps), 1.0 - eps)


def _theta_from_eta(eta: float, epsilon_r: float) -> int:
    """
    Compute the smallest integer theta satisfying

        P(AoI > theta) ≈ (1 - eta)^theta <= epsilon_r.

    Hence,

        theta >= log(epsilon_r) / log(1 - eta).

    Parameters
    ----------
    eta : float
        Effective per-slot successful update probability.
    epsilon_r : float
        Target false-trigger probability.

    Returns
    -------
    int
        AoI detection threshold.
    """
    eta = _clip_probability(eta)
    epsilon_r = _clip_probability(epsilon_r)

    theta = int(math.ceil(math.log(epsilon_r) / math.log(1.0 - eta)))

    # AoI threshold should be at least 1.
    return max(theta, 1)


def _expected_min_W_T(W_s: int, q_ss: float, q_s_other: float) -> float:
    """
    E[min(W_s, T_s)] where T_s is geometric with transition probability
    q_s_other and self-transition probability q_ss = 1 - q_s_other.

    This is the expected number of attempted AoII transmissions inside
    a state-s sojourn when the policy transmits for at most W_s slots.
    """
    W_s = int(W_s)
    if W_s <= 0:
        return 0.0

    q_ss = float(q_ss)
    q_s_other = float(q_s_other)

    return (1.0 - q_ss ** W_s) / q_s_other


def calibrate_baselines(
    p_t_star: float,
    r_prop: float,
    q01: float,
    q10: float,
    epsilon_bar_fn: Callable[[float], float],
    p_t_max: float,
) -> Dict:
    """
    Calibrate baseline parameters so each baseline matches the proposed
    policy's average energy budget E = p_t_star * r_prop.

    Parameters
    ----------
    p_t_star : float
        Optimal transmit power of the proposed policy.
    r_prop : float
        Average transmission rate of the proposed policy.
    q01 : float
        Markov surrogate transition probability 0 -> 1.
    q10 : float
        Markov surrogate transition probability 1 -> 0.
    epsilon_bar_fn : Callable[[float], float]
        Function returning the average packet error probability at power p_t.
    p_t_max : float
        Maximum allowed transmit power.

    Returns
    -------
    Dict
        Calibration result for predictive, event, probability, and AoII baselines.
    """
    p_t_star = float(p_t_star)
    r_prop = float(r_prop)
    q01 = float(q01)
    q10 = float(q10)
    p_t_max = float(p_t_max)

    if q01 <= 0.0 or q10 <= 0.0:
        raise ValueError(f"q01 and q10 must be positive, got q01={q01}, q10={q10}.")

    E = p_t_star * r_prop

    if E <= 0.0:
        raise ValueError(f"Energy budget must be positive, got E={E}.")

    if E > p_t_max:
        raise ValueError(
            f"Energy budget E={E:.6f} exceeds p_t_max={p_t_max:.6f}. "
            "The proposed design is infeasible under the given power constraint."
        )

    q00 = 1.0 - q01
    q11 = 1.0 - q10

    cycle_length = 1.0 / q01 + 1.0 / q10
    pi_0 = q10 / (q01 + q10)
    pi_1 = q01 / (q01 + q10)

    # ------------------------------------------------------------------
    # 1 & 2. Predictive-only / Event-triggered
    #
    # Under the two-state surrogate, both transmit once per sojourn.
    # Average transmission rate:
    #
    #   r_pred = 2 / (1/q01 + 1/q10)
    #
    # The transmit power is selected to match the energy budget:
    #
    #   p_t_pred * r_pred = E.
    # ------------------------------------------------------------------
    r_pred = 2.0 * q01 * q10 / (q01 + q10)

    p_t_pred_ideal = E / r_pred
    feasible_pred = bool(p_t_pred_ideal <= p_t_max)
    p_t_pred = min(p_t_pred_ideal, p_t_max)

    # ------------------------------------------------------------------
    # 3. Probability baseline
    #
    # Choose p_t to maximize successful delivery per unit power:
    #
    #   maximize (1 - eps_bar(p_t)) / p_t
    #
    # over p_t in [E, p_t_max].
    #
    # Then p_prob = E / p_t_prob so that:
    #
    #   p_prob * p_t_prob = E.
    # ------------------------------------------------------------------
    def _neg_efficiency(p_t: float) -> float:
        eps = float(epsilon_bar_fn(p_t))
        eps = _clip_probability(eps)
        return -((1.0 - eps) / float(p_t))

    search = minimize_scalar(
        _neg_efficiency,
        bounds=(E, p_t_max),
        method="bounded",
    )

    p_t_prob = float(search.x)
    p_prob = E / p_t_prob

    boundary_hit = bool(abs(p_t_prob - E) < 1e-6 * E + 1e-9)
    feasible_prob = bool(p_t_prob <= p_t_max and p_prob <= 1.0)

    assert abs(p_t_prob * p_prob - E) < 1e-6, (
        f"Energy check failed for probability baseline: "
        f"p_t={p_t_prob:.8f}, p={p_prob:.8f}, "
        f"product={p_t_prob * p_prob:.8f}, E={E:.8f}"
    )

    # ------------------------------------------------------------------
    # 4. AoII baseline
    #
    # After a transition into state s, AoII transmits for W_s slots.
    # Actual attempts in a sojourn are min(W_s, T_s).
    #
    # We choose W_0, W_1 to minimize the weighted miss probability under
    # the same energy budget.
    # ------------------------------------------------------------------
    W0_max = max(int(math.floor(1.0 / q01)), 1)
    W1_max = max(int(math.floor(1.0 / q10)), 1)

    def p_miss_state(W_s: int, q_ss: float, eps: float) -> float:
        """
        Probability that all AoII packets in a state-s sojourn fail.

        The number of attempts is min(W_s, T_s), where T_s is geometric.
        """
        W_s = int(W_s)
        eps = _clip_probability(eps)

        if W_s <= 0:
            return 1.0

        q_ss_eps = q_ss * eps

        if abs(1.0 - q_ss_eps) < 1e-15:
            return eps ** W_s

        term1 = (1.0 - q_ss) * eps * (1.0 - q_ss_eps ** W_s) / (1.0 - q_ss_eps)
        term2 = q_ss_eps ** W_s

        return term1 + term2

    best_W0 = None
    best_W1 = None
    best_p_miss_aoii = float("inf")
    best_p_t_aoii = None

    for W0_cand in range(1, W0_max + 1):
        for W1_cand in range(1, W1_max + 1):
            e_min_0 = _expected_min_W_T(W0_cand, q00, q01)
            e_min_1 = _expected_min_W_T(W1_cand, q11, q10)

            rate_cand = (e_min_0 + e_min_1) / cycle_length

            if rate_cand <= 0.0:
                continue

            p_t_cand = E / rate_cand

            if p_t_cand <= 0.0 or p_t_cand > p_t_max:
                continue

            eps = float(epsilon_bar_fn(p_t_cand))
            eps = _clip_probability(eps)

            pm0 = p_miss_state(W0_cand, q00, eps)
            pm1 = p_miss_state(W1_cand, q11, eps)

            p_miss_cand = pi_0 * pm0 + pi_1 * pm1

            if p_miss_cand < best_p_miss_aoii:
                best_p_miss_aoii = p_miss_cand
                best_W0 = W0_cand
                best_W1 = W1_cand
                best_p_t_aoii = p_t_cand

    if best_W0 is None:
        aoii_feasible = False

        best_W0 = 1
        best_W1 = 1

        e_min_0 = _expected_min_W_T(best_W0, q00, q01)
        e_min_1 = _expected_min_W_T(best_W1, q11, q10)

        rate_aoii_fallback = (e_min_0 + e_min_1) / cycle_length
        best_p_t_aoii = min(E / rate_aoii_fallback, p_t_max)

        eps = float(epsilon_bar_fn(best_p_t_aoii))
        eps = _clip_probability(eps)

        pm0 = p_miss_state(best_W0, q00, eps)
        pm1 = p_miss_state(best_W1, q11, eps)
        best_p_miss_aoii = pi_0 * pm0 + pi_1 * pm1
    else:
        aoii_feasible = True

    e_min_0_best = _expected_min_W_T(best_W0, q00, q01)
    e_min_1_best = _expected_min_W_T(best_W1, q11, q10)
    rate_aoii = (e_min_0_best + e_min_1_best) / cycle_length

    return {
        "predictive": {
            "p_t": p_t_pred,
            "rate": r_pred,
            "feasible": feasible_pred,
        },
        "event": {
            "p_t": p_t_pred,
            "rate": r_pred,
            "feasible": feasible_pred,
        },
        "probability": {
            "p_t": p_t_prob,
            "p": p_prob,
            "rate": p_prob,
            "feasible": feasible_prob,
            "boundary_hit": boundary_hit,
        },
        "aoii": {
            "p_t": best_p_t_aoii,
            "W_0": best_W0,
            "W_1": best_W1,
            "rate": rate_aoii,
            "p_miss": best_p_miss_aoii,
            "feasible": aoii_feasible,
        },
        "energy_budget": E,
    }


def compute_baseline_thresholds(
    q01: float,
    q10: float,
    epsilon_r: float,
    calibration_results: Dict,
    proposed_theta_0: int,
    proposed_theta_1: int,
    epsilon_bar_fn: Callable[[float], float],
) -> Dict:
    """
    Compute the AoI threshold each method uses to declare an outage.

    The proposed method keeps its optimized theta values.

    For baselines, the thresholds are computed using the effective successful
    update probability eta under normal operation:

        P(AoI > theta) ≈ (1 - eta)^theta.

    This is more consistent than directly using q_ss^theta for all baselines,
    because AoI is controlled by successful receptions, not only by the
    physical state sojourn duration.

    Baseline threshold rules
    ------------------------

    Proposed:
        theta_0, theta_1 are passed through from the proposed optimization.

    Predictive-only:
        Average update attempt rate:
            r_pred = 2 q01 q10 / (q01 + q10)

        Effective successful update rate:
            eta_pred = r_pred * (1 - eps_bar(p_t_pred))

        Single threshold:
            theta_pred = ceil(log(epsilon_r) / log(1 - eta_pred))

        The same value is used for both states.

    Event-triggered:
        Same rate and transmit power as predictive-only under the surrogate.
        Therefore it uses the same eta and theta as predictive-only.

    Probability:
        Effective successful update rate:
            eta_prob = p_prob * (1 - eps_bar(p_t_prob))

        Single threshold:
            theta_prob = ceil(log(epsilon_r) / log(1 - eta_prob))

    AoII:
        State-dependent attempt rates:
            attempt_rate_0 = E[min(W_0,T_0)] / E[T_0]
            attempt_rate_1 = E[min(W_1,T_1)] / E[T_1]

        Since E[T_0] = 1/q01 and E[T_1] = 1/q10,

            attempt_rate_0 = q01 * E[min(W_0,T_0)]
            attempt_rate_1 = q10 * E[min(W_1,T_1)]

        Effective successful update rates:
            eta_aoii_0 = attempt_rate_0 * (1 - eps_bar(p_t_aoii))
            eta_aoii_1 = attempt_rate_1 * (1 - eps_bar(p_t_aoii))

        State-dependent thresholds:
            theta_aoii_s = ceil(log(epsilon_r) / log(1 - eta_aoii_s))
    """
    q01 = float(q01)
    q10 = float(q10)
    epsilon_r = float(epsilon_r)

    if q01 <= 0.0 or q10 <= 0.0:
        raise ValueError(f"q01 and q10 must be positive, got q01={q01}, q10={q10}.")

    if not (0.0 < epsilon_r < 1.0):
        raise ValueError(f"epsilon_r must be in (0, 1), got {epsilon_r}.")

    q00 = 1.0 - q01
    q11 = 1.0 - q10

    # ------------------------------------------------------------------
    # Proposed method: pass-through from optimization.
    # ------------------------------------------------------------------
    theta_prop_0 = max(int(proposed_theta_0), 1)
    theta_prop_1 = max(int(proposed_theta_1), 1)

    # ------------------------------------------------------------------
    # Predictive-only and event-triggered.
    #
    # Use effective successful update rate:
    #
    #   eta = transmission_rate * success_probability.
    #
    # Under the surrogate, predictive and event both transmit once per sojourn.
    # ------------------------------------------------------------------
    r_pred = float(calibration_results["predictive"]["rate"])
    p_t_pred = float(calibration_results["predictive"]["p_t"])
    eps_pred = float(epsilon_bar_fn(p_t_pred))

    eta_pred = r_pred * (1.0 - eps_pred)
    theta_pred = _theta_from_eta(eta_pred, epsilon_r)

    # ------------------------------------------------------------------
    # Probability baseline.
    # ------------------------------------------------------------------
    p_prob = float(calibration_results["probability"]["p"])
    p_t_prob = float(calibration_results["probability"]["p_t"])
    eps_prob = float(epsilon_bar_fn(p_t_prob))

    eta_prob = p_prob * (1.0 - eps_prob)
    theta_prob = _theta_from_eta(eta_prob, epsilon_r)

    # ------------------------------------------------------------------
    # AoII baseline.
    #
    # AoII is state dependent. It transmits for W_s slots after entering
    # state s, but the sojourn may end early. Therefore the average number
    # of attempts in state s is E[min(W_s,T_s)].
    #
    # Effective per-slot attempt rate in state s:
    #
    #   E[min(W_s,T_s)] / E[T_s]
    #
    # Since E[T_0] = 1/q01 and E[T_1] = 1/q10:
    #
    #   attempt_rate_0 = q01 * E[min(W_0,T_0)]
    #   attempt_rate_1 = q10 * E[min(W_1,T_1)]
    # ------------------------------------------------------------------
    W_0 = int(calibration_results["aoii"]["W_0"])
    W_1 = int(calibration_results["aoii"]["W_1"])
    p_t_aoii = float(calibration_results["aoii"]["p_t"])
    eps_aoii = float(epsilon_bar_fn(p_t_aoii))

    e_min_0 = _expected_min_W_T(W_0, q00, q01)
    e_min_1 = _expected_min_W_T(W_1, q11, q10)

    attempt_rate_aoii_0 = q01 * e_min_0
    attempt_rate_aoii_1 = q10 * e_min_1

    eta_aoii_0 = attempt_rate_aoii_0 * (1.0 - eps_aoii)
    eta_aoii_1 = attempt_rate_aoii_1 * (1.0 - eps_aoii)

    theta_aoii_0 = _theta_from_eta(eta_aoii_0, epsilon_r)
    theta_aoii_1 = _theta_from_eta(eta_aoii_1, epsilon_r)

    result = {
        "proposed": {
            "theta_0": theta_prop_0,
            "theta_1": theta_prop_1,
        },
        "predictive": {
            "theta_0": theta_pred,
            "theta_1": theta_pred,
        },
        "event": {
            "theta_0": theta_pred,
            "theta_1": theta_pred,
        },
        "probability": {
            "theta": theta_prob,
        },
        "aoii": {
            "theta_0": theta_aoii_0,
            "theta_1": theta_aoii_1,
        },
        "debug": {
            "eta_pred": eta_pred,
            "eta_prob": eta_prob,
            "eta_aoii_0": eta_aoii_0,
            "eta_aoii_1": eta_aoii_1,
            "eps_pred": eps_pred,
            "eps_prob": eps_prob,
            "eps_aoii": eps_aoii,
        },
    }

    print(f"\n  === Per-method AoI detection thresholds (epsilon_r = {epsilon_r:.4f}) ===")
    print(
        f"    Proposed:       theta_0={result['proposed']['theta_0']}, "
        f"theta_1={result['proposed']['theta_1']}"
    )
    print(
        f"    Event-trigger:  theta_0={result['event']['theta_0']}, "
        f"theta_1={result['event']['theta_1']} "
        f"(eta={eta_pred:.6f})"
    )
    print(
        f"    Predictive:     theta_0={result['predictive']['theta_0']}, "
        f"theta_1={result['predictive']['theta_1']} "
        f"(eta={eta_pred:.6f})"
    )
    print(
        f"    Probability:    theta={result['probability']['theta']} "
        f"(eta={eta_prob:.6f})"
    )
    print(
        f"    AoII:           theta_0={result['aoii']['theta_0']}, "
        f"theta_1={result['aoii']['theta_1']} "
        f"(eta_0={eta_aoii_0:.6f}, eta_1={eta_aoii_1:.6f})"
    )

    return result
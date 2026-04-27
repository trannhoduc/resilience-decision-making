"""
Baseline calibration for the resilience-decision-making simulation.

Given the proposed policy's optimal transmit power p_t_star and its average
transmission rate r_prop, this module computes transmit powers (and, for the
probability baseline, a per-slot transmission probability) for three baselines
so that every baseline matches the same average energy budget

    E = p_t_star * r_prop   (packets/slot × power = energy/slot).

The three baselines are:
  1. Predictive-only      — one transmission per Markov sojourn, no resilience
                            updates (p_u0 = p_u1 = 0).
  2. Event-triggered      — same rate and power as predictive-only (one
                            transmission per sojourn under the surrogate).
  3. Probability-based    — Bernoulli-p per slot; the transmit power p_t is
                            chosen to maximise packet-success rate per unit
                            energy within [E, p_t_max], then p = E / p_t.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np
from scipy.optimize import minimize_scalar


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
    p_t_star      : Optimal transmit power of the proposed policy (scalar).
    r_prop        : Average transmission rate of the proposed policy
                    (packets/slot), computed from the two-state Markov
                    surrogate with resilience updates included.
    q01           : Markov surrogate transition probability 0→1.
    q10           : Markov surrogate transition probability 1→0.
    epsilon_bar_fn: Callable(p_t) → average PER; implements Eq. (8).
                    Should accept a single float and return a float in [0,1].
    p_t_max       : Maximum allowed transmit power.

    Returns
    -------
    dict with structure::

        {
          'predictive':  {'p_t': float, 'rate': float, 'feasible': bool},
          'event':       {'p_t': float, 'rate': float, 'feasible': bool},
          'probability': {'p_t': float, 'p': float,
                          'feasible': bool, 'boundary_hit': bool},
          'energy_budget': float,
        }

    Raises
    ------
    ValueError
        If the energy budget E exceeds p_t_max, making every single-slot
        transmission infeasible.
    """
    p_t_star = float(p_t_star)
    r_prop   = float(r_prop)
    q01      = float(q01)
    q10      = float(q10)
    p_t_max  = float(p_t_max)

    E = p_t_star * r_prop  # energy budget (energy per slot)

    if E > p_t_max:
        raise ValueError(
            f"Energy budget E={E:.6f} exceeds p_t_max={p_t_max:.6f}. "
            "The proposed design is infeasible under the given power constraint."
        )

    # ------------------------------------------------------------------
    # 1 & 2.  Predictive-only / Event-triggered
    #
    # Under the two-state Markov surrogate with no resilience updates
    # (p_u0 = p_u1 = 0), the average transmission rate reduces to:
    #
    #   r_pred = 2 / (1/q01 + 1/q10) = 2 q01 q10 / (q01 + q10)
    #
    # The transmit power is then set so that p_t_pred * r_pred = E.
    # ------------------------------------------------------------------
    r_pred          = 2.0 * q01 * q10 / (q01 + q10)
    p_t_pred_ideal  = E / r_pred                        # unconstrained power
    feasible_pred   = bool(p_t_pred_ideal <= p_t_max)
    # Cap to the hardware limit.  When infeasible the baseline runs at p_t_max
    # (energy slightly above budget); that is noted via feasible=False.
    p_t_pred = min(p_t_pred_ideal, p_t_max)

    # ------------------------------------------------------------------
    # 3.  Probability-based (Bernoulli-p per slot)
    #
    # Choose p_t to maximise packet-success rate per unit power:
    #
    #   p_t_prob = argmax_{p_t in [E, p_t_max]}  (1 - eps_bar(p_t)) / p_t
    #
    # The transmission probability is then p_prob = E / p_t_prob,
    # ensuring  p_prob * p_t_prob = E  (energy-budget constraint).
    #
    # The search lower bound is E because p_prob = E/p_t ≤ 1 requires
    # p_t ≥ E.  If the optimum sits at the lower boundary (p_t_prob ≈ E),
    # p_prob = 1 is optimal and we flag it as 'boundary_hit'.
    # ------------------------------------------------------------------
    def _neg_efficiency(p_t: float) -> float:
        eps = float(epsilon_bar_fn(p_t))
        return -((1.0 - eps) / p_t)

    search   = minimize_scalar(_neg_efficiency, bounds=(E, p_t_max), method="bounded")
    p_t_prob = float(search.x)
    p_prob   = E / p_t_prob

    # Boundary detection: p_t_prob indistinguishable from lower bound E
    boundary_hit  = bool(abs(p_t_prob - E) < 1e-6 * E + 1e-9)
    feasible_prob = bool(p_t_prob <= p_t_max)

    assert abs(p_t_prob * p_prob - E) < 1e-6, (
        f"Energy check failed for probability baseline: "
        f"p_t={p_t_prob:.8f}, p={p_prob:.8f}, "
        f"product={p_t_prob * p_prob:.8f}, E={E:.8f}"
    )

    # ------------------------------------------------------------------
    # 4. AoII — per-state retransmission windows (no ACK/NACK feedback)
    #
    # After transition INTO state s, sensor transmits for W_s slots.
    # Actual packets sent = min(W_s, T_s) since sojourn may end early.
    # W_s is capped at floor(E[T_s]) to prevent degeneration.
    #
    # E[min(W_s, T_s)] = (1 - q_ss^W_s) / q_{s,1-s}
    #   where q_ss = 1 - q_{s,1-s}
    #
    # Rate: r = (E[min(W0,T0)] + E[min(W1,T1)]) / cycle_length
    # Energy: p_t * r = E
    # Miss prob per state:
    #   P_miss_s = (1-q_ss)*eps*(1-(q_ss*eps)^W_s)/(1-q_ss*eps) + (q_ss*eps)^W_s
    # Weighted: P_miss = pi_0 * P_miss_0 + pi_1 * P_miss_1
    # ------------------------------------------------------------------
    cycle_length = 1.0 / q01 + 1.0 / q10
    q00 = 1.0 - q01
    q11 = 1.0 - q10
    pi_0 = q10 / (q01 + q10)
    pi_1 = q01 / (q01 + q10)

    W0_max = int(math.floor(1.0 / q01))   # floor(E[T_0])
    W1_max = int(math.floor(1.0 / q10))   # floor(E[T_1])
    # Ensure at least 1
    W0_max = max(W0_max, 1)
    W1_max = max(W1_max, 1)

    def expected_min_W_T(W_s, q_ss, q_s_other):
        """E[min(W_s, T_s)] where T_s ~ Geom(q_{s,1-s})"""
        if W_s <= 0:
            return 0.0
        return (1.0 - q_ss ** W_s) / q_s_other

    def p_miss_state(W_s, q_ss, eps):
        """P(all min(W_s, T_s) packets fail)"""
        if W_s <= 0 or eps <= 0:
            return 0.0
        if eps >= 1.0:
            return 1.0
        q_ss_eps = q_ss * eps
        if abs(1.0 - q_ss_eps) < 1e-15:
            # Degenerate case
            return eps ** W_s
        term1 = (1.0 - q_ss) * eps * (1.0 - q_ss_eps ** W_s) / (1.0 - q_ss_eps)
        term2 = q_ss_eps ** W_s
        return term1 + term2

    best_W0 = None
    best_W1 = None
    best_p_miss_aoii = 1.0
    best_p_t_aoii = None

    for W0_cand in range(1, W0_max + 1):
        for W1_cand in range(1, W1_max + 1):
            e_min_0 = expected_min_W_T(W0_cand, q00, q01)
            e_min_1 = expected_min_W_T(W1_cand, q11, q10)
            rate_cand = (e_min_0 + e_min_1) / cycle_length

            if rate_cand <= 0:
                continue

            p_t_cand = E / rate_cand
            if p_t_cand <= 0 or p_t_cand > p_t_max:
                continue

            eps = float(epsilon_bar_fn(p_t_cand))
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
        e0 = expected_min_W_T(1, q00, q01)
        e1 = expected_min_W_T(1, q11, q10)
        best_p_t_aoii = E / ((e0 + e1) / cycle_length)
    else:
        aoii_feasible = True

    e_min_0_best = expected_min_W_T(best_W0, q00, q01)
    e_min_1_best = expected_min_W_T(best_W1, q11, q10)
    rate_aoii = (e_min_0_best + e_min_1_best) / cycle_length

    return {
        "predictive": {
            "p_t":      p_t_pred,
            "rate":     r_pred,
            "feasible": feasible_pred,
        },
        "event": {
            "p_t":      p_t_pred,
            "rate":     r_pred,
            "feasible": feasible_pred,
        },
        "probability": {
            "p_t":          p_t_prob,
            "p":            p_prob,
            "feasible":     feasible_prob,
            "boundary_hit": boundary_hit,
        },
        "aoii": {
            "p_t":      best_p_t_aoii,
            "W_0":      best_W0,
            "W_1":      best_W1,
            "rate":     rate_aoii,
            "p_miss":   best_p_miss_aoii,
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
    Compute the AoI threshold each baseline uses to declare an outage.

    Each baseline's theta is derived from that baseline's own AoI distribution
    under normal operation, using the same false-trigger probability epsilon_r
    across all methods.

    Baseline-specific formulas
    --------------------------

    PROPOSED: pass-through — proposed_theta_0 and proposed_theta_1 are
    returned unchanged from Algorithm 2.

    EVENT-TRIGGERED and PREDICTIVE-ONLY (one packet per sojourn):
        P(AoI > theta | no outage) = q_ss^theta
        theta_s = ceil( log(epsilon_r) / log(q_ss) )
        where q00 = 1 - q01, q11 = 1 - q10.

    PROBABILITY (every slot, delivery probability eta = p_prob*(1-eps_bar(p_t_prob))):
        P(AoI > theta) = (1 - eta)^theta
        theta = ceil( log(epsilon_r) / log(1 - eta) )
        State-independent (single theta for both states).

    AoII (burst of W_s packets at sojourn start, then silent):
        P(AoI > theta | burst success) ≈ q_ss^(W_s + theta)
        theta_s = ceil( log(epsilon_r) / log(q_ss) - W_s )

    Parameters
    ----------
    q01               : Markov surrogate transition probability 0 → 1.
    q10               : Markov surrogate transition probability 1 → 0.
    epsilon_r         : False-trigger probability target (from the proposed
                        method's optimization result).
    calibration_results : Output of calibrate_baselines; provides p_prob,
                        p_t_prob (for probability) and W_0, W_1 (for AoII).
    proposed_theta_0  : Proposed method's theta for state 0 (pass-through).
    proposed_theta_1  : Proposed method's theta for state 1 (pass-through).
    epsilon_bar_fn    : Callable(p_t) → average PER; used to compute the
                        probability baseline's effective delivery rate.

    Returns
    -------
    dict::

        {
          'proposed':    {'theta_0': int, 'theta_1': int},
          'predictive':  {'theta_0': int, 'theta_1': int},
          'event':       {'theta_0': int, 'theta_1': int},
          'probability': {'theta':   int},
          'aoii':        {'theta_0': int, 'theta_1': int},
        }
    """
    q01       = float(q01)
    q10       = float(q10)
    epsilon_r = float(epsilon_r)

    if not (0.0 < epsilon_r < 1.0):
        raise ValueError(f"epsilon_r must be in (0, 1), got {epsilon_r}.")

    q00     = 1.0 - q01
    q11     = 1.0 - q10
    log_eps = math.log(epsilon_r)

    # Event-triggered and predictive-only: geometric sojourn quantiles
    theta_pred_0 = int(math.ceil(log_eps / math.log(q00)))
    theta_pred_1 = int(math.ceil(log_eps / math.log(q11)))

    # Probability baseline: derive effective delivery rate from calibration
    p_prob   = float(calibration_results["probability"]["p"])
    p_t_prob = float(calibration_results["probability"]["p_t"])
    eta      = p_prob * (1.0 - float(epsilon_bar_fn(p_t_prob)))
    theta_prob = int(math.ceil(log_eps / math.log(1.0 - eta)))

    # AoII: burst of W_s slots reduces the residual wait
    W_0 = int(calibration_results["aoii"]["W_0"])
    W_1 = int(calibration_results["aoii"]["W_1"])
    theta_aoii_0 = int(math.ceil(log_eps / math.log(q00) - W_0))
    theta_aoii_1 = int(math.ceil(log_eps / math.log(q11) - W_1))

    result = {
        "proposed":    {"theta_0": int(proposed_theta_0), "theta_1": int(proposed_theta_1)},
        "predictive":  {"theta_0": theta_pred_0,          "theta_1": theta_pred_1},
        "event":       {"theta_0": theta_pred_0,          "theta_1": theta_pred_1},
        "probability": {"theta":   theta_prob},
        "aoii":        {"theta_0": theta_aoii_0,          "theta_1": theta_aoii_1},
    }

    print(f"\n  === Per-method AoI detection thresholds (epsilon_r = {epsilon_r:.4f}) ===")
    print(f"    Proposed:       theta_0={result['proposed']['theta_0']}, theta_1={result['proposed']['theta_1']}")
    print(f"    Event-trigger:  theta_0={result['event']['theta_0']}, theta_1={result['event']['theta_1']}")
    print(f"    Predictive:     theta_0={result['predictive']['theta_0']}, theta_1={result['predictive']['theta_1']}")
    print(f"    Probability:    theta={result['probability']['theta']}")
    print(f"    AoII:           theta_0={result['aoii']['theta_0']}, theta_1={result['aoii']['theta_1']}")

    return result

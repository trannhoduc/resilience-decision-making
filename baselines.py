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
    # 4.  AoII — fixed W-slot retransmission window (no ACK/NACK feedback)
    #
    # Each transition triggers a burst of W transmissions.  The actual
    # average transmission rate is capped at 1 (at most one packet/slot):
    #
    #   actual_rate(W) = min(2W / cycle_length, 1)
    #
    # Energy constraint:  p_t * actual_rate = E  =>
    #
    #   p_t_aoii(W) = E / actual_rate(W)
    #               = E * cycle_length / (2W)        when 2W <= cycle_length
    #               = E                               when 2W >  cycle_length
    #
    # Joint optimisation: choose W to minimise the probability that ALL W
    # transmissions fail,  P_miss = eps_bar(p_t_aoii(W))^W.
    # ------------------------------------------------------------------
    cycle_length = 1.0 / q01 + 1.0 / q10

    best_W      = None
    best_p_miss = 1.0
    best_p_t_aoii = None

    for W_cand in range(1, 50):
        actual_rate = min(2.0 * W_cand / cycle_length, 1.0)
        p_t_cand    = E / actual_rate
        if p_t_cand <= 0 or p_t_cand > p_t_max:
            continue
        eps      = float(epsilon_bar_fn(p_t_cand))
        p_miss   = eps ** W_cand
        if p_miss < best_p_miss:
            best_p_miss   = p_miss
            best_W        = W_cand
            best_p_t_aoii = p_t_cand

    if best_W is None:
        aoii_feasible = False
        best_W        = 1
        best_p_t_aoii = E * cycle_length / 2.0
    else:
        aoii_feasible = True

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
            "W":        best_W,
            "rate":     min(2.0 * best_W / cycle_length, 1.0),
            "p_miss":   best_p_miss,
            "feasible": aoii_feasible,
        },
        "energy_budget": E,
    }


def compute_baseline_thresholds(
    q01: float,
    q10: float,
    eta_prob: float,
    epsilon_r: float,
) -> Dict:
    """
    Compute the AoI threshold each baseline uses to declare an outage.

    All baselines share the same reliability target ``epsilon_r`` that the
    proposed method achieves.  The thresholds are the smallest integer theta
    such that the probability of NOT detecting an outage within theta slots
    is <= epsilon_r.

    Baseline-specific formulas
    --------------------------

    1. **Probability baseline** — single threshold, state-independent.

       The per-slot effective detection rate is::

           eta = p_prob * (1 - eps_bar(p_t_prob))

       passed in as ``eta_prob``.  The threshold is then the
       (1 - epsilon_r)-quantile of the geometric waiting time::

           theta_prob = ceil( log(epsilon_r) / log(1 - eta_prob) )

    2. **Predictive-only and Event-triggered** — per-state thresholds,
       because sojourn times are Geometric with state-dependent parameter.

       AoI grows continuously (no reset on state transition for baselines).
       Within a sojourn in state s the probability that AoI has NOT yet
       reached theta_s after theta_s slots is q_ss^{theta_s} — matching
       the complementary CDF of the geometric sojourn.  Setting this equal
       to epsilon_r gives::

           theta_s = ceil( log(epsilon_r) / log(q_ss) )

       where q00 = 1 - q01 and q11 = 1 - q10.

    Parameters
    ----------
    q01       : Markov surrogate transition probability 0 → 1.
    q10       : Markov surrogate transition probability 1 → 0.
    eta_prob  : Effective per-slot detection rate for the probability
                baseline: ``p_prob * (1 - eps_bar(p_t_prob))``.
    epsilon_r : Reliability target (from the proposed method's design).

    Returns
    -------
    dict::

        {
          'predictive':  {'theta_0': int, 'theta_1': int},
          'event':       {'theta_0': int, 'theta_1': int},
          'probability': {'theta':   int},
        }
    """
    q01 = float(q01)
    q10 = float(q10)
    eta_prob = float(eta_prob)
    epsilon_r = float(epsilon_r)

    if not (0.0 < epsilon_r < 1.0):
        raise ValueError(f"epsilon_r must be in (0, 1), got {epsilon_r}.")

    # ------------------------------------------------------------------
    # Predictive-only / Event-triggered: geometric sojourn quantiles
    # ------------------------------------------------------------------
    q00 = 1.0 - q01
    q11 = 1.0 - q10

    if q00 <= 0.0 or q00 >= 1.0:
        theta_pred_0 = 1
    else:
        theta_pred_0 = int(math.ceil(math.log(epsilon_r) / math.log(q00)))

    if q11 <= 0.0 or q11 >= 1.0:
        theta_pred_1 = 1
    else:
        theta_pred_1 = int(math.ceil(math.log(epsilon_r) / math.log(q11)))

    # ------------------------------------------------------------------
    # Probability baseline: single geometric quantile with rate eta_prob
    # ------------------------------------------------------------------
    eta_prob = float(np.clip(eta_prob, 1e-9, 1.0 - 1e-9))
    theta_prob = int(math.ceil(math.log(epsilon_r) / math.log(1.0 - eta_prob)))

    # Ensure all thresholds are at least 1
    theta_pred_0 = max(1, theta_pred_0)
    theta_pred_1 = max(1, theta_pred_1)
    theta_prob   = max(1, theta_prob)

    return {
        "predictive":  {"theta_0": theta_pred_0, "theta_1": theta_pred_1},
        "event":       {"theta_0": theta_pred_0, "theta_1": theta_pred_1},
        "probability": {"theta":   theta_prob},
    }

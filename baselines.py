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
    r_pred   = 2.0 * q01 * q10 / (q01 + q10)
    p_t_pred = E / r_pred
    feasible_pred = bool(p_t_pred <= p_t_max)

    assert abs(p_t_pred * r_pred - E) < 1e-6, (
        f"Energy check failed for predictive/event baseline: "
        f"p_t={p_t_pred:.8f}, rate={r_pred:.8f}, "
        f"product={p_t_pred * r_pred:.8f}, E={E:.8f}"
    )

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
        "energy_budget": E,
    }

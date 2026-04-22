"""
recovery_metrics.py

TP/FP tracking for the outage-triggered recovery mechanism.

Each simulated policy attaches a ``RecoveryMetricsTracker`` to its
``RemoteEstimator``.  The tracker is fed two kinds of events during
the simulation loop:

  * ``log_recovery_trigger(disruption_active)``
        — AoI threshold crossing triggered recovery

Definitions
-----------
TP : recovery triggered while a real disruption was active at slot k
FP : recovery triggered with no real disruption active at slot k
"""

from __future__ import annotations

from typing import List


class RecoveryMetricsTracker:
    """
    Event-driven tracker for recovery trigger quality (TP / FP).

    Usage
    -----
    1. Attach one instance to each ``RemoteEstimator`` before simulation.
    2. During simulation, call ``log_recovery_trigger`` at each trigger event.
    3. Call ``compute_metrics()`` after the simulation loop finishes.
    """

    def __init__(self) -> None:
        self._trigger_is_tp: List[bool] = []

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------

    def log_recovery_trigger(self, disruption_active: bool) -> None:
        """
        Record one recovery episode.

        Parameters
        ----------
        disruption_active: True if a real disruption was active at the trigger slot
        """
        self._trigger_is_tp.append(bool(disruption_active))

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def compute_metrics(self) -> dict:
        """
        Compute TP and FP counts.

        Returns
        -------
        dict with keys: 'n_tp', 'n_fp', 'total_triggers'
        """
        n_tp = sum(self._trigger_is_tp)
        n_fp = len(self._trigger_is_tp) - n_tp

        return {
            "n_tp":           n_tp,
            "n_fp":           n_fp,
            "total_triggers": len(self._trigger_is_tp),
        }

import numpy as np
import matplotlib.pyplot as plt
from computation import evaluate_decision, predictive_transition_detection

class Model:
    """
    Sensor-side model:
    - evolves the true LTI system
    - generates measurements
    - runs the local Kalman filter at the sensor
    """

    def __init__(self, A, C, Q, R):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R

        self.n = A.shape[0]   # state dimension
        self.m = C.shape[0]   # measurement dimension

        self.x_true = None
        self.x_s = None       # sensor-side local estimate
        self.P_s = None       # sensor-side covariance

    def init_value(self, x_true0, x_s0, P_s0):
        self.x_true = np.asarray(x_true0, dtype=float).reshape(-1, 1)
        self.x_s = np.asarray(x_s0, dtype=float).reshape(-1, 1)
        self.P_s = np.asarray(P_s0, dtype=float)

    def evolve_true_state(self):
        """
        True process:
            x_k = A x_{k-1} + w_k
        """
        w_k = np.random.multivariate_normal(
            mean=np.zeros(self.n), cov=self.Q
        ).reshape(-1, 1)

        self.x_true = self.A @ self.x_true + w_k
        return self.x_true

    def measure(self):
        """
        Measurement:
            y_k = C x_k + v_k
        """
        v_k = np.random.multivariate_normal(
            mean=np.zeros(self.m), cov=self.R
        ).reshape(-1, 1)

        y_k = self.C @ self.x_true + v_k
        return y_k

    def local_kf_step(self, y_k):
        """
        Sensor-side Kalman filter:
            prediction + correction using local measurement y_k
        """
        # Prediction
        x_pred = self.A @ self.x_s
        P_pred = self.A @ self.P_s @ self.A.T + self.Q

        # Update
        S = self.C @ P_pred @ self.C.T + self.R
        K = P_pred @ self.C.T @ np.linalg.inv(S)

        innovation = y_k - self.C @ x_pred
        self.x_s = x_pred + K @ innovation

        I = np.eye(self.n)
        # Joseph form for better numerical stability
        self.P_s = (I - K @ self.C) @ P_pred @ (I - K @ self.C).T + K @ self.R @ K.T

        return self.x_s, self.P_s

    def step(self):
        """
        One full sensor-side time step:
          1) true state evolves
          2) measurement is generated
          3) local KF runs
        """
        x_true = self.evolve_true_state()
        y_k = self.measure()
        x_s, P_s = self.local_kf_step(y_k)
        return x_true, y_k, x_s, P_s


class RemoteEstimator:
    """
    Remote estimator:
    - receives the sensor estimate if transmission succeeds
    - otherwise uses prediction only
    - makes a reliability-aware binary decision using the
      feasible-decision rule from the paper

    Update rule:
        x_hat_k = delta_k * x_s_k + (1-delta_k) * A x_hat_{k-1}
        P_k     = delta_k * P_s_k + (1-delta_k) * (A P_{k-1} A^T + Q)
    """

    def __init__(self, A, Q, c, Delta, epsilon, alpha_fp, alpha_fn):
        self.A = A
        self.Q = Q
        self.c = np.asarray(c, dtype=float).reshape(-1, 1)
        self.Delta = float(Delta)
        self.epsilon = float(epsilon)

        self.alpha_fp = float(alpha_fp)
        self.alpha_fn = float(alpha_fn)

        self.n = A.shape[0]

        self.x_hat = None
        self.P = None

        # previous reliable decision at the estimator
        self.last_decision = 0

    def init_value(self, x_hat0, P0):
        self.x_hat = np.asarray(x_hat0, dtype=float).reshape(-1, 1)
        self.P = np.asarray(P0, dtype=float)

        # initialize previous decision using a simple threshold on the initial estimate
        self.last_decision = self.hard_decision(self.x_hat)

    def hard_decision(self, x):
        value = float(self.c.T @ x)
        return 1 if value >= self.Delta else 0

    def packet_success(self, transmit):
        """
        If sensor decides not to transmit => delta = 0
        If sensor transmits => success with probability 1-epsilon
        """
        if transmit == 0:
            return 0

        success = np.random.rand() > self.epsilon
        return 1 if success else 0

    def reliable_decision(self, x_hat=None, P=None):
        """
        Reliability-aware decision according to the feasible decision region:
        - pi = 1 if z <= z_minus
        - pi = 0 if z >= z_plus
        - keep previous decision in the confusion region
        """
        if x_hat is None:
            x_hat = self.x_hat
        if P is None:
            P = self.P

        info = evaluate_decision(
            x_hat=x_hat,
            P=P,
            c=self.c,
            Delta=self.Delta,
            alpha_fp=self.alpha_fp,
            alpha_fn=self.alpha_fn,
            previous_decision=self.last_decision,
        )
        return info

    def step(self, x_s, P_s, transmit):
        """
        Remote estimator update.

        transmit: sensor transmission decision (0 or 1)
        delta_k: successful reception indicator
        """
        delta_k = self.packet_success(transmit)

        x_pred = self.A @ self.x_hat
        P_pred = self.A @ self.P @ self.A.T + self.Q

        self.x_hat = delta_k * x_s + (1 - delta_k) * x_pred
        self.P = delta_k * P_s + (1 - delta_k) * P_pred

        decision_info = self.reliable_decision(self.x_hat, self.P)
        decision = int(decision_info["pi"])
        self.last_decision = decision

        return self.x_hat, self.P, delta_k, decision, decision_info

class PredictivePolicy:
    """
    Predictive-update-only transmission policy.

    Logic:
    - If a future predictive packet has already been sent and the local reliable
      decision has not changed yet, do not send again.
    - If the local reliable decision changes:
        * if there was NO pending predictive packet -> send immediately
          (current transition was not predicted before)
        * if there WAS a pending predictive packet -> consume/clear it,
          do NOT resend the current transition, and start predicting from the new state
    - If no current transition needs to be sent, run predictive transition detection
      to see whether a future transition within ell should be sent now.
    """

    def __init__(self, A, Q, c, Delta, alpha_fp, alpha_fn, ell, initial_decision=0):
        self.A = A
        self.Q = Q
        self.c = np.asarray(c, dtype=float).reshape(-1, 1)
        self.Delta = float(Delta)

        self.alpha_fp = float(alpha_fp)
        self.alpha_fn = float(alpha_fn)
        self.ell = int(ell)

        # m_{k-1}
        self.last_local_decision = int(initial_decision)

        # whether a predictive packet for a future transition has already been sent
        self.pending_predictive_packet = False

        # debug info
        self.last_prediction = None
        self.last_local_decision_info = None

    def __call__(self, x_true, x_s, P_s, x_hat_remote, P_remote, k):
        """
        Returns:
            1 -> transmit
            0 -> do not transmit

        Notes:
        - x_true is unused here; it stays only for interface compatibility.
        """

        # -------------------------------------------------
        # Step 1: compute current reliable local decision m_k
        # using the feasible-decision rule
        # -------------------------------------------------
        previous_decision = self.last_local_decision

        local_info = evaluate_decision(
            x_hat=x_s,
            P=P_s,
            c=self.c,
            Delta=self.Delta,
            alpha_fp=self.alpha_fp,
            alpha_fn=self.alpha_fn,
            previous_decision=previous_decision,
        )
        m_k = int(local_info["pi"])
        self.last_local_decision_info = local_info

        state_changed = (m_k != previous_decision)
        had_pending = self.pending_predictive_packet

        # -------------------------------------------------
        # Case 1:
        # A predictive packet was already sent, and the local decision
        # has not changed yet -> skip sending again
        # -------------------------------------------------
        if had_pending and not state_changed:
            self.last_prediction = {
                "found_transition": False,
                "reason": "pending_predictive_packet_active_same_state",
                "predicted_transition_time": None,
            }
            return 0

        # -------------------------------------------------
        # Case 2:
        # The local decision changed.
        # Then the old pending predictive packet, if any, has been consumed/realized.
        # So clear it immediately.
        # -------------------------------------------------
        if state_changed:
            self.pending_predictive_packet = False

            # -------------------------------------------------
            # Case 2a:
            # No prior predictive packet existed for this change
            # -> send immediately
            # -------------------------------------------------
            if not had_pending:
                self.last_prediction = {
                    "found_transition": True,
                    "reason": "unpredicted_current_transition",
                    "predicted_transition_time": 0,
                    "predicted_horizon": 0,
                    "decision_now": m_k,
                }

                self.last_local_decision = m_k
                return 1

            # -------------------------------------------------
            # Case 2b:
            # A predictive packet had already been sent for the old->new transition.
            # Do NOT resend the current transition.
            # Instead, continue from the NEW state and predict the NEXT transition.
            # -------------------------------------------------
            reference_decision = m_k

        else:
            # state unchanged, no pending currently
            reference_decision = previous_decision

        # -------------------------------------------------
        # Step 3:
        # From the current reference state, see whether a FUTURE transition
        # within ell can already be predicted.
        #
        # IMPORTANT:
        # if state_changed and had_pending=True, reference_decision = m_k
        # so we do not rediscover the already-realized current transition.
        # -------------------------------------------------
        pred = predictive_transition_detection(
            x_hat_sensor=x_s,
            P_sensor=P_s,
            A=self.A,
            Q=self.Q,
            c=self.c,
            Delta=self.Delta,
            alpha_fp=self.alpha_fp,
            alpha_fn=self.alpha_fn,
            previous_decision=reference_decision,
            ell=self.ell,
        )
        self.last_prediction = pred

        found = bool(pred.get("found_transition", False))
        horizon = pred.get("predicted_horizon", pred.get("predicted_transition_time", None))

        transmit = 0

        if found:
            # Only future predictive update should reach here.
            # If horizon == 0 happens here, still allow transmit,
            # but logically this should be rare after the case split above.
            transmit = 1

            if horizon is not None and horizon > 0:
                self.pending_predictive_packet = True
            else:
                self.pending_predictive_packet = False
        else:
            transmit = 0
            self.pending_predictive_packet = False

        # -------------------------------------------------
        # Update memory for next step
        # -------------------------------------------------
        self.last_local_decision = m_k
        return transmit

def build_predictive_policy(A, Q, c, Delta, alpha_fp, alpha_fn, ell, initial_decision=0):
    return PredictivePolicy(
        A=A,
        Q=Q,
        c=c,
        Delta=Delta,
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
        ell=ell,
        initial_decision=initial_decision,
    )

def simulate_system(sensor, estimator, T, transmission_policy):
    """
    Run the joint simulation.

    transmission_policy(...) must return 0 or 1.
    """

    true_values = []
    est_values = []
    true_events = []
    est_events = []
    transmit_hist = []
    delta_hist = []

    fp_idx = []
    fn_idx = []

    # optional debug histories
    z_hist = []
    region_hist = []
    predicted_horizon_hist = []

    for k in range(T):
        # Sensor side
        x_true, y_k, x_s, P_s = sensor.step()

        # Sensor decides whether to transmit
        transmit = int(transmission_policy(
            x_true=x_true,
            x_s=x_s,
            P_s=P_s,
            x_hat_remote=estimator.x_hat,
            P_remote=estimator.P,
            k=k
        ))

        # Remote estimator step
        x_hat, P, delta_k, decision_est, decision_info = estimator.step(
            x_s=x_s,
            P_s=P_s,
            transmit=transmit
        )

        # True event
        true_value = float(estimator.c.T @ x_true)
        est_value = float(estimator.c.T @ x_hat)

        true_event = 1 if true_value >= estimator.Delta else 0
        est_event = decision_est

        # Save histories
        true_values.append(true_value)
        est_values.append(est_value)
        true_events.append(true_event)
        est_events.append(est_event)
        transmit_hist.append(transmit)
        delta_hist.append(delta_k)

        z_hist.append(float(decision_info["z"]))
        region_hist.append(decision_info["region"])

        pred_info = getattr(transmission_policy, "last_prediction", None)
        if pred_info is None:
            predicted_horizon_hist.append(np.nan)
        else:
            horizon = pred_info.get("predicted_horizon", pred_info.get("predicted_transition_time", None))
            predicted_horizon_hist.append(np.nan if horizon is None else float(horizon))

        # Error bookkeeping
        if est_event == 1 and true_event == 0:
            fp_idx.append(k)

        if est_event == 0 and true_event == 1:
            fn_idx.append(k)

    true_values = np.array(true_values)
    est_values = np.array(est_values)
    true_events = np.array(true_events)
    est_events = np.array(est_events)
    transmit_hist = np.array(transmit_hist)
    delta_hist = np.array(delta_hist)
    z_hist = np.array(z_hist)
    predicted_horizon_hist = np.array(predicted_horizon_hist)

    # Rates
    num_negative_truth = np.sum(true_events == 0)
    num_positive_truth = np.sum(true_events == 1)

    false_positive_rate = len(fp_idx) / num_negative_truth if num_negative_truth > 0 else 0.0
    false_negative_rate = len(fn_idx) / num_positive_truth if num_positive_truth > 0 else 0.0

    print(f"False Positive Rate (FPR): {false_positive_rate:.4f}")
    print(f"False Negative Rate (FNR): {false_negative_rate:.4f}")
    print(f"Total successful receptions: {np.sum(delta_hist)} / {T}")
    print(f"Total transmission attempts: {np.sum(transmit_hist)} / {T}")

    results = {
        "true_values": true_values,
        "est_values": est_values,
        "true_events": true_events,
        "est_events": est_events,
        "transmit_hist": transmit_hist,
        "delta_hist": delta_hist,
        "fp_idx": np.array(fp_idx),
        "fn_idx": np.array(fn_idx),
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
        "z_hist": z_hist,
        "region_hist": region_hist,
        "predicted_horizon_hist": predicted_horizon_hist,
    }

    return results

def plot_results(results, Delta):
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.arange(len(results["true_values"]))

    true_values = results["true_values"]
    est_values = results["est_values"]

    transmit_hist = results["transmit_hist"]
    delta_hist = results["delta_hist"]

    fp_idx = results["fp_idx"]
    fn_idx = results["fn_idx"]

    fpr = results["fpr"]
    fnr = results["fnr"]

    # successful update: transmit=1 and delta=1
    success_idx = np.where(delta_hist == 1)[0]

    # failed update: transmit=1 but delta=0
    fail_idx = np.where((transmit_hist == 1) & (delta_hist == 0))[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [5, 1]}
    )

    # =========================
    # Top plot: signals
    # =========================
    ax1.plot(t, true_values, label=r"True $c^T x_k$", linewidth=2)
    ax1.plot(t, est_values, label=r"Estimator $c^T \hat{x}_k$", linewidth=2)
    ax1.axhline(Delta, linestyle="--", label=r"Threshold $\Delta$")

    # False positive: estimator says 1, truth says 0
    for i, k in enumerate(fp_idx):
        ax1.axvline(
            x=k,
            color="red",
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            label="False Positive" if i == 0 else ""
        )

    # False negative: estimator says 0, truth says 1
    for i, k in enumerate(fn_idx):
        ax1.axvline(
            x=k,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            label="False Negative" if i == 0 else ""
        )

    textstr = f"FPR = {fpr:.4f}\nFNR = {fnr:.4f}"
    ax1.text(
        0.02, 0.98, textstr,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax1.set_ylabel(r"Value of $c^T x$")
    ax1.set_title(r"True $c^T x_k$, estimator decision behavior, and update outcomes")
    ax1.grid(True)
    ax1.legend(loc="best")

    # =========================
    # Bottom plot: update events
    # =========================
    ax2.axhline(0, color="black", linewidth=1)

    if len(success_idx) > 0:
        ax2.scatter(
            success_idx,
            np.zeros_like(success_idx),
            marker="o",
            s=50,
            color="blue",
            label="Successful update"
        )

    if len(fail_idx) > 0:
        ax2.scatter(
            fail_idx,
            np.zeros_like(fail_idx),
            marker="x",
            s=60,
            color="orange",
            label="Failed update"
        )

    ax2.set_ylim(-1, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("Time step k")
    ax2.set_ylabel("Updates")
    ax2.grid(True, axis="x")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()

def probability_policy(x_true, x_s, P_s, x_hat_remote, P_remote, k):
    if np.random.uniform(0, 1) < 0.3:
        return 1
    return 0

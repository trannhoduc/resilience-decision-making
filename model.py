import math
import numpy as np
import matplotlib.pyplot as plt
from computation import evaluate_decision, predictive_transition_detection
from resilience_design import compute_instantaneous_packet_error
from recovery_metrics import RecoveryMetricsTracker

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

    def evolve_true_state(self, w_k=None):
        """
        True process:
            x_k = A x_{k-1} + w_k
        w_k: optional pre-generated noise vector (shape (n,) or (n,1)).
             If None, drawn fresh from N(0, Q).
        """
        if w_k is None:
            w_k = np.random.multivariate_normal(
                mean=np.zeros(self.n), cov=self.Q
            ).reshape(-1, 1)
        else:
            w_k = np.asarray(w_k, dtype=float).reshape(-1, 1)

        self.x_true = self.A @ self.x_true + w_k
        return self.x_true

    def measure(self, v_k=None):
        """
        Measurement:
            y_k = C x_k + v_k
        v_k: optional pre-generated noise vector (shape (m,) or (m,1)).
             If None, drawn fresh from N(0, R).
        """
        if v_k is None:
            v_k = np.random.multivariate_normal(
                mean=np.zeros(self.m), cov=self.R
            ).reshape(-1, 1)
        else:
            v_k = np.asarray(v_k, dtype=float).reshape(-1, 1)

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

    def step(self, w_k=None, v_k=None):
        """
        One full sensor-side time step:
          1) true state evolves
          2) measurement is generated
          3) local KF runs
        w_k, v_k: optional pre-generated noise (passed to evolve/measure).
        """
        x_true = self.evolve_true_state(w_k=w_k)
        y_k = self.measure(v_k=v_k)
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

    def __init__(self, A, Q, c, Delta, noise_var, blocklength_n, info_bits_l,
                 alpha_fp, alpha_fn, xi, ell,
                 q01=0.0, q10=0.0, p_r=0.0, theta_0=5, theta_1=5,
                 weibull_lambda=4.0, weibull_kappa=2.0):
        self.A = A
        self.Q = Q
        self.c = np.asarray(c, dtype=float).reshape(-1, 1)
        self.Delta = float(Delta)
        self.noise_var = float(noise_var)
        self.blocklength_n = int(blocklength_n)
        self.info_bits_l = int(info_bits_l)

        self.alpha_fp = float(alpha_fp)
        self.alpha_fn = float(alpha_fn)
        self.xi = float(xi)
        self.ell = int(ell)

        self._horizon_detected = False
        self._current_horizon = None
        self._horizon_history = []
        self._horizon_state_history = []
        self._sojourn_initialized = False

        self.n = A.shape[0]

        self.x_hat = None
        self.P = None

        # previous reliable decision at the estimator
        self.last_decision = 0

        # Resilience parameters
        # p_r0 = p_r / E[T_0] = p_r * q01  (per-sojourn disruption probability in state 0)
        # p_r1 = p_r / E[T_1] = p_r * q10  (per-sojourn disruption probability in state 1)
        self.q01 = float(q01)
        self.q10 = float(q10)
        self.p_r = float(p_r)
        self.theta_0 = int(theta_0)   # AoI threshold for state 0
        self.theta_1 = int(theta_1)   # AoI threshold for state 1
        self.weibull_lambda = float(weibull_lambda)
        self.weibull_kappa = float(weibull_kappa)

        # Sojourn / disruption / AoI runtime state (fully reset in init_value)
        self._sojourn_state = 0
        self._aoi = 0
        self._disruption_active = False
        self._recovery_end_k = None
        self._in_recovery = False

        # TP/FP tracker (always present; populated for all modes)
        self.metrics_tracker = RecoveryMetricsTracker()

        # Last reception indicator (1 = packet received, 0 = not received)
        self.delta_k = 0

    def init_value(self, x_hat0, P0):
        self.x_hat = np.asarray(x_hat0, dtype=float).reshape(-1, 1)
        self.P = np.asarray(P0, dtype=float)

        init_info = self.current_decision_from_fresh_update(self.x_hat, self.P)
        self.last_decision = int(init_info["pi"])

        self._horizon_detected = False
        self._current_horizon = None
        self._horizon_history = []
        self._horizon_state_history = []
        self._sojourn_initialized = False

        # Initialise sojourn tracking for time step 0
        self._init_sojourn()
        self._sojourn_initialized = True

    def _init_sojourn(self, delta_k=None):
        """
        Called at the start of every new sojourn (state transition detected or at init).
          - Active disruption persists across sojourn boundaries; it is cleared
            only by the recovery mechanism (not by the sojourn transition itself).
          - Active recovery window (_in_recovery) is NOT interrupted by a
            sojourn transition; it continues until t_h elapses.

        delta_k: reception indicator at the transition step (1=received, 0=not received).
          Used to distinguish on-time (h=0) from miss (h=-1) when no prediction was made.
        """
        if self._sojourn_initialized:
            from_state = self._sojourn_state   # state of the sojourn that just ended
            if self._horizon_detected:
                # Estimator predicted the transition ahead of time
                self._horizon_history.append(self._current_horizon)
            elif delta_k == 1:
                # No prior prediction, but packet received at the transition step
                self._horizon_history.append(0)
            else:
                # No prediction and no packet at the transition step — miss
                self._horizon_history.append(-1)
            self._horizon_state_history.append(from_state)

        self._sojourn_state = int(self.last_decision)
        self._horizon_detected = False
        self._current_horizon = None
        # For baseline: _aoi, _in_recovery, _recovery_end_k are intentionally preserved.

    def _sample_t_h(self):
        """
        Sample recovery duration t_h from the discrete Weibull(lambda, kappa) distribution.

        CDF: P(T_h <= tau) = 1 - exp(-((tau+1)/lambda)^kappa)
        Inverse CDF: tau = max(0, ceil(lambda * (-log(1-u))^(1/kappa) - 1))
        """
        u = np.random.rand()
        val = self.weibull_lambda * ((-math.log(max(1.0 - u, 1e-15))) ** (1.0 / self.weibull_kappa)) - 1.0
        return max(0, int(math.ceil(val)))

    def hard_decision(self, x):
        value = float(self.c.T @ x)
        return 1 if value >= self.Delta else 0

    def packet_success(self, transmit, p_t, h2=None, chan_u=None):
        """
        If sensor decides not to transmit => delta = 0.
        If a disruption is active => delta = 0 (outage blocks all receptions).
        Otherwise: draw h2 ~ Exp(1), compute instantaneous SNR gamma = h2 * gbar
        (gbar = p_t / noise_var), and succeed with probability 1 - epsilon_k.

        h2:     optional pre-generated Exp(1) channel gain.
        chan_u:  optional pre-generated U[0,1] for the success Bernoulli.
        """
        if transmit == 0:
            return 0
        if self._disruption_active:# or self._in_recovery:
            return 0
        h2 = np.random.exponential(scale=1.0) if h2 is None else float(h2)
        chan_u = np.random.rand() if chan_u is None else float(chan_u)
        gbar = p_t / self.noise_var
        gamma = h2 * gbar
        eps_k = compute_instantaneous_packet_error(gamma, self.blocklength_n, self.info_bits_l)
        success = chan_u > eps_k
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

    def current_decision_from_fresh_update(self, x_hat=None, P=None):
        """
        Estimator-side current decision when a fresh sensor update is available:
        - outside confusion region: reliable rule
        - inside confusion region: resolve by xi
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

        if info["region"] == "confusion":
            s_hat = float((self.c.T @ x_hat).item())
            info["pi"] = 1 if s_hat >= self.xi else 0
            info["used_xi_inside_confusion"] = True
        else:
            info["used_xi_inside_confusion"] = False

        return info

    def step(self, x_s, P_s, transmit, k, p_t=1.0,
             disruption_onset_k=None, h2=None, chan_u=None):
        """
        Remote estimator update at time step k.

        transmit           : sensor transmission decision (0 or 1)
        k                  : current time index (used for recovery scheduling)
        p_t                : transmit power used by the sensor this step
        disruption_onset_k : optional pre-generated bool — True if a new
                             disruption starts this slot (from env_seq).
                             pre-generated bool — True if a new disruption
                             starts this slot (from env_seq). All policies
                             share the same onset events.
        h2                 : optional pre-generated Exp(1) channel gain.
        chan_u              : optional pre-generated U[0,1] for channel
                             success Bernoulli.

        Disruption / AoI / recovery logic:
          - onset fires when disruption_onset_k is True AND no disruption
            is currently active; persists across sojourn transitions until
            cleared by the recovery mechanism.
          - While disruption_active: all receptions are blocked (delta = 0).
          - AoI increments each step without a successful reception.
          - When AoI >= theta_s AND disruption active AND not yet detected:
              t_h ~ DiscreteWeibull(lambda, kappa), recovery ends at k + t_h.
          - At k >= recovery_end: disruption clears, normal reception resumes.
        """
        # ------------------------------------------------------------------
        # Disruption onset — always driven by pre-generated external onset flag.
        # Fires only when flagged AND no disruption is currently active.
        # Persists across sojourn boundaries until cleared by recovery.
        # ------------------------------------------------------------------
        if disruption_onset_k and not self._disruption_active:
            self._disruption_active = True

        # ------------------------------------------------------------------
        # Reception (disruption blocks regardless of channel or transmit)
        # ------------------------------------------------------------------
        delta_k = self.packet_success(transmit, p_t, h2=h2, chan_u=chan_u)
        self.delta_k = delta_k

        # ------------------------------------------------------------------
        # Estimator update
        # ------------------------------------------------------------------
        x_pred = self.A @ self.x_hat
        P_pred = self.A @ self.P @ self.A.T + self.Q

        self.x_hat = delta_k * x_s + (1 - delta_k) * x_pred
        self.P = delta_k * P_s + (1 - delta_k) * P_pred

        # ------------------------------------------------------------------
        # AoI update (resets only on successful reception for ALL modes)
        # ------------------------------------------------------------------
        if delta_k == 1:
            self._aoi = 0
        else:
            self._aoi += 1

        # ------------------------------------------------------------------
        # Recovery trigger: fire whenever AoI >= threshold.
        # ------------------------------------------------------------------
        theta_s = self.theta_0 if self._sojourn_state == 0 else self.theta_1
        if not self._in_recovery and self._aoi >= theta_s:
            self._in_recovery = True
            t_h = self._sample_t_h()
            self._recovery_end_k = k + t_h
            self.metrics_tracker.log_recovery_trigger(bool(self._disruption_active))

        # ------------------------------------------------------------------
        # Recovery completion
        # ------------------------------------------------------------------
        if (self._in_recovery
                and self._recovery_end_k is not None
                and k >= self._recovery_end_k):
            self._in_recovery = False
            self._recovery_end_k = None
            self._disruption_active = False

        # ------------------------------------------------------------------
        # Decision
        # ------------------------------------------------------------------
        if delta_k == 1:
            # Fresh sensor update: xi is meaningful here
            decision_info = self.current_decision_from_fresh_update(self.x_hat, self.P)
        else:
            # No fresh update: use reliable rule only;
            # if still in confusion region, it automatically keeps the previous decision
            decision_info = self.reliable_decision(self.x_hat, self.P)

        decision = int(decision_info["pi"])

        # ------------------------------------------------------------------
        # Estimator-side predictive horizon measurement (observational only,
        # does NOT change the decision or any other estimator state).
        # Only run when a fresh packet was received (delta_k == 1) so that
        # KF drift on stale estimates (e.g. event-trigger steady state) does
        # not spuriously set _horizon_detected and inflate the horizon for
        # policies that never send predictive packets.
        # ------------------------------------------------------------------
        if decision == self.last_decision and not self._horizon_detected and delta_k == 1:
            _pred = predictive_transition_detection(
                x_hat_sensor=self.x_hat,
                P_sensor=self.P,
                A=self.A,
                Q=self.Q,
                c=self.c,
                Delta=self.Delta,
                alpha_fp=self.alpha_fp,
                alpha_fn=self.alpha_fn,
                previous_decision=decision,
                ell=self.ell,
                xi=self.xi,
            )
            if _pred.get("found_transition") and _pred.get("predicted_horizon") is not None:
                self._horizon_detected = True
                self._current_horizon = int(_pred["predicted_horizon"])

        # ------------------------------------------------------------------
        # Sojourn transition detection
        # A new sojourn begins when the decision changes; this can only be
        # known AFTER receiving the measurement and making the new decision.
        # The estimator has no access to the real process state — it compares
        # only its own current decision with its own previous decision.
        # ------------------------------------------------------------------
        prev_decision = self.last_decision
        self.last_decision = decision
        if decision != prev_decision:
            self._init_sojourn(delta_k=delta_k)

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

    def __init__(self, A, Q, c, Delta, alpha_fp, alpha_fn, ell, xi, initial_decision=0, p_t=20.0):
        self.A = A
        self.Q = Q
        self.c = np.asarray(c, dtype=float).reshape(-1, 1)
        self.Delta = float(Delta)
        self.xi = float(xi)

        self.alpha_fp = float(alpha_fp)
        self.alpha_fn = float(alpha_fn)
        self.ell = int(ell)
        self.p_t = float(p_t)

        # m_{k-1}
        self.last_local_decision = int(initial_decision)

        # whether a predictive packet for a future transition has already been sent
        self.pending_predictive_packet = False
        self.predicted_transition_step = None
        self.predicted_state = None
        self.predicted_horizon = None

        # debug info
        self.last_prediction = None
        self.last_local_decision_info = None


    def current_sensor_decision(self, x_s, P_s, previous_decision):
        """
        Current sensor-side decision:
        - outside confusion region: use reliable rule
        - inside confusion region: resolve by xi

        This is used ONLY for the current sensor-side classification.
        It is NOT used for predictive triggering.
        """
        info = evaluate_decision(
            x_hat=x_s,
            P=P_s,
            c=self.c,
            Delta=self.Delta,
            alpha_fp=self.alpha_fp,
            alpha_fn=self.alpha_fn,
            previous_decision=previous_decision,
        )

        if info["region"] == "confusion":
            s_hat = float((self.c.T @ x_s).item())
            pi_cur = 1 if s_hat >= self.xi else 0
            info["pi"] = pi_cur
            info["used_xi_inside_confusion"] = True
        else:
            info["used_xi_inside_confusion"] = False

        return info

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

        # -------------------------------------------------
        # Current sensor-side decision:
        # reliable outside confusion, xi-based inside confusion
        # -------------------------------------------------
        local_info = self.current_sensor_decision(
            x_s=x_s,
            P_s=P_s,
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
        if had_pending and k < self.predicted_transition_step:
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
        if had_pending and k == self.predicted_transition_step:
            self.pending_predictive_packet = False
            self.last_local_decision = self.predicted_state
            self.last_prediction = {
                "found_transition": True,
                "reason": "predicted_transition",
                "predicted_transition_time": k,
                "predicted_horizon": self.predicted_horizon,
                "decision_now": m_k,
            }

        # -------------------------------------------------
        # Case 3:
        # No prior predictive packet existed for this change
        # -> send immediately
        # -------------------------------------------------
        if not had_pending and state_changed:
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
        # Case 4:
        # No prior predictive packet existed
        # And no changed
        # -------------------------------------------------
        if not had_pending and not state_changed:
            pass
            #reference_decision = m_k


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
            previous_decision=self.last_local_decision,
            ell=self.ell,
            xi=self.xi,
        )
        self.last_prediction = pred

        found = bool(pred.get("found_transition", False))
        #
        transmit = 0

        if found:
            # Only future predictive update should reach here.
            # If horizon == 0 happens here, still allow transmit,
            # but logically this should be rare after the case split above.
            transmit = 1

            self.pending_predictive_packet = True
            self.predicted_transition_step = k + int(pred["predicted_horizon"])
            self.predicted_horizon = int(pred["predicted_horizon"])
            self.predicted_state = pred["predicted_decision"]
        else:
            transmit = 0
            self.pending_predictive_packet = False


        return transmit

def build_predictive_policy(A, Q, c, Delta, alpha_fp, alpha_fn, ell, xi, initial_decision=0, p_t=20.0):
    return PredictivePolicy(
        A=A,
        Q=Q,
        c=c,
        Delta=Delta,
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
        ell=ell,
        xi=xi,
        initial_decision=initial_decision,
        p_t=p_t,
    )

class ResilientPredictivePolicy(PredictivePolicy):
    """
    Resilient extension of PredictivePolicy.

    Identical to PredictivePolicy in all cases EXCEPT:

    1. At a state transition (state_changed = True):
         - Pending predictive packet already sent  -> consume it silently, no send.
         - No pending predictive packet            -> also do NOT send (resilience skipped
           at transition time; the probabilistic updates should have kept the estimator
           informed between transitions).

    2. When there is NO transition AND no new predictive update is triggered:
         -> send with probability p_u,s* where s = current local state
            (p_u0 for state 0, p_u1 for state 1).
            p_u0* and p_u1* are computed by compute_pu_star() in resilience_design.py.

    The probabilistic "resilience" send is therefore:
      - Active  : steady-state steps with no predictive trigger.
      - Inactive: any step where the local decision has just changed.
    """

    def __init__(self, A, Q, c, Delta, alpha_fp, alpha_fn, ell, xi,
                 p_u0, p_u1, initial_decision=0, p_t=20.0):
        super().__init__(A, Q, c, Delta, alpha_fp, alpha_fn, ell, xi,
                         initial_decision=initial_decision, p_t=p_t)
        self.p_u0 = float(p_u0)
        self.p_u1 = float(p_u1)

    def __call__(self, x_true, x_s, P_s, x_hat_remote, P_remote, k):
        previous_decision = self.last_local_decision

        local_info = self.current_sensor_decision(x_s, P_s, previous_decision)
        m_k = int(local_info["pi"])
        self.last_local_decision_info = local_info

        state_changed = (m_k != previous_decision)
        had_pending = self.pending_predictive_packet

        # ------------------------------------------------------------------
        # Case 1: pending packet active, no transition yet -> silent
        # ------------------------------------------------------------------
        if had_pending and k < self.predicted_transition_step:
            self.last_prediction = {
                "found_transition": False,
                "reason": "pending_predictive_packet_active_same_state",
                "predicted_transition_time": None,
            }
            p_u = self.p_u0 if self.last_local_decision == 1 else self.p_u1
            if np.random.rand() < p_u:
                return 1
            return 0

        # ------------------------------------------------------------------
        # Case 2: pending packet realized at this transition step
        # -> consume silently (estimator was pre-informed by predictive packet)
        # ------------------------------------------------------------------
        if had_pending and k == self.predicted_transition_step:
            self.pending_predictive_packet = False
            self.last_local_decision = self.predicted_state
            self.last_prediction = {
                "found_transition": True,
                "reason": "predicted_transition",
                "predicted_transition_time": k,
                "predicted_horizon": self.predicted_horizon,
                "decision_now": m_k,
            }

        # ------------------------------------------------------------------
        # Case 3: unpredicted transition (no pending packet, state just changed)
        # Send the update
        # ------------------------------------------------------------------
        if not had_pending and state_changed:
            self.last_local_decision = m_k
            self.last_prediction = {
                "found_transition": True,
                "reason": "unpredicted_current_transition_resilient_skip",
                "predicted_transition_time": 0,
                "predicted_horizon": 0,
                "decision_now": m_k,
            }
            return 1

        # ------------------------------------------------------------------
        # Step: run predictive transition detection for the next transition.
        # Reaches here from Case 2 fall-through or Case 4 (no change, no pending).
        # ------------------------------------------------------------------
        pred = predictive_transition_detection(
            x_hat_sensor=x_s,
            P_sensor=P_s,
            A=self.A,
            Q=self.Q,
            c=self.c,
            Delta=self.Delta,
            alpha_fp=self.alpha_fp,
            alpha_fn=self.alpha_fn,
            previous_decision=self.last_local_decision,
            ell=self.ell,
            xi=self.xi,
        )
        self.last_prediction = pred

        found = bool(pred.get("found_transition", False))

        if found:
            # Predictive update found -> send (same as PredictivePolicy)
            self.pending_predictive_packet = True
            self.predicted_transition_step = k + int(pred["predicted_horizon"])
            self.predicted_horizon = int(pred["predicted_horizon"])
            self.predicted_state = pred["predicted_decision"]
            return 1

        # ------------------------------------------------------------------
        # No predictive update found.
        # - If this step is a transition (Case 2 fall-through, state_changed=True)
        #   -> resilience is skipped; return 0.
        # - Otherwise (steady state, no transition)
        #   -> send with probability p_u,s* for the current local state.
        # ------------------------------------------------------------------
        self.pending_predictive_packet = False

        if not state_changed:
            p_u = self.p_u0 if self.last_local_decision == 0 else self.p_u1
            if np.random.rand() < p_u:
                return 1

        return 0

def build_resilient_predictive_policy(A, Q, c, Delta, alpha_fp, alpha_fn, ell, xi,
                                      p_u0, p_u1, initial_decision=0, p_t=20.0):
    return ResilientPredictivePolicy(
        A=A,
        Q=Q,
        c=c,
        Delta=Delta,
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
        ell=ell,
        xi=xi,
        p_u0=p_u0,
        p_u1=p_u1,
        initial_decision=initial_decision,
        p_t=p_t,
    )

# Supported policy_type values:
#   "predictive"            -> PredictivePolicy
#   "resilient_predictive"  -> ResilientPredictivePolicy  (requires p_u0, p_u1)
#   "probability"           -> ProbabilityPolicy          (requires p_prob)
#   "event_trigger"         -> EventTriggerPolicy         (transmit only on decision transition)
def build_policy(policy_type, A, Q, c, Delta, alpha_fp, alpha_fn, ell, xi,
                 initial_decision=0, p_t=20.0, p_u0=0.0, p_u1=0.0, p_prob=1.0):
    if policy_type == "predictive":
        return build_predictive_policy(
            A=A, Q=Q, c=c, Delta=Delta,
            alpha_fp=alpha_fp, alpha_fn=alpha_fn,
            ell=ell, xi=xi,
            initial_decision=initial_decision, p_t=p_t,
        )
    if policy_type == "resilient_predictive":
        return build_resilient_predictive_policy(
            A=A, Q=Q, c=c, Delta=Delta,
            alpha_fp=alpha_fp, alpha_fn=alpha_fn,
            ell=ell, xi=xi,
            p_u0=p_u0, p_u1=p_u1,
            initial_decision=initial_decision, p_t=p_t,
        )
    if policy_type == "probability":
        return ProbabilityPolicy(p=p_prob, p_t=p_t)
    if policy_type == "event_trigger":
        return build_event_trigger_policy(
            c=c, Delta=Delta,
            alpha_fp=alpha_fp, alpha_fn=alpha_fn,
            xi=xi,
            initial_decision=initial_decision, p_t=p_t,
        )
    raise ValueError(f"Unknown policy_type '{policy_type}'. "
                     f"Choose 'predictive', 'resilient_predictive', 'probability', or 'event_trigger'.")

def generate_env_sequences(T, n, m, Q, R, A, c, Delta, x_true0, p_r=0.0):
    """
    Pre-generate all environment randomness for T steps so that the same
    physical environment can be replayed identically across different policies.

    Disruption onsets are generated from the true physical process trajectory:
      1. Simulate x_true forward using the pre-generated process noise w.
      2. Determine the binary state at each step: state_k = 1 if c^T x_k >= Delta
         else 0. This is the exact model — the Markov chain (q01, q10) is only
         a surrogate used for closed-form analysis, not for simulation.
      3. Identify sojourns (contiguous runs in the same state).
      4. For each sojourn, draw Bernoulli(p_r): if hit, place the disruption
         onset at a step chosen uniformly at random within that sojourn.

    Returns a dict with arrays of shape:
        "w"                 : (T, n)  — process noise w_k ~ N(0, Q)
        "v"                 : (T, m)  — measurement noise v_k ~ N(0, R)
        "disruption_onset"  : (T,)    — bool, True = new disruption starts this step
        "h2"                : (T,)    — Exp(1) channel gain
        "chan"               : (T,)    — U[0,1] for channel success Bernoulli
        "state_seq"         : (T,)    — int, true binary state (0/1) at each step
    """
    w = np.random.multivariate_normal(np.zeros(n), Q, size=T)          # (T, n)
    v = np.random.multivariate_normal(np.zeros(m), R, size=T)          # (T, m)
    h2 = np.random.exponential(scale=1.0, size=T)                      # (T,)
    chan = np.random.rand(T)                                            # (T,)

    c_vec = np.asarray(c).reshape(-1)

    # --- Step 1 & 2: simulate x_true and derive binary state sequence ---
    state_seq = np.empty(T, dtype=int)
    x = np.asarray(x_true0, dtype=float).reshape(-1)
    for k in range(T):
        x = A @ x + w[k]
        state_seq[k] = 1 if float(c_vec @ x) >= Delta else 0

    # --- Step 3 & 4: one disruption per sojourn with prob p_r, uniform onset ---
    disruption_onset = np.zeros(T, dtype=bool)
    k = 0
    while k < T:
        sojourn_start = k
        current_state = state_seq[k]
        while k < T and state_seq[k] == current_state:
            k += 1
        sojourn_end = k          # sojourn covers [sojourn_start, sojourn_end - 1]
        if np.random.rand() < p_r:
            onset_k = np.random.randint(sojourn_start, sojourn_end)
            disruption_onset[onset_k] = True

    return {"w": w, "v": v, "disruption_onset": disruption_onset,
            "h2": h2, "chan": chan, "state_seq": state_seq}


def simulate_system(sensor, estimator, T, transmission_policy, blocklength_n, t_sym,
                    p_t_default=20.0, env_seq=None):
    """
    Run the joint simulation.

    transmission_policy(...) must return 0 or 1.
    Energy per transmission = blocklength_n * p_t * t_sym,
    where p_t is read from the policy (falls back to p_t_default).

    env_seq: optional dict returned by generate_env_sequences(). When provided,
             process noise, measurement noise, disruption onset draws, channel
             gain h2, and channel success draws are taken from the pre-generated
             arrays instead of sampled live — ensuring identical environment
             across different policies.
    """
    p_t = float(getattr(transmission_policy, "p_t", p_t_default))
    energy_per_tx = blocklength_n * p_t * t_sym

    true_values = []
    est_values = []
    true_events = []
    est_events = []
    transmit_hist = []
    delta_hist = []
    energy_hist = []
    disruption_hist = []

    fp_idx = []
    fn_idx = []

    # optional debug histories
    z_hist = []
    region_hist = []

    for k in range(T):
        # Sensor side — use pre-generated noise if env_seq provided
        if env_seq is not None:
            x_true, y_k, x_s, P_s = sensor.step(w_k=env_seq["w"][k], v_k=env_seq["v"][k])
        else:
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

        # Record disruption state BEFORE the step (reflects whether this step is blocked)
        disruption_hist.append(int(getattr(estimator, '_disruption_active', 0)))

        # Remote estimator step — use pre-generated channel/disruption draws if env_seq provided
        est_kwargs = dict(x_s=x_s, P_s=P_s, transmit=transmit, k=k, p_t=p_t)
        if env_seq is not None:
            est_kwargs["disruption_onset_k"] = bool(env_seq["disruption_onset"][k])
            est_kwargs["h2"]                 = env_seq["h2"][k]
            est_kwargs["chan_u"]             = env_seq["chan"][k]
        x_hat, P, delta_k, decision_est, decision_info = estimator.step(**est_kwargs)

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
        energy_hist.append(transmit * energy_per_tx)

        z_hist.append(float(decision_info["z"]))
        region_hist.append(decision_info["region"])

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
    energy_hist = np.array(energy_hist)
    disruption_hist = np.array(disruption_hist, dtype=int)
    z_hist = np.array(z_hist)

    # Rates
    num_negative_truth = np.sum(true_events == 0)
    num_positive_truth = np.sum(true_events == 1)

    false_positive_rate = len(fp_idx) / num_negative_truth if num_negative_truth > 0 else 0.0
    false_negative_rate = len(fn_idx) / num_positive_truth if num_positive_truth > 0 else 0.0

    # --- Estimator-side predictive horizon ---
    # horizon_history entries: -1 = miss, 0 = on-time, >0 = predicted h steps ahead.
    # One entry per estimator-detected sojourn transition.
    horizon_history       = estimator._horizon_history
    horizon_state_history = estimator._horizon_state_history

    pred_vals = [h for h in horizon_history if h > 0]
    # Average over transitions where a predictive horizon was actually detected
    avg_predicted_horizon_estimator = float(np.mean(pred_vals)) if pred_vals else float("nan")
    # Average over ALL estimator-detected transitions (misses and on-times contribute 0)
    n_true_transitions = len(horizon_history)
    avg_predicted_horizon_total = (
        float(sum(pred_vals)) / n_true_transitions
        if n_true_transitions > 0 else float("nan")
    )
    # --- Per-state lead-time distribution from horizon_history ---
    # Keyed by from_state (0 = alarm onset 0->1, 1 = alarm clearing 1->0).
    # Values: -1 = miss, 0 = on-time, >0 = predicted h steps ahead.
    def _state_summary(L_list):
        if not L_list:
            return {"n_total": 0, "n_pred": 0, "n_ontime": 0, "n_miss": 0,
                    "prob_L_geq_0": float("nan"), "avg_pred_horizon": float("nan")}
        n_total  = len(L_list)
        n_miss   = sum(1 for l in L_list if l < 0)
        n_ontime = sum(1 for l in L_list if l == 0)
        n_pred   = sum(1 for l in L_list if l > 0)
        prob     = (n_pred + n_ontime) / n_total
        pv       = [l for l in L_list if l > 0]
        avg_ph   = float(np.mean(pv)) if pv else 0.0
        return {"n_total": n_total, "n_pred": n_pred, "n_ontime": n_ontime,
                "n_miss": n_miss, "prob_L_geq_0": prob, "avg_pred_horizon": avg_ph}

    L_per_state = {0: [], 1: []}
    for h, s in zip(horizon_history, horizon_state_history):
        L_per_state[s].append(h)

    stats_s0 = _state_summary(L_per_state[0])   # 0->1 transitions
    stats_s1 = _state_summary(L_per_state[1])   # 1->0 transitions

    all_L = list(horizon_history)
    n_timely_overall = sum(1 for l in all_L if l >= 0)
    prob_L_geq_0 = (float(n_timely_overall) / n_true_transitions
                    if n_true_transitions > 0 else float("nan"))

    total_energy = float(np.sum(energy_hist))
    avg_energy = total_energy / T

    total_disruption_steps = int(np.sum(disruption_hist))

    def _fmt_nan(v, fmt=".4f"):
        return "nan" if math.isnan(v) else format(v, fmt)

    print(f"False Positive Rate (FPR): {false_positive_rate:.4f}")
    print(f"False Negative Rate (FNR): {false_negative_rate:.4f}")
    print(f"Avg Predicted Horizon (est-side, pred only):  {_fmt_nan(avg_predicted_horizon_estimator)}")
    print(f"Avg Predicted Horizon (over all transitions): {_fmt_nan(avg_predicted_horizon_total)}")
    print(f"P(L >= 0) overall:           {_fmt_nan(prob_L_geq_0)}")
    print(f"Transitions (est-detected):  {n_true_transitions}")
    for label, st in [("0->1 (alarm onset) ", stats_s0),
                      ("1->0 (alarm clear) ", stats_s1)]:
        print(f"  [{label}] total={st['n_total']:4d} "
              f"| pred(L>0)={st['n_pred']:4d} "
              f"| on-time(L=0)={st['n_ontime']:4d} "
              f"| miss={st['n_miss']:4d} "
              f"| P(L>=0)={_fmt_nan(st['prob_L_geq_0'])} "
              f"| E[L|L>0]={_fmt_nan(st['avg_pred_horizon'])}")
    print(f"Total Energy Consumption:  {total_energy:.6e} J")
    print(f"Avg Energy per Step:       {avg_energy:.6e} J")
    print(f"Disruption steps:          {total_disruption_steps} / {T}")
    print(f"Total successful receptions: {np.sum(delta_hist)} / {T}")
    print(f"Total transmission attempts: {np.sum(transmit_hist)} / {T}")

    # ------------------------------------------------------------------
    # TP / FP metrics from the tracker
    # ------------------------------------------------------------------
    rec_metrics = estimator.metrics_tracker.compute_metrics()

    theta_info = {"theta_0": estimator.theta_0, "theta_1": estimator.theta_1}

    results = {
        "true_values": true_values,
        "est_values": est_values,
        "true_events": true_events,
        "est_events": est_events,
        "transmit_hist": transmit_hist,
        "delta_hist": delta_hist,
        "energy_hist": energy_hist,
        "fp_idx": np.array(fp_idx),
        "fn_idx": np.array(fn_idx),
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
        "avg_predicted_horizon":       avg_predicted_horizon_estimator,
        "avg_predicted_horizon_total": avg_predicted_horizon_total,
        "total_energy": total_energy,
        "avg_energy": avg_energy,
        "disruption_hist": disruption_hist,
        "z_hist": z_hist,
        "region_hist": region_hist,
        "prob_L_geq_0":       prob_L_geq_0,
        "L_per_state":        L_per_state,
        "stats_s0":           stats_s0,
        "stats_s1":           stats_s1,
        "n_true_transitions": n_true_transitions,
        "horizon_history":    horizon_history,
        # Recovery detection quality
        "theta":               theta_info,
        "n_tp":                rec_metrics["n_tp"],
        "n_fp":                rec_metrics["n_fp"],
        "total_triggers":      rec_metrics["total_triggers"],
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
    total_energy = results.get("total_energy", float("nan"))
    avg_energy = results.get("avg_energy", float("nan"))
    disruption_hist = results.get("disruption_hist", np.zeros(len(t), dtype=int))

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
    # Shade disruption periods (contiguous runs of disruption_hist == 1)
    in_disruption = False
    d_start = None
    disruption_labeled = False
    for ki in range(len(disruption_hist)):
        if disruption_hist[ki] == 1 and not in_disruption:
            in_disruption = True
            d_start = ki
        elif disruption_hist[ki] == 0 and in_disruption:
            in_disruption = False
            label = "Disruption" if not disruption_labeled else ""
            ax1.axvspan(d_start, ki, color="purple", alpha=0.15, label=label)
            ax2.axvspan(d_start, ki, color="purple", alpha=0.15)
            disruption_labeled = True
    if in_disruption:
        label = "Disruption" if not disruption_labeled else ""
        ax1.axvspan(d_start, len(disruption_hist), color="purple", alpha=0.15, label=label)
        ax2.axvspan(d_start, len(disruption_hist), color="purple", alpha=0.15)

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

    total_disruption_steps = int(np.sum(disruption_hist))
    avg_ph_est   = results.get("avg_predicted_horizon",       float("nan"))
    avg_ph_total = results.get("avg_predicted_horizon_total", float("nan"))
    def _nan_fmt(v, fmt=".4f"):
        return "nan" if math.isnan(float(v)) else format(float(v), fmt)
    textstr = (
        f"FPR = {fpr:.4f}\n"
        f"FNR = {fnr:.4f}\n"
        f"E[L|L>0] (pred only)  = {_nan_fmt(avg_ph_est)}\n"
        f"E[L] (all transitions) = {_nan_fmt(avg_ph_total)}\n"
        f"Total Energy = {total_energy:.4e} J\n"
        f"Avg Energy/Step = {avg_energy:.4e} J\n"
        f"Disruption steps = {total_disruption_steps}"
    )
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


def plot_horizon_histograms(all_results, policy_names=None, x_cap=15):
    """
    For each policy, plot two subplots (0->1 and 1->0 transitions).
    Three bar categories per subplot:
      Blue  (L > 0) : individual bars at L=1,2,...,x_cap; values beyond x_cap
                      are collapsed into a single bar labelled "≥x_cap+1"
      Green (L = 0) : on-time — estimator first correct exactly at T_idx
      Red   (L < 0) : miss    — estimator still wrong at T_idx

    x_cap controls how far the x-axis extends before collapsing overflow.
    """
    import matplotlib.pyplot as plt

    if policy_names is None:
        policy_names = list(all_results.keys())

    n_policies   = len(policy_names)
    col_pred     = "#2196F3"   # blue  — L > 0
    col_on       = "#4CAF50"   # green — L = 0
    col_miss     = "#F44336"   # red   — miss
    col_overflow = "#90CAF9"   # light blue — overflow bucket

    fig, axes = plt.subplots(n_policies, 2,
                             figsize=(13, 3.5 * n_policies),
                             squeeze=False)

    state_labels = {0: "0→1  (alarm onset)", 1: "1→0  (alarm clearing)"}

    for row_idx, name in enumerate(policy_names):
        res         = all_results[name]
        L_per_state = res.get("L_per_state", {0: [], 1: []})

        for col_idx, from_state in enumerate([0, 1]):
            ax     = axes[row_idx][col_idx]
            L_list = L_per_state.get(from_state, [])
            st     = res["stats_s" + str(from_state)]

            if not L_list:
                ax.set_title(f"{name}\n{state_labels[from_state]}")
                ax.text(0.5, 0.5, "no transitions", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            pred_vals = [l for l in L_list if l > 0]
            n_ontime  = sum(1 for l in L_list if l == 0)
            n_miss    = sum(1 for l in L_list if l < 0)
            n_total   = len(L_list)
            assert len(pred_vals) + n_ontime + n_miss == n_total

            # Split pred values into in-range [1, x_cap] and overflow (>x_cap)
            pred_inrange  = [l for l in pred_vals if 1 <= l <= x_cap]
            pred_overflow = [l for l in pred_vals if l > x_cap]

            # ── bar positions ──────────────────────────────────────────────
            # miss at -2 (visual gap), on-time at 0, L=1..x_cap, overflow at x_cap+1
            bar_x      = []
            bar_h      = []
            bar_colors = []

            # miss bar (left of dashed separator)
            if n_miss > 0:
                bar_x.append(-2)
                bar_h.append(n_miss)
                bar_colors.append(col_miss)

            # on-time bar
            if n_ontime > 0:
                bar_x.append(0)
                bar_h.append(n_ontime)
                bar_colors.append(col_on)

            # individual L = 1 .. x_cap bars (skip zeros)
            for lv in range(1, x_cap + 1):
                cnt = pred_inrange.count(lv)
                if cnt > 0:
                    bar_x.append(lv)
                    bar_h.append(cnt)
                    bar_colors.append(col_pred)

            # overflow bucket
            if pred_overflow:
                bar_x.append(x_cap + 1)
                bar_h.append(len(pred_overflow))
                bar_colors.append(col_overflow)

            ax.bar(bar_x, bar_h, color=bar_colors, width=0.7, edgecolor="white")

            # dashed separator between miss and the rest
            if n_miss > 0:
                ax.axvline(x=-1, color="gray", linestyle="--", linewidth=0.9)

            # ── x-axis ticks and labels ────────────────────────────────────
            tick_pos    = []
            tick_labels = []
            if n_miss > 0:
                tick_pos.append(-2);          tick_labels.append("miss")
            if n_ontime > 0:
                tick_pos.append(0);           tick_labels.append("0")
            for lv in range(1, x_cap + 1):
                if pred_inrange.count(lv) > 0:
                    tick_pos.append(lv);      tick_labels.append(str(lv))
            if pred_overflow:
                tick_pos.append(x_cap + 1);  tick_labels.append(f"≥{x_cap+1}")

            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_xlim(-3, x_cap + 2)

            # ── legend entries ─────────────────────────────────────────────
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(color=col_miss,     label=f"miss: {n_miss}"),
                Patch(color=col_on,       label=f"L=0 (on-time): {n_ontime}"),
                Patch(color=col_pred,     label=f"L>0 (early): {len(pred_inrange)}"),
            ]
            if pred_overflow:
                legend_handles.append(
                    Patch(color=col_overflow,
                          label=f"L≥{x_cap+1} (early, overflow): {len(pred_overflow)}"))
            ax.legend(handles=legend_handles, fontsize=7, loc="upper right")

            ax.set_title(
                f"{name}  |  {state_labels[from_state]}\n"
                f"total={n_total}  "
                f"P(L≥0)={st['prob_L_geq_0']:.3f}  "
                f"E[L|L>0]={st['avg_pred_horizon']:.2f}",
                fontsize=9)
            ax.set_xlabel("Lead time L (steps)", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)

    fig.suptitle("Predicted Horizon Distribution per State and Policy",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


class ProbabilityPolicy:
    """
    Random transmission policy: transmit with probability p at each step.
    """

    def __init__(self, p=1.0, p_t=20.0):
        self.p = float(p)
        self.p_t = float(p_t)
        self.last_prediction = None

    def __call__(self, x_true, x_s, P_s, x_hat_remote, P_remote, k):
        return 1 if np.random.uniform(0, 1) <= self.p else 0

class EventTriggerPolicy:
    """
    Event-triggered transmission policy.

    The sensor only transmits when its local decision undergoes a transition
    (0->1 or 1->0). No packet is sent during steady-state steps.

    The local decision is computed with the same feasible-decision rule used by
    the other policies: reliable outside the confusion region, xi-based inside.
    """

    def __init__(self, c, Delta, alpha_fp, alpha_fn, xi, initial_decision=0, p_t=20.0):
        self.c = np.asarray(c, dtype=float).reshape(-1, 1)
        self.Delta = float(Delta)
        self.alpha_fp = float(alpha_fp)
        self.alpha_fn = float(alpha_fn)
        self.xi = float(xi)
        self.p_t = float(p_t)

        self.last_local_decision = int(initial_decision)
        self.last_prediction = None  # for interface compatibility with simulate_system

    def _current_sensor_decision(self, x_s, P_s):
        info = evaluate_decision(
            x_hat=x_s,
            P=P_s,
            c=self.c,
            Delta=self.Delta,
            alpha_fp=self.alpha_fp,
            alpha_fn=self.alpha_fn,
            previous_decision=self.last_local_decision,
        )
        if info["region"] == "confusion":
            s_hat = float((self.c.T @ x_s).item())
            info["pi"] = 1 if s_hat >= self.xi else 0
            info["used_xi_inside_confusion"] = True
        else:
            info["used_xi_inside_confusion"] = False
        return info

    def __call__(self, x_true, x_s, P_s, x_hat_remote, P_remote, k):
        local_info = self._current_sensor_decision(x_s, P_s)
        m_k = int(local_info["pi"])

        if m_k != self.last_local_decision:
            self.last_local_decision = m_k
            self.last_prediction = {
                "found_transition": True,
                "reason": "event_trigger_transition",
                "predicted_horizon": 0,
                "decision_now": m_k,
            }
            return 1

        self.last_prediction = {
            "found_transition": False,
            "reason": "no_transition",
            "predicted_horizon": None,
        }
        return 0

def build_event_trigger_policy(c, Delta, alpha_fp, alpha_fn, xi, initial_decision=0, p_t=20.0):
    return EventTriggerPolicy(
        c=c,
        Delta=Delta,
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
        xi=xi,
        initial_decision=initial_decision,
        p_t=p_t,
    )

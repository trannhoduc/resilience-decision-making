import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# System parameters
# ============================================================
A = np.array([[0.0, 1.0],
              [-0.9, 1.8]])

C = np.array([[0.5, 1.0]])

Q = np.array([[0.0, 0.0],
              [0.0, 1.0]])

R = np.array([[0.1]])

c = np.array([[1.0],
              [0.0]])

Delta = 5.0

alpha_fp = 0.3
alpha_fn = 0.3
xi = 5.0
ell = 10

T = 200
burn_in = 100
seed = 42

np.random.seed(seed)


# ============================================================
# Helpers
# ============================================================
def scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def compute_z_thresholds(alpha_fp, alpha_fn):
    """
    z_minus = Phi^{-1}(alpha_fp)
    z_plus  = Phi^{-1}(1 - alpha_fn)
    """
    if not (0.0 < alpha_fp < 1.0):
        raise ValueError("alpha_fp must be in (0,1).")
    if not (0.0 < alpha_fn < 1.0):
        raise ValueError("alpha_fn must be in (0,1).")

    z_minus = float(norm.ppf(alpha_fp))
    z_plus = float(norm.ppf(1.0 - alpha_fn))
    return z_minus, z_plus


def current_decision_with_xi(x_hat, c, xi):
    """
    Current decision uses ONLY xi.
    No prev-decision memory.
    """
    s_hat = scalar(c.T @ x_hat)
    return 1 if s_hat >= xi else 0


def reliable_decision_from_state(x_hat, P, c, Delta, alpha_fp, alpha_fn):
    """
    Reliable decision based on z-score thresholds.
    Returns:
        pi:
            1 if z <= z_minus
            0 if z >= z_plus
            None if in confusion region
        z:
            normalized distance to threshold
        region:
            'low', 'high', or 'confusion'
    """
    z_minus, z_plus = compute_z_thresholds(alpha_fp, alpha_fn)

    s_hat = scalar(c.T @ x_hat)
    sigma2 = scalar(c.T @ P @ c)
    sigma = np.sqrt(max(sigma2, 1e-14))

    z = (Delta - s_hat) / sigma

    if z <= z_minus:
        return 1, z, "low"
    elif z >= z_plus:
        return 0, z, "high"
    else:
        return None, z, "confusion"


def predictive_detection(x_hat, P, A, Q, c, Delta, alpha_fp, alpha_fn, ell, prev_decision):
    """
    Predictive detection logic:
    - propagate up to ell steps
    - if at any look-ahead step the predicted state enters confusion region,
      stop immediately and return False
    - otherwise, if reliable decision changes, return True with horizon i

    Returns:
        found: bool
        horizon: int or None
        predicted_step: int or None
        reason: str
    """
    z_minus, z_plus = compute_z_thresholds(alpha_fp, alpha_fn)

    x_tmp = x_hat.copy()
    P_tmp = P.copy()

    for i in range(ell + 1):
        pi, z, region = reliable_decision_from_state(
            x_tmp, P_tmp, c, Delta, alpha_fp, alpha_fn
        )

        # If we hit confusion at any look-ahead step, drop immediately
        if region == "confusion":
            return False, None, None, f"confusion_at_i={i}"

        # If reliable decision changes, we have a predictive trigger
        if pi != prev_decision:
            return True, i, i, f"decision_change_at_i={i}"

        # Propagate one step forward
        x_tmp = A @ x_tmp
        P_tmp = A @ P_tmp @ A.T + Q

    return False, None, None, "no_change"


# ============================================================
# Sensor-only simulation
# ============================================================
def run_sensor_debug():
    n = A.shape[0]
    m = C.shape[0]

    # Initial values
    x_true = np.array([[0.0], [1.0]])
    x_hat = np.zeros((n, 1))
    P = np.eye(n)

    # Burn-in
    for _ in range(burn_in):
        w = np.random.multivariate_normal(np.zeros(n), Q).reshape(-1, 1)
        x_true = A @ x_true + w

        v = np.random.multivariate_normal(np.zeros(m), R).reshape(-1, 1)
        y = C @ x_true + v

        x_pred = A @ x_hat
        P_pred = A @ P @ A.T + Q

        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)

        x_hat = x_pred + K @ (y - C @ x_pred)
        I = np.eye(n)
        P = (I - K @ C) @ P_pred @ (I - K @ C).T + K @ R @ K.T

    # Histories
    t_hist = []
    s_true_hist = []
    s_hat_hist = []
    detect_times = []
    predict_times = []

    # Decision history
    prev_current_pi = current_decision_with_xi(x_hat, c, xi)

    # State machine for prediction blocking
    prediction_active = False
    prediction_end = -1

    for k in range(T):
        # propagate true system
        w = np.random.multivariate_normal(np.zeros(n), Q).reshape(-1, 1)
        x_true = A @ x_true + w

        v = np.random.multivariate_normal(np.zeros(m), R).reshape(-1, 1)
        y = C @ x_true + v

        # KF prediction/update
        x_pred = A @ x_hat
        P_pred = A @ P @ A.T + Q

        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)

        x_hat = x_pred + K @ (y - C @ x_pred)
        I = np.eye(n)
        P = (I - K @ C) @ P_pred @ (I - K @ C).T + K @ R @ K.T

        # scalar signals
        s_true = scalar(c.T @ x_true)
        s_hat = scalar(c.T @ x_hat)

        t_hist.append(k)
        s_true_hist.append(s_true)
        s_hat_hist.append(s_hat)

        # current decision: xi only
        current_pi = current_decision_with_xi(x_hat, c, xi)

        # if there is an active predictive window, skip new prediction
        if prediction_active:
            if k >= prediction_end:
                prediction_active = False
        else:
            # use previous current decision as reference for predictive detection
            found, horizon_i, predicted_step, reason = predictive_detection(
                x_hat=x_hat,
                P=P,
                A=A,
                Q=Q,
                c=c,
                Delta=Delta,
                alpha_fp=alpha_fp,
                alpha_fn=alpha_fn,
                ell=ell,
                prev_decision=prev_current_pi,
            )

            if found and horizon_i is not None and horizon_i > 0:
                detect_times.append(k)
                predict_times.append(k + horizon_i)
                prediction_active = True
                prediction_end = k + horizon_i
                print(f"[k={k}] predict transition at {k + horizon_i}, horizon={horizon_i}, reason={reason}")

        prev_current_pi = current_pi

    # ========================================================
    # Plot
    # ========================================================
    plt.figure(figsize=(14, 6))
    plt.plot(t_hist, s_true_hist, label=r"True $s_k$", linewidth=2)
    plt.plot(t_hist, s_hat_hist, label=r"Estimate $\hat{s}_k$", linewidth=2)
    plt.axhline(Delta, linestyle="--", label=r"Threshold $\Delta$")

    if len(detect_times) > 0:
        plt.scatter(
            detect_times,
            [s_true_hist[t] for t in detect_times],
            color="green",
            label="Detection time (k)",
            zorder=5
        )

    for pt in predict_times:
        plt.axvline(pt, color="orange", linestyle=":", alpha=0.5)

    plt.title("Sensor predictive debug with xi-current decision and confusion-drop predictive rule")
    plt.xlabel("Time step k")
    plt.ylabel(r"Value of $s_k$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_sensor_debug()
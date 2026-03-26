from PARAMETERS import *
from model import *
from computation import evaluate_decision


if __name__ == "__main__":
    np.random.seed(42)

    # -------------------------------------------------
    # Design parameters used by the reliable decision rule
    # Use temporary defaults if they are not yet added to PARAMETERS.py
    # -------------------------------------------------
    alpha_fp = globals().get("ALPHA_FP", 0.05)
    alpha_fn = globals().get("ALPHA_FN", 0.05)
    lookahead_ell = globals().get("LOOKAHEAD_ELL", 5)

    # -------------------------------------------------
    # Sensor initialization
    # -------------------------------------------------
    x_true0 = np.array([0.0, 1.0])
    x_s0 = np.array([0.0, 0.0])
    P_s0 = np.eye(2)

    sensor = Model(A, C, Q, R)
    sensor.init_value(x_true0, x_s0, P_s0)

    # -------------------------------------------------
    # Remote estimator initialization
    # Here we assume the estimator starts with the same posterior
    # information as the sensor, as you requested.
    # -------------------------------------------------
    x_hat0 = np.array([0.0, 0.0])
    P0 = np.eye(2)

    estimator = RemoteEstimator(
        A=A,
        Q=Q,
        c=c,
        Delta=Delta,
        epsilon=epsilon,
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
    )
    estimator.init_value(x_hat0, P0)

    # -------------------------------------------------
    # Initialize the first reliable decision from the SENSOR posterior
    # This is better than blindly using estimator.last_decision,
    # because predictive scheduling in the paper is sensor-driven.
    #
    # We still need a fallback previous_decision for the very first call.
    # For that fallback only, use a simple hard threshold on x_s0.
    # -------------------------------------------------
    s0_value = (np.asarray(c).reshape(1, -1) @ np.asarray(x_s0).reshape(-1, 1)).item()
    fallback_initial_decision = 1 if s0_value >= Delta else 0

    initial_decision_info = evaluate_decision(
        x_hat=np.asarray(x_s0).reshape(-1, 1),
        P=np.asarray(P_s0),
        c=c,
        Delta=Delta,
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
        previous_decision=fallback_initial_decision,
    )
    initial_decision = int(initial_decision_info["pi"])

    # Keep estimator and policy synchronized at time 0
    estimator.last_decision = initial_decision

    # -------------------------------------------------
    # Build predictive-only policy
    # No resilience update here yet
    # -------------------------------------------------
    policy = build_predictive_policy(
        A=A,
        Q=Q,
        c=c,
        Delta=Delta,
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
        ell=lookahead_ell,
        initial_decision=initial_decision,
    )

    # -------------------------------------------------
    # Run simulation
    # -------------------------------------------------
    results = simulate_system(sensor, estimator, T, policy)

    print(f"Initial reliable decision = {initial_decision}")
    print(f"Initial decision region   = {initial_decision_info['region']}")
    print(f"Initial z-value           = {initial_decision_info['z']:.4f}")

    plot_results(results, Delta)
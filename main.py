from PARAMETERS import *
from model import *
from computation import evaluate_decision, precompute_all
from resilience_design import solve_resilience_design, compute_average_packet_error_closed_form
from baselines import calibrate_baselines


if __name__ == "__main__":
    np.random.seed(42)

    import PARAMETERS as P

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

    derived = precompute_all(P)
    xi = derived["steady_state_benchmark"]["xi"]

    # ------------------------------------------------------------------
    # Resilience design: solve for optimal theta_0, theta_1, p_u0*, p_u1*
    # ------------------------------------------------------------------
    solution = solve_resilience_design(derived=derived, params=P)
    if solution["feasible"]:
        best_design = solution["best_design"]
        theta_0  = int(best_design["theta0"])
        theta_1  = int(best_design["theta1"])
        pu0      = float(best_design["pu0_star"])
        pu1      = float(best_design["pu1_star"])
        p_t_star = float(best_design["pt"])
        r_prop   = float(best_design["average_rate"])
        print(f"Resilience design: theta_0={theta_0}, theta_1={theta_1}, "
              f"eps_r={best_design['eps_r']:.4f}, objective={best_design['objective']:.4f}")
        print(f"  p_u0* = {pu0:.4f},  p_u1* = {pu1:.4f}")
        print(f"  p_t*  = {p_t_star:.4f},  r_prop = {r_prop:.4f}  "
              f"(energy budget E = {p_t_star * r_prop:.4f})")
    else:
        theta_0  = 5
        theta_1  = 5
        pu0      = 0.0
        pu1      = 0.0
        p_t_star = float(P_T)
        r_prop   = None
        print("WARNING: No feasible resilience design found. Using defaults theta_0=5, theta_1=5.")

    # ------------------------------------------------------------------
    # Baseline calibration: match every baseline's energy to E = p_t* r_prop
    # ------------------------------------------------------------------
    # epsilon_bar_fn(p_t) -> average PER from Eq. (8), using current params.
    def epsilon_bar_fn(p_t: float) -> float:
        return compute_average_packet_error_closed_form(
            pt=p_t,
            noise_var=float(P.CHANNEL_NOISE_VAR),
            n=int(P.BLOCKLENGTH_N),
            l=int(P.INFO_BITS_L),
        )["eps_bar"]

    if r_prop is not None:
        calibration = calibrate_baselines(
            p_t_star=p_t_star,
            r_prop=r_prop,
            q01=float(derived["markov_surrogate"]["q01"]),
            q10=float(derived["markov_surrogate"]["q10"]),
            epsilon_bar_fn=epsilon_bar_fn,
            p_t_max=float(P.PT_MAX),
        )
        print("\nBaseline calibration (all share energy budget "
              f"E = {calibration['energy_budget']:.4f}):")
        pred_c = calibration["predictive"]
        prob_c = calibration["probability"]
        print(f"  predictive / event : p_t = {pred_c['p_t']:.4f}, "
              f"rate = {pred_c['rate']:.4f}, feasible = {pred_c['feasible']}")
        print(f"  probability        : p_t = {prob_c['p_t']:.4f}, "
              f"p = {prob_c['p']:.4f}, feasible = {prob_c['feasible']}"
              + (" [boundary: p=1 optimal]" if prob_c["boundary_hit"] else ""))

        # Per-policy transmit powers and probability
        policy_p_t = {
            "resilient_predictive": p_t_star,
            "predictive":           calibration["predictive"]["p_t"],
            "event_trigger":        calibration["event"]["p_t"],
            "probability":          calibration["probability"]["p_t"],
        }
        p_prob_calibrated = calibration["probability"]["p"]
    else:
        # Fallback when no feasible design was found
        policy_p_t = {k: float(P_T) for k in
                      ["resilient_predictive", "predictive", "event_trigger", "probability"]}
        p_prob_calibrated = 0.3

    q01 = float(derived["markov_surrogate"]["q01"])
    q10 = float(derived["markov_surrogate"]["q10"])

    # -------------------------------------------------
    # Pre-generate shared environment sequences (seed already set above).
    # All policies will see the same process evolution, sensor measurements,
    # channel gains h2, and disruption onset draws.
    # Recovery duration is NOT controlled here (detection time differs per policy).
    # -------------------------------------------------
    n_state = A.shape[0]
    m_obs   = C.shape[0]
    env_seq = generate_env_sequences(T, n_state, m_obs, Q, R)

    # -------------------------------------------------
    # Compute the initial decision once (same for all policies).
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

    print(f"Initial reliable decision = {initial_decision}")
    print(f"Initial decision region   = {initial_decision_info['region']}")
    print(f"Initial z-value           = {initial_decision_info['z']:.4f}")

    # -------------------------------------------------
    # Run all four policies over the same environment.
    # Sensor and estimator are reset before each policy run.
    # -------------------------------------------------
    all_results = {}

    for POLICY_TYPE in ["resilient_predictive", "predictive", "event_trigger", "probability"]:

        # Reset sensor state
        sensor = Model(A, C, Q, R)
        sensor.init_value(x_true0, x_s0, P_s0)

        # Reset estimator state
        estimator = RemoteEstimator(
            A=A,
            Q=Q,
            c=c,
            Delta=Delta,
            noise_var=float(P.CHANNEL_NOISE_VAR),
            blocklength_n=int(P.BLOCKLENGTH_N),
            info_bits_l=int(P.INFO_BITS_L),
            alpha_fp=alpha_fp,
            alpha_fn=alpha_fn,
            xi=xi,
            q01=q01,
            q10=q10,
            p_r=P_R,
            theta_0=theta_0,
            theta_1=theta_1,
            weibull_lambda=TH_RECOVERY_LAMBDA,
            weibull_kappa=TH_RECOVERY_KAPPA,
        )
        estimator.init_value(x_hat0, P0)
        estimator.last_decision = initial_decision

        p_t_for_policy = policy_p_t[POLICY_TYPE]

        policy = build_policy(
            policy_type=POLICY_TYPE,
            A=A,
            Q=Q,
            c=c,
            Delta=Delta,
            alpha_fp=alpha_fp,
            alpha_fn=alpha_fn,
            ell=lookahead_ell,
            xi=xi,
            initial_decision=initial_decision,
            p_t=p_t_for_policy,
            p_u0=pu0,
            p_u1=pu1,
            p_prob=p_prob_calibrated,
        )

        print(f"\nRunning policy '{POLICY_TYPE}' with p_t = {p_t_for_policy:.4f}"
              + (f", p_prob = {p_prob_calibrated:.4f}" if POLICY_TYPE == "probability" else ""))

        results = simulate_system(sensor, estimator, T, policy,
                                  blocklength_n=BLOCKLENGTH_N,
                                  t_sym=T_SYM,
                                  p_t_default=p_t_for_policy,
                                  env_seq=env_seq)

        all_results[POLICY_TYPE] = results

    plot_results(all_results, Delta)
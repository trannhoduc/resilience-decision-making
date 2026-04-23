import math

from PARAMETERS import *
from model import *
from computation import evaluate_decision, precompute_all
from resilience_design import solve_resilience_design, compute_average_packet_error_closed_form
from baselines import calibrate_baselines, compute_baseline_thresholds


if __name__ == "__main__":
    np.random.seed(42)

    import PARAMETERS as P

    # -------------------------------------------------
    # Design parameters
    # -------------------------------------------------
    alpha_fp = globals().get("ALPHA_FP", 0.05)
    alpha_fn = globals().get("ALPHA_FN", 0.05)
    lookahead_ell = globals().get("LOOKAHEAD_ELL", 5)

    # -------------------------------------------------
    # Sensor initialization
    # -------------------------------------------------
    x_true0 = np.array([0.0, 1.0])
    x_s0    = np.array([0.0, 0.0])
    P_s0    = np.eye(2)
    x_hat0  = np.array([0.0, 0.0])
    P0      = np.eye(2)

    sensor = Model(A, C, Q, R)
    sensor.init_value(x_true0, x_s0, P_s0)

    derived = precompute_all(P)
    xi = derived["steady_state_benchmark"]["xi"]

    # ------------------------------------------------------------------
    # Resilience design
    # ------------------------------------------------------------------
    solution = solve_resilience_design(derived=derived, params=P)
    if solution["feasible"]:
        best_design = solution["best_design"]
        theta_0   = int(best_design["theta0"])
        theta_1   = int(best_design["theta1"])
        pu0       = float(best_design["pu0_star"])
        pu1       = float(best_design["pu1_star"])
        p_t_star  = float(best_design["pt"])
        r_prop    = float(best_design["average_rate"])
        eps_r     = float(best_design["eps_r"])
        print(f"Resilience design: theta_0={theta_0}, theta_1={theta_1}, "
              f"eps_r={eps_r:.4f}, objective={best_design['objective']:.4f}")
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
        eps_r    = 0.05
        print("WARNING: No feasible resilience design found. Using defaults.")

    # ------------------------------------------------------------------
    # Baseline calibration — match every baseline's energy to E = p_t* r_prop
    # ------------------------------------------------------------------
    def epsilon_bar_fn(p_t: float) -> float:
        return compute_average_packet_error_closed_form(
            pt=p_t,
            noise_var=float(P.CHANNEL_NOISE_VAR),
            n=int(P.BLOCKLENGTH_N),
            l=int(P.INFO_BITS_L),
        )["eps_bar"]

    q01 = float(derived["markov_surrogate"]["q01"])
    q10 = float(derived["markov_surrogate"]["q10"])

    if r_prop is not None:
        calibration = calibrate_baselines(
            p_t_star=p_t_star,
            r_prop=r_prop,
            q01=q01,
            q10=q10,
            epsilon_bar_fn=epsilon_bar_fn,
            p_t_max=float(P.PT_MAX),
        )
        print("\nBaseline calibration (all share energy budget "
              f"E = {calibration['energy_budget']:.4f}):")
        pred_c = calibration["predictive"]
        prob_c = calibration["probability"]
        pred_warn = (" [CAPPED to PT_MAX — ideal p_t would exceed hardware limit]"
                     if not pred_c['feasible'] else "")
        print(f"  predictive / event : p_t = {pred_c['p_t']:.4f}, "
              f"rate = {pred_c['rate']:.4f}, feasible = {pred_c['feasible']}{pred_warn}")
        print(f"  probability        : p_t = {prob_c['p_t']:.4f}, "
              f"p = {prob_c['p']:.4f}, feasible = {prob_c['feasible']}"
              + (" [boundary: p=1 optimal]" if prob_c["boundary_hit"] else ""))

        policy_p_t = {
            "resilient_predictive": p_t_star,
            "predictive":           calibration["predictive"]["p_t"],
            "event_trigger":        calibration["event"]["p_t"],
            "probability":          calibration["probability"]["p_t"],
        }
        p_prob_calibrated = calibration["probability"]["p"]
        r_pred            = calibration["predictive"]["rate"]
    else:
        policy_p_t = {k: float(P_T) for k in
                      ["resilient_predictive", "predictive", "event_trigger", "probability"]}
        p_prob_calibrated = 0.3
        r_pred = 2.0 * q01 * q10 / (q01 + q10)

    # ------------------------------------------------------------------
    # Baseline AoI thresholds
    # ------------------------------------------------------------------
    p_t_prob      = policy_p_t["probability"]
    eps_bar_prob  = epsilon_bar_fn(p_t_prob)
    eta_prob      = p_prob_calibrated * (1.0 - eps_bar_prob)

    baseline_thresholds = compute_baseline_thresholds(
        q01=q01,
        q10=q10,
        eta_prob=eta_prob,
        epsilon_r=eps_r,
    )
    print(f"\nBaseline AoI thresholds (eps_r = {eps_r:.4f}):")
    pred_th = baseline_thresholds["predictive"]
    prob_th = baseline_thresholds["probability"]
    print(f"  predictive / event : theta_0={pred_th['theta_0']}, theta_1={pred_th['theta_1']}")
    print(f"  probability        : theta={prob_th['theta']}")

    # ------------------------------------------------------------------
    # Pre-generate shared environment sequences (all policies see the same
    # physical process, channel gains, and disruption draws)
    # ------------------------------------------------------------------
    env_seq = generate_env_sequences(T, A.shape[0], C.shape[0], Q, R,
                                     A=A, c=c, Delta=Delta, x_true0=x_true0,
                                     p_r=P_R)

    # ------------------------------------------------------------------
    # Initial reliable decision from sensor posterior (shared across runs)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Common estimator kwargs
    # ------------------------------------------------------------------
    _est_base_kwargs = dict(
        A=A, Q=Q, c=c, Delta=Delta,
        noise_var=float(P.CHANNEL_NOISE_VAR),
        blocklength_n=int(P.BLOCKLENGTH_N),
        info_bits_l=int(P.INFO_BITS_L),
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
        xi=xi,
        ell=lookahead_ell,
        q01=q01,
        q10=q10,
        p_r=P_R,
        weibull_lambda=TH_RECOVERY_LAMBDA,
        weibull_kappa=TH_RECOVERY_KAPPA,
    )

    # ------------------------------------------------------------------
    # Policy loop
    # ------------------------------------------------------------------
    # Each entry: (display_name, policy_type, extra_est_kwargs, extra_pol_kwargs)
    POLICY_SPECS = [
        (
            "resilient_predictive",
            "resilient_predictive",
            dict(theta_0=theta_0, theta_1=theta_1),
            dict(p_u0=pu0, p_u1=pu1),
        ),
        (
            "predictive",
            "predictive",
            dict(theta_0=pred_th["theta_0"], theta_1=pred_th["theta_1"]),
            {},
        ),
        (
            "event_trigger",
            "event_trigger",
            dict(theta_0=pred_th["theta_0"], theta_1=pred_th["theta_1"]),
            {},
        ),
        (
            "probability",
            "probability",
            #dict(theta_0=prob_th["theta"], theta_1=prob_th["theta"]),
            dict(theta_0=theta_0, theta_1=theta_1),
            dict(p_prob=p_prob_calibrated),
        ),
    ]

    all_results = {}

    for (name, policy_type, est_kwargs, pol_kwargs) in POLICY_SPECS:
        print(f"\n{'='*60}")
        print(f"Running policy: {name}")
        print(f"{'='*60}")

        # Reset sensor to identical initial state
        sensor.init_value(x_true0, x_s0, P_s0)

        p_t_for_policy = policy_p_t[policy_type]

        estimator = RemoteEstimator(
            **_est_base_kwargs,
            **est_kwargs,
        )
        estimator.init_value(x_hat0, P0)
        estimator.last_decision = initial_decision

        policy = build_policy(
            policy_type=policy_type,
            A=A, Q=Q, c=c, Delta=Delta,
            alpha_fp=alpha_fp,
            alpha_fn=alpha_fn,
            ell=lookahead_ell,
            xi=xi,
            initial_decision=initial_decision,
            p_t=p_t_for_policy,
            p_u0=pol_kwargs.get("p_u0", 0.0),
            p_u1=pol_kwargs.get("p_u1", 0.0),
            p_prob=pol_kwargs.get("p_prob", p_prob_calibrated),
        )

        results = simulate_system(
            sensor, estimator, T, policy,
            blocklength_n=BLOCKLENGTH_N,
            t_sym=T_SYM,
            p_t_default=p_t_for_policy,
            env_seq=env_seq,
        )

        # Attach calibration info for the table
        results["p_t"]  = p_t_for_policy
        results["rate"] = (r_prop if policy_type == "resilient_predictive"
                           else r_pred if policy_type in ("predictive", "event_trigger")
                           else p_prob_calibrated)
        all_results[name] = results

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    def _fmt(v, fmt=".4f"):
        if v is None:
            return "  N/A  "
        try:
            fv = float(v)
            if math.isnan(fv):
                return "  N/A  "
            return format(fv, fmt)
        except (TypeError, ValueError):
            return "  N/A  "

    header = (
        f"{'Method':<24} | {'p_t':>7} | {'rate':>7} | "
        f"{'FPR':>7} | {'FNR':>7} | {'E[avg]':>9} | "
        f"{'Prec':>7} | {'Recall':>7}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for name, res in all_results.items():
        th = res["theta"]
        if "theta" in th:
            theta_str = f"θ={th['theta']}"
        else:
            theta_str = f"θ₀={th['theta_0']},θ₁={th['theta_1']}"

        row = (
            f"{name:<24} | "
            f"{_fmt(res['p_t'])} | "
            f"{_fmt(res['rate'])} | "
            f"{_fmt(res['fpr'])} | "
            f"{_fmt(res['fnr'])} | "
            f"{_fmt(res['avg_energy'], '.3e')}"
        )
        print(row)
        print(f"  {theta_str:<22}   "
              f"TP={res['n_tp']:3d} FP={res['n_fp']:3d} total={res['total_triggers']:3d}")

    # Per-state P(L>=0) breakdown
    print("\nPer-state timeliness breakdown:")
    for name, res in all_results.items():
        s0 = res["stats_s0"]
        s1 = res["stats_s1"]
        print(f"  {name:<24} "
              f"0->1: pred={s0['n_pred']:4d} on-time={s0['n_ontime']:4d} miss={s0['n_miss']:4d} "
              f"P(L>=0)={_fmt(s0['prob_L_geq_0'])} E[L|L>0]={_fmt(s0['avg_pred_horizon'])}  |  "
              f"1->0: pred={s1['n_pred']:4d} on-time={s1['n_ontime']:4d} miss={s1['n_miss']:4d} "
              f"P(L>=0)={_fmt(s1['prob_L_geq_0'])} E[L|L>0]={_fmt(s1['avg_pred_horizon'])}")

    print(sep)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # plot_results(all_results["resilient_predictive"], Delta)
    # plot_results(all_results["probability"], Delta)
    plot_horizon_histograms(all_results)
    

"""
main_test.py — run 2 paradigms (filter_based + decision_adoption) and display
10 figure windows at once (5 per paradigm, one per method).

Each window is identical to plot_results with two extra marker types:
  Gold star  : predictive packet received (resilient_predictive / predictive only)
  Gold x     : predictive packet transmitted but lost
  Purple ◆   : filter_based only, probability/aoii policies only — successful
               reception where the Kalman estimator already predicts state 1
               while truth is still in state 0 (estimator ahead of truth).
               Not shown for event_trigger (fires after the transition, never ahead)
               nor in decision_adoption (no Kalman filter at the estimator).

Change SEED to vary the environment realisation.
"""
import math
import matplotlib.pyplot as plt

from PARAMETERS import *
from model import (
    Model, RemoteEstimator, RemoteEstimatorDecisionOnly,
    simulate_system, simulate_system_decision_only,
    generate_env_sequences, build_policy,
    plot_results_2,
)
from computation import evaluate_decision, precompute_all
from resillience_desgin_blackout import solve_resilience_design, compute_average_packet_error_closed_form
from baselines import calibrate_baselines, compute_baseline_thresholds
import PARAMETERS as P

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 22          # single shared seed for both paradigms

alpha_fp      = globals().get("ALPHA_FP",      0.05)
alpha_fn      = globals().get("ALPHA_FN",      0.05)
lookahead_ell = globals().get("LOOKAHEAD_ELL", 5)

x_true0 = np.array([0.0, 1.0])
x_s0    = np.array([0.0, 0.0])
P_s0    = np.eye(2)
x_hat0  = np.array([0.0, 0.0])
P0      = np.eye(2)


def _epsilon_bar(p_t: float) -> float:
    return compute_average_packet_error_closed_form(
        pt=p_t,
        noise_var=float(P.CHANNEL_NOISE_VAR),
        n=int(P.BLOCKLENGTH_N),
        l=int(P.INFO_BITS_L),
    )["eps_bar"]


def build_all_results(paradigm: str, seed: int) -> dict:
    """Run all 5 policies for one paradigm and seed; return all_results dict."""
    np.random.seed(seed)

    sensor = Model(A, C, Q, R)
    sensor.init_value(x_true0, x_s0, P_s0)

    derived = precompute_all(P)
    xi = derived["steady_state_benchmark"]["xi"]

    solution = solve_resilience_design(derived=derived, params=P)
    if solution["feasible"]:
        bd       = solution["best_design"]
        theta_0  = int(bd["theta0"])
        theta_1  = int(bd["theta1"])
        pu0      = float(bd["pu0_star"])
        pu1      = float(bd["pu1_star"])
        p_t_star = float(bd["pt"])
        r_prop   = float(bd["average_rate"])
        eps_r    = float(bd["eps_r"])
    else:
        theta_0  = 5;  theta_1  = 5
        pu0      = 0.0; pu1      = 0.0
        p_t_star = float(P_T); r_prop = None; eps_r = 0.05

    q01 = float(derived["markov_surrogate"]["q01"])
    q10 = float(derived["markov_surrogate"]["q10"])

    if r_prop is not None:
        calibration = calibrate_baselines(
            p_t_star=p_t_star, r_prop=r_prop, q01=q01, q10=q10,
            epsilon_bar_fn=_epsilon_bar, p_t_max=float(P.PT_MAX),
        )
        pred_c   = calibration["predictive"]
        prob_c   = calibration["probability"]
        aoii_c   = calibration["aoii"]
        policy_p_t = {
            "resilient_predictive": p_t_star,
            "predictive":           pred_c["p_t"],
            "event_trigger":        calibration["event"]["p_t"],
            "probability":          prob_c["p_t"],
            "aoii":                 aoii_c["p_t"],
        }
        p_prob_calibrated = prob_c["p"]
        r_pred   = pred_c["rate"]
        W0_aoii  = aoii_c["W_0"]; W1_aoii = aoii_c["W_1"]
        r_aoii   = aoii_c["rate"]
    else:
        policy_p_t        = {k: float(P_T) for k in
                             ["resilient_predictive","predictive","event_trigger","probability","aoii"]}
        p_prob_calibrated = 0.3
        r_pred   = 2.0 * q01 * q10 / (q01 + q10)
        W0_aoii  = 1; W1_aoii = 1; r_aoii = r_pred
        calibration = {"probability": {"p": p_prob_calibrated, "p_t": float(P_T)},
                       "aoii":        {"W_0": 1, "W_1": 1}}

    baseline_thresholds = compute_baseline_thresholds(
        q01=q01, q10=q10, epsilon_r=eps_r,
        calibration_results=calibration,
        proposed_theta_0=theta_0, proposed_theta_1=theta_1,
        epsilon_bar_fn=_epsilon_bar,
    )
    pred_th = baseline_thresholds["predictive"]
    aoii_th = baseline_thresholds["aoii"]

    env_seq = generate_env_sequences(
        T, A.shape[0], C.shape[0], Q, R,
        A=A, c=c, Delta=Delta, x_true0=x_true0, p_r=P_R,
    )

    s0_val = (np.asarray(c).reshape(1, -1) @ np.asarray(x_s0).reshape(-1, 1)).item()
    fallback = 1 if s0_val >= Delta else 0
    init_info = evaluate_decision(
        x_hat=np.asarray(x_s0).reshape(-1, 1), P=np.asarray(P_s0),
        c=c, Delta=Delta, alpha_fp=alpha_fp, alpha_fn=alpha_fn,
        previous_decision=fallback,
    )
    initial_decision = int(init_info["pi"])

    _est_base = dict(
        A=A, Q=Q, c=c, Delta=Delta,
        noise_var=float(P.CHANNEL_NOISE_VAR),
        blocklength_n=int(P.BLOCKLENGTH_N),
        info_bits_l=int(P.INFO_BITS_L),
        alpha_fp=alpha_fp, alpha_fn=alpha_fn, xi=xi,
        ell=lookahead_ell, q01=q01, q10=q10, p_r=P_R,
        weibull_lambda=TH_RECOVERY_LAMBDA, weibull_kappa=TH_RECOVERY_KAPPA,
    )

    POLICY_SPECS = [
        ("resilient_predictive", "resilient_predictive",
         dict(theta_0=theta_0, theta_1=theta_1), dict(p_u0=pu0, p_u1=pu1)),
        ("predictive", "predictive",
         dict(theta_0=pred_th["theta_0"], theta_1=pred_th["theta_1"]), {}),
        ("event_trigger", "event_trigger",
         dict(theta_0=pred_th["theta_0"], theta_1=pred_th["theta_1"]), {}),
        ("probability", "probability",
         dict(theta_0=baseline_thresholds["probability"]["theta"],
              theta_1=baseline_thresholds["probability"]["theta"]),
         dict(p_prob=p_prob_calibrated)),
        ("aoii", "aoii",
         dict(theta_0=aoii_th["theta_0"], theta_1=aoii_th["theta_1"]),
         dict(W_0=W0_aoii, W_1=W1_aoii)),
    ]

    all_results = {}
    for (name, policy_type, est_kwargs, pol_kwargs) in POLICY_SPECS:
        print(f"  [{name}]", end=" ", flush=True)
        sensor.init_value(x_true0, x_s0, P_s0)
        p_t_for_policy = policy_p_t[policy_type]

        if paradigm == "filter_based":
            estimator = RemoteEstimator(**_est_base, **est_kwargs)
        else:
            estimator = RemoteEstimatorDecisionOnly(**_est_base, **est_kwargs)
        estimator.init_value(x_hat0, P0)
        estimator.last_decision = initial_decision

        policy = build_policy(
            policy_type=policy_type, A=A, Q=Q, c=c, Delta=Delta,
            alpha_fp=alpha_fp, alpha_fn=alpha_fn, ell=lookahead_ell, xi=xi,
            initial_decision=initial_decision, p_t=p_t_for_policy,
            p_u0=pol_kwargs.get("p_u0", 0.0),
            p_u1=pol_kwargs.get("p_u1", 0.0),
            p_prob=pol_kwargs.get("p_prob", p_prob_calibrated),
            W_0=pol_kwargs.get("W_0", 1),
            W_1=pol_kwargs.get("W_1", 1),
        )

        if paradigm == "filter_based":
            results = simulate_system(
                sensor, estimator, T, policy,
                blocklength_n=BLOCKLENGTH_N, t_sym=T_SYM,
                p_t_default=p_t_for_policy, env_seq=env_seq,
            )
        else:
            results = simulate_system_decision_only(
                sensor, estimator, T, policy,
                blocklength_n=BLOCKLENGTH_N, t_sym=T_SYM,
                p_t_default=p_t_for_policy, env_seq=env_seq,
            )

        results["p_t"]  = p_t_for_policy
        results["rate"] = (r_prop  if policy_type == "resilient_predictive"
                           else r_pred if policy_type in ("predictive", "event_trigger")
                           else r_aoii if policy_type == "aoii"
                           else p_prob_calibrated)
        all_results[name] = results

    print()
    return all_results


if __name__ == "__main__":
    paradigms = [
        ("filter_based",     f"filter_based (seed={SEED})"),
        ("decision_adoption", f"decision_adoption (seed={SEED})"),
    ]

    for paradigm, scenario_name in paradigms:
        print(f"\n{'='*60}")
        print(f"Running {scenario_name}")
        print(f"{'='*60}")
        all_results = build_all_results(paradigm, SEED)
        plot_results_2(all_results, Delta=Delta, scenario_name=scenario_name)

    print("\nAll 10 figures ready. Close them to exit.")
    plt.show()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

import PARAMETERS as P
from computation import predictive_transition_detection, evaluate_decision
from model import Model


# ============================================================
# Helpers
# ============================================================

def _get_param(params: Any, name: str, default: Any = None) -> Any:
    return getattr(params, name, default)


def _as_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _as_column(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _scalar(x: Any) -> float:
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"Expected scalar-like array, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def _true_state(x_true: np.ndarray, c: np.ndarray, Delta: float) -> int:
    return 1 if _scalar(c.T @ x_true) >= Delta else 0


def current_sensor_decision_with_xi(
    x_hat: np.ndarray,
    P: np.ndarray,
    c: np.ndarray,
    Delta: float,
    alpha_fp: float,
    alpha_fn: float,
    xi: float,
    previous_decision: int,
) -> tuple[int, Dict[str, Any], bool]:
    """
    Reliable rule outside confusion region.
    Inside confusion region, use xi.
    """
    info = evaluate_decision(
        x_hat=x_hat,
        P=P,
        c=c,
        Delta=Delta,
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
        previous_decision=previous_decision,
    )

    used_xi = False
    if info["region"] == "confusion":
        s_hat = _scalar(c.T @ x_hat)
        pi = 1 if s_hat >= xi else 0
        used_xi = True
    else:
        pi = int(info["pi"])

    return pi, info, used_xi


# ============================================================
# Records
# ============================================================

@dataclass
class PredictionEvent:
    detect_step: int
    predicted_step: int
    horizon_i: int
    used: bool = True


@dataclass
class DecisionChangeEvent:
    step: int
    previous_pi: int
    current_pi: int
    used_xi: bool
    I: int


@dataclass
class TransitionRecord:
    transition_step: int
    state_before: int
    matched_prediction: bool
    detect_step: Optional[int]
    predicted_step: Optional[int]
    horizon_i: int
    prediction_error: Optional[int]


# ============================================================
# Simulation
# ============================================================

def simulate_sensor_only(params: Any) -> Dict[str, Any]:
    """
    Sensor-only simulation.

    Rules:
    1) Compute current sensor decision.
       - outside confusion: reliable rule
       - inside confusion: use xi

    2) If reliable decision equals previous reliable decision, try predictive search.
       - predictive search uses only reliable rule (no xi)
       - if prediction found, store detect time and predicted time k+I
       - until k+I, do not launch a new prediction

    3) If current decision changes and there is no predictive plan, record I = 0.

    4) If a true transition happens while a prediction is active:
       - store the prediction horizon I
       - store prediction error = actual_transition_step - predicted_step
       - if no prediction active, I = 0
    """

    A = _as_array(_get_param(params, "A"))
    C = _as_array(_get_param(params, "C"))
    Q = _as_array(_get_param(params, "Q"))
    R = _as_array(_get_param(params, "R"))
    c = _as_column(_get_param(params, "c"))
    Delta = float(_get_param(params, "Delta"))

    alpha_fp = float(_get_param(params, "ALPHA_FP", 0.2))
    alpha_fn = float(_get_param(params, "ALPHA_FN", 0.2))
    ell = int(_get_param(params, "LOOKAHEAD_ELL", 10))

    # xi is the auxiliary threshold used only in confusion region
    xi = float(_get_param(params, "XI_VALUE", Delta))

    burn_in = int(_get_param(params, "DEBUG_BURN_IN", 100))
    max_steps = int(_get_param(params, "DEBUG_MAX_STEPS", 500))
    seed = int(_get_param(params, "DEBUG_SEED", 42))
    verbose_steps = int(_get_param(params, "DEBUG_VERBOSE_STEPS", 0))

    x_true0 = _as_column(_get_param(params, "X_TRUE0", np.array([0.0, 1.0])))
    x_s0 = _as_column(_get_param(params, "X_S0", np.array([0.0, 0.0])))
    P_s0 = _as_array(_get_param(params, "P_S0", np.eye(A.shape[0])))

    np.random.seed(seed)

    sensor = Model(A, C, Q, R)
    sensor.init_value(x_true0, x_s0, P_s0)

    # --------------------------------------------------------
    # Burn-in
    # --------------------------------------------------------
    prev_reliable_pi = 1 if _scalar(c.T @ sensor.x_s) >= Delta else 0
    prev_current_pi = prev_reliable_pi

    for _ in range(burn_in):
        sensor.step()
        reliable_info = evaluate_decision(
            x_hat=sensor.x_s,
            P=sensor.P_s,
            c=c,
            Delta=Delta,
            alpha_fp=alpha_fp,
            alpha_fn=alpha_fn,
            previous_decision=prev_reliable_pi,
        )
        prev_reliable_pi = int(reliable_info["pi"])

        current_pi, _, _ = current_sensor_decision_with_xi(
            x_hat=sensor.x_s,
            P=sensor.P_s,
            c=c,
            Delta=Delta,
            alpha_fp=alpha_fp,
            alpha_fn=alpha_fn,
            xi=xi,
            previous_decision=prev_current_pi,
        )
        prev_current_pi = int(current_pi)

    # --------------------------------------------------------
    # Main logs
    # --------------------------------------------------------
    t_hist: List[int] = [0]
    s_true_hist: List[float] = [_scalar(c.T @ sensor.x_true)]
    s_hat_hist: List[float] = [_scalar(c.T @ sensor.x_s)]

    pred_events: List[PredictionEvent] = []
    decision_events: List[DecisionChangeEvent] = []
    transition_records: List[TransitionRecord] = []

    # Store only the prediction times for plotting
    pred_detect_steps: List[int] = []
    pred_pred_steps: List[int] = []

    # Active prediction plan
    # Contains: detect_step, predicted_step, horizon_i
    active_prediction: Optional[PredictionEvent] = None

    current_time = 0
    current_true_state = _true_state(sensor.x_true, c, Delta)

    # For statistics
    I_samples: List[int] = []
    pred_errors: List[int] = []
    matched_prediction_count = 0
    missed_prediction_count = 0
    zero_I_count = 0

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------
    while current_time < max_steps:
        s_true_now = _scalar(c.T @ sensor.x_true)
        s_hat_now = _scalar(c.T @ sensor.x_s)

        # Reliable decision for the predictive search logic
        reliable_info = evaluate_decision(
            x_hat=sensor.x_s,
            P=sensor.P_s,
            c=c,
            Delta=Delta,
            alpha_fp=alpha_fp,
            alpha_fn=alpha_fn,
            previous_decision=prev_reliable_pi,
        )
        reliable_pi = int(reliable_info["pi"])

        # Current decision used by the sensor itself
        current_pi, current_info, used_xi = current_sensor_decision_with_xi(
            x_hat=sensor.x_s,
            P=sensor.P_s,
            c=c,
            Delta=Delta,
            alpha_fp=alpha_fp,
            alpha_fn=alpha_fn,
            xi=xi,
            previous_decision=prev_current_pi,
        )

        # Log decision changes
        if current_pi != prev_current_pi:
            # If there is an active prediction and we are at its predicted step,
            # we treat this decision change as a predicted one; otherwise I = 0.
            if active_prediction is not None and current_time == active_prediction.predicted_step:
                I_here = active_prediction.horizon_i
            else:
                I_here = 0

            decision_events.append(
                DecisionChangeEvent(
                    step=current_time,
                    previous_pi=prev_current_pi,
                    current_pi=current_pi,
                    used_xi=used_xi,
                    I=I_here,
                )
            )
            prev_current_pi = current_pi

        # Predictive search:
        # - only when current reliable decision does not change
        # - only when no active prediction is waiting
        # - prediction uses reliable rule only (no xi)
        if active_prediction is None and reliable_pi == prev_reliable_pi:
            pred = predictive_transition_detection(
                x_hat=sensor.x_s,
                P=sensor.P_s,
                A=A,
                Q=Q,
                c=c,
                Delta=Delta,
                alpha_fp=alpha_fp,
                alpha_fn=alpha_fn,
                previous_decision=prev_reliable_pi,
                ell=ell,
            )

            found = bool(pred.get("found_transition", False))
            horizon_i = pred.get("predicted_horizon", pred.get("predicted_transition_time", None))

            if found and horizon_i is not None:
                horizon_i = int(horizon_i)

                # Only keep genuine predictive updates
                if horizon_i > 0:
                    predicted_step = current_time + horizon_i
                    active_prediction = PredictionEvent(
                        detect_step=current_time,
                        predicted_step=predicted_step,
                        horizon_i=horizon_i,
                        used=True,
                    )
                    pred_events.append(active_prediction)
                    pred_detect_steps.append(current_time)
                    pred_pred_steps.append(predicted_step)

                    if verbose_steps > 0 and current_time < verbose_steps:
                        print(
                            f"[PRED] k={current_time:4d} | "
                            f"state={current_true_state} | "
                            f"I={horizon_i} | "
                            f"pred_step={predicted_step}"
                        )

        # Optional verbose logging
        if verbose_steps > 0 and current_time < verbose_steps:
            print(
                f"k={current_time:4d} | "
                f"s_true={s_true_now:9.4f} | "
                f"s_hat={s_hat_now:9.4f} | "
                f"true_state={current_true_state} | "
                f"reliable_pi={reliable_pi} | "
                f"current_pi={current_pi} | "
                f"xi_used={used_xi} | "
                f"active_pred={None if active_prediction is None else (active_prediction.detect_step, active_prediction.predicted_step, active_prediction.horizon_i)}"
            )

        # Advance sensor by one step
        sensor.step()
        current_time += 1

        # Store histories
        t_hist.append(current_time)
        s_true_hist.append(_scalar(c.T @ sensor.x_true))
        s_hat_hist.append(_scalar(c.T @ sensor.x_s))

        # Check true transition after advancing the system
        new_true_state = _true_state(sensor.x_true, c, Delta)
        if new_true_state != current_true_state:
            actual_transition_step = current_time

            if active_prediction is not None:
                I_sample = int(active_prediction.horizon_i)
                pred_error = actual_transition_step - active_prediction.predicted_step
                matched = True
                matched_prediction_count += 1
                pred_errors.append(pred_error)
            else:
                I_sample = 0
                pred_error = None
                matched = False
                zero_I_count += 1

            I_samples.append(I_sample)

            transition_records.append(
                TransitionRecord(
                    transition_step=actual_transition_step,
                    state_before=current_true_state,
                    matched_prediction=matched,
                    detect_step=None if active_prediction is None else active_prediction.detect_step,
                    predicted_step=None if active_prediction is None else active_prediction.predicted_step,
                    horizon_i=I_sample,
                    prediction_error=pred_error,
                )
            )

            # Once a true transition happens, clear the active prediction
            active_prediction = None
            current_true_state = new_true_state

        else:
            # If no transition happened and the active prediction window has passed,
            # clear it and allow a new prediction later.
            if active_prediction is not None and current_time > active_prediction.predicted_step:
                missed_prediction_count += 1
                active_prediction = None

        # Update previous reliable decision after the step
        reliable_info_after = evaluate_decision(
            x_hat=sensor.x_s,
            P=sensor.P_s,
            c=c,
            Delta=Delta,
            alpha_fp=alpha_fp,
            alpha_fn=alpha_fn,
            previous_decision=prev_reliable_pi,
        )
        prev_reliable_pi = int(reliable_info_after["pi"])

        if current_time >= max_steps:
            break

    I_arr = np.asarray(I_samples, dtype=float) if len(I_samples) > 0 else np.array([], dtype=float)
    pred_err_arr = np.asarray(pred_errors, dtype=float) if len(pred_errors) > 0 else np.array([], dtype=float)

    print("\n=== Summary ===")
    print(f"num_predictions = {len(pred_events)}")
    print(f"num_transition_records = {len(transition_records)}")
    print(f"matched_predictions = {matched_prediction_count}")
    print(f"missed_predictions = {missed_prediction_count}")
    print(f"zero_I_transitions = {zero_I_count}")
    if len(I_arr) > 0:
        print(f"E[I] = {float(np.mean(I_arr)):.6f}")
        print(f"min(I) = {float(np.min(I_arr)):.3f}, max(I) = {float(np.max(I_arr)):.3f}")
    if len(pred_err_arr) > 0:
        print(f"mean prediction error = {float(np.mean(pred_err_arr)):.6f}")
        print(f"mean abs prediction error = {float(np.mean(np.abs(pred_err_arr))):.6f}")

    return {
        "time": np.asarray(t_hist, dtype=int),
        "s_true": np.asarray(s_true_hist, dtype=float),
        "s_hat": np.asarray(s_hat_hist, dtype=float),
        "pred_detect_steps": np.asarray(pred_detect_steps, dtype=int),
        "pred_pred_steps": np.asarray(pred_pred_steps, dtype=int),
        "prediction_events": pred_events,
        "decision_events": decision_events,
        "transition_records": transition_records,
        "I_samples": I_arr,
        "pred_errors": pred_err_arr,
        "params": {
            "alpha_fp": alpha_fp,
            "alpha_fn": alpha_fn,
            "ell": ell,
            "xi": xi,
            "burn_in": burn_in,
            "max_steps": max_steps,
            "seed": seed,
        },
    }


# ============================================================
# Summary printing
# ============================================================

def print_transition_summary(results: Dict[str, Any], max_rows: int = 20) -> None:
    records: List[TransitionRecord] = results["transition_records"]
    print("\n=== Transition summary ===")
    print("idx | state_before | transition_step | detect_step | pred_step | I | pred_err | matched")
    print("-" * 96)
    for i, rec in enumerate(records[:max_rows]):
        print(
            f"{i:3d} | "
            f"{rec.state_before:12d} | "
            f"{rec.transition_step:15d} | "
            f"{str(rec.detect_step):10s} | "
            f"{str(rec.predicted_step):9s} | "
            f"{rec.horizon_i:1d} | "
            f"{str(rec.prediction_error):8s} | "
            f"{rec.matched_prediction}"
        )


# ============================================================
# Plotting
# ============================================================

def plot_results(
    results: Dict[str, Any],
    Delta: float,
    alpha_fp: float,
    alpha_fn: float,
    ell: int,
    xi: float,
    out_path: Optional[str] = None,
) -> None:
    t = results["time"]
    s_true = results["s_true"]
    s_hat = results["s_hat"]
    pred_detect_steps = results["pred_detect_steps"]
    pred_pred_steps = results["pred_pred_steps"]
    pred_events: List[PredictionEvent] = results["prediction_events"]
    pred_errors = results["pred_errors"]

    fig, ax = plt.subplots(1, 1, figsize=(18, 7))

    ax.plot(t, s_true, linewidth=2, label=r"True $s_k = c^\top x_k$")
    ax.plot(t, s_hat, linewidth=2, label=r"Sensor estimate $c^\top \hat x^s_{k|k}$")
    ax.axhline(Delta, linestyle="--", linewidth=1.8, label=r"Threshold $\Delta$")

    # Predictive detection times
    if len(pred_detect_steps) > 0:
        ax.scatter(
            pred_detect_steps,
            s_true[pred_detect_steps],
            s=35,
            color="green",
            marker="o",
            label="Predictive detection time",
            zorder=6,
        )

    # Predicted transition times
    if len(pred_pred_steps) > 0:
        for x in pred_pred_steps:
            ax.axvline(x=x, color="green", alpha=0.15, linestyle=":", linewidth=1.0)

    ax.set_title("Sensor-only predictive debug")
    ax.set_xlabel("Time step k")
    ax.set_ylabel(r"Value of $s_k$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    mean_I = float(np.mean(results["I_samples"])) if len(results["I_samples"]) > 0 else float("nan")
    mean_err = float(np.mean(pred_errors)) if len(pred_errors) > 0 else float("nan")
    mean_abs_err = float(np.mean(np.abs(pred_errors))) if len(pred_errors) > 0 else float("nan")

    txt = (
        rf"$\alpha_{{FP}}={alpha_fp}$, $\alpha_{{FN}}={alpha_fn}$, $\ell={ell}$, $\xi={xi}$"
        + "\n"
        + rf"$E[I]={mean_I:.3f}$, mean err={mean_err:.3f}, mean |err|={mean_abs_err:.3f}"
    )
    ax.text(
        0.012, 0.02, txt,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.show()


# ============================================================
# Main
# ============================================================

def main() -> None:
    alpha_fp = float(_get_param(P, "ALPHA_FP", 0.2))
    alpha_fn = float(_get_param(P, "ALPHA_FN", 0.2))
    ell = int(_get_param(P, "LOOKAHEAD_ELL", 10))
    xi = float(_get_param(P, "XI_VALUE", float(_get_param(P, "Delta"))))

    # Debug settings
    setattr(P, "DEBUG_BURN_IN", int(_get_param(P, "DEBUG_BURN_IN", 100)))
    setattr(P, "DEBUG_MAX_STEPS", int(_get_param(P, "DEBUG_MAX_STEPS", 500)))
    setattr(P, "DEBUG_SEED", int(_get_param(P, "DEBUG_SEED", 42)))
    setattr(P, "DEBUG_VERBOSE_STEPS", int(_get_param(P, "DEBUG_VERBOSE_STEPS", 30)))

    print("=== Sensor-only debug configuration ===")
    print(f"alpha_fp={alpha_fp}, alpha_fn={alpha_fn}, ell={ell}, xi={xi}")
    print(
        f"burn_in={_get_param(P, 'DEBUG_BURN_IN', 100)}, "
        f"max_steps={_get_param(P, 'DEBUG_MAX_STEPS', 500)}, "
        f"seed={_get_param(P, 'DEBUG_SEED', 42)}"
    )

    results = simulate_sensor_only(P)
    print_transition_summary(results, max_rows=25)

    plot_results(
        results,
        Delta=float(_get_param(P, "Delta")),
        alpha_fp=alpha_fp,
        alpha_fn=alpha_fn,
        ell=ell,
        xi=xi,
        out_path=_get_param(P, "DEBUG_OUT_PATH", "sensor_predictive_debug.png"),
    )


if __name__ == "__main__":
    main()
"""Plan §15 evaluation metrics (Section IV.B) plus closed-loop inference checks."""

from __future__ import annotations

from typing import Any

from ts_jepa.plan.enforce import plan_enforced

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_INCOMPLETE = "INCOMPLETE"

# Plan §15 paper metrics. Closed-loop stability exercises §14 packet-loss inference.
PLAN_BASELINE_VALIDATION: dict[str, Any] = {
    "checks": (
        "embedding_quality_tsne",
        "consecutive_frame_mape",
        "fig4_sampling_rate_mape",
        "actor_prediction_nmae",
        "control_performance",
        "horizon_prediction_1_to_Kp",
        "communication_reduction",
        "closed_loop_stability",
    ),
    "required_report_keys": (
        "embedding_tsne",
        "consecutive_frame_mape",
        "fig4_mape",
        "actor_nmae",
        "control",
        "prediction_horizon_nmae",
        "communication_bits",
        "stability",
    ),
    "wireless_requires_all_pass": True,
    "control_position_tol": 0.05,
    "control_angle_tol": 0.05,
    "acceptable_control_score_band": [0.74, 1.0],  # scalability plots only; not a pass/fail gate
}


def assert_plan_baseline_validation_config(config: dict[str, Any]) -> None:
    """Raise when evaluation config cannot support plan §15 metrics."""
    if not plan_enforced(config):
        return
    errors: list[str] = []
    ev = config.get("evaluation", {})
    bv = ev.get("baseline_validation", {})

    if bv and bv.get("enabled") is False:
        errors.append("evaluation.baseline_validation.enabled must be true for plan §15")

    pos = float(ev.get("control_position_tol", float("nan")))
    ang = float(ev.get("control_angle_tol", float("nan")))
    if abs(pos - PLAN_BASELINE_VALIDATION["control_position_tol"]) > 1e-12:
        errors.append(
            f"evaluation.control_position_tol: expected "
            f"{PLAN_BASELINE_VALIDATION['control_position_tol']}, got {pos}"
        )
    if abs(ang - PLAN_BASELINE_VALIDATION["control_angle_tol"]) > 1e-12:
        errors.append(
            f"evaluation.control_angle_tol: expected "
            f"{PLAN_BASELINE_VALIDATION['control_angle_tol']}, got {ang}"
        )

    kp = int(config.get("ts_jepa", {}).get("prediction_horizon", {}).get("Kp", -1))
    if kp < 1:
        errors.append("ts_jepa.prediction_horizon.Kp must be >= 1 for horizon NMAE (plan §15 Eq. 27)")

    # Fig. 4 sampling-rate axis is NOT SPECIFIED; the IC key must exist (plan §18).
    if "fig4_sampling_interval_ms" not in config.get("experiments", {}):
        errors.append(
            "experiments.fig4_sampling_interval_ms must exist as an implementation choice "
            "(Fig. 4 rates are NOT SPECIFIED by the paper)"
        )

    if errors:
        raise ValueError("Plan §15 evaluation config mismatch:\n  - " + "\n  - ".join(errors))

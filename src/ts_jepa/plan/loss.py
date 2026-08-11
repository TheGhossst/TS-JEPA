"""Plan §11 JEPA loss specification and config verification."""

from __future__ import annotations

from typing import Any

PLAN_JEPA_LOSS: dict[str, Any] = {
    "paper_objective": "cosine_similarity_alignment",
    "formula": "L_JEPA = -(1/Kp) * sum_{j=1}^{Kp} cos_sim(z_pred_j, z_target_j)",
    "cosine_similarity": "(z_pred · z_target) / (||z_pred||_2 * ||z_target||_2)",
    "implementation": "negative_mean_cosine_similarity",
    "minimizing_form": "one_minus_mean_cosine_similarity",
    "target_source": "ema_target_encoder",
    "target_stop_gradient": True,
    "horizon_axis": "Kp",
}


def assert_plan_jepa_loss_config(config: dict[str, Any]) -> None:
    """Raise ValueError when JEPA loss settings deviate from plan §11."""
    loss = config.get("ts_jepa", {}).get("loss", {})
    target_enc = config.get("ts_jepa", {}).get("target_encoder", {})
    errors: list[str] = []

    paper_obj = str(loss.get("paper_objective", ""))
    if paper_obj != PLAN_JEPA_LOSS["paper_objective"]:
        errors.append(
            f"ts_jepa.loss.paper_objective: expected {PLAN_JEPA_LOSS['paper_objective']!r}, got {paper_obj!r}"
        )

    impl = str(loss.get("implementation", ""))
    allowed_impl = {
        PLAN_JEPA_LOSS["implementation"],
        PLAN_JEPA_LOSS["minimizing_form"],
        "one_minus_cosine_similarity",
    }
    if impl not in allowed_impl:
        errors.append(
            f"ts_jepa.loss.implementation: expected one of {sorted(allowed_impl)}, got {impl!r}"
        )

    if not bool(target_enc.get("stop_gradient", False)):
        errors.append("ts_jepa.target_encoder.stop_gradient must be true (plan §11 targets via EMA encoder)")
    if str(target_enc.get("update_rule", "")) != "ema":
        errors.append("ts_jepa.target_encoder.update_rule must be 'ema' for plan §11 target embeddings")

    kp = int(config.get("ts_jepa", {}).get("prediction_horizon", {}).get("Kp", -1))
    if kp <= 0:
        errors.append("ts_jepa.prediction_horizon.Kp must be positive for plan §11 horizon sum")

    if errors:
        raise ValueError("Plan §11 JEPA loss config mismatch:\n  - " + "\n  - ".join(errors))

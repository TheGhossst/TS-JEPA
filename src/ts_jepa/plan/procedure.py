"""Plan §10 Algorithm 1 training procedure specification."""

from __future__ import annotations

from typing import Any

PLAN_JEPA_PROCEDURE: dict[str, Any] = {
    "steps": (
        "context_encoding",
        "target_encoding_stop_gradient",
        "autoregressive_prediction",
        "jepa_loss",
        "sgd_update_theta_phi",
        "ema_update_target",
    ),
    "optimized_modules": ("context_encoder", "predictor"),
    "frozen_modules": ("target_encoder",),
    "loss": "jepa_loss",
    "optimizer": "SGD",
    "ema_after_optimizer_step": True,
    "target_stop_gradient": True,
}


def assert_plan_jepa_procedure_config(config: dict[str, Any]) -> None:
    """
    Validate config documents plan §10 Algorithm 1 training procedure invariants.

    Call from baseline entry scripts. Does not enforce batch/Kp (smoke-safe).
    """
    errors: list[str] = []
    jepa = config.get("ts_jepa", {})
    opt = jepa.get("optimizer", {})
    tgt = jepa.get("target_encoder", {})
    loss = jepa.get("loss", {})
    proc = jepa.get("training_procedure", {})

    if str(opt.get("type", "")) != "SGD":
        errors.append(f"plan §10 Algorithm 1 step 5 requires SGD; got optimizer.type={opt.get('type')!r}")
    if not bool(tgt.get("stop_gradient", False)):
        errors.append("plan §10 Algorithm 1 step 2 requires target_encoder.stop_gradient=true")
    if str(tgt.get("update_rule", "")) != "ema":
        errors.append("plan §10 Algorithm 1 step 6 requires target_encoder.update_rule='ema'")
    if str(loss.get("implementation", "")) not in {
        "negative_mean_cosine_similarity",
        "one_minus_mean_cosine_similarity",
        "one_minus_cosine_similarity",
    }:
        errors.append("plan §10 Algorithm 1 step 4 requires cosine JEPA loss implementation")

    if proc:
        steps = tuple(proc.get("steps", ()))
        if steps and steps != PLAN_JEPA_PROCEDURE["steps"]:
            errors.append(
                f"ts_jepa.training_procedure.steps must match plan §10 Algorithm 1 order; got {list(steps)}"
            )
        if proc.get("ema_after_optimizer_step") is False:
            errors.append("plan §10 requires ema_after_optimizer_step=true")
        if proc.get("target_stop_gradient") is False:
            errors.append("plan §10 requires training_procedure.target_stop_gradient=true")

    if errors:
        raise ValueError("Plan §10 Algorithm 1 training procedure mismatch:\n  - " + "\n  - ".join(errors))

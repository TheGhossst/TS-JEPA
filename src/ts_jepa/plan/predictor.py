"""Plan §9 predictor specification and config verification."""

from __future__ import annotations

from typing import Any

from ts_jepa.plan.enforce import plan_enforced

PLAN_PREDICTOR: dict[str, Any] = {
    "type": "MLP",
    "hidden_dim": 1024,
    "output_dim": 256,
    "autoregressive": True,
    "forbidden_inputs": [
        "virtual_inputs",
        "virtual_channel_variables",
        "virtual_channel_embeddings",
        "unspecified_future_channel_features",
    ],
}

# Hidden-layer nonlinearity is NOT SPECIFIED in plan §9 (actor ReLU is §12).
IC_PREDICTOR_ACTIVATION = "ReLU"
# Hidden BatchNorm1d is NOT SPECIFIED. Baseline uses it to condition SGD 0.2 (IC).
IC_PREDICTOR_HIDDEN_BATCH_NORM = True


def assert_plan_predictor_config(config: dict[str, Any]) -> None:
    """Raise ValueError when predictor settings deviate from plan §9."""
    if not plan_enforced(config):
        return
    pred = config.get("ts_jepa", {}).get("predictor", {})
    errors: list[str] = []

    if str(pred.get("type")) != PLAN_PREDICTOR["type"]:
        errors.append(f"ts_jepa.predictor.type: expected MLP, got {pred.get('type')}")

    hidden = pred.get("hidden_dim")
    if hidden is None or int(hidden) != PLAN_PREDICTOR["hidden_dim"]:
        errors.append(f"ts_jepa.predictor.hidden_dim: expected 1024, got {hidden}")

    out_dim = pred.get("output_dim")
    allow_grid = bool(config.get("experiments", {}).get("allow_non_baseline_embedding_dim", False))
    if out_dim is None or (not allow_grid and int(out_dim) != PLAN_PREDICTOR["output_dim"]):
        errors.append(f"ts_jepa.predictor.output_dim: expected 256, got {out_dim}")

    if pred.get("autoregressive") is False:
        errors.append("ts_jepa.predictor.autoregressive must be true (plan §9)")

    emb = int(config.get("ts_jepa", {}).get("encoder", {}).get("embedding_dim", -1))
    if out_dim is not None and int(out_dim) != emb:
        errors.append(f"predictor.output_dim ({out_dim}) must match encoder.embedding_dim ({emb})")

    inputs = [str(x) for x in pred.get("inputs", [])]
    for forbidden in PLAN_PREDICTOR["forbidden_inputs"]:
        if forbidden in inputs:
            errors.append(f"predictor.inputs must not include {forbidden!r} (plan §9)")

    command_source = pred.get("command_source")
    if command_source is None:
        errors.append(
            "ts_jepa.predictor.command_source must be set and labeled an implementation choice "
            "(plan §9: pretraining ũ is NOT SPECIFIED)"
        )

    if errors:
        raise ValueError("Plan §9 predictor config mismatch:\n  - " + "\n  - ".join(errors))

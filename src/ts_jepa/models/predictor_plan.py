"""Plan §9 predictor specification and config verification."""

from __future__ import annotations

from typing import Any

PLAN_PREDICTOR: dict[str, Any] = {
    "type": "MLP",
    "hidden_dim": 1024,
    "output_dim": 256,
    "activation": "ReLU",
    "input_construction": "concat_embedding_and_command",
    "forbidden_inputs": [
        "virtual_inputs",
        "virtual_channel_variables",
        "virtual_channel_embeddings",
        "unspecified_future_channel_features",
    ],
    # Plan §10 OPEN — must be recorded, not claimed paper-exact.
    "default_command_source": "teacher_dp",
}


def assert_plan_predictor_config(config: dict[str, Any]) -> None:
    """Raise ValueError when predictor settings deviate from plan §9."""
    pred = config.get("ts_jepa", {}).get("predictor", {})
    errors: list[str] = []

    if str(pred.get("type")) != PLAN_PREDICTOR["type"]:
        errors.append(f"ts_jepa.predictor.type: expected MLP, got {pred.get('type')}")

    hidden = pred.get("hidden_dim")
    if hidden is None or int(hidden) != PLAN_PREDICTOR["hidden_dim"]:
        errors.append(f"ts_jepa.predictor.hidden_dim: expected 1024, got {hidden}")

    out_dim = pred.get("output_dim")
    if out_dim is None or int(out_dim) != PLAN_PREDICTOR["output_dim"]:
        errors.append(f"ts_jepa.predictor.output_dim: expected 256, got {out_dim}")

    emb = int(config.get("ts_jepa", {}).get("encoder", {}).get("embedding_dim", -1))
    if out_dim is not None and int(out_dim) != emb:
        errors.append(f"predictor.output_dim ({out_dim}) must match encoder.embedding_dim ({emb})")

    construction = pred.get("input_tensor_construction")
    if construction != PLAN_PREDICTOR["input_construction"]:
        errors.append(
            f"ts_jepa.predictor.input_tensor_construction: expected "
            f"{PLAN_PREDICTOR['input_construction']!r}, got {construction!r}"
        )

    command_source = pred.get("command_source")
    if command_source is None:
        errors.append("ts_jepa.predictor.command_source must be set (plan §10 records the choice)")

    if errors:
        raise ValueError("Plan §9 predictor config mismatch:\n  - " + "\n  - ".join(errors))

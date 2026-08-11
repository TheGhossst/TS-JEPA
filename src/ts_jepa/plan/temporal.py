"""Plan §8 temporal configuration and sample indexing."""

from __future__ import annotations

from typing import Any

PLAN_TEMPORAL: dict[str, Any] = {
    "kappa": 2,
    "Kp": 15,
    "embedding_dim": 256,
}


def assert_plan_temporal_config(config: dict[str, Any]) -> None:
    """Raise ValueError when temporal settings deviate from plan §8."""
    errors: list[str] = []
    kappa = int(config.get("input", {}).get("kappa", -1))
    kp = int(config.get("ts_jepa", {}).get("prediction_horizon", {}).get("Kp", -1))
    emb = int(config.get("ts_jepa", {}).get("encoder", {}).get("embedding_dim", -1))

    if kappa != PLAN_TEMPORAL["kappa"]:
        errors.append(f"input.kappa: expected {PLAN_TEMPORAL['kappa']}, got {kappa}")
    if kp != PLAN_TEMPORAL["Kp"]:
        errors.append(f"ts_jepa.prediction_horizon.Kp: expected {PLAN_TEMPORAL['Kp']}, got {kp}")
    if emb != PLAN_TEMPORAL["embedding_dim"]:
        errors.append(f"ts_jepa.encoder.embedding_dim: expected {PLAN_TEMPORAL['embedding_dim']}, got {emb}")

    if errors:
        raise ValueError("Plan §8 temporal config mismatch:\n  - " + "\n  - ".join(errors))

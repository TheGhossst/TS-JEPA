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


def context_frame_indices(time_index: int, kappa: int) -> list[int]:
    """
    Plan §8 context at step k with κ consecutive frames ending at k:
      [x_i,k-1, x_i,k] for κ=2.
    """
    k = int(time_index)
    kappa = int(kappa)
    start = max(0, k - kappa + 1)
    indices = list(range(start, k + 1))
    while len(indices) < kappa:
        indices.insert(0, indices[0])
    return indices


def target_end_indices(time_index: int, kp: int) -> list[int]:
    """
    Plan §8 target times k+1 .. k+Kp.

    Each target is encoded from a κ-frame window ending at the listed index
    (IMPLEMENTATION CHOICE — consistent κ packing per docs/IMPLEMENTATION_CHOICES.md).
    """
    k = int(time_index)
    kp = int(kp)
    return [k + offset for offset in range(1, kp + 1)]


def command_indices(time_index: int, kp: int) -> list[int]:
    """Plan §8 control sequence [u_i,k, ..., u_i,k+Kp-1]."""
    k = int(time_index)
    kp = int(kp)
    return list(range(k, k + kp))


def max_valid_time_index(trajectory_length: int, kp: int) -> int:
    """Largest k such that targets through k+Kp and commands through k+Kp-1 fit."""
    return int(trajectory_length) - int(kp) - 1


def describe_temporal_sample(time_index: int, kappa: int, kp: int) -> dict[str, Any]:
    """Human/machine-readable summary of one plan §8 training sample at time k."""
    targets = target_end_indices(time_index, kp)
    return {
        "time_index_k": int(time_index),
        "kappa": int(kappa),
        "Kp": int(kp),
        "context_frame_indices": context_frame_indices(time_index, kappa),
        "command_indices": command_indices(time_index, kp),
        "target_end_indices": targets,
        "predicted_latent_indices": targets,
    }

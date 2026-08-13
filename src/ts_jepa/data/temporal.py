"""Temporal sample indexing (plan §6)."""

from __future__ import annotations

from typing import Any

def context_frame_indices(time_index: int, kappa: int) -> list[int]:
    """
    Plan §6 context at step k with κ consecutive frames ending at k:
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
    Plan §6 target times k+1 .. k+Kp.

    Each target is the single RGB frame at the listed index (Algorithm 1).
    """
    k = int(time_index)
    kp = int(kp)
    return [k + offset for offset in range(1, kp + 1)]


def command_indices(time_index: int, kp: int) -> list[int]:
    """Plan §6 predictor conditioning [u_i,k, ..., u_i,k+Kp-1]."""
    k = int(time_index)
    kp = int(kp)
    return list(range(k, k + kp))


def predicted_command_indices(time_index: int, kp: int) -> list[int]:
    """
    Commands aligned with predicted latents z̃_{k+1} .. z̃_{k+Kp}:

      u_{k+1} .. u_{k+Kp}

    Used for horizon NMAE: actor(z̃_{k+j}) vs u_{k+j}. Distinct from predictor
    conditioning command_indices (u_k .. u_{k+Kp-1}).
    """
    k = int(time_index)
    kp = int(kp)
    return list(range(k + 1, k + kp + 1))


def max_valid_time_index(trajectory_length: int, kp: int) -> int:
    """Largest k such that frames/commands through k+Kp fit in [0, length)."""
    return int(trajectory_length) - int(kp) - 1


def describe_temporal_sample(time_index: int, kappa: int, kp: int) -> dict[str, Any]:
    """Human/machine-readable summary of one plan §6 training sample at time k."""
    targets = target_end_indices(time_index, kp)
    return {
        "time_index_k": int(time_index),
        "kappa": int(kappa),
        "Kp": int(kp),
        "context_frame_indices": context_frame_indices(time_index, kappa),
        "command_indices": command_indices(time_index, kp),
        "predicted_command_indices": predicted_command_indices(time_index, kp),
        "target_end_indices": targets,
        "predicted_latent_indices": targets,
    }

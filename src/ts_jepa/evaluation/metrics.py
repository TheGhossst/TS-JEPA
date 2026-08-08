from __future__ import annotations

from typing import Any

import numpy as np


def control_score(
    state: np.ndarray,
    desired_x: float = 0.0,
    position_tol: float = 0.05,
    angle_tol: float = 0.05,
) -> int:
    """Binary control score R_i,k from paper evaluation."""
    x_ok = abs(float(state[0]) - desired_x) <= position_tol
    theta_ok = abs(float(state[2])) <= angle_tol
    return int(x_ok and theta_ok)


def nmae(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Normalized mean absolute error between predicted and target commands."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    denom = np.mean(np.abs(target)) + eps
    return float(np.mean(np.abs(pred - target)) / denom)


def communication_bits_rgb(height: int, width: int, channels: int = 3, bits_per_channel: int = 8) -> int:
    return int(height * width * channels * bits_per_channel)


def communication_bits_embedding(embedding_dim: int = 256, bits_per_value: int = 32) -> int:
    return int(embedding_dim * bits_per_value)


def summarize_scores(scores: list[float]) -> dict[str, Any]:
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "best": float(arr.max()) if len(arr) else 0.0,
        "worst": float(arr.min()) if len(arr) else 0.0,
        "repetitions": len(arr),
    }

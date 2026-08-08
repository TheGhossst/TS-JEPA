from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CommandNormalizer:
    """Z-score command normalization (PAPER-SPECIFIED). μ/σ storage is IC."""

    mean: float
    std: float

    @classmethod
    def from_array(cls, commands: np.ndarray, eps: float = 1e-8) -> "CommandNormalizer":
        mean = float(np.mean(commands))
        std = float(np.std(commands))
        if std < eps:
            std = 1.0
        return cls(mean=mean, std=std)

    def normalize(self, commands: np.ndarray) -> np.ndarray:
        return ((commands.astype(np.float32) - self.mean) / self.std).astype(np.float32)

    def denormalize(self, commands_norm: np.ndarray) -> np.ndarray:
        return (commands_norm.astype(np.float32) * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "CommandNormalizer":
        return cls(mean=float(payload["mean"]), std=float(payload["std"]))

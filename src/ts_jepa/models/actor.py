"""Plan §14 semantic actor Cε: embedding → scalar control command."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ts_jepa.plan.actor import PLAN_SEMANTIC_ACTOR


class SemanticActor(nn.Module):
    """
    Semantic actor Cε (plan §14): 256-D embedding → scalar cart control.

    Architecture:
        256 → Linear(1024) → ReLU → Linear(256) → ReLU → Linear(1)

    Plan §15 adds Dropout(0.2) after each hidden ReLU during training.
    Network output is in the normalized command domain; physical clip is applied
    after denormalization at runtime (IMPLEMENTATION CHOICE).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 256),
        dropout: float = 0.2,
        command_dim: int = 1,
    ) -> None:
        super().__init__()
        hidden = tuple(int(h) for h in hidden_dims)
        if len(hidden) != 2:
            raise ValueError(
                f"plan §14 requires exactly two hidden layers (1024, 256); got {hidden}"
            )
        self.embedding_dim = int(embedding_dim)
        self.hidden_dims = hidden
        self.command_dim = int(command_dim)
        self.dropout_p = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(hidden[1], self.command_dim),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SemanticActor":
        """Build Cε from config using plan §14 / §15 fields."""
        arch = config["semantic_actor"]["architecture"]
        emb = int(config["ts_jepa"]["encoder"]["embedding_dim"])
        return cls(
            embedding_dim=emb,
            hidden_dims=tuple(arch["hidden_dims"]),
            dropout=float(arch.get("dropout", 0.2)),
            command_dim=int(PLAN_SEMANTIC_ACTOR["output_dim"]),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Map embedding z → predicted command ũ (normalized domain)."""
        return self.net(embedding)

    def assert_plan_architecture(self) -> None:
        """Raise if this instance does not match plan §14 layer sizes."""
        expected_hidden = tuple(PLAN_SEMANTIC_ACTOR["hidden_dims"])
        if self.embedding_dim != PLAN_SEMANTIC_ACTOR["input_dim"]:
            raise ValueError(
                f"plan §14 input dim: expected {PLAN_SEMANTIC_ACTOR['input_dim']}, "
                f"got {self.embedding_dim}"
            )
        if self.hidden_dims != expected_hidden:
            raise ValueError(
                f"plan §14 hidden dims: expected {expected_hidden}, got {self.hidden_dims}"
            )
        if self.command_dim != PLAN_SEMANTIC_ACTOR["output_dim"]:
            raise ValueError(
                f"plan §14 output dim: expected {PLAN_SEMANTIC_ACTOR['output_dim']}, "
                f"got {self.command_dim}"
            )
        # Linear layer order: 0, 3, 6 in Sequential (Linear, ReLU, Dropout)*2 + Linear
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        if len(linears) != 3:
            raise ValueError(f"plan §14 requires 3 Linear layers; got {len(linears)}")
        shapes = [(m.in_features, m.out_features) for m in linears]
        expected_shapes = [
            (256, 1024),
            (1024, 256),
            (256, 1),
        ]
        if shapes != expected_shapes:
            raise ValueError(f"plan §14 Linear shapes: expected {expected_shapes}, got {shapes}")

"""Plan §12 semantic actor Cε: embedding → scalar control command."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ts_jepa.plan.actor import PLAN_SEMANTIC_ACTOR


class SemanticActor(nn.Module):
    """
    Semantic actor Cε (plan §12): 256-D embedding → scalar cart-pole force.

    Architecture:
        256 → Linear(1024) → ReLU → Dropout → Linear(256) → ReLU → Dropout → Linear(1)

    Dropout 0.2 is Table III. Placement after each hidden ReLU is IMPLEMENTATION CHOICE.
    Last layer is linear (plan §18 IC). Loss domain is physical Newtons (Eq. 15).
    Desired state x_d is not an input.
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
                f"plan §12 requires exactly two hidden layers (1024, 256); got {hidden}"
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
        """Build Cε from config using plan §12 fields."""
        arch = config["semantic_actor"]["architecture"]
        emb = int(config["ts_jepa"]["encoder"]["embedding_dim"])
        return cls(
            embedding_dim=emb,
            hidden_dims=tuple(arch["hidden_dims"]),
            dropout=float(arch.get("dropout", 0.2)),
            command_dim=int(PLAN_SEMANTIC_ACTOR["output_dim"]),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Map embedding z → predicted command ũ in Newtons (Eq. 15). Embedding only."""
        return self.net(embedding)

    def assert_plan_architecture(self) -> None:
        """Raise if hidden sizes / ReLU count mismatch plan §12, or the linear-head IC (plan §18)."""
        expected_hidden = tuple(PLAN_SEMANTIC_ACTOR["hidden_dims"])
        if self.embedding_dim != PLAN_SEMANTIC_ACTOR["input_dim"]:
            # Fig. 7 grid: input dim tracks encoder embedding_dim (plan §17).
            if self.embedding_dim < 1:
                raise ValueError(f"actor embedding_dim must be positive, got {self.embedding_dim}")
        if self.hidden_dims != expected_hidden:
            raise ValueError(
                f"plan §12 hidden dims: expected {expected_hidden}, got {self.hidden_dims}"
            )
        if self.command_dim != PLAN_SEMANTIC_ACTOR["output_dim"]:
            raise ValueError(
                f"plan §12 output dim: expected {PLAN_SEMANTIC_ACTOR['output_dim']}, "
                f"got {self.command_dim}"
            )
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        if len(linears) != 3:
            raise ValueError(f"plan §12 requires 3 Linear layers; got {len(linears)}")
        shapes = [(m.in_features, m.out_features) for m in linears]
        expected_shapes = [
            (self.embedding_dim, 1024),
            (1024, 256),
            (256, 1),
        ]
        if shapes != expected_shapes:
            raise ValueError(f"plan §12 Linear shapes: expected {expected_shapes}, got {shapes}")
        relus = [m for m in self.net if isinstance(m, nn.ReLU)]
        if len(relus) != 2:
            raise ValueError(f"plan §12 requires ReLU after each hidden layer; got {len(relus)} ReLU")
        forbidden = (nn.Tanh, nn.Sigmoid, nn.Hardtanh, nn.ReLU6)
        if any(isinstance(m, forbidden) for m in self.net):
            raise ValueError(
                "baseline IC (plan §18): Cε uses a linear head; tanh/sigmoid/clip "
                "are not paper-specified and are not used here"
            )
        if not isinstance(self.net[-1], nn.Linear):
            raise ValueError(
                "baseline IC (plan §18): Cε last layer is Linear (paper: scalar force; "
                "output nonlinearity is NOT SPECIFIED)"
            )

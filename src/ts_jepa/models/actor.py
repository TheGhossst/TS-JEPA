from __future__ import annotations

import torch
import torch.nn as nn


class SemanticActor(nn.Module):
    """
    Semantic actor Cε: embedding → control command (normalized domain during training).

    Paper-specified: Linear(1024)-ReLU, Linear(256)-ReLU, then control output.
    Dropout 0.2 from training hyperparams. Output activation is IMPLEMENTATION CHOICE
    (linear here; physical clip applied after denormalization at runtime).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dims: tuple[int, int] = (1024, 256),
        dropout: float = 0.2,
        command_dim: int = 1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], command_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)

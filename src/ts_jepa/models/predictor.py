from __future__ import annotations

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    Embedding-only MLP predictor Pφ.

    Paper-specified: hidden 1024, output 256.
    Autoregressive over the latent sequence without teacher-command conditioning.
    """

    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward_step(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)

    def forward(
        self,
        embedding: torch.Tensor,
        horizon: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive prediction over Kp steps.

        returns: [B, Kp, D]
        """
        kp = int(horizon) if horizon is not None else 1
        z = embedding
        preds = []
        for _ in range(kp):
            z = self.forward_step(z)
            preds.append(z)
        return torch.stack(preds, dim=1)

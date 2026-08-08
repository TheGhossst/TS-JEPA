from __future__ import annotations

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    Action-conditioned MLP predictor Pφ.

    Paper-specified: hidden 1024, output 256.
    Input concat(z, u_norm) is IMPLEMENTATION CHOICE.
    """

    def __init__(self, embedding_dim: int = 256, command_dim: int = 1, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.command_dim = command_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + command_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward_step(self, embedding: torch.Tensor, command_norm: torch.Tensor) -> torch.Tensor:
        if command_norm.ndim == 1:
            command_norm = command_norm.unsqueeze(-1)
        if command_norm.ndim == 2 and command_norm.shape[-1] != self.command_dim:
            command_norm = command_norm.view(command_norm.shape[0], self.command_dim)
        x = torch.cat([embedding, command_norm], dim=-1)
        return self.net(x)

    def forward(
        self,
        embedding: torch.Tensor,
        commands_norm: torch.Tensor,
        horizon: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive prediction over Kp steps.

        commands_norm: [B, Kp] or [B, Kp, 1]
        returns: [B, Kp, D]
        """
        if commands_norm.ndim == 2:
            commands_norm = commands_norm.unsqueeze(-1)
        kp = commands_norm.shape[1] if horizon is None else int(horizon)
        z = embedding
        preds = []
        for t in range(kp):
            z = self.forward_step(z, commands_norm[:, t])
            preds.append(z)
        return torch.stack(preds, dim=1)

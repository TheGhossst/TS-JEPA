from __future__ import annotations

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    Action-conditioned MLP predictor Pφ (plan §9).

    Paper-specified: hidden 1024, output 256, autoregressive, command-conditioned.

    Hidden ReLU, concat fusion, and input width 257 are IMPLEMENTATION CHOICES (plan §9 / §18).
    Virtual-channel inputs are forbidden.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        command_dim: int = 1,
        hidden_dim: int = 1024,
        output_dim: int | None = None,
        *,
        strict_baseline_dim: bool = True,
    ) -> None:
        super().__init__()
        if int(hidden_dim) != 1024:
            raise ValueError(f"plan §9 predictor hidden_dim must be 1024, got {hidden_dim}")
        out_dim = int(output_dim if output_dim is not None else embedding_dim)
        if strict_baseline_dim:
            if out_dim != 256:
                raise ValueError(f"plan §9 predictor output_dim must be 256, got {out_dim}")
            if int(embedding_dim) != 256:
                raise ValueError(f"plan §9 predictor embedding_dim must be 256, got {embedding_dim}")
        elif out_dim != int(embedding_dim):
            raise ValueError(f"predictor output_dim ({out_dim}) must match embedding_dim ({embedding_dim})")
        if int(command_dim) != 1:
            raise ValueError(f"plan §9 cart-pole control is scalar; command_dim must be 1, got {command_dim}")

        self.embedding_dim = int(embedding_dim)
        self.command_dim = int(command_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = out_dim
        self.input_dim = self.embedding_dim + self.command_dim

        # IC fusion: concat(z, u) → Linear(257, 1024); concat width is not paper-specified.
        self.fc_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"predictor input dim must be {self.input_dim}, got {x.shape[-1]}")
        return self.fc_out(self.relu(self.fc_in(x)))

    def forward_step(self, embedding: torch.Tensor, command_norm: torch.Tensor) -> torch.Tensor:
        """
        One autoregressive predictor step (plan §9).

        Inputs: z_current [B, D] and u_j [B, 1] (normalized scalar control).
        """
        if command_norm.ndim == 1:
            command_norm = command_norm.unsqueeze(-1)
        if command_norm.ndim == 2 and command_norm.shape[-1] != self.command_dim:
            command_norm = command_norm.view(command_norm.shape[0], self.command_dim)
        x = torch.cat([embedding, command_norm], dim=-1)
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"predictor input dim must be {self.input_dim}, got {x.shape[-1]}")
        return self.forward_mlp(x)

    def forward(
        self,
        embedding: torch.Tensor,
        commands_norm: torch.Tensor,
        horizon: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive prediction over Kp steps (plan §9).

        commands_norm: [B, Kp] or [B, Kp, 1]
        returns: [B, Kp, D] with z̃_{k+1}, ..., z̃_{k+Kp}
        """
        if commands_norm.ndim == 2:
            commands_norm = commands_norm.unsqueeze(-1)
        kp = commands_norm.shape[1] if horizon is None else int(horizon)
        z_current = embedding
        preds: list[torch.Tensor] = []
        for j in range(kp):
            z_next = self.forward_step(z_current, commands_norm[:, j])
            preds.append(z_next)
            z_current = z_next
        return torch.stack(preds, dim=1)

    def architecture_summary(self) -> dict[str, int | str]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "stack": "Linear-1024-ReLU-Linear-256",
            "autoregressive": True,
            "inputs": "concat(embedding, command)",
        }

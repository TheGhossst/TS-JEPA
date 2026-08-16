from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    """
    Action-conditioned MLP predictor Pφ (plan §9).

    Paper-specified: hidden 1024, output 256, autoregressive, command-conditioned.

    Hidden ReLU, concat fusion, input width 257, and hidden BatchNorm1d are
    IMPLEMENTATION CHOICES (plan §9 / §18). Virtual-channel inputs are forbidden.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        command_dim: int = 1,
        hidden_dim: int = 1024,
        output_dim: int | None = None,
        *,
        hidden_batch_norm: bool = True,
        strict_baseline_dim: bool = True,
        l2_normalize_output: bool = True,
        command_scale: float = 1.0,
        conditioning: str = "concat",
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
        self.hidden_batch_norm = bool(hidden_batch_norm)
        self.l2_normalize_output = bool(l2_normalize_output)
        self.command_scale = float(command_scale)
        if self.command_scale <= 0.0:
            raise ValueError(f"predictor command_scale must be > 0, got {self.command_scale}")
        cond = str(conditioning).strip().lower()
        if cond not in {"concat", "film"}:
            raise ValueError(f"predictor conditioning must be 'concat' or 'film', got {conditioning!r}")
        self.conditioning = cond

        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)
        if self.conditioning == "film":
            # BN on z-features only, then FiLM(u). Concat-then-BN drowns scalar u.
            self.fc_in = nn.Linear(self.embedding_dim, self.hidden_dim)
            self.bn = nn.BatchNorm1d(self.hidden_dim) if self.hidden_batch_norm else nn.Identity()
            self.film = nn.Linear(self.command_dim, 2 * self.hidden_dim)
            nn.init.normal_(self.film.weight, std=0.02)
            nn.init.zeros_(self.film.bias)
            self.input_dim = self.embedding_dim
        else:
            # IC fusion: concat(z, u) → Linear(257, 1024). Paper overlay stays here.
            self.fc_in = nn.Linear(self.input_dim, self.hidden_dim)
            self.bn = nn.BatchNorm1d(self.hidden_dim) if self.hidden_batch_norm else nn.Identity()
            self.film = None

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "film":
            raise ValueError("forward_mlp is concat-only; use forward_step for FiLM")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"predictor input dim must be {self.input_dim}, got {x.shape[-1]}")
        return self.fc_out(self.relu(self.bn(self.fc_in(x))))

    def _film_mlp(self, embedding: torch.Tensor, command_norm: torch.Tensor) -> torch.Tensor:
        assert self.film is not None
        hidden = self.bn(self.fc_in(embedding))
        gamma_beta = self.film(command_norm)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        hidden = (1.0 + gamma) * hidden + beta
        return self.fc_out(self.relu(hidden))

    def forward_step(self, embedding: torch.Tensor, command_norm: torch.Tensor) -> torch.Tensor:
        """
        One autoregressive predictor step (plan §9).

        Inputs: z_current [B, D] and u_j [B, 1] (normalized scalar control).
        """
        if command_norm.ndim == 1:
            command_norm = command_norm.unsqueeze(-1)
        if command_norm.ndim == 2 and command_norm.shape[-1] != self.command_dim:
            command_norm = command_norm.view(command_norm.shape[0], self.command_dim)
        if self.conditioning == "film":
            out = self._film_mlp(embedding, command_norm)
        else:
            if self.command_scale != 1.0:
                command_norm = command_norm * self.command_scale
            x = torch.cat([embedding, command_norm], dim=-1)
            if x.shape[-1] != self.input_dim:
                raise ValueError(f"predictor input dim must be {self.input_dim}, got {x.shape[-1]}")
            out = self.forward_mlp(x)
        if self.l2_normalize_output:
            return F.normalize(out, dim=-1, eps=1e-8)
        return out

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
            # IC: Eq. (12) is still unrolled. Eq. (13) is one-step cosine; full
            # 15-step BPTT through Pφ is not specified and oscillates under Table II
            # SGD 0.2. Detach so Ψθ is trained on ẑ_{k+1} and Pφ on every horizon
            # step without backprop-through-time.
            z_current = z_next.detach()
        return torch.stack(preds, dim=1)

    def architecture_summary(self) -> dict[str, int | str | bool]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "stack": (
                "Linear-1024-BN-ReLU-Linear-256-L2Norm"
                if self.hidden_batch_norm
                else "Linear-1024-ReLU-Linear-256-L2Norm"
            ),
            "hidden_batch_norm": self.hidden_batch_norm,
            "l2_normalize_output": self.l2_normalize_output,
            "command_scale": self.command_scale,
            "conditioning": self.conditioning,
            "autoregressive": True,
            "detach_autoregressive_state": True,
            "inputs": "film(embedding, command)" if self.conditioning == "film" else "concat(embedding, command)",
        }

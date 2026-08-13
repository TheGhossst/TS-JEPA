"""§16 supervised RGB→command and generative autoencoder (architectures are IC)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ts_jepa.models.encoder import ContextEncoder


class SupervisedRGBToCommand(nn.Module):
    """
    Plan §16 baseline 2: high-dimensional state (κ RGB frames) → command.

    IMPLEMENTATION CHOICE: ResNet widths 64/128/256 (same as TS-JEPA encoder, paper-silent
    for this baseline) plus MLP 1024→256→1. Not Table III (that table is the semantic actor).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        embedding_dim: int = 256,
        hidden_dims: tuple[int, ...] = (1024, 256),
        dropout: float = 0.2,
        blocks_per_stage: int = 2,
        spatial_pool_hw: tuple[int, int] = (4, 8),
    ) -> None:
        super().__init__()
        self.encoder = ContextEncoder(
            in_channels=in_channels,
            embedding_dim=int(embedding_dim),
            blocks_per_stage=blocks_per_stage,
            spatial_pool_hw=spatial_pool_hw,
            strict_baseline_dim=False,
        )
        h0, h1 = hidden_dims
        self.head = nn.Sequential(
            nn.Linear(int(embedding_dim), h0),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(h0, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(h1, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(context))

    @classmethod
    def from_config(cls, config: dict[str, Any], *, kappa: int) -> "SupervisedRGBToCommand":
        ch = int(config["input"]["channels_per_rgb_frame"]) * int(kappa)
        enc = config["ts_jepa"]["encoder"]
        ic = config.get("experiments", {}).get("supervised_architecture", {})
        return cls(
            in_channels=ch,
            embedding_dim=int(ic.get("embedding_dim", enc.get("embedding_dim", 256))),
            hidden_dims=tuple(ic.get("hidden_dims", [1024, 256])),
            dropout=float(ic.get("dropout", 0.2)),
            blocks_per_stage=int(enc.get("blocks_per_stage", 2)),
            spatial_pool_hw=tuple(enc.get("spatial_pool_hw", [4, 8])),
        )


class GenerativeAutoencoder(nn.Module):
    """
    Plan §16 baseline 3: encode RGB → bottleneck, reconstruct high-dimensional state,
    then nonlinear DP on a decoded 4D plant state.

    IMPLEMENTATION CHOICE: encoder copies TS-JEPA ResNet; RGB decoder is a linear reshape;
    a 4D state head exists because DP cannot run on RGB (paper silent on AE tensors).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        height: int,
        width: int,
        embedding_dim: int = 256,
        blocks_per_stage: int = 2,
        spatial_pool_hw: tuple[int, int] = (4, 8),
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.height = int(height)
        self.width = int(width)
        self.embedding_dim = int(embedding_dim)
        self.encoder = ContextEncoder(
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            blocks_per_stage=blocks_per_stage,
            spatial_pool_hw=spatial_pool_hw,
            strict_baseline_dim=False,
        )
        recon_dim = self.in_channels * self.height * self.width
        self.rgb_decoder = nn.Linear(self.embedding_dim, recon_dim)
        self.state_head = nn.Linear(self.embedding_dim, 4)

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        return self.encoder(context)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(context)
        recon = self.rgb_decoder(z).view(-1, self.in_channels, self.height, self.width)
        state_hat = self.state_head(z)
        return z, recon, state_hat

    @classmethod
    def from_config(cls, config: dict[str, Any], *, kappa: int) -> "GenerativeAutoencoder":
        h, w = config["input"]["resize"]
        ch = int(config["input"]["channels_per_rgb_frame"]) * int(kappa)
        enc = config["ts_jepa"]["encoder"]
        ic = config.get("experiments", {}).get("autoencoder_architecture", {})
        return cls(
            in_channels=ch,
            height=int(h),
            width=int(w),
            embedding_dim=int(ic.get("embedding_dim", enc.get("embedding_dim", 256))),
            blocks_per_stage=int(enc.get("blocks_per_stage", 2)),
            spatial_pool_hw=tuple(enc.get("spatial_pool_hw", [4, 8])),
        )

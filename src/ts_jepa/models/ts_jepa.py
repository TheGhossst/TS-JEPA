from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ts_jepa.models.ema import clone_encoder, ema_update
from ts_jepa.models.encoder import ContextEncoder
from ts_jepa.models.predictor import Predictor


class TSJEPA(nn.Module):
    """Context encoder + target encoder + predictor."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        inp = config["input"]
        enc_cfg = config["ts_jepa"]["encoder"]
        pred_cfg = config["ts_jepa"]["predictor"]
        in_channels = int(inp["channels_per_rgb_frame"]) * int(inp["kappa"])
        embedding_dim = int(enc_cfg["embedding_dim"])
        self.context_encoder = ContextEncoder(
            in_channels=in_channels,
            widths=list(enc_cfg["widths"]),
            embedding_dim=embedding_dim,
            blocks_per_stage=int(enc_cfg.get("blocks_per_stage", 2)),
        )
        self.target_encoder = clone_encoder(self.context_encoder)
        self.predictor = Predictor(
            embedding_dim=embedding_dim,
            command_dim=1,
            hidden_dim=int(pred_cfg["hidden_dim"]),
        )
        self.ema_decay = float(config["ts_jepa"]["target_encoder"]["ema_decay"])
        self.kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(context)

    @torch.no_grad()
    def encode_targets(self, future_frames: torch.Tensor) -> torch.Tensor:
        """future_frames: [B, Kp, C_kappa, H, W] → [B, Kp, D]."""
        b, kp, c, h, w = future_frames.shape
        flat = future_frames.reshape(b * kp, c, h, w)
        emb = self.target_encoder(flat)
        return emb.view(b, kp, -1)

    def predict(self, embedding: torch.Tensor, commands_norm: torch.Tensor) -> torch.Tensor:
        return self.predictor(embedding, commands_norm, horizon=self.kp)

    def ema_step(self) -> None:
        ema_update(self.target_encoder, self.context_encoder, decay=self.ema_decay)

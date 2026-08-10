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
        resize_hw = tuple(inp["resize"])
        input_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.context_encoder = ContextEncoder(
            in_channels=in_channels,
            widths=list(enc_cfg["widths"]),
            embedding_dim=embedding_dim,
            blocks_per_stage=int(enc_cfg.get("blocks_per_stage", 2)),
            input_hw=input_hw,
        )
        self.target_encoder = clone_encoder(self.context_encoder)
        self.predictor = Predictor(
            embedding_dim=embedding_dim,
            hidden_dim=int(pred_cfg["hidden_dim"]),
        )
        self.ema_decay = float(config["ts_jepa"]["target_encoder"]["ema_decay"])
        self.kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(context)

    @torch.no_grad()
    def encode_targets(self, future_frames: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
        """
        future_frames: [B, Kp, C_kappa, H, W] → [B, Kp, D].

        Chunked forward is an IMPLEMENTATION CHOICE for GPU memory. Target encoder
        runs in eval/no-grad, so BatchNorm uses running stats and chunking does not
        change outputs vs a single mega-batch.
        """
        b, kp, c, h, w = future_frames.shape
        flat = future_frames.reshape(b * kp, c, h, w)
        chunks = []
        for start in range(0, flat.shape[0], int(chunk_size)):
            chunks.append(self.target_encoder(flat[start : start + int(chunk_size)]))
        emb = torch.cat(chunks, dim=0)
        return emb.view(b, kp, -1)

    def predict(self, embedding: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
        return self.predictor(embedding, horizon=horizon if horizon is not None else self.kp)

    def ema_step(self) -> None:
        ema_update(self.target_encoder, self.context_encoder, decay=self.ema_decay)

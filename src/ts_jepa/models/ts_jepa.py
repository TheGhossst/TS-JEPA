from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ts_jepa.models.ema import (
    assert_target_initialized_from_context,
    clone_encoder,
    ema_update,
    initialize_target_from_context,
)
from ts_jepa.models.encoder import ContextEncoder
from ts_jepa.models.encoder_plan import assert_plan_encoder_config
from ts_jepa.models.predictor_plan import assert_plan_predictor_config
from ts_jepa.models.predictor_command_resolution import (
    PredictorCommandResolution,
    assert_plan_predictor_command_resolution,
    load_predictor_command_resolution,
)
from ts_jepa.models.predictor import Predictor


class TSJEPA(nn.Module):
    """Context encoder + target encoder + predictor."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        assert_plan_encoder_config(config)
        assert_plan_predictor_config(config)
        assert_plan_predictor_command_resolution(config)
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
        self.target_encoder = initialize_target_from_context(self.context_encoder)
        assert_target_initialized_from_context(self.context_encoder, self.target_encoder)
        self.predictor = Predictor(
            embedding_dim=embedding_dim,
            command_dim=1,
            hidden_dim=int(pred_cfg["hidden_dim"]),
            output_dim=int(pred_cfg["output_dim"]),
        )
        self.command_source = str(pred_cfg.get("command_source", "teacher_dp"))
        self.command_resolution: PredictorCommandResolution = load_predictor_command_resolution(config)
        self.ema_decay = float(config["ts_jepa"]["target_encoder"]["ema_decay"])
        self.kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(context)

    @torch.no_grad()
    def encode_targets(self, future_frames: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
        """
        Plan §7: target encoder forward with stop-gradient.

        future_frames: [B, Kp, C_kappa, H, W] → [B, Kp, D].
        """
        b, kp, c, h, w = future_frames.shape
        flat = future_frames.reshape(b * kp, c, h, w)
        chunks = []
        for start in range(0, flat.shape[0], int(chunk_size)):
            chunks.append(self.target_encoder(flat[start : start + int(chunk_size)]))
        emb = torch.cat(chunks, dim=0)
        return emb.view(b, kp, -1)

    def predict(self, embedding: torch.Tensor, commands_norm: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
        """
        Autoregressive latent prediction conditioned on `commands_norm`.

        Plan §10: during baseline training these are DP teacher trajectory commands
        (command_source=teacher_dp), not claimed to be paper ũ until §10 is resolved.
        """
        return self.predictor(embedding, commands_norm, horizon=horizon or self.kp)

    def ema_step(self) -> None:
        ema_update(self.target_encoder, self.context_encoder, decay=self.ema_decay)

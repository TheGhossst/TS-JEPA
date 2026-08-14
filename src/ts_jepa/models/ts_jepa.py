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
        # Algorithm 1: Ψ(x_{i,k}) on one RGB frame (3 channels). κ concat is for
        # supervised / AE baselines only (Section IV.D.3).
        in_channels = int(inp["channels_per_rgb_frame"])
        embedding_dim = int(enc_cfg["embedding_dim"])
        strict_dim = not bool(config.get("experiments", {}).get("allow_non_baseline_embedding_dim", False))
        self.context_encoder = ContextEncoder(
            in_channels=in_channels,
            widths=list(enc_cfg["widths"]),
            embedding_dim=embedding_dim,
            blocks_per_stage=int(enc_cfg.get("blocks_per_stage", 2)),
            spatial_pool_hw=tuple(enc_cfg.get("spatial_pool_hw", [4, 8])),
            strict_baseline_dim=strict_dim,
        )
        self.target_encoder = initialize_target_from_context(self.context_encoder)
        assert_target_initialized_from_context(self.context_encoder, self.target_encoder)
        self.predictor = Predictor(
            embedding_dim=embedding_dim,
            command_dim=1,
            hidden_dim=int(pred_cfg["hidden_dim"]),
            output_dim=int(pred_cfg["output_dim"]),
            hidden_batch_norm=bool(pred_cfg.get("hidden_batch_norm", True)),
            strict_baseline_dim=strict_dim,
        )
        self.command_source = str(pred_cfg.get("command_source", "teacher_dp"))
        self.command_resolution: PredictorCommandResolution = load_predictor_command_resolution(config)
        self.ema_decay = float(config["ts_jepa"]["target_encoder"]["ema_decay"])
        self.kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])

    def train(self, mode: bool = True) -> TSJEPA:
        """Keep Ψθ̄ in eval. Plan §8: target is stop-grad + EMA, not a trained BN branch."""
        super().train(mode)
        self.target_encoder.eval()
        return self

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(context)

    @torch.no_grad()
    def encode_targets(self, future_frames: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
        """
        Plan §8: target encoder forward with stop-gradient.

        future_frames: [B, Kp, 3, H, W] → [B, Kp, D] (one RGB frame per target step).
        """
        self.target_encoder.eval()
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

        Plan §9: during baseline training these are DP teacher trajectory commands
        (command_source=teacher_dp), not claimed to be paper ũ (pretraining ũ is OPEN).
        """
        return self.predictor(embedding, commands_norm, horizon=horizon or self.kp)

    def ema_step(self) -> None:
        ema_update(self.target_encoder, self.context_encoder, decay=self.ema_decay)

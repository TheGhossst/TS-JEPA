from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.preprocessing.pipeline import PreprocessPipeline


@dataclass
class RuntimeCommandStats:
    mean: float
    std: float
    force_min: float = -20.0
    force_max: float = 20.0

    def denormalize_and_clip(self, u_norm: float | np.ndarray | torch.Tensor) -> float:
        if isinstance(u_norm, torch.Tensor):
            u_norm_v = float(u_norm.detach().cpu().reshape(-1)[0].item())
        elif isinstance(u_norm, np.ndarray):
            u_norm_v = float(np.asarray(u_norm).reshape(-1)[0])
        else:
            u_norm_v = float(u_norm)
        u = u_norm_v * self.std + self.mean
        return float(np.clip(u, self.force_min, self.force_max))


class FrozenRuntimeController:
    """
    Frozen TS-JEPA runtime with packet received / packet lost paths.

    Packet received: x → Ψθ → z → Cε → denorm → clip
    Packet lost:     z̃,ũ → Pφ → z̃' → Cε → denorm → clip
    """

    def __init__(
        self,
        config: dict[str, Any],
        jepa: TSJEPA,
        actor: SemanticActor,
        normalizer: CommandNormalizer,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cpu")
        self.jepa = jepa.to(self.device).eval()
        self.actor = actor.to(self.device).eval()
        for module in (self.jepa, self.actor):
            for p in module.parameters():
                p.requires_grad_(False)
        self.pipeline = PreprocessPipeline(config, training=False)
        self.stats = RuntimeCommandStats(
            mean=normalizer.mean,
            std=normalizer.std,
            force_min=float(config["simulation"]["control_min_N"]),
            force_max=float(config["simulation"]["control_max_N"]),
        )
        self.latent: torch.Tensor | None = None
        self.last_command_norm: float = 0.0
        self.frame_buffer: list[np.ndarray] = []

    @classmethod
    def from_checkpoints(
        cls,
        config: dict[str, Any],
        jepa_ckpt: Path,
        actor_ckpt: Path,
        device: torch.device | None = None,
    ) -> "FrozenRuntimeController":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jepa_payload = torch.load(jepa_ckpt, map_location=device, weights_only=False)
        actor_payload = torch.load(actor_ckpt, map_location=device, weights_only=False)
        jepa = TSJEPA(config)
        jepa.load_state_dict(jepa_payload["model"])
        actor = SemanticActor(
            embedding_dim=int(config["ts_jepa"]["encoder"]["embedding_dim"]),
            hidden_dims=tuple(config["semantic_actor"]["architecture"]["hidden_dims"]),
            dropout=float(config["semantic_actor"]["architecture"]["dropout"]),
        )
        actor.load_state_dict(actor_payload["actor"])
        normalizer = CommandNormalizer.from_dict(
            actor_payload.get("normalizer") or jepa_payload["normalizer"]
        )
        return cls(config, jepa, actor, normalizer, device=device)

    def _context_from_buffer(self) -> torch.Tensor:
        frames = np.stack(self.frame_buffer, axis=0)
        t = len(frames) - 1
        context = self.pipeline.make_context_tensor(frames, t, self.config["input"]["kappa"])
        return context.unsqueeze(0).to(self.device)

    def observe_frame(self, frame: np.ndarray) -> None:
        self.frame_buffer.append(frame.copy())
        kappa = int(self.config["input"]["kappa"])
        if len(self.frame_buffer) > max(kappa + 5, 8):
            self.frame_buffer = self.frame_buffer[-(kappa + 5) :]

    @torch.no_grad()
    def step_packet_received(self, frame: np.ndarray) -> float:
        self.observe_frame(frame)
        context = self._context_from_buffer()
        z = self.jepa.encode_context(context)
        u_norm = self.actor(z)
        force = self.stats.denormalize_and_clip(u_norm)
        self.latent = z
        self.last_command_norm = float(u_norm.reshape(-1)[0].item())
        return force

    @torch.no_grad()
    def step_packet_lost(self) -> float:
        if self.latent is None:
            return 0.0
        cmd = torch.tensor([[self.last_command_norm]], dtype=torch.float32, device=self.device)
        z_next = self.jepa.predictor.forward_step(self.latent, cmd)
        u_norm = self.actor(z_next)
        force = self.stats.denormalize_and_clip(u_norm)
        self.latent = z_next
        self.last_command_norm = float(u_norm.reshape(-1)[0].item())
        return force

    def step(self, frame: np.ndarray | None, packet_received: bool) -> float:
        if packet_received:
            assert frame is not None
            return self.step_packet_received(frame)
        return self.step_packet_lost()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.device import select_device
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.preprocessing.pipeline import PreprocessPipeline


@dataclass
class RuntimeCommandStats:
    mean: float
    std: float
    force_min: float = -20.0
    force_max: float = 20.0

    def denormalize(self, u_norm: float | np.ndarray | torch.Tensor) -> float:
        """Plan §14: invert command z-score to Newtons."""
        if isinstance(u_norm, torch.Tensor):
            u_norm_v = float(u_norm.detach().cpu().reshape(-1)[0].item())
        elif isinstance(u_norm, np.ndarray):
            u_norm_v = float(np.asarray(u_norm).reshape(-1)[0])
        else:
            u_norm_v = float(u_norm)
        return float(u_norm_v * self.std + self.mean)

    def apply_plant_force_limits(self, force_n: float) -> float:
        """IMPLEMENTATION CHOICE: clip to paper u_min/u_max at the actuator."""
        return float(np.clip(force_n, self.force_min, self.force_max))

    def normalize(self, force_n: float) -> float:
        return float((float(force_n) - self.mean) / self.std)

    def actor_output_to_force_and_norm(self, u_actor: float | np.ndarray | torch.Tensor) -> tuple[float, float]:
        """Eq. 15 actor emits Newtons; clip at the plant; z-score only for Pφ."""
        if isinstance(u_actor, torch.Tensor):
            raw = float(u_actor.detach().cpu().reshape(-1)[0].item())
        elif isinstance(u_actor, np.ndarray):
            raw = float(np.asarray(u_actor).reshape(-1)[0])
        else:
            raw = float(u_actor)
        force = self.apply_plant_force_limits(raw)
        return force, self.normalize(force)

    def denormalize_and_clip(self, u_norm: float | np.ndarray | torch.Tensor) -> float:
        return self.apply_plant_force_limits(self.denormalize(u_norm))


class FrozenRuntimeController:
    """
    Plan §14 closed-loop remote control (weights not updated).

    Device / context encoder Ψθ encodes the RGB state when an embedding is received.
    Predictor Pφ and actor Cε run on the remote side.

    Packet received: x_k → Ψθ → z → Cε → Newtons (clip)
    Packet lost:     z̃, ũ → Pφ → z̃' → Cε → Newtons (clip)
      (predicted commands for Pφ = z-score of last applied force; first-step miss → 0 N is IC)

    The encoder lives on the device, so the RGB buffer is updated every
    timestep. Transmission success only gates whether the remote uses the
    new z or rolls Pφ.

    Plant clip to [u_min, u_max] is IMPLEMENTATION CHOICE (env also clips).
    """

    miss_behavior = "predict"

    def __init__(
        self,
        config: dict[str, Any],
        jepa: TSJEPA,
        actor: SemanticActor,
        normalizer: CommandNormalizer,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = select_device(device)
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

    def reset_episode(self) -> None:
        """Clear temporal runtime state before starting an independent rollout."""
        self.latent = None
        self.last_command_norm = 0.0
        self.frame_buffer.clear()

    @classmethod
    def from_checkpoints(
        cls,
        config: dict[str, Any],
        jepa_ckpt: Path,
        actor_ckpt: Path,
        device: torch.device | None = None,
    ) -> "FrozenRuntimeController":
        device = select_device(device)
        jepa_payload = torch.load(jepa_ckpt, map_location=device, weights_only=False)
        actor_payload = torch.load(actor_ckpt, map_location=device, weights_only=False)
        jepa = TSJEPA(config)
        jepa.load_state_dict(jepa_payload["model"])
        actor = SemanticActor.from_config(config)
        actor.load_state_dict(actor_payload["actor"])
        normalizer = CommandNormalizer.from_dict(
            actor_payload.get("normalizer") or jepa_payload["normalizer"]
        )
        return cls(config, jepa, actor, normalizer, device=device)

    def _context_from_buffer(self) -> torch.Tensor:
        frames = np.stack(self.frame_buffer, axis=0)
        t = len(frames) - 1
        context = self.pipeline.make_jepa_input(frames, t)
        return context.unsqueeze(0).to(self.device)

    def observe_frame(self, frame: np.ndarray) -> None:
        self.frame_buffer.append(frame.copy())
        kappa = int(self.config["input"]["kappa"])
        if len(self.frame_buffer) > max(kappa + 5, 8):
            self.frame_buffer = self.frame_buffer[-(kappa + 5) :]

    @torch.no_grad()
    def _act_from_device_encoding(self) -> float:
        context = self._context_from_buffer()
        z = self.jepa.encode_context(context)
        u_actor = self.actor(z)
        force, u_norm = self.stats.actor_output_to_force_and_norm(u_actor)
        self.latent = z
        self.last_command_norm = u_norm
        return force

    @torch.no_grad()
    def step_packet_received(self, frame: np.ndarray) -> float:
        self.observe_frame(frame)
        return self._act_from_device_encoding()

    @torch.no_grad()
    def step_packet_lost(self) -> float:
        if self.latent is None:
            # IMPLEMENTATION CHOICE: no embedding yet, so no predicted command.
            return 0.0
        cmd = torch.tensor([[self.last_command_norm]], dtype=torch.float32, device=self.device)
        z_next = self.jepa.predictor.forward_step(self.latent, cmd)
        u_actor = self.actor(z_next)
        force, u_norm = self.stats.actor_output_to_force_and_norm(u_actor)
        self.latent = z_next
        self.last_command_norm = u_norm
        return force

    def step(
        self,
        frame: np.ndarray | None,
        packet_received: bool,
        plant_state: np.ndarray | None = None,
    ) -> float:
        # Device always observes RGB; the remote only receives z on success.
        del plant_state  # TS-JEPA control is embedding-only (plan §12/§14).
        if frame is not None:
            self.observe_frame(frame)
        if packet_received:
            return self._act_from_device_encoding()
        return self.step_packet_lost()

"""Plan §16 controllers: DP / supervised / AE + hold-last; TS-JEPA remains predictive."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from ts_jepa.baselines.models import GenerativeAutoencoder, SupervisedRGBToCommand
from ts_jepa.control.dp_teacher import DPControlTeacher
from ts_jepa.device import select_device
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.preprocessing.pipeline import PreprocessPipeline


class ControlLoopAgent(Protocol):
    miss_behavior: str

    def reset_episode(self) -> None: ...

    def step(
        self,
        frame: np.ndarray | None,
        packet_received: bool,
        plant_state: np.ndarray | None = None,
    ) -> float: ...


class HoldLastMixin:
    """
    Plan §16: when unscheduled, apply the most recently received command.

    The same hold is used when a scheduled packet is not delivered (outage / extra
    drop). That extension is IMPLEMENTATION CHOICE; plan §16 names unscheduled hold.
    """

    miss_behavior = "hold_last_command"

    def __init__(self, *, force_min: float, force_max: float, initial_force: float = 0.0) -> None:
        self.force_min = float(force_min)
        self.force_max = float(force_max)
        self.initial_force = float(initial_force)
        self.last_force = float(initial_force)

    def reset_episode(self) -> None:
        self.last_force = self.initial_force

    def _clip(self, force: float) -> float:
        return float(np.clip(force, self.force_min, self.force_max))

    def _on_miss(self) -> float:
        return float(self.last_force)


class ZeroOnMissMixin(HoldLastMixin):
    miss_behavior = "zero_action"

    def _on_miss(self) -> float:
        return 0.0


class NonlinearDPController(HoldLastMixin):
    """
    Plan §16 baseline 1: receive high-dimensional observation, apply nonlinear DP.

    IMPLEMENTATION CHOICE: DP uses the privileged 4D plant state available in simulation.
    The paper transmits the RGB frame; DP equations are on x, not pixels.
    """

    def __init__(self, config: dict[str, Any], teacher: DPControlTeacher) -> None:
        sim = config["simulation"]
        super().__init__(force_min=float(sim["control_min_N"]), force_max=float(sim["control_max_N"]))
        self.teacher = teacher

    def step(
        self,
        frame: np.ndarray | None,
        packet_received: bool,
        plant_state: np.ndarray | None = None,
    ) -> float:
        if not packet_received:
            return self._on_miss()
        if plant_state is None:
            raise ValueError("NonlinearDPController requires plant_state on a received packet")
        self.last_force = self._clip(float(self.teacher.act(plant_state)))
        return self.last_force


class SupervisedController(HoldLastMixin):
    """Plan §16 baseline 2: RGB(κ) → command. Hold last command if not delivered."""

    def __init__(
        self,
        config: dict[str, Any],
        model: SupervisedRGBToCommand,
        normalizer: CommandNormalizer,
        *,
        kappa: int,
        device: torch.device | None = None,
    ) -> None:
        sim = config["simulation"]
        super().__init__(force_min=float(sim["control_min_N"]), force_max=float(sim["control_max_N"]))
        self.device = select_device(device)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.normalizer = normalizer
        self.kappa = int(kappa)
        cfg = {**config, "input": {**config["input"], "kappa": self.kappa}}
        self.pipeline = PreprocessPipeline(cfg, training=False)
        self.frame_buffer: list[np.ndarray] = []

    def reset_episode(self) -> None:
        super().reset_episode()
        self.frame_buffer.clear()

    @torch.no_grad()
    def step(
        self,
        frame: np.ndarray | None,
        packet_received: bool,
        plant_state: np.ndarray | None = None,
    ) -> float:
        if frame is not None:
            self.frame_buffer.append(frame.copy())
            if len(self.frame_buffer) > max(self.kappa + 5, 8):
                self.frame_buffer = self.frame_buffer[-(self.kappa + 5) :]
        if not packet_received:
            return self._on_miss()
        frames = np.stack(self.frame_buffer, axis=0)
        t = len(frames) - 1
        context = self.pipeline.make_context_tensor(frames, t, self.kappa).unsqueeze(0).to(self.device)
        u_norm = self.model(context)
        u_phys = self.normalizer.denormalize(u_norm.detach().cpu().numpy().reshape(-1))
        force = self._clip(float(u_phys.reshape(-1)[0]))
        self.last_force = force
        return force


class AutoencoderDPController(HoldLastMixin):
    """
    Plan §16 baseline 3: transmit AE embedding, reconstruct, then nonlinear DP.

    Hold last command if the embedding is not delivered (conventional, not TS-JEPA predict).
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: GenerativeAutoencoder,
        teacher: DPControlTeacher,
        *,
        kappa: int,
        device: torch.device | None = None,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
    ) -> None:
        sim = config["simulation"]
        super().__init__(force_min=float(sim["control_min_N"]), force_max=float(sim["control_max_N"]))
        self.device = select_device(device)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.teacher = teacher
        self.kappa = int(kappa)
        cfg = {**config, "input": {**config["input"], "kappa": self.kappa}}
        self.pipeline = PreprocessPipeline(cfg, training=False)
        self.frame_buffer: list[np.ndarray] = []
        self.state_mean = np.zeros(4, dtype=np.float64) if state_mean is None else np.asarray(state_mean, dtype=np.float64)
        std = np.ones(4, dtype=np.float64) if state_std is None else np.asarray(state_std, dtype=np.float64)
        self.state_std = np.maximum(std, 1e-6)

    def reset_episode(self) -> None:
        super().reset_episode()
        self.frame_buffer.clear()

    @torch.no_grad()
    def step(
        self,
        frame: np.ndarray | None,
        packet_received: bool,
        plant_state: np.ndarray | None = None,
    ) -> float:
        if frame is not None:
            self.frame_buffer.append(frame.copy())
            if len(self.frame_buffer) > max(self.kappa + 5, 8):
                self.frame_buffer = self.frame_buffer[-(self.kappa + 5) :]
        if not packet_received:
            return self._on_miss()
        frames = np.stack(self.frame_buffer, axis=0)
        t = len(frames) - 1
        context = self.pipeline.make_context_tensor(frames, t, self.kappa).unsqueeze(0).to(self.device)
        _, _, state_hat = self.model(context)
        hat = state_hat.detach().cpu().numpy().reshape(-1)
        plant = hat * self.state_std + self.state_mean
        self.last_force = self._clip(float(self.teacher.act(plant)))
        return self.last_force


class ZeroActionController(ZeroOnMissMixin):
    """Fig. 6 prediction-case zero-action conventional fallback."""

    def __init__(self, config: dict[str, Any], inner: ControlLoopAgent) -> None:
        sim = config["simulation"]
        super().__init__(force_min=float(sim["control_min_N"]), force_max=float(sim["control_max_N"]))
        self.inner = inner

    def reset_episode(self) -> None:
        super().reset_episode()
        self.inner.reset_episode()

    def step(
        self,
        frame: np.ndarray | None,
        packet_received: bool,
        plant_state: np.ndarray | None = None,
    ) -> float:
        if packet_received:
            force = self.inner.step(frame, True, plant_state=plant_state)
            self.last_force = force
            return force
        return self._on_miss()

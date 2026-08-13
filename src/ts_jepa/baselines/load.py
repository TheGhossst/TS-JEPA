"""Load §16 baseline checkpoints into hold-last controllers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_jepa.baselines.controllers import AutoencoderDPController, NonlinearDPController, SupervisedController
from ts_jepa.baselines.models import GenerativeAutoencoder, SupervisedRGBToCommand
from ts_jepa.data.trajectory_generator import build_env_and_teacher
from ts_jepa.preprocessing.command_stats import CommandNormalizer


def load_supervised_controller(
    config: dict[str, Any],
    checkpoint: Path,
    *,
    kappa: int,
    device: torch.device | None = None,
) -> SupervisedController:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    kappa = int(payload.get("kappa", kappa))
    model = SupervisedRGBToCommand.from_config(config, kappa=kappa)
    model.load_state_dict(payload["model"])
    normalizer = CommandNormalizer.from_dict(payload["normalizer"])
    return SupervisedController(config, model, normalizer, kappa=kappa, device=device)


def load_autoencoder_controller(
    config: dict[str, Any],
    checkpoint: Path,
    *,
    kappa: int = 2,
    device: torch.device | None = None,
) -> AutoencoderDPController:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    kappa = int(payload.get("kappa", kappa))
    model = GenerativeAutoencoder.from_config(config, kappa=kappa)
    model.load_state_dict(payload["model"])
    _, teacher = build_env_and_teacher(config)
    return AutoencoderDPController(
        config,
        model,
        teacher,
        kappa=kappa,
        device=device,
        state_mean=np.asarray(payload.get("state_mean", [0, 0, 0, 0]), dtype=np.float64),
        state_std=np.asarray(payload.get("state_std", [1, 1, 1, 1]), dtype=np.float64),
    )


def build_dp_controller(config: dict[str, Any]) -> NonlinearDPController:
    _, teacher = build_env_and_teacher(config)
    return NonlinearDPController(config, teacher)

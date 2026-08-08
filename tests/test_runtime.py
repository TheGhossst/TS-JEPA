from __future__ import annotations

import numpy as np
import torch

from ts_jepa.config import load_config
from ts_jepa.inference.infer import FrozenRuntimeController, RuntimeCommandStats
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.preprocessing.command_stats import CommandNormalizer


def test_runtime_recv_and_lost_paths():
    config = load_config()
    jepa = TSJEPA(config)
    actor = SemanticActor()
    normalizer = CommandNormalizer(mean=0.0, std=1.0)
    ctrl = FrozenRuntimeController(config, jepa, actor, normalizer, device=torch.device("cpu"))
    frame = np.random.randint(0, 255, size=(96, 192, 3), dtype=np.uint8)
    force_recv = ctrl.step_packet_received(frame)
    force_lost = ctrl.step_packet_lost()
    assert -20.0 <= force_recv <= 20.0
    assert -20.0 <= force_lost <= 20.0


def test_denorm_clip():
    stats = RuntimeCommandStats(mean=0.0, std=10.0, force_min=-20.0, force_max=20.0)
    assert stats.denormalize_and_clip(3.0) == 20.0
    assert stats.denormalize_and_clip(-3.0) == -20.0

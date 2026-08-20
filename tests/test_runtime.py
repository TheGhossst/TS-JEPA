from __future__ import annotations

import numpy as np
import torch

from ts_jepa.config import load_config
from ts_jepa.inference.infer import FrozenRuntimeController, RuntimeCommandStats
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.runtime import (
    CUDAPrefetcher,
    recommended_jepa_microbatch,
    resolve_amp_dtype,
    resolve_num_workers,
)
from torch.utils.data import DataLoader, Dataset


def test_recommended_jepa_microbatch_cpu_keeps_laptop_fallback():
    assert recommended_jepa_microbatch(256, torch.device("cpu")) == 16
    assert recommended_jepa_microbatch(2, torch.device("cpu")) == 2


def test_resolve_amp_dtype_off_on_cpu():
    assert resolve_amp_dtype(torch.device("cpu"), {"amp": "auto"}) is None
    assert resolve_amp_dtype(torch.device("cpu"), {"amp": "bf16"}) is None


def test_resolve_num_workers_cpu_is_zero():
    assert resolve_num_workers({"runtime": {"num_workers": "auto"}}, torch.device("cpu")) == 0
    assert resolve_num_workers({"runtime": {"num_workers": 8}}, torch.device("cpu")) == 0


def test_recommended_jepa_microbatch_48gb_ada_uses_full_paper_batch(monkeypatch):
    from ts_jepa import runtime as rt

    monkeypatch.setattr(rt, "gpu_total_memory_gb", lambda _device: 48.0)
    assert rt.recommended_jepa_microbatch(256, torch.device("cuda")) == 256
    assert rt.recommended_jepa_microbatch(256, torch.device("cuda")) % 256 == 0


class _DictTensorDataset(Dataset):
    def __init__(self, n: int = 8) -> None:
        self.x = torch.arange(n).float()

    def __len__(self) -> int:
        return int(self.x.numel())

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[idx]}


def test_cuda_prefetcher_cpu_close_drops_iterator():
    loader = DataLoader(_DictTensorDataset(8), batch_size=4)
    pref = CUDAPrefetcher(loader, torch.device("cpu"))
    it = iter(pref)
    first = next(it)
    assert first["x"].shape[0] == 4
    pref.close()
    try:
        next(it)
        raised = False
    except StopIteration:
        raised = True
    assert raised
    n = sum(1 for _ in CUDAPrefetcher(loader, torch.device("cpu")))
    assert n == 2


def test_runtime_recv_and_lost_paths():
    config = load_config()
    jepa = TSJEPA(config)
    actor = SemanticActor()
    normalizer = CommandNormalizer(mean=0.0, std=1.0)
    ctrl = FrozenRuntimeController(config, jepa, actor, normalizer, device=torch.device("cpu"))
    frame = np.random.randint(0, 255, size=(128, 256, 3), dtype=np.uint8)
    assert ctrl.step(frame, packet_received=False) == 0.0  # IC: no latent yet
    force_recv = ctrl.step_packet_received(frame)
    force_lost = ctrl.step_packet_lost()
    assert -20.0 <= force_recv <= 20.0
    assert -20.0 <= force_lost <= 20.0


def test_runtime_device_buffer_updates_when_packet_lost():
    config = load_config()
    jepa = TSJEPA(config)
    actor = SemanticActor()
    normalizer = CommandNormalizer(mean=0.0, std=1.0)
    ctrl = FrozenRuntimeController(config, jepa, actor, normalizer, device=torch.device("cpu"))
    frames = [np.random.randint(0, 255, size=(128, 256, 3), dtype=np.uint8) for _ in range(3)]
    ctrl.step(frames[0], packet_received=True)
    ctrl.step(frames[1], packet_received=False)
    ctrl.step(frames[2], packet_received=True)
    assert len(ctrl.frame_buffer) == 3


def test_denorm_clip():
    stats = RuntimeCommandStats(mean=0.0, std=10.0, force_min=-20.0, force_max=20.0)
    assert stats.denormalize_and_clip(3.0) == 20.0
    assert stats.denormalize_and_clip(-3.0) == -20.0

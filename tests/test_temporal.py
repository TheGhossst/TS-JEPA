"""Plan §6 temporal configuration tests."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from ts_jepa.config import load_config
from ts_jepa.data.datasets import TrajectoryDataset
from ts_jepa.data.temporal_plan import (
    PLAN_TEMPORAL,
    assert_plan_temporal_config,
    command_indices,
    context_frame_indices,
    describe_temporal_sample,
    max_valid_time_index,
    predicted_command_indices,
    target_end_indices,
)
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.preprocessing.command_stats import CommandNormalizer


def test_plan_temporal_config_matches_baseline_yaml():
    config = load_config()
    assert_plan_temporal_config(config)
    assert config["input"]["kappa"] == PLAN_TEMPORAL["kappa"]
    assert config["ts_jepa"]["prediction_horizon"]["Kp"] == PLAN_TEMPORAL["Kp"]
    assert config["temporal"]["kappa"] == 2
    assert config["temporal"]["Kp"] == 15


def test_plan_temporal_indexing_at_k5():
    sample = describe_temporal_sample(time_index=5, kappa=2, kp=15)
    assert sample["context_frame_indices"] == [4, 5]
    assert sample["command_indices"] == list(range(5, 20))
    assert sample["predicted_command_indices"] == list(range(6, 21))
    assert sample["target_end_indices"] == list(range(6, 21))
    assert sample["predicted_latent_indices"] == sample["target_end_indices"]


def test_max_valid_time_index_for_length_100():
    assert max_valid_time_index(100, 15) == 84
    assert target_end_indices(84, 15)[-1] == 99
    assert command_indices(84, 15)[-1] == 98
    assert predicted_command_indices(84, 15)[-1] == 99


def test_trajectory_dataset_temporal_fields(tmp_path):
    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 20
    config["ts_jepa"]["prediction_horizon"]["Kp"] = 4
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    steps = 20
    frames = np.zeros((steps, 128, 256, 3), dtype=np.uint8)
    commands = np.arange(steps, dtype=np.float32)
    states = np.zeros((steps, 4), dtype=np.float64)
    np.savez_compressed(
        traj_dir / "00000.npz",
        frames=frames,
        commands=commands,
        states=states,
        split=np.asarray("jepa_train"),
        seed=np.asarray(10000),
        trajectory_index=np.asarray(0),
        control_teacher=np.asarray("dp_nonlinear"),
        sampling_interval_ms=np.asarray(1.0),
        dt=np.asarray(0.001),
        render_height=np.asarray(128),
        render_width=np.asarray(256),
    )
    normalizer = CommandNormalizer(mean=0.0, std=1.0)
    ds = TrajectoryDataset(traj_dir, config, normalizer, training=False, kp=4)

    k = 5
    sample = ds[ds.index_map.index((0, k))]
    assert int(sample["time_index"]) == k
    assert sample["state"].shape == (4,)
    assert sample["teacher_commands"].shape == (4,)
    assert torch.allclose(sample["teacher_commands"], torch.tensor([5.0, 6.0, 7.0, 8.0]))
    assert torch.allclose(sample["target_commands"], torch.tensor([6.0, 7.0, 8.0, 9.0]))
    assert sample["future_frames"].shape[0] == 4
    assert sample["context"].shape == (3, 64, 128)
    assert sample["future_frames"].shape == (4, 3, 64, 128)


def test_tsjepa_predicts_kp_latents():
    config = load_config()
    model = TSJEPA(config)
    b = 2
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    context = torch.randn(b, 3, 64, 128)
    future = torch.randn(b, kp, 3, 64, 128)
    commands = torch.randn(b, kp)
    z = model.encode_context(context)
    z_tgt = model.encode_targets(future)
    z_pred = model.predict(z, commands)
    assert z.shape == (b, 256)
    assert z_tgt.shape == (b, kp, 256)
    assert z_pred.shape == (b, kp, 256)


def test_plan_temporal_config_rejects_wrong_kappa():
    config = copy.deepcopy(load_config())
    config["input"]["kappa"] = 1
    with pytest.raises(ValueError, match="temporal config mismatch"):
        assert_plan_temporal_config(config)

"""Tests for production trajectory generation with observation cadence."""

from __future__ import annotations

import numpy as np

from ts_jepa.config import load_config
from ts_jepa.data.trajectory_generator import build_env_and_teacher, generate_dataset_split
from ts_jepa.env.sampling import (
    jepa_training_windows_per_trajectory,
    resolve_observation_cadence_from_config,
)


def test_baseline_cadence_unchanged(tmp_path):
    config = load_config("configs/ts_jepa_baseline.yaml")
    cadence = resolve_observation_cadence_from_config(config)
    assert cadence["num_observations"] == 100
    assert cadence["observation_stride"] == 1
    assert cadence["budget_truncated"] is False

    env, teacher = build_env_and_teacher(config)
    out = tmp_path / "train"
    generate_dataset_split(config, "jepa_train", 1, 0, out, env, teacher)
    with np.load(out / "00000.npz") as data:
        assert data["frames"].shape[0] == 100
        assert int(np.asarray(data["observation_stride"])) == 1
        assert int(np.asarray(data["physics_steps_total"])) == 100


def test_dp_35ms_config_produces_20_observations(tmp_path):
    config = load_config("configs/ts_jepa_dp_35ms.yaml")
    cadence = resolve_observation_cadence_from_config(config)
    assert cadence["num_observations"] == 20
    assert cadence["observation_stride"] == 35
    assert cadence["physics_budget_effective"] == 700
    assert cadence["budget_truncated"] is False
    assert jepa_training_windows_per_trajectory(20, 15) == 5

    env, teacher = build_env_and_teacher(config)
    out = tmp_path / "train"
    generate_dataset_split(config, "jepa_train", 1, 0, out, env, teacher)
    with np.load(out / "00000.npz") as data:
        assert data["frames"].shape == (20, 96, 192, 3)
        assert data["commands"].shape[0] == 20
        assert data["states"].shape[0] == 20
        assert int(np.asarray(data["observation_stride"])) == 35
        assert int(np.asarray(data["physics_steps_total"])) == 700
        assert int(np.asarray(data["sampling_interval_ms"])) == 35
        ts = np.asarray(data["timestamps_s"])
        assert np.isclose(ts[1] - ts[0], 0.035)

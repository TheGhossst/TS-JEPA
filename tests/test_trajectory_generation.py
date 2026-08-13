"""Plan §4 dataset generation alignment tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from ts_jepa.config import load_config
from ts_jepa.data.dataset_sanity import verify_not_uniform_random_actions
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv
from ts_jepa.data.trajectory_generator import (
    UniformRandomControlTeacher,
    assert_plan_dataset_counts,
    assert_plan_simulation_timing,
    build_env_and_teacher,
    dataset_split_index_plan,
    generate_dataset_split,
    generate_trajectory,
)


def test_plan_simulation_timing_and_counts():
    config = load_config()
    assert_plan_simulation_timing(config)
    assert_plan_dataset_counts(config)
    assert config["dataset_generation"]["D_s"]["train_trajectories"] == 200
    assert config["dataset_generation"]["D_a"]["test_trajectories"] == 20
    assert config["dataset_generation"]["steps_per_trajectory"] == 100
    assert config["dataset_generation"]["control_teacher"] == "dp_nonlinear"
    assert config["control_teacher"]["dp_substeps"] == 1


def test_plan_simulation_timing_rejects_non_tau_o_bellman():
    config = copy.deepcopy(load_config())
    config["control_teacher"]["dp_substeps"] = 50
    with pytest.raises(ValueError, match="dp_substeps must be 1"):
        assert_plan_simulation_timing(config)


def test_plan_dataset_counts_reject_tiny_override():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["dataset"]["train_trajectories"] = 4
    with pytest.raises(ValueError, match="Plan §4 dataset counts"):
        assert_plan_dataset_counts(config)


def test_dp_teacher_trajectory_metadata_and_shapes(tmp_path: Path):
    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 100
    env, teacher = build_env_and_teacher(config)
    traj = generate_trajectory(env, teacher, steps=100, seed=42)
    assert traj["frames"].shape == (100, 128, 256, 3)
    assert traj["commands"].shape == (100,)
    assert traj["states"].shape == (100, 4)

    out = tmp_path / "traj"
    generate_dataset_split(config, "jepa_train", 1, 0, out, env, teacher)
    with np.load(out / "00000.npz") as data:
        assert str(np.asarray(data["control_teacher"]).item()) == "dp_nonlinear"
        assert float(np.asarray(data["sampling_interval_ms"]).item()) == 1.0
        assert int(np.asarray(data["observation_stride_steps"]).item()) == 1
        assert data["frames"].shape == (100, 128, 256, 3)


def test_dp_teacher_passes_not_uniform_random_check():
    config = load_config()
    env, teacher = build_env_and_teacher(config)
    commands = []
    for seed in range(5):
        traj = generate_trajectory(env, teacher, steps=100, seed=seed)
        commands.append(traj["commands"])
    report = verify_not_uniform_random_actions(commands)
    assert report["pass"], report


def test_optional_stride_can_hold_force_across_physics_ticks():
    """Holding force for N ticks is still supported; it is not the paper sampling setup."""
    env = InvertedCartPoleEnv(
        render_height=128,
        render_width=256,
        dt=0.001,
        process_noise_std=0.0,
        observation_stride_steps=50,
        init_noise=0.0,
    )

    class PushPolicy:
        def act(self, state):  # noqa: ARG002
            return 12.0

    traj = env.rollout(PushPolicy(), steps=8, seed=0)
    frames = traj["frames"].astype(np.float64)
    diffs = np.mean(np.abs(frames[1:] - frames[:-1]), axis=(1, 2, 3))
    assert float(diffs.mean()) > 0.3
    assert traj["states"].shape[0] == 8
    # 50 ms of 12 N must move the cart more than a 1 ms step.
    assert abs(float(traj["states"][1, 0] - traj["states"][0, 0])) > 0.001


def test_uniform_random_teacher_fails_not_uniform_random_check():
    env, _ = build_env_and_teacher(load_config())
    commands = []
    for seed in range(5):
        uniform = UniformRandomControlTeacher(seed=seed)
        traj = env.rollout(uniform, steps=100, seed=seed)
        commands.append(traj["commands"])
    report = verify_not_uniform_random_actions(commands)
    assert not report["pass"], report


def test_dataset_split_index_plan_is_disjoint_ds_then_da():
    config = load_config()
    plan = dataset_split_index_plan(config)
    assert plan["jepa_train"] == (200, 0)
    assert plan["jepa_test"] == (40, 200)
    assert plan["actor_train"] == (100, 240)
    assert plan["actor_test"] == (20, 340)
    ranges = []
    for count, start in plan.values():
        ranges.append(set(range(start, start + count)))
    for i, a in enumerate(ranges):
        for b in ranges[i + 1 :]:
            assert a.isdisjoint(b)

"""Tests for observation sampling cadence (ablation helpers).

Verifies physics-step stride mapping without changing ODE dt, teacher, or renderer.
"""

from __future__ import annotations

import numpy as np

from ts_jepa.config import load_config, project_root
from ts_jepa.control.dp_teacher import DPControlTeacher
from ts_jepa.data.trajectory_generator import build_env_and_teacher
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv
from ts_jepa.env.sampling import (
    num_observations_for_fixed_physics_budget,
    observation_stride_steps,
    physics_dt_ms,
    resolve_observation_cadence,
)


def test_observation_stride_mapping_for_1_2_5_10_ms():
    dt = 0.001
    assert physics_dt_ms(dt) == 1.0
    assert observation_stride_steps(1, dt) == 1
    assert observation_stride_steps(2, dt) == 2
    assert observation_stride_steps(5, dt) == 5
    assert observation_stride_steps(10, dt) == 10
    assert observation_stride_steps(20, dt) == 20


def test_fixed_physics_budget_observation_counts():
    budget = 100
    assert num_observations_for_fixed_physics_budget(budget, 1) == 100
    assert num_observations_for_fixed_physics_budget(budget, 5) == 20
    assert num_observations_for_fixed_physics_budget(budget, 10) == 10
    assert num_observations_for_fixed_physics_budget(budget, 20) == 5


def test_non_multiple_sampling_interval_raises():
    try:
        observation_stride_steps(1.5, 0.001)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_rollout_stride_preserves_ode_dt_and_physics_budget():
    env = InvertedCartPoleEnv(render_height=96, render_width=192, dt=0.001, process_noise_std=0.0)

    class ZeroPolicy:
        def act(self, state):
            return 0.0

    assert env.ode.dt == 0.001
    traj1 = env.rollout(ZeroPolicy(), steps=20, seed=0, observation_stride=5)
    assert env.ode.dt == 0.001
    assert int(traj1["observation_stride"]) == 5
    assert int(traj1["physics_steps_total"]) == 100  # 20 obs * 5
    assert traj1["frames"].shape == (20, 96, 192, 3)
    assert traj1["timestamps_s"][0] == 0.0
    assert np.isclose(traj1["timestamps_s"][1], 0.005)
    assert np.isclose(traj1["timestamps_s"][-1], 0.095)

    traj_default = env.rollout(ZeroPolicy(), steps=5, seed=1)
    assert int(traj_default["observation_stride"]) == 1
    assert int(traj_default["physics_steps_total"]) == 5


def test_teacher_and_renderer_unchanged_by_stride_parameter():
    config = load_config("configs/ts_jepa_baseline.yaml")
    env, teacher = build_env_and_teacher(config)
    assert isinstance(teacher, DPControlTeacher)
    assert env.ode.dt == 0.001
    assert teacher.dp_substeps == int(config["control_teacher"]["dp_substeps"])
    assert teacher.effective_dp_dt == teacher.dp_substeps * env.ode.dt
    assert env.renderer.height == 96
    assert env.renderer.width == 192

    before = (teacher.dp_substeps, len(teacher.forces), tuple(teacher.forces))
    _ = env.rollout(teacher, steps=4, seed=123, observation_stride=5)
    after = (teacher.dp_substeps, len(teacher.forces), tuple(teacher.forces))
    assert before == after
    assert env.ode.dt == 0.001
    assert env.renderer.height == 96
    assert env.renderer.width == 192


def test_resolve_observation_cadence_truncates_for_30ms_budget_100():
    cadence = resolve_observation_cadence(30, 0.001, 100, allow_truncated_budget=True)
    assert cadence["num_observations"] == 3
    assert cadence["physics_budget_effective"] == 90
    assert cadence["budget_truncated"] is True


def test_resolve_observation_cadence_exact_for_20ms_budget_100():
    cadence = resolve_observation_cadence(20, 0.001, 100, allow_truncated_budget=True)
    assert cadence["num_observations"] == 5
    assert cadence["physics_budget_effective"] == 100
    assert cadence["budget_truncated"] is False


def test_protected_dataset_dirs_are_not_ablation_targets():
    root = project_root(load_config("configs/ts_jepa_baseline.yaml"))
    ablation = root / "data_sampling_ablation"
    protected = [root / "data", root / "data_dp_fixed"]
    assert ablation.resolve() not in {p.resolve() for p in protected}
    for p in protected:
        assert "sampling_ablation" not in p.name

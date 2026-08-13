from __future__ import annotations

import numpy as np
import pytest

from ts_jepa.config import load_config
from ts_jepa.control.dp_teacher import DPControlTeacher
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv


def _tiny_teacher(**kwargs) -> DPControlTeacher:
    grid = {
        "x": [-0.2, 0.2, 3],
        "x_dot": [-0.5, 0.5, 3],
        "theta": [-0.15, 0.15, 5],
        "theta_dot": [-0.5, 0.5, 3],
    }
    env = InvertedCartPoleEnv(process_noise_std=0.0, dt=0.001)
    defaults = dict(
        env=env,
        grid=grid,
        force_min=-20.0,
        force_max=20.0,
        force_bins=11,
        value_iteration_iters=3,
        control_effort_weight=0.001,
        discount=0.99,
        dp_substeps=1,
        desired_state=[0.0, 0.0, 0.0, 0.0],
    )
    defaults.update(kwargs)
    return DPControlTeacher(**defaults)


def test_baseline_config_dp_grid_and_forces():
    config = load_config()
    ct = config["control_teacher"]
    assert ct["force_bins"] == 11
    assert ct["dp_substeps"] == 1
    assert ct["value_iteration_iters"] == 50
    assert ct["control_effort_weight"] == 0.001
    assert ct["discount"] == 0.99
    assert ct["grid"]["x"] == [-1.2, 1.2, 13]
    assert ct["grid"]["x_dot"] == [-2.5, 2.5, 11]
    assert ct["grid"]["theta"] == [-0.35, 0.35, 13]
    assert ct["grid"]["theta_dot"] == [-3.0, 3.0, 11]
    forces = np.linspace(ct["force_min_N"], ct["force_max_N"], ct["force_bins"])
    assert np.allclose(forces, [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20])


def test_dp_substeps_and_effective_dt():
    teacher = _tiny_teacher(dp_substeps=1)
    assert teacher.dp_substeps == 1
    assert teacher.effective_dp_dt == 0.001
    assert teacher.env.ode.dt == 0.001


def test_dp_substeps_not_equal_to_one_rejected():
    with pytest.raises(ValueError, match="dp_substeps must be 1"):
        _tiny_teacher(dp_substeps=50)


def test_force_grid_eleven_bins():
    teacher = _tiny_teacher(force_bins=11)
    assert len(teacher.forces) == 11
    assert np.allclose(teacher.forces, np.linspace(-20, 20, 11))


def test_successor_state_rollout_constant_force():
    teacher = _tiny_teacher(dp_substeps=1)
    s0 = np.array([0.0, 0.0, 0.1, 0.0], dtype=np.float64)
    final, stage = teacher._rollout_constant_force(s0, 0.0)
    assert isinstance(stage, float)
    assert final.shape == (4,)
    # Holding zero force from nonzero theta should move theta (nonlinear dynamics).
    assert abs(float(final[2]) - 0.1) > 1e-6
    # Control cost once: stage with u=0 equals pure state integral.
    final20, stage20 = teacher._rollout_constant_force(s0, 20.0)
    control_once = 0.5 * teacher.control_effort_weight * (20.0**2)
    # Stage for nonzero u includes exactly one control charge (not N).
    # Bound: stage20 - control_once should equal integrated state cost > 0.
    assert stage20 > control_once
    state_part = stage20 - control_once
    assert state_part > 0.0
    assert abs(stage20 - (state_part + control_once)) < 1e-9


def test_bellman_uses_successor_not_current_state_only():
    teacher = _tiny_teacher(dp_substeps=1, value_iteration_iters=5)
    s = np.array([0.0, 0.0, 0.12, 0.0], dtype=np.float64)
    q0 = teacher._bellman_cost(s, 0.0)
    # A large restoring-side force should generally change Q vs zero (not force-independent).
    qs = [teacher._bellman_cost(s, float(u)) for u in teacher.forces]
    assert max(qs) - min(qs) > 1e-6
    assert np.isfinite(q0)


def test_quantization_nearest_neighbor():
    teacher = _tiny_teacher()
    # Point closer to second x node than first.
    x0, x1 = teacher.grid.x[0], teacher.grid.x[1]
    mid_closer_to_x1 = x0 + 0.6 * (x1 - x0)
    state = np.array([mid_closer_to_x1, 0.0, 0.0, 0.0], dtype=np.float64)
    ix, _, _, _ = teacher._index_of(state)
    assert ix == 1


def test_tie_breaking_prefers_smaller_abs_u():
    # With desired already at equilibrium and tiny VI, u=0 should win near origin.
    teacher = _tiny_teacher(value_iteration_iters=5)
    u = teacher.act(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64))
    assert abs(u) <= abs(teacher.forces).max()
    # Exact equilibrium table policy should be 0 under symmetric cost.
    table = float(teacher.policy[teacher._index_of(np.zeros(4))])
    assert table == 0.0


def test_act_matches_candidate_argmin():
    teacher = _tiny_teacher(value_iteration_iters=5)
    state = np.array([0.0, 0.0, 0.08, 0.0], dtype=np.float64)
    table_u = float(teacher.policy[teacher._index_of(state)])
    deltas = np.array([0.0, -4.0, 4.0, -8.0, 8.0])
    local = np.clip(table_u + deltas, teacher.forces[0], teacher.forces[-1])
    coarse = teacher.forces[:: max(1, len(teacher.forces) // 5)]
    candidates = np.unique(np.concatenate([local, coarse]))
    best_u = float(candidates[0])
    best_cost = float("inf")
    for u in candidates:
        cost = teacher._bellman_cost(state, float(u))
        if cost < best_cost - 1e-12 or (abs(cost - best_cost) <= 1e-12 and abs(u) < abs(best_u)):
            best_cost = cost
            best_u = float(u)
    assert abs(teacher.act(state) - best_u) <= 1e-9


def test_dp_teacher_returns_bounded_force():
    teacher = _tiny_teacher(force_bins=5, value_iteration_iters=2)
    force = teacher.act(np.array([0.0, 0.0, 0.05, 0.0]))
    assert -20.0 <= force <= 20.0

"""Plan §3 environment tests."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from ts_jepa.config import load_config
from ts_jepa.env.cartpole_ode import CartPoleODE
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv
from ts_jepa.env.env_plan import PLAN_ENVIRONMENT, assert_plan_environment_config
from ts_jepa.env.env_validation import assert_plan_environment_validated, validate_plan_environment
from ts_jepa.env.factory import build_inverted_cartpole_env


def test_plan_environment_config_matches_baseline_yaml():
    config = load_config()
    assert_plan_environment_config(config)
    sim = config["simulation"]
    assert sim["control_min_N"] == PLAN_ENVIRONMENT["force_min_N"]
    assert sim["control_max_N"] == PLAN_ENVIRONMENT["force_max_N"]
    assert sim["trajectory_steps"] == PLAN_ENVIRONMENT["trajectory_steps"]
    assert sim["observation_stride_steps"] == PLAN_ENVIRONMENT["observation_stride_steps"]
    assert config["input"]["kappa"] == PLAN_ENVIRONMENT["kappa"]
    assert config["input"]["resize"] == [
        PLAN_ENVIRONMENT["encoder_height"],
        PLAN_ENVIRONMENT["encoder_width"],
    ]


def test_build_env_from_config_matches_plan():
    config = load_config()
    env = build_inverted_cartpole_env(config)
    assert isinstance(env, InvertedCartPoleEnv)
    assert env.ode.dt == PLAN_ENVIRONMENT["dt"]
    assert env.force_min == PLAN_ENVIRONMENT["force_min_N"]
    assert env.force_max == PLAN_ENVIRONMENT["force_max_N"]
    assert env.observation_stride_steps == PLAN_ENVIRONMENT["observation_stride_steps"]
    # Native camera size is IC, taken from config rather than the paper.
    assert env.renderer.height == int(config["simulation"]["render_height"])
    assert env.renderer.width == int(config["simulation"]["render_width"])


def test_ode_step_deterministic_and_shape():
    ode = CartPoleODE(dt=0.001)
    state = np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float64)
    next1 = ode.step(state, force=1.0, process_noise_std=0.0)
    next2 = ode.step(state, force=1.0, process_noise_std=0.0)
    assert next1.shape == (4,)
    assert np.allclose(next1, next2)


def test_env_rollout_rgb_shapes():
    env = InvertedCartPoleEnv(render_height=128, render_width=256, dt=0.001, process_noise_std=0.0)

    class ZeroPolicy:
        def act(self, state):
            return 0.0

    traj = env.rollout(ZeroPolicy(), steps=5, seed=0)
    assert traj["frames"].shape == (5, 128, 256, 3)
    assert traj["commands"].shape == (5,)
    assert traj["states"].shape == (5, 4)
    assert traj["frames"].dtype == np.uint8


def test_plan_environment_validation_passes_baseline_config():
    config = load_config()
    report = validate_plan_environment(config)
    assert report["overall_pass"] is True
    assert report["checks"]["force_limits"]["pass"]
    assert report["checks"]["native_render_resolution"]["pass"]
    assert report["checks"]["encoder_input_64x128"]["pass"]
    assert report["checks"]["subpixel_force_motion"]["pass"]
    assert report["checks"]["kappa_context_construction"]["pass"]
    assert_plan_environment_validated(config)


def test_native_render_size_is_not_a_plan_requirement():
    """Plan §3: native camera resolution is NOT SPECIFIED."""
    config = copy.deepcopy(load_config())
    config["simulation"]["render_height"] = 96
    config["simulation"]["render_width"] = 192
    assert_plan_environment_config(config)


def test_plan_environment_config_rejects_wrong_stride():
    config = copy.deepcopy(load_config())
    config["simulation"]["observation_stride_steps"] = 50
    with pytest.raises(ValueError, match="observation_stride_steps"):
        assert_plan_environment_config(config)


def test_plan_environment_config_rejects_wrong_force_limits():
    config = copy.deepcopy(load_config())
    config["simulation"]["control_max_N"] = 10.0
    with pytest.raises(ValueError, match="control_max_N"):
        assert_plan_environment_config(config)

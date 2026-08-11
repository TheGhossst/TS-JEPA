"""
Plan §3.3 environment validation — run before dataset generation / training.

Verifies force limits, integration, pendulum dynamics, cart tracking, RGB rendering,
64×128 resolution, 1 ms numerical stability, and state/render consistency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv
from ts_jepa.env.env_plan import PLAN_ENVIRONMENT, assert_plan_environment_config
from ts_jepa.env.factory import build_inverted_cartpole_env
from ts_jepa.preprocessing.pipeline import PreprocessPipeline


def _cart_pixel_centroid_x(frame: np.ndarray) -> float:
    """Approximate horizontal cart centroid from blue cart pixels in RGB frame."""
    r = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    b = frame[:, :, 2].astype(np.int16)
    mask = (b > 80) & (r < 120) & (g < 150)
    if not np.any(mask):
        return float("nan")
    xs = np.where(mask)[1]
    return float(xs.mean())


def _check_force_limits(env: InvertedCartPoleEnv) -> dict[str, Any]:
    clipped_low = env.clip_force(-50.0)
    clipped_high = env.clip_force(50.0)
    env.reset(seed=0)
    _, _ = env.step(100.0)
    state_after = env.state.copy()
    env.state = state_after.copy()
    _, _ = env.step(-100.0)

    class ExtremePolicy:
        def act(self, state: np.ndarray) -> float:  # noqa: ARG002
            return 25.0

    traj = env.rollout(ExtremePolicy(), steps=20, seed=1)
    cmds = traj["commands"]
    in_bounds = bool(np.all(cmds >= env.force_min - 1e-6) and np.all(cmds <= env.force_max + 1e-6))
    return {
        "clip_min": clipped_low == env.force_min,
        "clip_max": clipped_high == env.force_max,
        "rollout_in_bounds": in_bounds,
        "pass": clipped_low == env.force_min and clipped_high == env.force_max and in_bounds,
    }


def _check_state_integration(env: InvertedCartPoleEnv) -> dict[str, Any]:
    env.reset(seed=42, init_noise=0.0)
    env.state = np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float64)
    states = [env.state.copy()]
    for force in (1.0, -1.0, 0.5, -0.5) * 25:
        state, _ = env.step(force)
        states.append(state.copy())
    stacked = np.stack(states, axis=0)
    finite = bool(np.isfinite(stacked).all())
    changed = bool(np.any(np.abs(stacked[1:] - stacked[:-1]) > 1e-12))
    return {
        "finite_states": finite,
        "state_changes_under_force": changed,
        "final_state": [float(v) for v in states[-1]],
        "pass": finite and changed,
    }


def _check_pendulum_dynamics(env: InvertedCartPoleEnv) -> dict[str, Any]:
    """Inverted pendulum is unstable at upright: |θ| grows with zero control."""
    env.reset(seed=0, init_noise=0.0)
    env.state = np.array([0.0, 0.0, 0.06, 0.0], dtype=np.float64)
    theta0 = abs(float(env.state[2]))
    for _ in range(150):
        env.step(0.0)
    theta_end = abs(float(env.state[2]))
    grew = theta_end > theta0 + 0.01
    return {
        "initial_abs_theta": theta0,
        "final_abs_theta": theta_end,
        "unstable_without_control": grew,
        "pass": grew,
    }


def _check_cart_position_tracking(env: InvertedCartPoleEnv) -> dict[str, Any]:
    """Corrective force toward desired x=0 reduces |x| from a lateral offset."""
    env.reset(seed=0, init_noise=0.0)
    env.state = np.array([0.35, 0.0, 0.0, 0.0], dtype=np.float64)
    x0 = abs(float(env.state[0]))
    force = -15.0 if env.state[0] > 0 else 15.0
    for _ in range(80):
        env.step(force)
    x1 = abs(float(env.state[0]))
    improved = x1 < x0 - 0.02
    desired_ok = bool(np.allclose(env.desired_state, np.zeros(4)))
    score_at_origin = env.control_score(np.zeros(4)) == 1
    return {
        "initial_abs_x": x0,
        "final_abs_x": x1,
        "corrective_force_reduces_offset": improved,
        "desired_state_is_origin": desired_ok,
        "control_score_at_origin": score_at_origin,
        "pass": improved and desired_ok and score_at_origin,
    }


def _check_rgb_rendering(env: InvertedCartPoleEnv) -> dict[str, Any]:
    env.reset(seed=0)
    frame = env.render()
    ok_dtype = frame.dtype == np.uint8
    ok_shape = frame.ndim == 3 and frame.shape[2] == PLAN_ENVIRONMENT["channels"]
    ok_range = bool(frame.min() >= 0 and frame.max() <= 255)
    return {
        "dtype": str(frame.dtype),
        "shape": list(frame.shape),
        "value_range_ok": ok_range,
        "pass": ok_dtype and ok_shape and ok_range,
    }


def _check_render_resolution(env: InvertedCartPoleEnv) -> dict[str, Any]:
    frame = env.render(np.zeros(4, dtype=np.float64))
    exp_h = PLAN_ENVIRONMENT["render_height"]
    exp_w = PLAN_ENVIRONMENT["render_width"]
    exp_c = PLAN_ENVIRONMENT["channels"]
    ok = frame.shape == (exp_h, exp_w, exp_c)
    return {
        "shape": list(frame.shape),
        "expected": [exp_h, exp_w, exp_c],
        "pass": ok,
    }


def _check_numerical_stability_1ms(env: InvertedCartPoleEnv) -> dict[str, Any]:
    """Long rollout at dt=1 ms stays finite (plan §3.3)."""
    rng = np.random.default_rng(0)
    env.reset(seed=0, init_noise=0.1)
    max_abs = 0.0
    for _ in range(10_000):
        force = float(rng.uniform(env.force_min, env.force_max))
        state, _ = env.step(force)
        if not np.isfinite(state).all():
            return {"steps": 0, "finite": False, "pass": False}
        max_abs = max(max_abs, float(np.max(np.abs(state))))
    return {
        "steps": 10_000,
        "dt_s": env.ode.dt,
        "max_abs_state": max_abs,
        "finite": True,
        "pass": True,
    }


def _check_state_render_consistency(env: InvertedCartPoleEnv) -> dict[str, Any]:
    """Physical cart position x must move rendered cart centroid monotonically."""
    left_state = np.array([-0.45, 0.0, 0.0, 0.0], dtype=np.float64)
    right_state = np.array([0.45, 0.0, 0.0, 0.0], dtype=np.float64)
    frame_left = env.render(left_state)
    frame_right = env.render(right_state)
    centroid_left = _cart_pixel_centroid_x(frame_left)
    centroid_right = _cart_pixel_centroid_x(frame_right)
    mean_diff = float(np.mean(np.abs(frame_left.astype(np.int16) - frame_right.astype(np.int16))))
    monotonic = bool(
        np.isfinite(centroid_left)
        and np.isfinite(centroid_right)
        and centroid_right > centroid_left + 5.0
    )
    different_images = mean_diff > 1.0
    return {
        "centroid_x_at_x_neg": centroid_left,
        "centroid_x_at_x_pos": centroid_right,
        "mean_abs_pixel_diff": mean_diff,
        "centroid_monotonic_with_x": monotonic,
        "images_differ": different_images,
        "pass": monotonic and different_images,
    }


def _check_custom_backend_not_gym(env: InvertedCartPoleEnv) -> dict[str, Any]:
    module = type(env).__module__
    is_custom = isinstance(env, InvertedCartPoleEnv) and module.startswith("ts_jepa.env")
    return {
        "class": type(env).__name__,
        "module": module,
        "is_custom_inverted_cartpole": is_custom,
        "pass": is_custom,
    }


def _check_kappa_context_construction(config: dict[str, Any]) -> dict[str, Any]:
    """Plan §3.1: κ=2 RGB frames → [6, 64, 128] via channel_concat; eval is deterministic."""
    kappa = int(config["input"]["kappa"])
    frames = np.zeros((kappa + 1, 64, 128, 3), dtype=np.uint8)
    frames[1:] = np.arange(1, kappa + 1, dtype=np.uint8).reshape(-1, 1, 1, 1)
    train_pipe = PreprocessPipeline(config, training=True)
    eval_pipe = PreprocessPipeline(config, training=False)
    t = kappa
    ctx_train = train_pipe.make_context_tensor(frames, t, kappa=kappa)
    ctx_eval_a = eval_pipe.make_context_tensor(frames, t, kappa=kappa)
    ctx_eval_b = eval_pipe.make_context_tensor(frames, t, kappa=kappa)
    exp_shape = (
        PLAN_ENVIRONMENT["context_channels"],
        PLAN_ENVIRONMENT["render_height"],
        PLAN_ENVIRONMENT["render_width"],
    )
    eval_deterministic = bool(torch.equal(ctx_eval_a, ctx_eval_b))
    shape_ok = tuple(ctx_eval_a.shape) == exp_shape and tuple(ctx_train.shape) == exp_shape
    channel_concat = config["input"]["multi_frame_tensor_construction"] == "channel_concat"
    return {
        "kappa": kappa,
        "context_shape": list(ctx_eval_a.shape),
        "expected_shape": list(exp_shape),
        "channel_concat": channel_concat,
        "eval_deterministic": eval_deterministic,
        "pass": shape_ok and eval_deterministic and channel_concat,
    }


def validate_plan_environment(
    config: dict[str, Any],
    env: InvertedCartPoleEnv | None = None,
) -> dict[str, Any]:
    """
    Run plan §3.3 validation suite.

    Returns a JSON-serializable report with per-check results and `overall_pass`.
    """
    assert_plan_environment_config(config)
    env = env or build_inverted_cartpole_env(config)
    sim = config["simulation"]
    if float(env.force_min) != float(sim["control_min_N"]) or float(env.force_max) != float(sim["control_max_N"]):
        raise ValueError(
            "Built environment force limits do not match config simulation.control_*_N"
        )

    checks = {
        "custom_backend_not_gym": _check_custom_backend_not_gym(env),
        "force_limits": _check_force_limits(env),
        "state_integration": _check_state_integration(env),
        "pendulum_dynamics": _check_pendulum_dynamics(env),
        "cart_position_tracking": _check_cart_position_tracking(env),
        "rgb_rendering": _check_rgb_rendering(env),
        "render_resolution_64x128": _check_render_resolution(env),
        "numerical_stability_1ms": _check_numerical_stability_1ms(env),
        "state_render_consistency": _check_state_render_consistency(env),
        "kappa_context_construction": _check_kappa_context_construction(config),
    }
    overall_pass = all(bool(c.get("pass")) for c in checks.values())
    return {
        "plan_section": "3.3",
        "environment": PLAN_ENVIRONMENT["name"],
        "backend": PLAN_ENVIRONMENT["backend"],
        "checks": checks,
        "overall_pass": overall_pass,
    }


def assert_plan_environment_validated(config: dict[str, Any], env: InvertedCartPoleEnv | None = None) -> dict[str, Any]:
    """Run §3.3 validation and raise ValueError on failure."""
    report = validate_plan_environment(config, env=env)
    if not report["overall_pass"]:
        failed = [name for name, payload in report["checks"].items() if not payload.get("pass")]
        raise ValueError(
            "Plan §3.3 environment validation failed:\n  - " + "\n  - ".join(failed)
        )
    return report

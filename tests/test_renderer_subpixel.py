"""Subpixel / anti-aliased cart-pole renderer (IMPLEMENTATION CHOICE)."""

from __future__ import annotations

import numpy as np

from ts_jepa.env.cartpole_ode import CartPoleODE
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv
from ts_jepa.env.renderer import CartPoleRenderer


def test_twelve_newton_one_ms_changes_rgb():
    env = InvertedCartPoleEnv(render_height=64, render_width=128, dt=0.001, process_noise_std=0.0)
    env.reset(seed=0, init_noise=0.0)
    env.state = np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float64)
    frame_k = env.render(env.state)
    next_state, _ = env.step(12.0)
    frame_k1 = env.render(next_state)
    mape = float(
        np.mean(np.abs(frame_k.astype(np.float64) - frame_k1.astype(np.float64)) / np.maximum(frame_k.astype(np.float64), 1.0))
    )
    assert not np.array_equal(frame_k, frame_k1)
    assert mape > 0.0
    assert next_state[0] != 0.0


def test_identical_state_is_deterministic():
    renderer = CartPoleRenderer(height=64, width=128)
    state = np.array([0.12, 0.0, 0.04, 0.0], dtype=np.float64)
    a = renderer.render(state)
    b = renderer.render(state)
    assert np.array_equal(a, b)
    assert a.dtype == np.uint8
    assert a.shape == (64, 128, 3)


def test_ode_12n_displacement_is_sub_pixel_scale():
    ode = CartPoleODE(dt=0.001)
    s0 = np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float64)
    s1 = ode.step(s0, force=12.0, process_noise_std=0.0)
    assert abs(float(s1[0])) < 0.01

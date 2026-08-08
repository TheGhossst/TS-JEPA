from __future__ import annotations

import numpy as np

from ts_jepa.env.cartpole_ode import CartPoleODE
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv


def test_ode_step_deterministic_and_shape():
    ode = CartPoleODE(dt=0.001)
    state = np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float64)
    next1 = ode.step(state, force=1.0, process_noise_std=0.0)
    next2 = ode.step(state, force=1.0, process_noise_std=0.0)
    assert next1.shape == (4,)
    assert np.allclose(next1, next2)


def test_env_rollout_rgb_shapes():
    env = InvertedCartPoleEnv(render_height=96, render_width=192, dt=0.001, process_noise_std=0.0)

    class ZeroPolicy:
        def act(self, state):
            return 0.0

    traj = env.rollout(ZeroPolicy(), steps=5, seed=0)
    assert traj["frames"].shape == (5, 96, 192, 3)
    assert traj["commands"].shape == (5,)
    assert traj["states"].shape == (5, 4)
    assert traj["frames"].dtype == np.uint8

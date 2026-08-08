from __future__ import annotations

import numpy as np

from ts_jepa.config import load_config
from ts_jepa.control.dp_teacher import DPControlTeacher
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv


def test_dp_teacher_returns_bounded_force():
    config = load_config()
    # Tiny grid for unit-test speed (still exercises DP path).
    grid = {
        "x": [-0.2, 0.2, 3],
        "x_dot": [-0.5, 0.5, 3],
        "theta": [-0.1, 0.1, 3],
        "theta_dot": [-0.5, 0.5, 3],
    }
    env = InvertedCartPoleEnv(process_noise_std=0.0)
    teacher = DPControlTeacher(
        env=env,
        grid=grid,
        force_bins=5,
        value_iteration_iters=2,
        control_effort_weight=0.001,
    )
    force = teacher.act(np.array([0.0, 0.0, 0.05, 0.0]))
    assert -20.0 <= force <= 20.0

#!/usr/bin/env python
"""Verify 1 ms / 12 N cart motion changes the RGB frame (subpixel renderer IC)."""

from __future__ import annotations

import json

import numpy as np

from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv


def frame_mape(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64)
    bb = b.astype(np.float64)
    return float(np.mean(np.abs(aa - bb) / np.maximum(np.abs(aa), 1.0)))


def main() -> None:
    env = InvertedCartPoleEnv(render_height=64, render_width=128, dt=0.001, process_noise_std=0.0)
    env.reset(seed=0, init_noise=0.0)
    env.state = np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float64)
    frame_k = env.render(env.state)
    state_k1, _ = env.step(12.0)
    frame_k1 = env.render(state_k1)
    mape = frame_mape(frame_k, frame_k1)
    n_diff = int(np.sum(frame_k != frame_k1))
    report = {
        "force_N": 12.0,
        "dt_s": 0.001,
        "state_k": [0.0, 0.0, 0.05, 0.0],
        "state_k1": [float(v) for v in state_k1],
        "delta_x_m": float(state_k1[0] - 0.0),
        "mape": mape,
        "num_changed_pixels": n_diff,
        "frames_identical": bool(np.array_equal(frame_k, frame_k1)),
        "pass": mape > 0.0 and n_diff > 0,
    }
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("Renderer MAPE is zero: 1 ms motion is still invisible.")


if __name__ == "__main__":
    main()

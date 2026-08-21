"""How much actor-force noise can the closed loop tolerate?

True-state LQR scores ~0.52 on eval seeds while every embedding-based actor
scores ~0. Embedding decode noise is ~2-3 N through the current high-gain LQR
(K theta ~ -114). This sweeps additive Gaussian force noise, softer Q/R gain
sets, and command EMA smoothing to see which closed-loop path can survive the
representation noise floor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ts_jepa.config import load_config  # noqa: E402
from ts_jepa.control.lqr import (  # noqa: E402
    discrete_linearization,
    discrete_lqr_gain,
    default_lqr_weights,
    lqr_force,
)
from ts_jepa.env.factory import build_inverted_cartpole_env  # noqa: E402
from ts_jepa.evaluation.evaluate import _apply_held_force, _observation_stride  # noqa: E402
from ts_jepa.evaluation.metrics import control_score  # noqa: E402


def rollout_score(
    config,
    gain: np.ndarray,
    *,
    seed: int,
    noise_std: float,
    ema: float,
    rng: np.random.Generator,
) -> float:
    env = build_inverted_cartpole_env(config)
    steps = int(config["simulation"]["trajectory_steps"])
    stride = _observation_stride(config)
    lo = float(config["simulation"]["control_min_N"])
    hi = float(config["simulation"]["control_max_N"])
    state = env.reset(seed=seed)
    scores = []
    u_prev = 0.0
    for _ in range(steps):
        u = lqr_force(gain, state, lo, hi)
        u = u + float(rng.normal(0.0, noise_std)) if noise_std > 0 else u
        u = float(np.clip(u, lo, hi))
        if ema < 1.0:
            u = ema * u + (1.0 - ema) * u_prev
        u_prev = u
        scores.append(
            control_score(
                state,
                desired_x=float(config["simulation"]["desired_state"][0]),
                position_tol=float(config["evaluation"]["control_position_tol"]),
                angle_tol=float(config["evaluation"]["control_angle_tol"]),
            )
        )
        state = _apply_held_force(env, u, stride)
    return float(np.mean(scores))


def main() -> None:
    config = load_config("configs/ts_jepa_working.yaml")
    env = build_inverted_cartpole_env(config)
    stride = _observation_stride(config)
    a, b = discrete_linearization(env.ode, stride)

    gain_sets = {
        "baseline_Q40_4_80_4_R0.05": discrete_lqr_gain(a, b, *default_lqr_weights()),
        "soft_Q10_2_40_2_R0.5": discrete_lqr_gain(
            a, b, np.diag([10.0, 2.0, 40.0, 2.0]), np.array([[0.5]])
        ),
        "softer_Q4_1_20_1_R2": discrete_lqr_gain(
            a, b, np.diag([4.0, 1.0, 20.0, 1.0]), np.array([[2.0]])
        ),
    }
    seeds = [100, 101, 102]
    noise_levels = [0.0, 1.0, 2.0, 3.0, 5.0]
    emas = [1.0, 0.5]
    reps = 3

    report = {}
    for gname, gain in gain_sets.items():
        report[gname] = {"gain": np.asarray(gain).reshape(-1).tolist(), "grid": {}}
        for noise in noise_levels:
            for ema in emas:
                vals = []
                for seed in seeds:
                    for r in range(reps if noise > 0 else 1):
                        rng = np.random.default_rng(9000 + seed * 10 + r)
                        vals.append(
                            rollout_score(
                                config, gain, seed=seed, noise_std=noise, ema=ema, rng=rng
                            )
                        )
                key = f"noise={noise}_ema={ema}"
                report[gname]["grid"][key] = {
                    "mean": float(np.mean(vals)),
                    "per_rollout": [round(v, 3) for v in vals],
                }
                print(f"{gname:28s} {key:20s} mean={np.mean(vals):.3f}")

    out = Path("runs/eval/lqr_noise_sensitivity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

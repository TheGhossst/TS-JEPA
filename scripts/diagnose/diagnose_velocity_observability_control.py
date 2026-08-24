"""Which state-estimate quality suffices to close the loop?

White force noise <=5 N barely hurts LQR (see diagnose_lqr_noise_sensitivity).
This isolates the velocity channel: LQR gets exact/noisy positions but
degraded velocities, matching what a frozen-z decoder can provide:
  x MAE ~0.008 m, theta MAE ~0.004 rad (MLP probe),
  x_dot/theta_dot regression R^2 ~0.45 (attenuation toward the mean).
Modes:
  true_vel            : upper bound (positions noisy at decode level)
  attenuated_vel      : v_hat = a*v + noise (memoryless z decode of velocity)
  fd_vel              : v_hat = finite difference of noisy positions
  alpha_beta_vel      : alpha-beta observer on noisy positions (temporal filter)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ts_jepa.config import load_config  # noqa: E402
from ts_jepa.control.lqr import (  # noqa: E402
    default_lqr_weights,
    discrete_linearization,
    discrete_lqr_gain,
    lqr_force,
)
from ts_jepa.env.factory import build_inverted_cartpole_env  # noqa: E402
from ts_jepa.evaluation.evaluate import _apply_held_force, _observation_stride  # noqa: E402
from ts_jepa.evaluation.metrics import control_score  # noqa: E402

POS_NOISE = 0.008  # m, MLP probe MAE for x
ANG_NOISE = 0.004  # rad, MLP probe MAE for theta


def rollout(config, gain, *, seed, mode, vel_atten, vel_noise, rng):
    env = build_inverted_cartpole_env(config)
    steps = int(config["simulation"]["trajectory_steps"])
    stride = _observation_stride(config)
    dt_obs = stride * float(config["simulation"]["dt"])
    lo = float(config["simulation"]["control_min_N"])
    hi = float(config["simulation"]["control_max_N"])
    state = env.reset(seed=seed)
    scores = []
    prev_meas = None
    # alpha-beta observer state: [x, x_dot, theta, theta_dot]
    est = None
    alpha, beta_g = 0.5, 0.3
    for _ in range(steps):
        x_m = float(state[0]) + float(rng.normal(0.0, POS_NOISE))
        th_m = float(state[2]) + float(rng.normal(0.0, ANG_NOISE))
        if mode == "true_vel":
            s_hat = np.array([x_m, state[1], th_m, state[3]])
        elif mode == "attenuated_vel":
            xd = vel_atten * float(state[1]) + float(rng.normal(0.0, vel_noise))
            thd = vel_atten * float(state[3]) + float(rng.normal(0.0, vel_noise))
            s_hat = np.array([x_m, xd, th_m, thd])
        elif mode == "fd_vel":
            if prev_meas is None:
                s_hat = np.array([x_m, 0.0, th_m, 0.0])
            else:
                s_hat = np.array(
                    [
                        x_m,
                        (x_m - prev_meas[0]) / dt_obs,
                        th_m,
                        (th_m - prev_meas[1]) / dt_obs,
                    ]
                )
            prev_meas = (x_m, th_m)
        elif mode == "alpha_beta_vel":
            if est is None:
                est = np.array([x_m, 0.0, th_m, 0.0])
            else:
                # predict
                est[0] += est[1] * dt_obs
                est[2] += est[3] * dt_obs
                # correct
                rx = x_m - est[0]
                rth = th_m - est[2]
                est[0] += alpha * rx
                est[1] += beta_g * rx / dt_obs
                est[2] += alpha * rth
                est[3] += beta_g * rth / dt_obs
            s_hat = est.copy()
        else:
            raise ValueError(mode)
        u = lqr_force(gain, s_hat, lo, hi)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_working.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--out", type=str, default="runs/eval/velocity_observability_control.json")
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_inverted_cartpole_env(config)
    stride = _observation_stride(config)
    a, b = discrete_linearization(env.ode, stride)
    gain = discrete_lqr_gain(a, b, *default_lqr_weights())
    seeds = [int(s) for s in args.seeds]
    reps = 3

    cases = [
        ("true_vel", {}),
        ("attenuated_vel_a0.45_n0.5", {"mode": "attenuated_vel", "vel_atten": 0.45, "vel_noise": 0.5}),
        ("attenuated_vel_a0.45_n0.2", {"mode": "attenuated_vel", "vel_atten": 0.45, "vel_noise": 0.2}),
        ("attenuated_vel_a0.8_n0.2", {"mode": "attenuated_vel", "vel_atten": 0.8, "vel_noise": 0.2}),
        ("attenuated_vel_a1.0_n0.5", {"mode": "attenuated_vel", "vel_atten": 1.0, "vel_noise": 0.5}),
        ("fd_vel", {"mode": "fd_vel"}),
        ("alpha_beta_vel", {"mode": "alpha_beta_vel"}),
    ]
    report = {}
    for name, kw in cases:
        mode = kw.get("mode", "true_vel")
        vals = []
        for seed in seeds:
            for r in range(reps):
                rng = np.random.default_rng(7000 + seed * 10 + r)
                vals.append(
                    rollout(
                        config,
                        gain,
                        seed=seed,
                        mode=mode,
                        vel_atten=float(kw.get("vel_atten", 1.0)),
                        vel_noise=float(kw.get("vel_noise", 0.0)),
                        rng=rng,
                    )
                )
        report[name] = {"mean": float(np.mean(vals)), "per_rollout": [round(v, 3) for v in vals]}
        print(f"{name:30s} mean={np.mean(vals):.3f} per={[round(v,2) for v in vals]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

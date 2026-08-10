#!/usr/bin/env python
"""
Compare DP-teacher / control behavior on short (100ms-budget) vs extended-duration trajectories.

Read-only diagnostic: does not modify teacher, renderer, or production data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ts_jepa.config import load_config, project_root
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv


def _trajectory_stats(states: np.ndarray, commands: np.ndarray, *, track_limit: float) -> dict[str, Any]:
    x = states[:, 0]
    theta = states[:, 2]
    cmds = commands.astype(np.float64)
    pos_frac = float(np.mean(np.abs(x) > 0.9 * track_limit))
    large_angle = float(np.mean(np.abs(theta) > 0.35))
    return {
        "num_steps": int(states.shape[0]),
        "cart_x_min": float(x.min()),
        "cart_x_max": float(x.max()),
        "cart_x_abs_max": float(np.max(np.abs(x))),
        "theta_min_rad": float(theta.min()),
        "theta_max_rad": float(theta.max()),
        "theta_abs_max_rad": float(np.max(np.abs(theta))),
        "fraction_near_track_limit": pos_frac,
        "fraction_angle_beyond_dp_grid": large_angle,
        "command_min_N": float(cmds.min()),
        "command_max_N": float(cmds.max()),
        "command_std_N": float(cmds.std()),
        "command_mean_N": float(cmds.mean()),
        "command_positive_fraction": float(np.mean(cmds > 0)),
        "command_unique_count": int(len(np.unique(np.round(cmds, 6)))),
    }


def _aggregate(files: list[Path], *, track_limit: float, max_duration_s: float | None = None) -> dict[str, Any]:
    per_traj: list[dict[str, Any]] = []
    for path in files:
        with np.load(path) as data:
            states = np.asarray(data["states"], dtype=np.float64)
            commands = np.asarray(data["commands"], dtype=np.float64)
            dt_s = float(np.asarray(data["dt_s"])) if "dt_s" in data else 0.001
            stride = int(np.asarray(data["observation_stride"])) if "observation_stride" in data else 1
            if max_duration_s is not None:
                max_steps = int(max_duration_s / (dt_s * stride)) + 1
                max_steps = min(max_steps, states.shape[0])
                states = states[:max_steps]
                commands = commands[:max_steps]
        per_traj.append(_trajectory_stats(states, commands, track_limit=track_limit))

    def _mean(key: str) -> float:
        return float(np.mean([t[key] for t in per_traj]))

    return {
        "num_trajectories": len(per_traj),
        "per_trajectory": per_traj,
        "aggregate": {
            "mean_cart_x_abs_max": _mean("cart_x_abs_max"),
            "mean_theta_abs_max_rad": _mean("theta_abs_max_rad"),
            "mean_fraction_near_track_limit": _mean("fraction_near_track_limit"),
            "mean_fraction_angle_beyond_dp_grid": _mean("fraction_angle_beyond_dp_grid"),
            "mean_command_std_N": _mean("command_std_N"),
            "mean_command_positive_fraction": _mean("command_positive_fraction"),
            "mean_command_unique_count": _mean("command_unique_count"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-dir", type=str, required=True, help="e.g. data_sampling_ablation/35ms")
    parser.add_argument("--extended-dir", type=str, required=True, help="e.g. data_sampling_ablation_extended/35ms")
    parser.add_argument("--out", type=str, default="runs/eval/extended_duration_control.json")
    parser.add_argument(
        "--compare-first-duration-s",
        type=float,
        default=0.1,
        help="Also slice extended trajectories to first N seconds for apples-to-apples comparison.",
    )
    args = parser.parse_args()

    root = project_root(load_config())
    config = load_config()
    track_limit = float(config["simulation"]["physics"]["track_limit"])

    short_dir = Path(args.short_dir)
    extended_dir = Path(args.extended_dir)
    if not short_dir.is_absolute():
        short_dir = root / short_dir
    if not extended_dir.is_absolute():
        extended_dir = root / extended_dir

    short_files = sorted((short_dir / "trajectories").glob("*.npz"))
    extended_files = sorted((extended_dir / "trajectories").glob("*.npz"))

    report = {
        "short_dir": str(short_dir),
        "extended_dir": str(extended_dir),
        "track_limit_m": track_limit,
        "dp_teacher_theta_grid_max_rad": 0.35,
        "short_full": _aggregate(short_files, track_limit=track_limit),
        "extended_full": _aggregate(extended_files, track_limit=track_limit),
        "extended_first_window_s": float(args.compare_first_duration_s),
        "extended_first_window": _aggregate(
            extended_files,
            track_limit=track_limit,
            max_duration_s=float(args.compare_first_duration_s),
        ),
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    s = report["short_full"]["aggregate"]
    e = report["extended_full"]["aggregate"]
    ew = report["extended_first_window"]["aggregate"]
    print("=== Extended duration control comparison ===")
    print(f"short:  x_abs_max={s['mean_cart_x_abs_max']:.3f} theta_abs_max={s['mean_theta_abs_max_rad']:.3f} cmd_std={s['mean_command_std_N']:.2f}")
    print(f"ext:    x_abs_max={e['mean_cart_x_abs_max']:.3f} theta_abs_max={e['mean_theta_abs_max_rad']:.3f} cmd_std={e['mean_command_std_N']:.2f}")
    print(f"ext@100ms: x_abs_max={ew['mean_cart_x_abs_max']:.3f} theta_abs_max={ew['mean_theta_abs_max_rad']:.3f} cmd_std={ew['mean_command_std_N']:.2f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

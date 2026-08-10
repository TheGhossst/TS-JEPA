#!/usr/bin/env python
"""
Generate SMALL diagnostic trajectories for sampling-interval ablation.

Writes ONLY under data_sampling_ablation/{1ms,2ms,5ms,10ms,20ms,30ms}/.
Never touches data/ or data_dp_fixed/.

Physics ODE dt stays 0.001 s. Observation cadence = sampling_interval_ms / dt_ms
physics steps between stored RGB frames. Total physical duration is held fixed
via trajectory_steps physics-step budget from the baseline config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ts_jepa.config import load_config, project_root
from ts_jepa.data.trajectory_generator import build_env_and_teacher, generate_trajectory
from ts_jepa.env.sampling import (
    resolve_observation_cadence,
)


DEFAULT_INTERVALS_MS = (1, 2, 5, 10, 20, 30, 35, 40)


def generate_ablation_condition(
    config: dict,
    *,
    sampling_interval_ms: int,
    data_root: Path,
    num_trajectories: int,
    seed_base: int,
    physics_budget_steps: int | None = None,
) -> dict:
    sim = config["simulation"]
    dt = float(sim["dt"])
    physics_budget = int(physics_budget_steps if physics_budget_steps is not None else sim["trajectory_steps"])
    cadence = resolve_observation_cadence(
        sampling_interval_ms,
        dt,
        physics_budget,
        allow_truncated_budget=True,
    )
    stride = int(cadence["observation_stride"])
    num_obs = int(cadence["num_observations"])
    effective_budget = int(cadence["physics_budget_effective"])

    out_dir = data_root / f"{int(sampling_interval_ms)}ms" / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    env, teacher = build_env_and_teacher(config)
    # Sanity: teacher / ODE / renderer unchanged by this ablation parameter.
    assert float(env.ode.dt) == dt
    assert int(env.renderer.height) == int(sim["render_height"])
    assert int(env.renderer.width) == int(sim["render_width"])
    assert int(teacher.dp_substeps) == int(config["control_teacher"].get("dp_substeps", 50))

    meta_trajs = []
    for i in range(num_trajectories):
        trajectory_index = i
        seed = seed_base + trajectory_index
        traj = generate_trajectory(
            env,
            teacher,
            steps=num_obs,
            seed=seed,
            observation_stride=stride,
        )
        assert int(traj["observation_stride"]) == stride
        assert int(traj["physics_steps_total"]) == num_obs * stride
        assert traj["frames"].shape == (num_obs, sim["render_height"], sim["render_width"], 3)

        path = out_dir / f"{trajectory_index:05d}.npz"
        np.savez_compressed(
            path,
            frames=traj["frames"],
            commands=traj["commands"],
            states=traj["states"],
            observation_indices=traj["observation_indices"],
            timestamps_s=traj["timestamps_s"],
            observation_stride=traj["observation_stride"],
            physics_steps_total=traj["physics_steps_total"],
            sampling_interval_ms=np.asarray(int(sampling_interval_ms), dtype=np.int64),
            dt_s=np.asarray(dt, dtype=np.float64),
            split=np.asarray("sampling_ablation"),
            seed=np.asarray(seed),
            trajectory_index=np.asarray(trajectory_index),
        )
        meta_trajs.append(
            {
                "trajectory_index": trajectory_index,
                "seed": seed,
                "path": str(path),
                "num_observations": num_obs,
                "physics_steps_total": int(traj["physics_steps_total"]),
            }
        )

    manifest = {
        "sampling_interval_ms": int(sampling_interval_ms),
        "dt_s": dt,
        "observation_stride_physics_steps": stride,
        "physics_budget_steps": physics_budget,
        "physics_budget_effective_steps": effective_budget,
        "budget_truncated": bool(cadence["budget_truncated"]),
        "physical_duration_s": float(cadence["physical_duration_s"]),
        "num_observations_per_trajectory": num_obs,
        "num_trajectories": num_trajectories,
        "render_height": int(sim["render_height"]),
        "render_width": int(sim["render_width"]),
        "dp_substeps": int(teacher.dp_substeps),
        "ode_dt_unchanged": True,
        "teacher_unchanged": True,
        "trajectories": meta_trajs,
    }
    with (data_root / f"{int(sampling_interval_ms)}ms" / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_baseline.yaml")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data_sampling_ablation",
        help="Must NOT be data/ or data_dp_fixed/.",
    )
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument(
        "--physics-budget-steps",
        type=int,
        default=None,
        help="Override simulation.trajectory_steps physics budget (for extended-duration diagnostics).",
    )
    parser.add_argument(
        "--intervals-ms",
        type=int,
        nargs="+",
        default=list(DEFAULT_INTERVALS_MS),
    )
    args = parser.parse_args()

    root = project_root(load_config(args.config))
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root

    forbidden = {root / "data", root / "data_dp_fixed"}
    if data_root.resolve() in {p.resolve() for p in forbidden}:
        raise SystemExit(f"Refusing to write into protected dataset root: {data_root}")

    config = load_config(args.config)
    data_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for ms in args.intervals_ms:
        print(f"Generating sampling_interval_ms={ms} -> {data_root / f'{ms}ms'}")
        summaries[f"{ms}ms"] = generate_ablation_condition(
            config,
            sampling_interval_ms=int(ms),
            data_root=data_root,
            num_trajectories=int(args.num_trajectories),
            seed_base=int(args.seed_base),
            physics_budget_steps=args.physics_budget_steps,
        )

    overview = {
        "data_root": str(data_root),
        "config": args.config,
        "seed_base": int(args.seed_base),
        "num_trajectories_per_condition": int(args.num_trajectories),
        "conditions": summaries,
        "note": (
            "ODE dt fixed at config simulation.dt; observation_stride = "
            "sampling_interval_ms / (dt*1000); physical duration fixed via "
            "trajectory_steps physics budget."
        ),
    }
    with (data_root / "overview.json").open("w", encoding="utf-8") as handle:
        json.dump(overview, handle, indent=2)
    print(f"Wrote ablation diagnostic data under {data_root}")


if __name__ == "__main__":
    main()

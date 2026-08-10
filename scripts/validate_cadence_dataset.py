#!/usr/bin/env python
"""
Validate a cadence-aware dataset (e.g. 35 ms / 700 ms) before JEPA training.

Checks trajectory structure, JEPA window counts, and observability metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ts_jepa.config import load_config, project_root
from ts_jepa.data.dataset_sanity import sanity_check_trajectory_root
from ts_jepa.data.datasets import TrajectoryDataset, load_command_normalizer
from ts_jepa.diagnostics.observability import analyze_condition_observability
from ts_jepa.env.sampling import (
    actor_training_samples_per_trajectory,
    jepa_training_windows_per_trajectory,
    resolve_observation_cadence_from_config,
)


def _inspect_npz_files(files: list[Path], *, expected_obs: int, expected_stride: int, dt: float) -> dict[str, Any]:
    errors: list[str] = []
    interval_ms_checks: list[float] = []
    for path in files:
        with np.load(path) as data:
            frames = np.asarray(data["frames"])
            commands = np.asarray(data["commands"])
            states = np.asarray(data["states"])
            stride = int(np.asarray(data["observation_stride"])) if "observation_stride" in data else None
            physics_total = int(np.asarray(data["physics_steps_total"])) if "physics_steps_total" in data else None
            timestamps = np.asarray(data["timestamps_s"]) if "timestamps_s" in data else None
            file_dt = float(np.asarray(data["dt_s"])) if "dt_s" in data else None

        n = int(frames.shape[0])
        if n != expected_obs:
            errors.append(f"{path.name}: expected {expected_obs} observations, got {n}")
        if commands.shape[0] != n or states.shape[0] != n:
            errors.append(f"{path.name}: frames/commands/states length mismatch")
        if stride is not None and stride != expected_stride:
            errors.append(f"{path.name}: observation_stride {stride} != {expected_stride}")
        if physics_total is not None and physics_total != n * expected_stride:
            errors.append(f"{path.name}: physics_steps_total {physics_total} != {n * expected_stride}")
        if file_dt is not None and not np.isclose(file_dt, dt):
            errors.append(f"{path.name}: dt_s {file_dt} != {dt}")
        if timestamps is not None and timestamps.shape[0] >= 2:
            deltas_s = np.diff(timestamps)
            deltas_ms = deltas_s * 1000.0
            interval_ms_checks.extend(deltas_ms.tolist())
            expected_delta_s = expected_stride * dt
            if not np.allclose(deltas_s, expected_delta_s, rtol=0, atol=1e-9):
                errors.append(f"{path.name}: timestamp deltas not {expected_delta_s * 1000} ms")

    return {
        "files_checked": len(files),
        "mean_observation_interval_ms": float(np.mean(interval_ms_checks)) if interval_ms_checks else None,
        "errors": errors,
        "pass": len(errors) == 0,
    }


def _observability_on_split(split_dir: Path, label: str) -> dict[str, Any]:
    """Run observability metrics on a flat trajectory directory via temp wrapper."""
    with tempfile.TemporaryDirectory() as tmp:
        cond = Path(tmp) / label
        traj = cond / "trajectories"
        traj.mkdir(parents=True)
        for src in sorted(split_dir.glob("*.npz")):
            shutil.copy2(src, traj / src.name)
        return analyze_condition_observability(cond, label=label, kappa_values=(2, 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_35ms.yaml")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--pilot-count",
        type=int,
        default=None,
        help="If set, expect this many trajectories per split (pilot validation).",
    )
    parser.add_argument("--out", type=str, default="runs/eval/dp_35ms_dataset_validation.json")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(args.data_root) if args.data_root else (project_root(config) / config["paths"]["data_root"])
    if not root.is_absolute():
        root = project_root(config) / root

    cadence = resolve_observation_cadence_from_config(config, allow_truncated_budget=False)
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    kappa = int(config["input"]["kappa"])
    expected_obs = int(cadence["num_observations"])
    expected_stride = int(cadence["observation_stride"])
    dt = float(config["simulation"]["dt"])
    windows_per_traj = jepa_training_windows_per_trajectory(expected_obs, kp)

    jepa_train_dir = root / "trajectories" / "jepa" / "train"
    train_files = sorted(jepa_train_dir.glob("*.npz"))

    if args.pilot_count is not None:
        pilot_config = json.loads(json.dumps(config))
        for cfg_key in ("ts_jepa", "semantic_actor"):
            pilot_config[cfg_key]["dataset"]["train_trajectories"] = int(args.pilot_count)
            pilot_config[cfg_key]["dataset"]["test_trajectories"] = int(args.pilot_count)
        sanity = sanity_check_trajectory_root(pilot_config, root, fit_normalizer=True)
    else:
        sanity = sanity_check_trajectory_root(config, root, fit_normalizer=True)
    structure = _inspect_npz_files(train_files, expected_obs=expected_obs, expected_stride=expected_stride, dt=dt)

    normalizer = load_command_normalizer(config, data_root=root)
    jepa_ds = TrajectoryDataset(jepa_train_dir, config, normalizer, training=False)
    actual_windows = len(jepa_ds)
    expected_windows_pilot = windows_per_traj * len(train_files)

    obs = _observability_on_split(jepa_train_dir, "pilot_jepa_train")
    rd = obs["raw_frame_diversity"]
    cf = obs["consecutive_frame_change"]
    k2 = obs["kappa_context_analysis"]["kappa_2"]
    col = k2["context_collisions"]
    total_samples = int(rd["total_frames"])
    collision_rate = float(col["samples_in_groups_with_multiple_commands"]) / max(total_samples, 1)

    report: dict[str, Any] = {
        "config": args.config,
        "data_root": str(root),
        "cadence": cadence,
        "kappa": kappa,
        "kp": kp,
        "jepa_windows_per_trajectory": windows_per_traj,
        "actor_samples_per_trajectory": actor_training_samples_per_trajectory(expected_obs),
        "pilot_jepa_train_files": len(train_files),
        "pilot_jepa_train_windows_actual": actual_windows,
        "pilot_jepa_train_windows_expected": expected_windows_pilot,
        "windows_match": actual_windows == expected_windows_pilot,
        "sanity_check": sanity,
        "structure_check": structure,
        "observability": {
            "unique_frame_ratio": rd["unique_frame_ratio"],
            "fraction_zero_consecutive_pixels": cf["fraction_exact_zero_pixel_difference"],
            "unique_kappa2_context_ratio": k2["unique_context_ratio"],
            "context_collision_samples": col["samples_in_groups_with_multiple_commands"],
            "context_collision_rate_per_sample": collision_rate,
            "max_command_span_in_collisions": col["command_span_stats_within_multi_command_groups"]["max"],
        },
        "zero_consecutive_gate_status": {
            "target_lt": 0.2,
            "value": cf["fraction_exact_zero_pixel_difference"],
            "met": bool(cf["fraction_exact_zero_pixel_difference"] < 0.2),
            "note": "OPEN — gate not met in prior extended diagnostic; re-report at every stage.",
        },
        "overall_pass": bool(
            sanity.get("overall_pass")
            and structure["pass"]
            and actual_windows == expected_windows_pilot
        ),
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = project_root(config) / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=== Cadence dataset validation ===")
    print(f"cadence: {cadence['sampling_interval_ms']} ms, {expected_obs} obs, {cadence['physical_duration_s']} s")
    print(f"JEPA windows/trajectory: {windows_per_traj} (pilot actual={actual_windows}, expected={expected_windows_pilot})")
    print(f"observability: uniq_frame={rd['unique_frame_ratio']:.3f} zero_consec={cf['fraction_exact_zero_pixel_difference']:.3f}")
    print(f"collision_rate={collision_rate:.4f} overall_pass={report['overall_pass']}")
    print(f"Wrote {out}")
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

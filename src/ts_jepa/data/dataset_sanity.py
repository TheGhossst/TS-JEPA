"""Post-generation trajectory dataset sanity checks (IMPLEMENTATION CHOICE)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ts_jepa.config import project_root
from ts_jepa.data.datasets import fit_command_normalizer
from ts_jepa.env.sampling import resolve_observation_cadence_from_config
from ts_jepa.preprocessing.command_stats import CommandNormalizer


SPLIT_SPECS = (
    ("jepa", "train", "ts_jepa", "train_trajectories"),
    ("jepa", "test", "ts_jepa", "test_trajectories"),
    ("actor", "train", "semantic_actor", "train_trajectories"),
    ("actor", "test", "semantic_actor", "test_trajectories"),
)


def _command_stats(commands: np.ndarray) -> dict[str, Any]:
    c = np.asarray(commands, dtype=np.float64).reshape(-1)
    uniq = np.unique(c)
    return {
        "command_min": float(c.min()) if c.size else float("nan"),
        "command_max": float(c.max()) if c.size else float("nan"),
        "command_mean": float(c.mean()) if c.size else float("nan"),
        "command_std": float(c.std()) if c.size else float("nan"),
        "num_unique_commands": int(uniq.size),
        "unique_commands": [float(u) for u in uniq],
        "zero_fraction": float(np.mean(c == 0.0)) if c.size else float("nan"),
        "positive_fraction": float(np.mean(c > 0.0)) if c.size else float("nan"),
        "negative_fraction": float(np.mean(c < 0.0)) if c.size else float("nan"),
        "num_values": int(c.size),
    }


def _split_pass(stats: dict[str, Any]) -> dict[str, bool]:
    return {
        "std_gt_0": bool(stats["command_std"] > 0.0),
        "positive_fraction_gt_0": bool(stats["positive_fraction"] > 0.0),
        "negative_fraction_gt_0": bool(stats["negative_fraction"] > 0.0),
        "unique_commands_ge_3": bool(stats["num_unique_commands"] >= 3),
    }


def inspect_split(
    split_dir: Path,
    *,
    expected_count: int,
    expected_steps: int,
) -> dict[str, Any]:
    files = sorted(split_dir.glob("*.npz"))
    out: dict[str, Any] = {
        "path": str(split_dir),
        "n_files": len(files),
        "expected_count": int(expected_count),
        "expected_steps": int(expected_steps),
        "count_ok": len(files) == int(expected_count),
    }
    if not files:
        out["command_stats"] = _command_stats(np.array([]))
        out["pass"] = {k: False for k in _split_pass(out["command_stats"])}
        out["pass"]["count_ok"] = False
        out["errors"] = ["no trajectory files"]
        return out

    commands = []
    ids: list[int] = []
    errors: list[str] = []
    length_ok = True
    finite_ok = True
    for path in files:
        with np.load(path) as data:
            frames = np.asarray(data["frames"])
            cmds = np.asarray(data["commands"])
            states = np.asarray(data["states"])
            traj_id = int(np.asarray(data["trajectory_index"]).item())
            ids.append(traj_id)
            if frames.shape[0] != expected_steps or cmds.shape[0] != expected_steps or states.shape[0] != expected_steps:
                length_ok = False
                errors.append(f"{path.name}: length mismatch frames/cmds/states vs {expected_steps}")
            if not (np.isfinite(cmds).all() and np.isfinite(states).all()):
                finite_ok = False
                errors.append(f"{path.name}: non-finite commands/states")
            if frames.size == 0 or not np.isfinite(frames.astype(np.float64)).all():
                # uint8 frames are finite by construction; still guard empty
                finite_ok = False
                errors.append(f"{path.name}: empty/non-finite frames")
            if frames.shape[0] != cmds.shape[0] or cmds.shape[0] != states.shape[0]:
                errors.append(f"{path.name}: frames/commands/states length mismatch within file")
            commands.append(cmds.reshape(-1))

    all_cmds = np.concatenate(commands, axis=0)
    stats = _command_stats(all_cmds)
    checks = _split_pass(stats)
    checks["count_ok"] = bool(out["count_ok"])
    checks["length_ok"] = bool(length_ok)
    checks["finite_ok"] = bool(finite_ok)
    out["command_stats"] = stats
    out["trajectory_ids"] = sorted(ids)
    out["pass"] = checks
    out["errors"] = errors
    out["split_pass"] = all(checks.values()) and not errors
    return out


def sanity_check_trajectory_root(
    config: dict[str, Any],
    data_root: Path | str,
    *,
    fit_normalizer: bool = True,
) -> dict[str, Any]:
    """
    Validate JEPA/actor train+test splits under data_root.

    Does not regenerate data. Optionally fits command_norm.json under this root only.
    """
    root = Path(data_root)
    cadence = resolve_observation_cadence_from_config(config, allow_truncated_budget=False)
    observations_per_trajectory = int(cadence["num_observations"])
    report: dict[str, Any] = {
        "data_root": str(root),
        "trajectory_steps_physics_budget": int(config["simulation"]["trajectory_steps"]),
        "observations_per_trajectory": observations_per_trajectory,
        "sampling_interval_ms": int(cadence["sampling_interval_ms"]),
        "observation_stride": int(cadence["observation_stride"]),
        "init_noise": float(config["simulation"]["init_noise"]),
        "splits": {},
        "id_overlap": {},
        "command_normalizer": None,
        "pass": {},
    }

    for family, split, cfg_key, count_key in SPLIT_SPECS:
        expected = int(config[cfg_key]["dataset"][count_key])
        split_dir = root / "trajectories" / family / split
        key = f"{family}_{split}"
        report["splits"][key] = inspect_split(
            split_dir,
            expected_count=expected,
            expected_steps=observations_per_trajectory,
        )

    # Train/test ID separation within each family (paper protocol).
    for family in ("jepa", "actor"):
        train_ids = set(report["splits"][f"{family}_train"].get("trajectory_ids") or [])
        test_ids = set(report["splits"][f"{family}_test"].get("trajectory_ids") or [])
        overlap = sorted(train_ids & test_ids)
        report["id_overlap"][family] = {
            "overlap_ids": overlap,
            "no_overlap": len(overlap) == 0,
        }

    # Command normalizer from JEPA train only (same as training pipeline).
    if fit_normalizer and report["splits"]["jepa_train"]["n_files"] > 0:
        normalizer = fit_command_normalizer(config, data_root=root)
        report["command_normalizer"] = {
            "mean": float(normalizer.mean),
            "std": float(normalizer.std),
            "path": str(root / "stats" / "command_norm.json"),
            "nondegenerate": bool(np.isfinite(normalizer.mean) and np.isfinite(normalizer.std) and normalizer.std > 1e-8),
        }
    else:
        stats_path = root / "stats" / "command_norm.json"
        if stats_path.exists():
            with stats_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            normalizer = CommandNormalizer.from_dict(payload)
            report["command_normalizer"] = {
                "mean": float(normalizer.mean),
                "std": float(normalizer.std),
                "path": str(stats_path),
                "nondegenerate": bool(
                    np.isfinite(normalizer.mean) and np.isfinite(normalizer.std) and normalizer.std > 1e-8
                ),
            }

    split_ok = all(report["splits"][k]["split_pass"] for k in report["splits"])
    overlap_ok = all(report["id_overlap"][f]["no_overlap"] for f in ("jepa", "actor"))
    norm = report["command_normalizer"] or {}
    norm_ok = bool(norm.get("nondegenerate"))
    report["pass"] = {
        "all_splits": split_ok,
        "no_train_test_id_overlap": overlap_ok,
        "command_normalizer_nondegenerate": norm_ok,
    }
    report["overall_pass"] = bool(split_ok and overlap_ok and norm_ok)
    return report

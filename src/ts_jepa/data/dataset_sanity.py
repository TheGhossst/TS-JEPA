"""Post-generation trajectory dataset sanity checks (plan §4 + IMPLEMENTATION CHOICE)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ts_jepa.config import project_root
from ts_jepa.data.datasets import fit_command_normalizer
from ts_jepa.data.trajectory_generator import CONTROL_TEACHER_TYPE
from ts_jepa.preprocessing.command_stats import CommandNormalizer


SPLIT_SPECS = (
    ("jepa", "train", "ts_jepa", "train_trajectories"),
    ("jepa", "test", "ts_jepa", "test_trajectories"),
    ("actor", "train", "semantic_actor", "train_trajectories"),
    ("actor", "test", "semantic_actor", "test_trajectories"),
)


def _lag1_autocorr(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 3:
        return float("nan")
    a = x[:-1] - x[:-1].mean()
    b = x[1:] - x[1:].mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def verify_not_uniform_random_actions(
    commands_by_trajectory: list[np.ndarray],
    *,
    force_min: float = -20.0,
    force_max: float = 20.0,
    seed: int = 0,
    autocorr_margin: float = 0.02,
) -> dict[str, Any]:
    """
    Plan §4.1: explicitly verify commands are not i.i.d. Uniform(force_min, force_max).

    Uses lag-1 autocorrelation and histogram structure vs a synthetic uniform-iid reference.
    """
    if not commands_by_trajectory:
        return {
            "policy_mean_lag1_autocorr": float("nan"),
            "uniform_iid_mean_lag1_autocorr": float("nan"),
            "checks": {"has_trajectories": False},
            "pass": False,
        }

    rng = np.random.default_rng(seed)
    policy_autocorrs = [_lag1_autocorr(c) for c in commands_by_trajectory if np.asarray(c).size >= 3]
    policy_mean_ac = float(np.nanmean(policy_autocorrs)) if policy_autocorrs else float("nan")

    uniform_autocorrs = []
    for cmds in commands_by_trajectory:
        c = np.asarray(cmds).reshape(-1)
        if c.size < 3:
            continue
        u = rng.uniform(force_min, force_max, size=c.size)
        uniform_autocorrs.append(_lag1_autocorr(u))
    uniform_mean_ac = float(np.nanmean(uniform_autocorrs)) if uniform_autocorrs else float("nan")

    all_cmds = np.concatenate([np.asarray(c).reshape(-1) for c in commands_by_trajectory])
    bins = np.linspace(force_min, force_max, 12)
    counts, _ = np.histogram(all_cmds, bins=bins)
    counts = counts.astype(np.float64) + 1e-6
    hist_cv = float(counts.std() / counts.mean())

    ref = rng.uniform(force_min, force_max, size=all_cmds.size)
    ref_counts, _ = np.histogram(ref, bins=bins)
    ref_counts = ref_counts.astype(np.float64) + 1e-6
    ref_cv = float(ref_counts.std() / ref_counts.mean())

    checks = {
        "has_trajectories": True,
        "command_std_gt_0": float(np.std(all_cmds)) > 0.0,
        "autocorr_exceeds_uniform_iid": abs(policy_mean_ac) > abs(uniform_mean_ac) + autocorr_margin,
        "histogram_more_structured_than_uniform_iid": hist_cv > ref_cv * 0.75,
    }
    return {
        "policy_mean_lag1_autocorr": policy_mean_ac,
        "uniform_iid_mean_lag1_autocorr": uniform_mean_ac,
        "command_histogram_cv": hist_cv,
        "uniform_reference_histogram_cv": ref_cv,
        "checks": checks,
        "pass": all(checks.values()),
    }


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
    expected_render_hw: tuple[int, int] | None = None,
    expected_teacher: str = CONTROL_TEACHER_TYPE,
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
    metadata_ok = True
    frame_shape_ok = True
    for path in files:
        with np.load(path) as data:
            frames = np.asarray(data["frames"])
            cmds = np.asarray(data["commands"])
            states = np.asarray(data["states"])
            traj_id = int(np.asarray(data["trajectory_index"]).item())
            ids.append(traj_id)
            if "control_teacher" in data:
                teacher = str(np.asarray(data["control_teacher"]).item())
                if teacher != expected_teacher:
                    metadata_ok = False
                    errors.append(f"{path.name}: control_teacher={teacher!r}, expected {expected_teacher!r}")
            else:
                metadata_ok = False
                errors.append(f"{path.name}: missing control_teacher metadata (regenerate dataset)")
            if expected_render_hw is not None:
                exp_h, exp_w = expected_render_hw
                if frames.ndim != 4 or frames.shape[1:3] != (exp_h, exp_w) or frames.shape[3] != 3:
                    frame_shape_ok = False
                    errors.append(f"{path.name}: frame shape {frames.shape}, expected (*, {exp_h}, {exp_w}, 3)")
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
    checks["metadata_ok"] = bool(metadata_ok)
    checks["frame_shape_ok"] = bool(frame_shape_ok)
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
    steps = int(config["simulation"]["trajectory_steps"])
    render_hw = (int(config["simulation"]["render_height"]), int(config["simulation"]["render_width"]))
    report: dict[str, Any] = {
        "data_root": str(root),
        "trajectory_steps": steps,
        "sampling_interval_ms": float(config["simulation"]["sampling_interval_ms"]),
        "dt": float(config["simulation"]["dt"]),
        "control_teacher": CONTROL_TEACHER_TYPE,
        "init_noise": float(config["simulation"]["init_noise"]),
        "splits": {},
        "id_overlap": {},
        "teacher_not_uniform_random": None,
        "command_normalizer": None,
        "pass": {},
    }

    commands_by_traj: list[np.ndarray] = []
    for family, split, cfg_key, count_key in SPLIT_SPECS:
        expected = int(config[cfg_key]["dataset"][count_key])
        split_dir = root / "trajectories" / family / split
        key = f"{family}_{split}"
        report["splits"][key] = inspect_split(
            split_dir,
            expected_count=expected,
            expected_steps=steps,
            expected_render_hw=render_hw,
        )
        for path in sorted(split_dir.glob("*.npz")):
            with np.load(path) as data:
                commands_by_traj.append(np.asarray(data["commands"]))

    teacher_check = verify_not_uniform_random_actions(
        commands_by_traj,
        force_min=float(config["simulation"]["control_min_N"]),
        force_max=float(config["simulation"]["control_max_N"]),
    )
    report["teacher_not_uniform_random"] = teacher_check

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
    teacher_ok = bool(teacher_check.get("pass"))
    report["pass"] = {
        "all_splits": split_ok,
        "no_train_test_id_overlap": overlap_ok,
        "command_normalizer_nondegenerate": norm_ok,
        "teacher_not_uniform_random": teacher_ok,
    }
    report["overall_pass"] = bool(split_ok and overlap_ok and norm_ok and teacher_ok)
    return report

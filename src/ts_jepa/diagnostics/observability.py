"""Read-only observability metrics for RGB / κ-context collision analysis."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def hash_frame(frame: np.ndarray) -> str:
    arr = np.ascontiguousarray(frame)
    h = hashlib.sha1()
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()


def raw_context_frame_indices(time_index: int, kappa: int, num_frames: int) -> list[int]:
    start = max(0, time_index - kappa + 1)
    indices = list(range(start, time_index + 1))
    while len(indices) < kappa:
        indices.insert(0, indices[0])
    return [min(max(0, t), num_frames - 1) for t in indices]


def hash_raw_context(frames: np.ndarray, time_index: int, kappa: int) -> str:
    h = hashlib.sha1()
    for t in raw_context_frame_indices(time_index, kappa, frames.shape[0]):
        arr = np.ascontiguousarray(frames[t])
        h.update(np.array(arr.shape, dtype=np.int64).tobytes())
        h.update(arr.tobytes())
    return h.hexdigest()


def consecutive_pixel_abs_diff(frames: np.ndarray) -> np.ndarray:
    if frames.shape[0] < 2:
        return np.array([], dtype=np.float64)
    a = frames[:-1].astype(np.float64)
    b = frames[1:].astype(np.float64)
    return np.mean(np.abs(b - a), axis=(1, 2, 3))


def _stats(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "median": None, "max": None, "min": None}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "max": float(a.max()),
        "min": float(a.min()),
    }


def _collision_group_stats(
    groups: dict[str, list[dict[str, Any]]],
    *,
    max_representatives: int = 10,
) -> dict[str, Any]:
    multi = {h: items for h, items in groups.items() if len(items) > 1}
    groups_multi_command = 0
    groups_multi_state = 0
    command_spans: list[float] = []
    state_spans: list[float] = []
    samples_in_diff_command_groups = 0
    representative: list[dict[str, Any]] = []

    for h, items in sorted(multi.items(), key=lambda kv: -len(kv[1])):
        st = np.asarray([it["state"] for it in items], dtype=np.float64)
        cmds = np.asarray([it["command"] for it in items], dtype=np.float64)
        uniq_states = {tuple(np.round(s, 8)) for s in st}
        uniq_cmds = np.unique(cmds)
        state_span = float(np.linalg.norm(st.max(axis=0) - st.min(axis=0)))
        cmd_span = float(uniq_cmds.max() - uniq_cmds.min()) if uniq_cmds.size else 0.0
        state_spans.append(state_span)
        if len(uniq_states) > 1:
            groups_multi_state += 1
        if uniq_cmds.size > 1:
            groups_multi_command += 1
            command_spans.append(cmd_span)
            samples_in_diff_command_groups += len(items)
        entry = {
            "hash_prefix": h[:12],
            "num_samples": len(items),
            "num_unique_states_8dp": len(uniq_states),
            "num_unique_commands": int(uniq_cmds.size),
            "state_l2_span": state_span,
            "command_span_N": cmd_span,
            "command_unique_values": [float(c) for c in uniq_cmds[:20]],
        }
        if len(representative) < max_representatives:
            representative.append(entry)

    return {
        "num_collision_groups": len(multi),
        "groups_with_multiple_distinct_states_8dp": groups_multi_state,
        "groups_with_multiple_commands": groups_multi_command,
        "samples_in_groups_with_multiple_commands": samples_in_diff_command_groups,
        "state_l2_span_stats": _stats(state_spans),
        "command_span_stats_within_multi_command_groups": _stats(command_spans),
        "fraction_collision_groups_multi_command": float(
            groups_multi_command / max(len(multi), 1)
        ),
        "representative_groups": representative,
    }


def analyze_condition_observability(
    condition_dir: Path,
    *,
    label: str,
    kappa_values: list[int] | tuple[int, ...] = (2, 4),
) -> dict[str, Any]:
    traj_dir = condition_dir / "trajectories"
    files = sorted(traj_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No trajectories under {traj_dir}")

    all_frame_hashes: list[str] = []
    per_traj_unique_frames: list[int] = []
    per_traj_total_frames: list[int] = []
    pixel_diffs: list[float] = []
    state_l2: list[float] = []
    zero_pixel_pairs = 0
    total_pairs = 0

    kappa_results: dict[str, dict[str, Any]] = {}
    context_groups_by_kappa: dict[int, dict[str, list[dict[str, Any]]]] = {
        int(k): defaultdict(list) for k in kappa_values
    }

    manifest_path = condition_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    stride = None
    dt_s = None

    for path in files:
        with np.load(path) as data:
            frames = np.asarray(data["frames"])
            states = np.asarray(data["states"], dtype=np.float64)
            commands = np.asarray(data["commands"], dtype=np.float64)
            traj_id = int(np.asarray(data["trajectory_index"])) if "trajectory_index" in data else int(path.stem)
            if stride is None and "observation_stride" in data:
                stride = int(np.asarray(data["observation_stride"]))
            if dt_s is None and "dt_s" in data:
                dt_s = float(np.asarray(data["dt_s"]))

        frame_hashes = [hash_frame(frames[t]) for t in range(frames.shape[0])]
        all_frame_hashes.extend(frame_hashes)
        per_traj_unique_frames.append(len(set(frame_hashes)))
        per_traj_total_frames.append(len(frame_hashes))

        diffs = consecutive_pixel_abs_diff(frames)
        for d in diffs:
            pixel_diffs.append(float(d))
            total_pairs += 1
            if d == 0.0:
                zero_pixel_pairs += 1

        for t in range(states.shape[0] - 1):
            delta = states[t + 1] - states[t]
            state_l2.append(float(np.linalg.norm(delta)))

        for kappa in kappa_values:
            k = int(kappa)
            ctx_hashes: list[str] = []
            for t in range(frames.shape[0]):
                ctx_h = hash_raw_context(frames, t, k)
                ctx_hashes.append(ctx_h)
                context_groups_by_kappa[k][ctx_h].append(
                    {
                        "trajectory_index": traj_id,
                        "t": t,
                        "state": states[t].tolist(),
                        "command": float(commands[t]),
                    }
                )

            per_traj_unique_ctx = len(set(ctx_hashes))
            k_key = f"kappa_{k}"
            if k_key not in kappa_results:
                kappa_results[k_key] = {
                    "kappa": k,
                    "per_trajectory_unique_contexts": [],
                    "total_contexts": 0,
                    "all_context_hashes": [],
                }
            kappa_results[k_key]["per_trajectory_unique_contexts"].append(per_traj_unique_ctx)
            kappa_results[k_key]["total_contexts"] += len(ctx_hashes)
            kappa_results[k_key]["all_context_hashes"].extend(ctx_hashes)

    frame_counts = Counter(all_frame_hashes)
    n_frames = len(all_frame_hashes)
    n_unique_frames = len(frame_counts)

    pixel_arr = np.asarray(pixel_diffs, dtype=np.float64)
    state_arr = np.asarray(state_l2, dtype=np.float64)
    if pixel_arr.size >= 2 and state_arr.size == pixel_arr.size and np.std(pixel_arr) > 0 and np.std(state_arr) > 0:
        corr = float(np.corrcoef(state_arr, pixel_arr)[0, 1])
    else:
        corr = float("nan")

    # Single-frame collision groups (duplicate raw RGB)
    frame_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in files:
        with np.load(path) as data:
            frames = np.asarray(data["frames"])
            states = np.asarray(data["states"], dtype=np.float64)
            commands = np.asarray(data["commands"], dtype=np.float64)
            traj_id = int(np.asarray(data["trajectory_index"])) if "trajectory_index" in data else int(path.stem)
        for t, frame in enumerate(frames):
            frame_groups[hash_frame(frame)].append(
                {
                    "trajectory_index": traj_id,
                    "t": t,
                    "state": states[t].tolist(),
                    "command": float(commands[t]),
                }
            )

    for k_key, payload in kappa_results.items():
        k = int(payload["kappa"])
        ctx_counts = Counter(payload.pop("all_context_hashes"))
        total_ctx = int(payload["total_contexts"])
        unique_ctx = len(ctx_counts)
        payload["unique_context_ratio"] = float(unique_ctx / max(total_ctx, 1))
        payload["duplicate_context_fraction"] = float(1.0 - unique_ctx / max(total_ctx, 1))
        payload["exact_unique_context_hashes"] = unique_ctx
        payload["mean_unique_contexts_per_trajectory"] = float(
            np.mean(payload["per_trajectory_unique_contexts"])
        )
        payload["median_unique_contexts_per_trajectory"] = float(
            np.median(payload["per_trajectory_unique_contexts"])
        )
        payload["context_collisions"] = _collision_group_stats(context_groups_by_kappa[k])

    return {
        "label": label,
        "condition_dir": str(condition_dir),
        "manifest": manifest,
        "num_trajectories": len(files),
        "inferred_observation_stride": stride,
        "inferred_dt_s": dt_s,
        "raw_frame_diversity": {
            "total_frames": n_frames,
            "exact_unique_frame_hashes": n_unique_frames,
            "unique_frame_ratio": float(n_unique_frames / max(n_frames, 1)),
            "duplicate_fraction": float(1.0 - n_unique_frames / max(n_frames, 1)),
            "largest_duplicate_group": int(max(frame_counts.values()) if frame_counts else 0),
            "mean_unique_frames_per_trajectory": float(np.mean(per_traj_unique_frames)),
            "median_unique_frames_per_trajectory": float(np.median(per_traj_unique_frames)),
        },
        "consecutive_frame_change": {
            "num_pairs": total_pairs,
            "mean_abs_pixel_difference": float(np.mean(pixel_diffs)) if pixel_diffs else None,
            "median_abs_pixel_difference": float(np.median(pixel_diffs)) if pixel_diffs else None,
            "max_abs_pixel_difference": float(np.max(pixel_diffs)) if pixel_diffs else None,
            "fraction_exact_zero_pixel_difference": float(zero_pixel_pairs / max(total_pairs, 1)),
            "fraction_changed": float((total_pairs - zero_pixel_pairs) / max(total_pairs, 1)),
        },
        "physical_state_change": {
            "num_pairs": len(state_l2),
            "mean_state_l2": float(np.mean(state_l2)) if state_l2 else None,
            "median_state_l2": float(np.median(state_l2)) if state_l2 else None,
            "max_state_l2": float(np.max(state_l2)) if state_l2 else None,
        },
        "duplicate_rgb_collisions": _collision_group_stats(frame_groups),
        "kappa_context_analysis": kappa_results,
        "rasterization_sensitivity": {
            "corr_consecutive_state_l2_vs_mean_abs_pixel_diff": corr,
        },
    }

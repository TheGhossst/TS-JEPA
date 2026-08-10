#!/usr/bin/env python
"""
Read-only observation-diversity diagnostic for sampling-interval ablation.

Compares data_sampling_ablation/{1ms,2ms,5ms,10ms}/.
For κ-context collision metrics use scripts/diagnose_observation_observability.py.
Does NOT touch data/, data_dp_fixed/, checkpoints, or training.
Does NOT train JEPA or the actor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _hash_frame(frame: np.ndarray) -> str:
    arr = np.ascontiguousarray(frame)
    h = hashlib.sha1()
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()


def _consecutive_pixel_abs_diff(frames: np.ndarray) -> np.ndarray:
    if frames.shape[0] < 2:
        return np.array([], dtype=np.float64)
    a = frames[:-1].astype(np.float64)
    b = frames[1:].astype(np.float64)
    return np.mean(np.abs(b - a), axis=(1, 2, 3))


def _diagnose_condition(condition_dir: Path, label: str) -> dict[str, Any]:
    traj_dir = condition_dir / "trajectories"
    files = sorted(traj_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No trajectories under {traj_dir}")

    all_hashes: list[str] = []
    per_traj_unique: list[int] = []
    per_traj_total: list[int] = []
    pixel_diffs: list[float] = []
    state_l2: list[float] = []
    abs_dx: list[float] = []
    abs_dtheta: list[float] = []
    zero_pixel_pairs = 0
    changed_pixel_pairs = 0
    total_pairs = 0

    # hash -> list of (traj_id, t, state, command)
    hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    manifest_path = condition_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    for path in files:
        with np.load(path) as data:
            frames = np.asarray(data["frames"])
            states = np.asarray(data["states"], dtype=np.float64)
            commands = np.asarray(data["commands"], dtype=np.float64)
            traj_id = int(np.asarray(data["trajectory_index"])) if "trajectory_index" in data else int(path.stem)
            stride = int(np.asarray(data["observation_stride"])) if "observation_stride" in data else None
            dt_s = float(np.asarray(data["dt_s"])) if "dt_s" in data else None

        hashes = [_hash_frame(frames[t]) for t in range(frames.shape[0])]
        all_hashes.extend(hashes)
        per_traj_unique.append(len(set(hashes)))
        per_traj_total.append(len(hashes))

        diffs = _consecutive_pixel_abs_diff(frames)
        for d in diffs:
            pixel_diffs.append(float(d))
            total_pairs += 1
            if d == 0.0:
                zero_pixel_pairs += 1
            else:
                changed_pixel_pairs += 1

        for t in range(states.shape[0] - 1):
            delta = states[t + 1] - states[t]
            state_l2.append(float(np.linalg.norm(delta)))
            abs_dx.append(float(abs(delta[0])))
            abs_dtheta.append(float(abs(delta[2])))

        for t, h in enumerate(hashes):
            hash_groups[h].append(
                {
                    "trajectory_index": traj_id,
                    "t": t,
                    "state": states[t].tolist(),
                    "command": float(commands[t]),
                }
            )

    counts = Counter(all_hashes)
    n_frames = len(all_hashes)
    n_unique = len(counts)
    largest = max(counts.values()) if counts else 0

    # Duplicate-frame groups with multiple physical states / commands
    multi_sample_groups = {h: items for h, items in hash_groups.items() if len(items) > 1}
    groups_multi_state = 0
    groups_multi_command = 0
    state_spans: list[float] = []
    command_spans: list[dict[str, Any]] = []
    representative = []

    for h, items in sorted(multi_sample_groups.items(), key=lambda kv: -len(kv[1])):
        st = np.asarray([it["state"] for it in items], dtype=np.float64)
        cmds = np.asarray([it["command"] for it in items], dtype=np.float64)
        uniq_states = {tuple(np.round(s, 8)) for s in st}
        uniq_cmds = np.unique(cmds)
        span = float(np.linalg.norm(st.max(axis=0) - st.min(axis=0)))
        state_spans.append(span)
        if len(uniq_states) > 1:
            groups_multi_state += 1
        if uniq_cmds.size > 1:
            groups_multi_command += 1
        command_spans.append(
            {
                "hash_prefix": h[:12],
                "num_samples": len(items),
                "num_unique_states_8dp": len(uniq_states),
                "num_unique_commands": int(uniq_cmds.size),
                "state_l2_span": span,
                "command_unique_values": [float(c) for c in uniq_cmds],
            }
        )
        if len(representative) < 10:
            representative.append(command_spans[-1])

    # Rasterization sensitivity: corr(|Δstate|, pixel diff) over consecutive pairs
    pixel_arr = np.asarray(pixel_diffs, dtype=np.float64)
    state_arr = np.asarray(state_l2, dtype=np.float64)
    if pixel_arr.size >= 2 and state_arr.size == pixel_arr.size and np.std(pixel_arr) > 0 and np.std(state_arr) > 0:
        corr = float(np.corrcoef(state_arr, pixel_arr)[0, 1])
    else:
        corr = float("nan")

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

    return {
        "label": label,
        "condition_dir": str(condition_dir),
        "manifest": manifest,
        "num_trajectories": len(files),
        "total_frames": n_frames,
        "raw_frame_diversity": {
            "total_frames": n_frames,
            "exact_unique_frame_hashes": n_unique,
            "unique_frame_ratio": float(n_unique / max(n_frames, 1)),
            "duplicate_fraction": float(1.0 - n_unique / max(n_frames, 1)),
            "largest_duplicate_group": int(largest),
            "mean_unique_frames_per_trajectory": float(np.mean(per_traj_unique)),
            "median_unique_frames_per_trajectory": float(np.median(per_traj_unique)),
            "mean_frames_per_trajectory": float(np.mean(per_traj_total)),
        },
        "consecutive_frame_change": {
            "num_pairs": total_pairs,
            "mean_abs_pixel_difference": float(np.mean(pixel_diffs)) if pixel_diffs else None,
            "median_abs_pixel_difference": float(np.median(pixel_diffs)) if pixel_diffs else None,
            "max_abs_pixel_difference": float(np.max(pixel_diffs)) if pixel_diffs else None,
            "fraction_exact_zero_pixel_difference": float(zero_pixel_pairs / max(total_pairs, 1)),
            "fraction_changed": float(changed_pixel_pairs / max(total_pairs, 1)),
        },
        "physical_state_change": {
            "num_pairs": len(state_l2),
            "mean_state_l2": float(np.mean(state_l2)) if state_l2 else None,
            "median_state_l2": float(np.median(state_l2)) if state_l2 else None,
            "max_state_l2": float(np.max(state_l2)) if state_l2 else None,
            "mean_abs_dx": float(np.mean(abs_dx)) if abs_dx else None,
            "mean_abs_dtheta": float(np.mean(abs_dtheta)) if abs_dtheta else None,
            "max_abs_dx": float(np.max(abs_dx)) if abs_dx else None,
            "max_abs_dtheta": float(np.max(abs_dtheta)) if abs_dtheta else None,
        },
        "duplicate_rgb_to_state_command": {
            "num_duplicate_frame_groups": len(multi_sample_groups),
            "groups_with_multiple_distinct_states_8dp": groups_multi_state,
            "groups_with_multiple_commands": groups_multi_command,
            "state_l2_span_stats": _stats(state_spans),
            "fraction_duplicate_groups_multi_state": float(
                groups_multi_state / max(len(multi_sample_groups), 1)
            ),
            "fraction_duplicate_groups_multi_command": float(
                groups_multi_command / max(len(multi_sample_groups), 1)
            ),
            "representative_groups": representative,
        },
        "rasterization_sensitivity": {
            "corr_consecutive_state_l2_vs_mean_abs_pixel_diff": corr,
            "note": (
                "Correlation across consecutive observation pairs within trajectories. "
                "Higher values mean pixel change tracks physical motion better."
            ),
        },
        "inferred_stride": stride if files else None,
        "inferred_dt_s": dt_s if files else None,
    }


def _criterion_notes(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Transparent, non-arbitrary reporting aids — not pass/fail declarations."""
    baseline = results.get("1ms", {})
    base_dup = baseline.get("raw_frame_diversity", {}).get("duplicate_fraction")
    base_zero = baseline.get("consecutive_frame_change", {}).get("fraction_exact_zero_pixel_difference")
    base_unique = baseline.get("raw_frame_diversity", {}).get("unique_frame_ratio")

    comparisons = {}
    for label, rep in results.items():
        rd = rep["raw_frame_diversity"]
        cf = rep["consecutive_frame_change"]
        dup = rd["duplicate_fraction"]
        zero = cf["fraction_exact_zero_pixel_difference"]
        uniq = rd["unique_frame_ratio"]
        comparisons[label] = {
            "unique_frame_ratio": uniq,
            "duplicate_fraction": dup,
            "fraction_exact_zero_consecutive": zero,
            "delta_duplicate_fraction_vs_1ms": None
            if base_dup is None
            else float(dup - base_dup),
            "delta_zero_consecutive_vs_1ms": None
            if base_zero is None
            else float(zero - base_zero),
            "unique_ratio_relative_to_1ms": None
            if not base_unique
            else float(uniq / base_unique),
            # Transparent heuristic flags (informational only):
            "flag_duplicate_fraction_below_0_5": bool(dup < 0.5),
            "flag_zero_consecutive_below_0_2": bool(zero < 0.2),
            "flag_unique_ratio_at_least_3x_1ms": bool(base_unique and uniq >= 3.0 * base_unique),
        }

    # Rank by unique ratio then by lower zero-consecutive fraction
    ranked = sorted(
        results.keys(),
        key=lambda k: (
            results[k]["raw_frame_diversity"]["unique_frame_ratio"],
            -results[k]["consecutive_frame_change"]["fraction_exact_zero_pixel_difference"],
        ),
        reverse=True,
    )
    return {
        "ranking_by_unique_frame_ratio": ranked,
        "best_observation_diversity_by_unique_ratio": ranked[0] if ranked else None,
        "comparisons_vs_1ms": comparisons,
        "criterion_definition": {
            "substantially_better_flags": [
                "duplicate_fraction < 0.5",
                "fraction_exact_zero_consecutive < 0.2",
                "unique_frame_ratio >= 3x the 1ms baseline",
            ],
            "note": (
                "Flags are transparent reporting aids only; they do not declare experiment success. "
                "Decide from the raw tables."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="data_sampling_ablation")
    parser.add_argument(
        "--intervals-ms",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10],
    )
    parser.add_argument("--out", type=str, default="runs/eval/sampling_interval_ablation.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out

    results: dict[str, Any] = {}
    for ms in args.intervals_ms:
        label = f"{int(ms)}ms"
        condition_dir = data_root / label
        print(f"Diagnosing {condition_dir} ...")
        results[label] = _diagnose_condition(condition_dir, label)

    notes = _criterion_notes(results)

    # Hypothesis support language from numbers only
    r1 = results["1ms"]["raw_frame_diversity"]["duplicate_fraction"]
    r5 = results.get("5ms", {}).get("raw_frame_diversity", {}).get("duplicate_fraction")
    r10 = results.get("10ms", {}).get("raw_frame_diversity", {}).get("duplicate_fraction")
    r20 = results.get("20ms", {}).get("raw_frame_diversity", {}).get("duplicate_fraction")
    z1 = results["1ms"]["consecutive_frame_change"]["fraction_exact_zero_pixel_difference"]

    aliasing_supported = bool(r1 is not None and r1 >= 0.5 and z1 is not None and z1 >= 0.5)
    summary = {
        "best_observation_diversity": notes["best_observation_diversity_by_unique_ratio"],
        "ranking_by_unique_frame_ratio": notes["ranking_by_unique_frame_ratio"],
        "duplicate_fractions": {
            k: results[k]["raw_frame_diversity"]["duplicate_fraction"] for k in results
        },
        "zero_consecutive_fractions": {
            k: results[k]["consecutive_frame_change"]["fraction_exact_zero_pixel_difference"]
            for k in results
        },
        "flags": notes["comparisons_vs_1ms"],
        "hypothesis_1ms_causes_observation_aliasing": {
            "supported_by_1ms_numbers": aliasing_supported,
            "1ms_duplicate_fraction": r1,
            "1ms_zero_consecutive_fraction": z1,
            "interpretation": (
                "Supported if 1ms shows high duplicate-frame fraction and many exact "
                "zero consecutive pixel differences while physical state still moves."
            ),
        },
        "does_increasing_interval_reduce_aliasing": {
            "5ms_duplicate_fraction": r5,
            "10ms_duplicate_fraction": r10,
            "20ms_duplicate_fraction": r20,
            "note": "Compare duplicate fractions and zero-consecutive fractions in the tables; do not treat flags as pass/fail.",
        },
    }

    report = {
        "experiment": "sampling_interval_ablation",
        "data_root": str(data_root),
        "protected_paths_untouched": ["data/", "data_dp_fixed/"],
        "results": results,
        "comparison_notes": notes,
        "summary": summary,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=== Sampling interval ablation summary ===")
    for label, rep in results.items():
        rd = rep["raw_frame_diversity"]
        cf = rep["consecutive_frame_change"]
        print(
            f"{label}: frames={rd['total_frames']} unique={rd['exact_unique_frame_hashes']} "
            f"uniq_ratio={rd['unique_frame_ratio']:.4f} dup_frac={rd['duplicate_fraction']:.4f} "
            f"zero_consec={cf['fraction_exact_zero_pixel_difference']:.4f} "
            f"mean_|dpixel|={cf['mean_abs_pixel_difference']}"
        )
    print(f"best by unique ratio: {summary['best_observation_diversity']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

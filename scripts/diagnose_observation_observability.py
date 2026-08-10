#!/usr/bin/env python
"""
Read-only observability diagnostic for sampling-interval and kappa ablations.

For each condition (e.g. data_sampling_ablation/5ms/) and each kappa in {2, 4}:
  - unique raw frames / trajectory
  - mean consecutive pixel difference
  - fraction of consecutive identical frames
  - unique κ-frame raw contexts
  - same-observation / different-command collision counts (single-frame and κ-context)
  - command spread within collision groups

Does NOT train JEPA, touch data/, data_dp_fixed/, or existing checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ts_jepa.diagnostics.plateau import (
    evaluate_success_gates,
    extrapolate_unique_frame_ratio,
    interpolate_threshold_crossing,
    observation_cadence_table,
    rate_of_change_analysis,
)
from ts_jepa.diagnostics.observability import analyze_condition_observability


DEFAULT_INTERVALS_MS = (1, 2, 5, 10, 20, 30, 35, 40)
DEFAULT_KAPPA = (2, 4)


def _zero_consecutive_crossing_estimate(
    intervals_ms: list[int],
    values: list[float],
    *,
    target: float,
) -> dict[str, Any]:
    """Estimate when a decreasing metric reaches target (first value <= target)."""
    for ms, val in zip(intervals_ms, values):
        if val <= target:
            return {
                "reached_at_or_below_tested_interval_ms": int(ms),
                "value": float(val),
                "method": "exact_at_tested_point",
            }
    for i in range(len(intervals_ms) - 1):
        x0, x1 = float(intervals_ms[i]), float(intervals_ms[i + 1])
        y0, y1 = float(values[i]), float(values[i + 1])
        if y0 > target >= y1 and y0 != y1:
            frac = (y0 - target) / (y0 - y1)
            crossing = x0 + frac * (x1 - x0)
            return {
                "bracket_interval_ms": [int(intervals_ms[i]), int(intervals_ms[i + 1])],
                "values_at_bracket": [y0, y1],
                "interpolated_crossing_ms": float(crossing),
                "method": "linear_between_bracket",
            }
    # extrapolate linearly from last segment if still above target
    if len(intervals_ms) >= 2:
        x0, x1 = float(intervals_ms[-2]), float(intervals_ms[-1])
        y0, y1 = float(values[-2]), float(values[-1])
        if y1 != y0 and y1 > target:
            slope = (y1 - y0) / (x1 - x0)
            if slope < 0:
                extrap = x1 + (target - y1) / slope
                return {
                    "last_tested_interval_ms": int(intervals_ms[-1]),
                    "last_tested_value": float(y1),
                    "linear_extrapolation_ms": float(extrap),
                    "method": "linear_extrapolation_beyond_last_segment",
                    "note": "zero_consecutive still far above target at last tested interval",
                }
    return {
        "reached_within_tested_range": False,
        "last_tested_interval_ms": int(intervals_ms[-1]),
        "last_tested_value": float(values[-1]),
        "margin_above_target": float(values[-1] - target),
        "method": "not_reached_in_range",
    }


def _comparison_table(results: dict[str, dict[str, Any]], kappa: int) -> dict[str, Any]:
    k_key = f"kappa_{kappa}"
    rows = {}
    for label, rep in results.items():
        rd = rep["raw_frame_diversity"]
        cf = rep["consecutive_frame_change"]
        dup_rgb = rep["duplicate_rgb_collisions"]
        kctx = rep["kappa_context_analysis"][k_key]
        ctx_col = kctx["context_collisions"]
        rows[label] = {
            "unique_frame_ratio": rd["unique_frame_ratio"],
            "duplicate_frame_fraction": rd["duplicate_fraction"],
            "fraction_zero_consecutive_pixels": cf["fraction_exact_zero_pixel_difference"],
            "mean_abs_consecutive_pixel_diff": cf["mean_abs_pixel_difference"],
            "single_frame_collision_groups_multi_command": dup_rgb["groups_with_multiple_commands"],
            "single_frame_samples_in_diff_command_groups": dup_rgb["samples_in_groups_with_multiple_commands"],
            "unique_context_ratio": kctx["unique_context_ratio"],
            "context_collision_groups_multi_command": ctx_col["groups_with_multiple_commands"],
            "context_samples_in_diff_command_groups": ctx_col["samples_in_groups_with_multiple_commands"],
            "max_command_span_in_context_collision_groups": ctx_col["command_span_stats_within_multi_command_groups"]["max"],
        }
    return rows


def _rank_conditions(results: dict[str, dict[str, Any]], kappa: int) -> list[str]:
    k_key = f"kappa_{kappa}"

    def score(label: str) -> tuple[float, float, float]:
        rep = results[label]
        ctx = rep["kappa_context_analysis"][k_key]["context_collisions"]
        return (
            rep["kappa_context_analysis"][k_key]["unique_context_ratio"],
            -ctx["samples_in_groups_with_multiple_commands"],
            -rep["consecutive_frame_change"]["fraction_exact_zero_pixel_difference"],
        )

    return sorted(results.keys(), key=score, reverse=True)


def _answers_for_condition(rep: dict[str, Any], kappa: int) -> dict[str, str]:
    k_key = f"kappa_{kappa}"
    rd = rep["raw_frame_diversity"]
    cf = rep["consecutive_frame_change"]
    kctx = rep["kappa_context_analysis"][k_key]
    ctx_col = kctx["context_collisions"]

    visually_distinct = rd["unique_frame_ratio"] > 0.5 and cf["fraction_exact_zero_pixel_difference"] < 0.5
    fewer_collisions = (
        ctx_col["groups_with_multiple_commands"] < rep["duplicate_rgb_collisions"]["groups_with_multiple_commands"]
        or ctx_col["samples_in_groups_with_multiple_commands"]
        < rep["duplicate_rgb_collisions"]["samples_in_groups_with_multiple_commands"]
    )
    temporal_info = kctx["unique_context_ratio"] > rd["unique_frame_ratio"] * 0.9

    return {
        "A_physically_different_states_visually_distinguishable": (
            "yes" if visually_distinct else "no / marginal"
        ),
        "B_raw_and_context_collisions_decreased_vs_baseline": "compare tables",
        "C_same_observation_different_command_collisions": (
            f"{ctx_col['groups_with_multiple_commands']} context groups, "
            f"{ctx_col['samples_in_groups_with_multiple_commands']} samples"
        ),
        "D_temporal_information_potentially_sufficient": (
            "promising" if temporal_info and fewer_collisions else "unlikely at this setting"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_baseline.yaml")
    parser.add_argument("--data-root", type=str, default="data_sampling_ablation")
    parser.add_argument(
        "--intervals-ms",
        type=int,
        nargs="+",
        default=list(DEFAULT_INTERVALS_MS),
    )
    parser.add_argument(
        "--kappa",
        type=int,
        nargs="+",
        default=list(DEFAULT_KAPPA),
        help="Kappa values for raw κ-context collision analysis (no JEPA required).",
    )
    parser.add_argument("--out", type=str, default="runs/eval/observation_observability.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out

    from ts_jepa.config import load_config

    config = load_config(args.config)
    sim = config["simulation"]
    dt_s = float(sim["dt"])
    physics_budget = int(sim["trajectory_steps"])
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])

    kappa_values = [int(k) for k in args.kappa]
    intervals_ms = [int(ms) for ms in args.intervals_ms]
    results: dict[str, Any] = {}
    for ms in intervals_ms:
        label = f"{int(ms)}ms"
        condition_dir = data_root / label
        print(f"Analyzing {condition_dir} (kappa={kappa_values}) ...")
        results[label] = analyze_condition_observability(
            condition_dir,
            label=label,
            kappa_values=kappa_values,
        )

    baseline = results.get("1ms")
    summary: dict[str, Any] = {
        "recommended_first_test": None,
        "ranking_kappa_2": _rank_conditions(results, 2) if results else [],
        "ranking_kappa_4": _rank_conditions(results, 4) if 4 in kappa_values and results else [],
        "comparison_kappa_2": _comparison_table(results, 2),
        "comparison_kappa_4": _comparison_table(results, 4) if 4 in kappa_values else None,
        "per_condition_answers": {
            label: {f"kappa_{k}": _answers_for_condition(rep, k) for k in kappa_values}
            for label, rep in results.items()
        },
    }

    if summary["ranking_kappa_2"]:
        best = summary["ranking_kappa_2"][0]
        if best != "1ms":
            summary["recommended_first_test"] = {
                "sampling_interval": best,
                "kappa": 2,
                "reason": (
                    "Highest unique kappa=2 context ratio with fewest same-context/different-command "
                    "collisions among tested intervals; retrain JEPA only after this passes observability gates."
                ),
            }
        else:
            summary["recommended_first_test"] = {
                "sampling_interval": "5ms or 10ms",
                "kappa": 2,
                "reason": (
                    "1ms still ranks best among tested intervals — try coarser cadence first; "
                    "kappa=4 is secondary (needs JEPA retrain) and cannot fix identical single frames."
                ),
            }

    if baseline:
        b2 = baseline["kappa_context_analysis"]["kappa_2"]["context_collisions"]
        summary["baseline_1ms_kappa2_context_collisions"] = {
            "groups_multi_command": b2["groups_with_multiple_commands"],
            "samples_in_diff_command_groups": b2["samples_in_groups_with_multiple_commands"],
        }

    summary["observation_cadence_table"] = observation_cadence_table(
        intervals_ms,
        dt_s=dt_s,
        physics_budget=physics_budget,
        kp=kp,
    )
    summary["rate_of_change_analysis"] = rate_of_change_analysis(
        intervals_ms,
        summary["comparison_kappa_2"],
    )
    summary["unique_frame_ratio_extrapolation"] = extrapolate_unique_frame_ratio(
        intervals_ms,
        summary["comparison_kappa_2"],
    )
    summary["success_gate_evaluation"] = evaluate_success_gates(
        intervals_ms,
        summary["comparison_kappa_2"],
    )
    ufr_series = [
        summary["comparison_kappa_2"][f"{ms}ms"]["unique_frame_ratio"] for ms in intervals_ms
    ]
    zc_series = [
        summary["comparison_kappa_2"][f"{ms}ms"]["fraction_zero_consecutive_pixels"]
        for ms in intervals_ms
    ]
    summary["empirical_threshold_crossings"] = {
        "unique_frame_ratio_0_5": interpolate_threshold_crossing(
            intervals_ms, ufr_series, threshold=0.5
        ),
        "zero_consecutive_reaches_0_2": _zero_consecutive_crossing_estimate(
            intervals_ms, zc_series, target=0.2
        ),
    }

    report = {
        "experiment": "observation_observability_ablation",
        "data_root": str(data_root),
        "kappa_values": kappa_values,
        "protected_paths_untouched": ["data/", "data_dp_fixed/"],
        "success_criteria": {
            "unique_frame_ratio": "> 0.5 (target)",
            "fraction_zero_consecutive_pixels": "< 0.2 (target)",
            "context_collision_samples_multi_command": "substantially below 1ms κ=2 baseline",
            "unique_context_ratio": ">= unique_frame_ratio (kappa>1 should not reduce diversity)",
            "note": "Meeting these is necessary but not sufficient for actor success; retrain JEPA after passing.",
        },
        "results": results,
        "summary": summary,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=== Observation observability summary ===")
    for ms in intervals_ms:
        label = f"{ms}ms"
        rep = results[label]
        rd = rep["raw_frame_diversity"]
        cf = rep["consecutive_frame_change"]
        k2 = rep["kappa_context_analysis"]["kappa_2"]
        col = k2["context_collisions"]
        print(
            f"{label}: uniq_frame={rd['unique_frame_ratio']:.3f} "
            f"zero_consec={cf['fraction_exact_zero_pixel_difference']:.3f} "
            f"uniq_ctx_k2={k2['unique_context_ratio']:.3f} "
            f"ctx_coll_samples={col['samples_in_groups_with_multiple_commands']}"
        )
    gates = summary["success_gate_evaluation"]
    print(f"any interval passes all gates: {gates['any_interval_passes_all_gates']}")
    print(f"closest interval: {gates['closest_interval']['interval']} ({gates['closest_interval']['gates_passed']}/4 gates)")
    extrap = summary["unique_frame_ratio_extrapolation"]["consensus"]
    print(f"extrapolation verdict: {extrap['verdict']} — {extrap.get('detail', '')}")
    print(f"recommended first test: {summary['recommended_first_test']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

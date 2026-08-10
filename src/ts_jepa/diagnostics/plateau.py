"""Trend / plateau analysis for observation-cadence diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from ts_jepa.env.sampling import resolve_observation_cadence

# JEPA needs time_index in [0, T - Kp - 1] => T >= Kp + 1
DEFAULT_KP = 15
DEFAULT_SUCCESS_GATES = {
    "unique_frame_ratio": {"op": "gt", "target": 0.5},
    "fraction_zero_consecutive_pixels": {"op": "lt", "target": 0.2},
    "unique_context_ratio_kappa_2": {"op": "gte_unique_frame_ratio", "target": None},
    "context_collision_samples_vs_1ms_baseline": {"op": "lt_fraction_of_baseline", "target": 0.1},
}

TRACKED_METRIC_KEYS = (
    "unique_frame_ratio",
    "fraction_zero_consecutive_pixels",
    "context_collision_samples",
    "max_command_span_in_collisions",
)


def observation_cadence_table(
    intervals_ms: list[int] | tuple[int, ...],
    *,
    dt_s: float = 0.001,
    physics_budget: int = 100,
    kp: int = DEFAULT_KP,
    allow_truncated_budget: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    min_obs_for_kp = int(kp) + 1
    for ms in intervals_ms:
        cadence = resolve_observation_cadence(
            ms,
            dt_s,
            physics_budget,
            allow_truncated_budget=allow_truncated_budget,
        )
        num_obs = int(cadence["num_observations"])
        rows.append(
            {
                "interval_ms": int(ms),
                "observation_stride": int(cadence["observation_stride"]),
                "num_observations_per_trajectory": num_obs,
                "physics_budget_nominal": int(cadence["physics_budget_nominal"]),
                "physics_budget_effective": int(cadence["physics_budget_effective"]),
                "physical_duration_s": float(cadence["physical_duration_s"]),
                "budget_truncated": bool(cadence["budget_truncated"]),
                "kp_feasible": num_obs >= min_obs_for_kp,
                "min_observations_required_for_kp": min_obs_for_kp,
                "observations_deficit_for_kp": max(0, min_obs_for_kp - num_obs),
            }
        )
    return rows


def _metric_series(comparison_kappa_2: dict[str, dict[str, Any]], intervals_ms: list[int]) -> dict[str, list[float | None]]:
    series: dict[str, list[float | None]] = {k: [] for k in TRACKED_METRIC_KEYS}
    for ms in intervals_ms:
        label = f"{ms}ms"
        row = comparison_kappa_2.get(label)
        if row is None:
            for key in TRACKED_METRIC_KEYS:
                series[key].append(None)
            continue
        series["unique_frame_ratio"].append(float(row["unique_frame_ratio"]))
        series["fraction_zero_consecutive_pixels"].append(float(row["fraction_zero_consecutive_pixels"]))
        series["context_collision_samples"].append(
            float(row["context_samples_in_diff_command_groups"])
        )
        series["max_command_span_in_collisions"].append(
            float(row["max_command_span_in_context_collision_groups"])
            if row["max_command_span_in_context_collision_groups"] is not None
            else None
        )
    return series


def rate_of_change_analysis(
    intervals_ms: list[int],
    comparison_kappa_2: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    series = _metric_series(comparison_kappa_2, intervals_ms)
    steps: list[dict[str, Any]] = []
    for i in range(len(intervals_ms) - 1):
        ms0, ms1 = intervals_ms[i], intervals_ms[i + 1]
        delta_ms = float(ms1 - ms0)
        step: dict[str, Any] = {
            "from_ms": ms0,
            "to_ms": ms1,
            "delta_interval_ms": delta_ms,
            "metrics": {},
        }
        for key in TRACKED_METRIC_KEYS:
            v0, v1 = series[key][i], series[key][i + 1]
            if v0 is None or v1 is None:
                step["metrics"][key] = {
                    "absolute_change": None,
                    "per_ms_change": None,
                }
                continue
            abs_change = float(v1 - v0)
            step["metrics"][key] = {
                "absolute_change": abs_change,
                "per_ms_change": float(abs_change / delta_ms) if delta_ms else None,
            }
        steps.append(step)

    # Simple plateau heuristic: last step per-ms gain vs first step per-ms gain
    plateau_notes: dict[str, str] = {}
    if len(steps) >= 2:
        for key in TRACKED_METRIC_KEYS:
            first = steps[0]["metrics"][key]["per_ms_change"]
            last = steps[-1]["metrics"][key]["per_ms_change"]
            if first is None or last is None:
                plateau_notes[key] = "insufficient data"
                continue
            if key in ("unique_frame_ratio",):
                if abs(last) < 0.25 * abs(first):
                    plateau_notes[key] = "flattening (last per-ms gain < 25% of first)"
                elif last > first:
                    plateau_notes[key] = "still accelerating"
                else:
                    plateau_notes[key] = "improving but decelerating"
            elif key == "fraction_zero_consecutive_pixels":
                # decreasing is good
                if abs(last) < 0.25 * abs(first):
                    plateau_notes[key] = "flattening (last per-ms gain < 25% of first)"
                else:
                    plateau_notes[key] = "still improving"
            elif key == "context_collision_samples":
                if abs(last) < 0.25 * abs(first):
                    plateau_notes[key] = "flattening (last per-ms reduction < 25% of first)"
                else:
                    plateau_notes[key] = "still improving"
            else:
                if abs(last) < 0.25 * abs(first):
                    plateau_notes[key] = "flattening"
                else:
                    plateau_notes[key] = "still improving"

    return {
        "intervals_ms": intervals_ms,
        "metric_series": series,
        "consecutive_steps": steps,
        "plateau_heuristic_notes": plateau_notes,
    }


def _fit_saturating_exponential(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Fit y = a * (1 - exp(-b*x)) with grid search over b (numpy only)."""
    if x.size < 2:
        return {"success": False, "reason": "need at least 2 points"}

    y = np.clip(y, 0.0, None)
    best: dict[str, Any] | None = None
    for b in np.logspace(-4, 0.5, 400):
        z = 1.0 - np.exp(-b * x)
        denom = float(np.dot(z, z))
        if denom <= 1e-12:
            continue
        a = float(np.dot(y, z) / denom)
        a = max(a, 1e-9)
        residual = float(np.sum((y - a * z) ** 2))
        if best is None or residual < best["residual_sum_squares"]:
            best = {
                "a": a,
                "b": float(b),
                "residual_sum_squares": residual,
            }
    if best is None:
        return {"success": False, "reason": "fit failed"}

    a = float(best["a"])
    b = float(best["b"])
    y_hat = a * (1.0 - np.exp(-b * x))
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    target = 0.5
    projected_ms: float | None
    reachable: bool
    if a <= target + 1e-6:
        projected_ms = None
        reachable = False
        note = f"fitted asymptote a={a:.4f} <= target {target}; unique-frame gate likely not reachable by cadence alone"
    else:
        frac = target / a
        if frac >= 1.0:
            projected_ms = None
            reachable = False
            note = "target above fitted asymptote"
        else:
            projected_ms = float(-np.log(1.0 - frac) / b)
            reachable = projected_ms <= 200.0
            note = (
                f"projected interval to reach unique_frame_ratio={target} is ~{projected_ms:.1f} ms"
                if reachable
                else f"projected crossing at ~{projected_ms:.1f} ms exceeds practical 200 ms diagnostic range"
            )

    return {
        "success": True,
        "model": "a * (1 - exp(-b * interval_ms))",
        "a_asymptote": a,
        "b_rate": b,
        "r_squared": r2,
        "residual_sum_squares": best["residual_sum_squares"],
        "target_unique_frame_ratio": target,
        "projected_interval_ms_for_target": projected_ms,
        "target_reachable_within_200ms": reachable,
        "interpretation": note,
    }


def _log_linear_extrapolation(x: np.ndarray, y: np.ndarray, *, target: float = 0.5) -> dict[str, Any]:
    eps = 1e-6
    y_pos = np.clip(y, eps, None)
    if np.any(x <= 0):
        return {"success": False, "reason": "log-linear requires positive intervals"}
    coeffs = np.polyfit(np.log(x), np.log(y_pos), 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    if slope <= 0:
        return {
            "success": True,
            "model": "log-linear on unique_frame_ratio",
            "slope_loglog": slope,
            "intercept_loglog": intercept,
            "projected_interval_ms_for_target": None,
            "interpretation": "log-linear slope <= 0; trend not increasing — target not reachable this way",
        }
    projected = float(np.exp((np.log(target) - intercept) / slope))
    return {
        "success": True,
        "model": "log-linear on unique_frame_ratio",
        "slope_loglog": slope,
        "intercept_loglog": intercept,
        "target_unique_frame_ratio": target,
        "projected_interval_ms_for_target": projected,
        "interpretation": f"log-linear extrapolation crosses {target} at ~{projected:.1f} ms",
    }


def extrapolate_unique_frame_ratio(
    intervals_ms: list[int],
    comparison_kappa_2: dict[str, dict[str, Any]],
    *,
    target: float = 0.5,
) -> dict[str, Any]:
    x = np.asarray(intervals_ms, dtype=np.float64)
    y = np.asarray(
        [comparison_kappa_2[f"{ms}ms"]["unique_frame_ratio"] for ms in intervals_ms],
        dtype=np.float64,
    )
    sat = _fit_saturating_exponential(x, y)
    loglin = _log_linear_extrapolation(x, y, target=target)
    return {
        "intervals_ms": intervals_ms,
        "observed_unique_frame_ratio": y.tolist(),
        "saturating_exponential_fit": sat,
        "log_linear_extrapolation": loglin,
        "consensus": _extrapolation_consensus(sat, loglin, target=target),
    }


def _extrapolation_consensus(sat: dict[str, Any], loglin: dict[str, Any], *, target: float) -> dict[str, Any]:
    projections: list[float] = []
    if sat.get("success") and sat.get("projected_interval_ms_for_target") is not None:
        projections.append(float(sat["projected_interval_ms_for_target"]))
    if loglin.get("success") and loglin.get("projected_interval_ms_for_target") is not None:
        projections.append(float(loglin["projected_interval_ms_for_target"]))

    if sat.get("success") and not sat.get("target_reachable_within_200ms", True):
        if sat.get("a_asymptote", 0.0) <= target:
            return {
                "verdict": "likely_asymptoting_below_target",
                "detail": sat.get("interpretation"),
                "fitted_asymptote_unique_frame_ratio": sat.get("a_asymptote"),
            }

    if not projections:
        return {
            "verdict": "indeterminate",
            "detail": "no finite crossing projected by either model",
        }

    lo, hi = min(projections), max(projections)
    return {
        "verdict": "cadence_increase_may_reach_target",
        "projected_interval_ms_range": [lo, hi],
        "detail": f"models bracket crossing of unique_frame_ratio={target} between ~{lo:.1f} ms and ~{hi:.1f} ms",
    }


def interpolate_threshold_crossing(
    intervals_ms: list[int],
    values: list[float],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Linear interpolation of threshold crossing between consecutive tested intervals."""
    if len(intervals_ms) != len(values):
        raise ValueError("intervals_ms and values must have the same length")

    for i in range(len(intervals_ms) - 1):
        x0, x1 = float(intervals_ms[i]), float(intervals_ms[i + 1])
        y0, y1 = float(values[i]), float(values[i + 1])
        if y0 < threshold <= y1 and y1 != y0:
            frac = (threshold - y0) / (y1 - y0)
            crossing = x0 + frac * (x1 - x0)
            return {
                "bracket_interval_ms": [int(intervals_ms[i]), int(intervals_ms[i + 1])],
                "values_at_bracket": [y0, y1],
                "interpolated_crossing_ms": float(crossing),
                "method": "linear_between_bracket",
            }

    for ms, val in zip(intervals_ms, values):
        if val >= threshold:
            return {
                "crossed_at_or_below_tested_interval_ms": int(ms),
                "value_at_crossing_interval": float(val),
                "interpolated_crossing_ms": float(ms),
                "method": "exact_at_tested_point",
            }

    last_ms, last_val = intervals_ms[-1], float(values[-1])
    return {
        "crossed_within_tested_range": False,
        "last_tested_interval_ms": int(last_ms),
        "last_tested_value": last_val,
        "margin_to_threshold": float(threshold - last_val),
        "method": "not_crossed_in_range",
    }


def evaluate_success_gates(
    intervals_ms: list[int],
    comparison_kappa_2: dict[str, dict[str, Any]],
    *,
    baseline_label: str = "1ms",
    collision_fraction_of_baseline: float = 0.1,
) -> dict[str, Any]:
    baseline = comparison_kappa_2.get(baseline_label)
    baseline_collisions = (
        float(baseline["context_samples_in_diff_command_groups"]) if baseline else None
    )
    per_interval: dict[str, Any] = {}
    any_pass_all = False
    closest: dict[str, Any] | None = None

    for ms in intervals_ms:
        label = f"{ms}ms"
        row = comparison_kappa_2[label]
        gates = {
            "unique_frame_ratio_gt_0_5": {
                "pass": bool(row["unique_frame_ratio"] > 0.5),
                "value": row["unique_frame_ratio"],
                "target": 0.5,
                "margin": float(row["unique_frame_ratio"] - 0.5),
            },
            "zero_consecutive_fraction_lt_0_2": {
                "pass": bool(row["fraction_zero_consecutive_pixels"] < 0.2),
                "value": row["fraction_zero_consecutive_pixels"],
                "target": 0.2,
                "margin": float(0.2 - row["fraction_zero_consecutive_pixels"]),
            },
            "unique_context_ratio_gte_unique_frame_ratio": {
                "pass": bool(row["unique_context_ratio"] >= row["unique_frame_ratio"]),
                "value": row["unique_context_ratio"],
                "reference": row["unique_frame_ratio"],
                "margin": float(row["unique_context_ratio"] - row["unique_frame_ratio"]),
            },
            "context_collisions_lt_10pct_of_1ms_baseline": {
                "pass": bool(
                    baseline_collisions is not None
                    and row["context_samples_in_diff_command_groups"]
                    < collision_fraction_of_baseline * baseline_collisions
                ),
                "value": row["context_samples_in_diff_command_groups"],
                "target": (
                    collision_fraction_of_baseline * baseline_collisions
                    if baseline_collisions is not None
                    else None
                ),
                "margin": (
                    float(collision_fraction_of_baseline * baseline_collisions - row["context_samples_in_diff_command_groups"])
                    if baseline_collisions is not None
                    else None
                ),
            },
        }
        passed = sum(1 for g in gates.values() if g["pass"])
        all_pass = passed == len(gates)
        any_pass_all = any_pass_all or all_pass
        per_interval[label] = {
            "gates": gates,
            "gates_passed": passed,
            "gates_total": len(gates),
            "all_gates_pass": all_pass,
        }
        score = passed + row["unique_frame_ratio"]  # tie-break toward higher uniqueness
        if closest is None or score > closest["score"]:
            closest = {"interval": label, "score": score, "gates_passed": passed, "row": row}

    return {
        "per_interval": per_interval,
        "any_interval_passes_all_gates": any_pass_all,
        "closest_interval": closest,
        "collision_baseline_1ms_samples": baseline_collisions,
        "collision_gate_fraction_of_baseline": collision_fraction_of_baseline,
    }

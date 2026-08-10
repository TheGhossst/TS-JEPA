"""Tests for observation-cadence plateau / trend analysis."""

from __future__ import annotations

from ts_jepa.diagnostics.plateau import (
    evaluate_success_gates,
    extrapolate_unique_frame_ratio,
    interpolate_threshold_crossing,
    observation_cadence_table,
    rate_of_change_analysis,
)


def _synthetic_comparison(intervals_ms: list[int]) -> dict:
    rows = {}
    for i, ms in enumerate(intervals_ms):
        label = f"{ms}ms"
        ratio = 0.01 * ms
        rows[label] = {
            "unique_frame_ratio": ratio,
            "fraction_zero_consecutive_pixels": max(0.0, 1.0 - 0.02 * ms),
            "context_samples_in_diff_command_groups": float(400 - 30 * i),
            "max_command_span_in_context_collision_groups": float(max(0.0, 20 - 2 * i)),
            "unique_context_ratio": ratio * 1.1,
        }
    return rows


def test_observation_cadence_table_kp_feasibility():
    rows = observation_cadence_table([1, 5, 10, 20, 30], physics_budget=100, kp=15)
    by_ms = {r["interval_ms"]: r for r in rows}
    assert by_ms[1]["num_observations_per_trajectory"] == 100
    assert by_ms[1]["kp_feasible"] is True
    assert by_ms[10]["num_observations_per_trajectory"] == 10
    assert by_ms[10]["kp_feasible"] is False
    assert by_ms[20]["num_observations_per_trajectory"] == 5
    assert by_ms[30]["num_observations_per_trajectory"] == 3
    assert by_ms[30]["budget_truncated"] is True


def test_rate_of_change_analysis_has_consecutive_steps():
    intervals = [1, 2, 5, 10]
    comp = _synthetic_comparison(intervals)
    out = rate_of_change_analysis(intervals, comp)
    assert len(out["consecutive_steps"]) == 3
    step = out["consecutive_steps"][0]
    assert step["from_ms"] == 1 and step["to_ms"] == 2
    assert step["metrics"]["unique_frame_ratio"]["absolute_change"] is not None


def test_extrapolate_unique_frame_ratio_runs():
    intervals = [1, 2, 5, 10, 20]
    comp = _synthetic_comparison(intervals)
    out = extrapolate_unique_frame_ratio(intervals, comp)
    assert out["saturating_exponential_fit"]["success"] is True
    assert "consensus" in out


def test_interpolate_threshold_crossing_brackets():
    out = interpolate_threshold_crossing([30, 35, 40], [0.43, 0.48, 0.55], threshold=0.5)
    assert out["method"] == "linear_between_bracket"
    assert 35 < out["interpolated_crossing_ms"] < 40


def test_evaluate_success_gates_flags_none_passing():
    intervals = [1, 2, 5, 10]
    comp = _synthetic_comparison(intervals)
    out = evaluate_success_gates(intervals, comp)
    assert out["any_interval_passes_all_gates"] is False
    assert out["closest_interval"]["interval"] is not None

"""Full-information closed-loop diagnostic tests."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from ts_jepa.config import load_config
from ts_jepa.evaluation.evaluate import (
    evaluate_closed_loop_full_information_diagnostic,
    validate_baseline,
)
from ts_jepa.plan.baseline_validation import STATUS_PASS


class _StubTeacher:
    def act(self, state: np.ndarray) -> float:
        return float(state[0]) * 5.0


class _TrackingController:
    def __init__(self, force: float = 1.5) -> None:
        self.device = "cpu"
        self.force = force
        self.reset_calls = 0
        self.recv_calls = 0
        self.lost_calls = 0
        self.step_calls = 0

    def reset_episode(self) -> None:
        self.reset_calls += 1

    def step_packet_received(self, frame: np.ndarray) -> float:
        self.recv_calls += 1
        assert frame is not None
        return self.force

    def step_packet_lost(self) -> float:
        self.lost_calls += 1
        raise AssertionError("predictor path must not be used in full-info diagnostic")

    def step(self, frame: np.ndarray | None, packet_received: bool) -> float:
        self.step_calls += 1
        raise AssertionError("step() must not be used in full-info diagnostic")


def test_full_info_diagnostic_uses_packet_received_only():
    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 4
    config["evaluation"]["repetitions"] = 2
    controller = _TrackingController(force=2.0)
    report = evaluate_closed_loop_full_information_diagnostic(
        config,
        controller,  # type: ignore[arg-type]
        seeds=[10, 11],
        teacher=_StubTeacher(),
    )
    assert controller.reset_calls == 2
    assert controller.recv_calls == 8
    assert controller.lost_calls == 0
    assert controller.step_calls == 0
    assert report["packet_loss"] is False
    assert report["predictor_rollout"] is False
    assert report["controller_path"] == "step_packet_received"
    assert len(report["rollouts"]) == 2


def test_full_info_diagnostic_rollout_metrics():
    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 3
    config["evaluation"]["repetitions"] = 1
    controller = _TrackingController(force=-4.0)
    report = evaluate_closed_loop_full_information_diagnostic(
        config,
        controller,  # type: ignore[arg-type]
        seeds=[42],
        teacher=_StubTeacher(),
    )
    rollout = report["rollouts"][0]
    assert len(rollout["initial_state"]) == 4
    assert len(rollout["final_state"]) == 4
    assert rollout["num_steps"] == 3
    assert rollout["force_min"] == pytest.approx(-4.0)
    assert rollout["force_max"] == pytest.approx(-4.0)
    assert rollout["force_mean"] == pytest.approx(-4.0)
    assert rollout["force_mean_abs"] == pytest.approx(4.0)
    assert rollout["x_min"] <= rollout["x_max"]
    assert rollout["theta_min"] <= rollout["theta_max"]
    assert 0 <= rollout["controlled_steps"] <= 3
    assert 0.0 <= rollout["mean_control_score"] <= 1.0
    assert rollout["nan_or_inf"] is False


def test_full_info_diagnostic_teacher_comparison_on_same_states():
    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 2
    config["evaluation"]["repetitions"] = 1
    controller = _TrackingController(force=3.0)
    report = evaluate_closed_loop_full_information_diagnostic(
        config,
        controller,  # type: ignore[arg-type]
        seeds=[7],
        teacher=_StubTeacher(),
    )
    cmp_ = report["rollouts"][0]["teacher_comparison"]
    assert len(cmp_["forces_actor"]) == 2
    assert len(cmp_["forces_teacher"]) == 2
    assert len(cmp_["abs_force_errors"]) == 2
    assert cmp_["forces_actor"] == [3.0, 3.0]
    assert cmp_["abs_force_errors"] == [
        abs(3.0 - cmp_["forces_teacher"][0]),
        abs(3.0 - cmp_["forces_teacher"][1]),
    ]
    assert cmp_["mean_abs_force_error"] == pytest.approx(float(np.mean(cmp_["abs_force_errors"])))
    assert report["summary"]["mean_abs_force_error_vs_teacher"] == pytest.approx(cmp_["mean_abs_force_error"])


def test_full_info_diagnostic_flags_nan_or_inf(monkeypatch: pytest.MonkeyPatch):
    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 2
    config["evaluation"]["repetitions"] = 1

    class _NanController(_TrackingController):
        def step_packet_received(self, frame: np.ndarray) -> float:
            self.recv_calls += 1
            return float("nan")

    report = evaluate_closed_loop_full_information_diagnostic(
        config,
        _NanController(),  # type: ignore[arg-type]
        seeds=[0],
        teacher=_StubTeacher(),
    )
    assert report["rollouts"][0]["nan_or_inf"] is True
    assert report["summary"]["any_nan_or_inf"] is True


def test_full_info_diagnostic_not_part_of_baseline_validation_gate():
    config = load_config()
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    report: dict[str, Any] = {
        "embedding_tsne": {
            "num_samples": 10,
            "coords": [[0.0, 0.0]] * 10,
            "cart_positions": [0.0] * 10,
        },
        "actor_nmae": {"nmae": 0.4, "num_values": 10, "nmae_denominator": 40.0},
        "control": {"mean": 0.5, "repetitions": 5},
        "prediction_horizon_nmae": {
            "nmae": 0.3,
            "nmae_by_horizon": {str(h): 0.3 for h in range(1, kp + 1)},
            "latent_cosine_mean": 0.2,
            "latent_cosine_by_horizon": {str(h): 0.2 for h in range(1, kp + 1)},
        },
        "communication_bits": {
            "rgb_bits": 1000,
            "embedding_bits": 100,
            "reduction_ratio": 10.0,
        },
        "stability": {
            "passed": True,
            "full_receive": {"mean_control_score": 0.5},
            "intermittent_loss": {"mean_control_score": 0.4},
        },
        "test_losses": {
            "jepa": {"best_test_loss": 0.01},
            "semantic_actor": {"best_test_loss": 0.7},
        },
        "closed_loop_full_info_diagnostic": {
            "summary": {"mean_control_score": 0.0, "any_nan_or_inf": False},
        },
        "consecutive_frame_mape": {"mean_mape_percent": 1.2, "num_pairs": 10},
        "fig4_mape": {
            "by_sampling_interval_ms": {
                "1.0": {
                    "without_augmentation": {"mean_mape_percent": 1.0, "num_pairs": 10},
                    "with_augmentation": {"mean_mape_percent": 2.0, "num_pairs": 10},
                },
                "2.0": {
                    "without_augmentation": {"mean_mape_percent": 1.5, "num_pairs": 8},
                    "with_augmentation": {"mean_mape_percent": 2.5, "num_pairs": 8},
                },
            }
        },
    }
    result = validate_baseline(report, config)
    assert "closed_loop_full_info_diagnostic" not in result["checks"]
    assert result["status"] == STATUS_PASS

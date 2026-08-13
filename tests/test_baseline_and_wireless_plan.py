"""Plan §16 baseline validation and §13 wireless channel tests."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from ts_jepa.config import load_config
from ts_jepa.evaluation.evaluate import evaluate_prediction_horizon_nmae, validate_baseline
from ts_jepa.evaluation.metrics import communication_reduction_report, latent_cosine_by_horizon, nmae
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.plan.baseline_validation import (
    PLAN_BASELINE_VALIDATION,
    STATUS_FAIL,
    STATUS_INCOMPLETE,
    STATUS_PASS,
    assert_plan_baseline_validation_config,
)
from ts_jepa.plan.wireless import PLAN_WIRELESS, assert_plan_wireless_config
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.wireless.channel import WirelessChannelModel


def _valid_baseline_report(config: dict) -> dict:
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    return {
        "embedding_tsne": {
            "num_samples": 10,
            "coords": [[float(i), float(i)] for i in range(10)],
            "cart_positions": [0.0] * 10,
        },
        "consecutive_frame_mape": {
            "mean_mape_percent": 1.2,
            "num_pairs": 10,
        },
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
        "actor_nmae": {
            "nmae": 0.4,
            "num_values": 10,
            "nmae_denominator": 40.0,
            "denormalized": True,
        },
        "control": {"mean": 0.5, "repetitions": 5, "best": 0.6, "worst": 0.4},
        "prediction_horizon_nmae": {
            "nmae": 0.3,
            "nmae_by_horizon": {str(h): 0.3 for h in range(1, kp + 1)},
            "latent_cosine_mean": 0.2,
            "latent_cosine_by_horizon": {str(h): 0.2 for h in range(1, kp + 1)},
            "kp": kp,
        },
        "communication_bits": communication_reduction_report(height=64, width=128, embedding_dim=256),
        "stability": {
            "passed": True,
            "full_receive": {"mean_control_score": 0.4},
            "intermittent_loss": {"mean_control_score": 0.3},
        },
        "test_losses": {
            "jepa": {"best_test_loss": 0.01},
            "semantic_actor": {"best_test_loss": 0.7},
        },
    }


def test_baseline_and_wireless_config_match_plan():
    config = load_config()
    assert_plan_baseline_validation_config(config)
    assert_plan_wireless_config(config)
    config_dp = load_config("configs/ts_jepa_dp_fixed.yaml")
    assert_plan_baseline_validation_config(config_dp)
    assert_plan_wireless_config(config_dp)


def test_plan_wireless_rejects_wrong_table_iv():
    config = copy.deepcopy(load_config())
    config["wireless"]["carrier_frequency_hz"] = 2.4e9
    with pytest.raises(ValueError, match="carrier_frequency"):
        assert_plan_wireless_config(config)


def test_plan_wireless_rejects_wrong_snr_targets():
    config = copy.deepcopy(load_config())
    config["wireless"]["snr_targets_db"] = [5, 15]
    with pytest.raises(ValueError, match="snr_targets"):
        assert_plan_wireless_config(config)


def test_communication_reduction_rgb_vs_embedding():
    report = communication_reduction_report(height=64, width=128, embedding_dim=256)
    assert report["rgb_bits"] == 64 * 128 * 3 * 8
    assert report["embedding_bits"] == 256 * 8
    assert report["rgb_bits"] == 64 * 128 * 3 * 8
    assert report["reduction_percent"] == pytest.approx(98.958333, rel=1e-5)
    assert report["reduction_ratio"] == pytest.approx(report["rgb_bits"] / report["embedding_bits"])
    assert report["reduction_ratio"] > 1.0


def test_channel_capacity_and_outage_formulas():
    config = load_config()
    model = WirelessChannelModel(config)
    gamma = model.snr_linear(tx_power_watt=0.1, h_power=1.0, path_loss_db=80.0)
    capacity = model.channel_capacity_bps(gamma)
    expected = model.bandwidth_hz * np.log2(1.0 + gamma)
    assert capacity == pytest.approx(expected)
    assert model.bandwidth_hz == PLAN_WIRELESS["total_bandwidth_hz"]
    assert model.is_outage(snr_db=3.0, gamma_th_db=5.0) is True
    assert model.is_outage(snr_db=10.0, gamma_th_db=5.0) is False


def test_outage_probability_estimate_for_snr_targets():
    config = load_config()
    model = WirelessChannelModel(config)
    for snr in PLAN_WIRELESS["snr_targets_db"]:
        out = model.estimate_outage_probability(float(snr), num_samples=200, seed=0)
        assert 0.0 <= out["outage_probability"] <= 1.0
        assert out["mean_capacity_bps"] > 0.0
        assert out["bandwidth_hz"] == model.bandwidth_hz


def test_validate_baseline_requires_all_section15_checks():
    config = load_config()
    result = validate_baseline(_valid_baseline_report(config), config)
    assert result["status"] == STATUS_PASS
    assert result["plan_section_15_status"] == STATUS_PASS
    assert result["plan_section_15_passed"] is True
    assert result["wireless_allowed"] is True
    for key in PLAN_BASELINE_VALIDATION["checks"]:
        assert result["checks"][key] is True
        assert result["check_states"][key] == "ok"


def test_validate_baseline_blocks_wireless_without_stability():
    config = load_config()
    report = _valid_baseline_report(config)
    report["stability"]["passed"] = False
    result = validate_baseline(report, config)
    assert result["checks"]["closed_loop_stability"] is False
    assert result["check_states"]["closed_loop_stability"] == "invalid"
    assert result["status"] == STATUS_FAIL
    assert result["wireless_allowed"] is False


def test_validate_baseline_nan_metrics_fail():
    config = load_config()
    report = _valid_baseline_report(config)
    report["actor_nmae"]["nmae"] = float("nan")
    result = validate_baseline(report, config)
    assert result["status"] == STATUS_FAIL
    assert result["check_states"]["actor_prediction_nmae"] == "invalid"
    assert result["wireless_allowed"] is False


def test_validate_baseline_inf_horizon_fails():
    config = load_config()
    report = _valid_baseline_report(config)
    report["prediction_horizon_nmae"]["nmae_by_horizon"]["7"] = float("inf")
    result = validate_baseline(report, config)
    assert result["status"] == STATUS_FAIL
    assert result["check_states"]["horizon_prediction_1_to_Kp"] == "invalid"
    assert result["wireless_allowed"] is False


def test_validate_baseline_missing_metrics_incomplete():
    config = load_config()
    report = _valid_baseline_report(config)
    del report["actor_nmae"]
    result = validate_baseline(report, config)
    assert result["status"] == STATUS_INCOMPLETE
    assert result["check_states"]["actor_prediction_nmae"] == "missing"
    assert result["wireless_allowed"] is False


def test_validate_baseline_missing_horizon_key_incomplete():
    config = load_config()
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    assert kp == 15
    report = _valid_baseline_report(config)
    del report["prediction_horizon_nmae"]["nmae_by_horizon"]["7"]
    result = validate_baseline(report, config)
    assert result["status"] == STATUS_INCOMPLETE
    assert result["check_states"]["horizon_prediction_1_to_Kp"] == "missing"
    assert result["wireless_allowed"] is False


def test_validate_baseline_empty_control_does_not_pass():
    config = load_config()
    report = _valid_baseline_report(config)
    report["control"] = {"mean": 0.0, "repetitions": 0}
    result = validate_baseline(report, config)
    assert result["checks"]["control_performance"] is False
    assert result["wireless_allowed"] is False


def test_validate_baseline_rejects_zero_control_score() -> None:
    config = load_config()
    report = _valid_baseline_report(config)
    report["control"] = {"mean": 0.0, "repetitions": 5, "best": 0.0, "worst": 0.0}

    result = validate_baseline(report, config)

    assert result["check_states"]["control_performance"] == "invalid"
    assert result["status"] == STATUS_FAIL
    assert result["wireless_allowed"] is False
    assert "no evaluated state met the control criterion" in result["reasons"]["control_performance"]


def test_validate_baseline_rejects_zero_stability_scores_even_if_marked_passed() -> None:
    config = load_config()
    report = _valid_baseline_report(config)
    report["stability"] = {
        "passed": True,
        "full_receive": {"mean_control_score": 0.0},
        "intermittent_loss": {"mean_control_score": 0.0},
    }

    result = validate_baseline(report, config)

    assert result["check_states"]["closed_loop_stability"] == "invalid"
    assert result["status"] == STATUS_FAIL
    assert result["wireless_allowed"] is False
    assert "zero means no controlled states" in result["reasons"]["closed_loop_stability"]


def test_latent_cosine_covers_all_kp15_horizons():
    config = load_config()
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    assert kp == 15
    rng = np.random.default_rng(0)
    z_pred = rng.normal(size=(4, kp, 8))
    z_tgt = z_pred + 0.1 * rng.normal(size=(4, kp, 8))
    by_h = latent_cosine_by_horizon(z_pred, z_tgt)
    assert set(by_h) == {str(h) for h in range(1, kp + 1)}
    assert all(np.isfinite(v) for v in by_h.values())
    # Command NMAE is a separate metric and must not be used as latent quality.
    u_pred = rng.normal(size=(4 * kp,))
    u_true = rng.normal(size=(4 * kp,))
    command = nmae(u_pred, u_true)
    assert command != pytest.approx(float(np.mean(list(by_h.values()))))


def test_evaluate_prediction_horizon_reports_command_and_latent_for_kp15(monkeypatch):
    config = load_config()
    assert int(config["ts_jepa"]["prediction_horizon"]["Kp"]) == 15

    class _TinyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                "context": torch.zeros(3, 64, 128),
                "future_frames": torch.zeros(15, 3, 64, 128),
                "teacher_commands": torch.arange(15, dtype=torch.float32),
                "teacher_commands_norm": torch.zeros(15),
                "target_commands": torch.arange(1, 16, dtype=torch.float32),
                "target_commands_norm": torch.zeros(15),
            }

    monkeypatch.setattr(
        "ts_jepa.evaluation.evaluate.load_command_normalizer",
        lambda *args, **kwargs: CommandNormalizer(mean=0.0, std=1.0),
    )
    monkeypatch.setattr("ts_jepa.evaluation.evaluate.TrajectoryDataset", lambda *args, **kwargs: _TinyDataset())

    jepa = TSJEPA(config)
    actor = SemanticActor.from_config(config)
    controller = FrozenRuntimeController(
        config, jepa, actor, CommandNormalizer(mean=0.0, std=1.0), device=torch.device("cpu")
    )
    out = evaluate_prediction_horizon_nmae(config, controller)
    keys = {str(h) for h in range(1, 16)}
    assert out["kp"] == 15
    assert set(out["nmae_by_horizon"]) == keys
    assert set(out["latent_cosine_by_horizon"]) == keys
    assert np.isfinite(out["nmae"])
    assert np.isfinite(out["latent_cosine_mean"])
    assert out["nmae_formula"] == "mean(abs(u_pred - u_true)) / 40"
    assert out["nmae_denominator"] == pytest.approx(40.0)
    assert out["denormalized"] is True
    assert all(np.isfinite(v) for v in out["nmae_by_horizon"].values())
    assert all(np.isfinite(v) for v in out["latent_cosine_by_horizon"].values())
    # Distinct metrics: command NMAE is not a cosine.
    assert out["nmae"] != pytest.approx(out["latent_cosine_mean"])

from __future__ import annotations

import numpy as np
import pytest

from ts_jepa.config import load_config
from ts_jepa.evaluation.metrics import (
    DEFAULT_FORCE_RANGE_N,
    NMAE_FORMULA,
    PAPER_COMMUNICATION_REDUCTION,
    communication_reduction_report,
    consecutive_frame_mape,
    nmae,
    physical_force_range_n,
)
from ts_jepa.preprocessing.command_stats import CommandNormalizer


def test_nmae_zero_when_equal():
    x = np.array([1.0, -2.0, 3.0])
    assert nmae(x, x) == 0.0


def test_nmae_is_mean_abs_over_40():
    pred = np.array([0.0, 0.0])
    target = np.array([40.0, 0.0])
    # mean(|40-0|, |0-0|) / 40 = 20/40 = 0.5
    assert nmae(pred, target) == pytest.approx(0.5)
    # Old mean(|target|) normalization would give 20/20 = 1.0
    assert nmae(pred, target) != pytest.approx(np.mean(np.abs(pred - target)) / (np.mean(np.abs(target))))


def test_nmae_denominator_is_force_range_not_target_mean():
    pred = np.array([10.0, 10.0, 10.0])
    target = np.array([10.0, 10.0, 10.0])
    assert nmae(pred, target) == 0.0
    pred = np.array([0.0])
    target = np.array([1.0])
    assert nmae(pred, target) == pytest.approx(1.0 / 40.0)


def test_nmae_formula_constant_and_config_force_range():
    config = load_config()
    assert physical_force_range_n(config) == pytest.approx(40.0)
    assert physical_force_range_n(config) == pytest.approx(DEFAULT_FORCE_RANGE_N)
    assert NMAE_FORMULA == "mean(abs(u_pred - u_true)) / 40"


def test_nmae_requires_denormalization_before_metric():
    normalizer = CommandNormalizer(mean=0.0, std=10.0)
    u_true = np.array([20.0, -20.0])
    u_pred = np.array([0.0, 0.0])
    physical = nmae(u_pred, u_true)
    assert physical == pytest.approx(20.0 / 40.0)

    u_pred_norm = normalizer.normalize(u_pred)
    u_true_norm = normalizer.normalize(u_true)
    # Z-scored commands must not be scored as Newtons.
    assert nmae(u_pred_norm, u_true_norm) == pytest.approx(2.0 / 40.0)
    denorm_pred = normalizer.denormalize(u_pred_norm)
    denorm_true = normalizer.denormalize(u_true_norm)
    assert nmae(denorm_pred, denorm_true) == pytest.approx(physical)


def test_consecutive_frame_mape_eq26_percent_and_skips_zeros():
    prev = np.array([10.0, 0.0, 5.0])
    curr = np.array([12.0, 99.0, 5.0])
    # skip the zero denom: mean(|2|/10, |0|/5) * 100 = 10
    assert consecutive_frame_mape(curr, prev) == pytest.approx(10.0)
    assert np.isnan(consecutive_frame_mape(np.zeros(3), np.zeros(3)))


def test_mean_command_baseline_mse():
    from ts_jepa.training.actor_helpers import mean_command_baseline_mse

    y = np.array([1.0, 3.0, 5.0])
    assert mean_command_baseline_mse(y, 3.0) == pytest.approx(8.0 / 3.0)


def test_communication_bits_match_paper_98_95():
    report = communication_reduction_report(height=64, width=128, embedding_dim=256)
    assert report["rgb_bits"] == 64 * 128 * 3 * 8
    assert report["embedding_bits"] == 256 * 8
    assert report["reduction_percent"] == pytest.approx(PAPER_COMMUNICATION_REDUCTION * 100.0, rel=1e-4)
    assert report["bits_per_value"] == 8
    assert report["bitwidth_paper_exact"] is False
    assert report["embedding_bitwidth_source"] == "recovered_from_paper_98.95_percent_reduction"


def test_fig4_sampling_intervals_are_ic_not_plan_defaults():
    import copy

    from ts_jepa.evaluation.evaluate import fig4_sampling_intervals_ms

    config = load_config()
    rates = fig4_sampling_intervals_ms(config)
    assert len(rates) >= 2
    assert 1.0 in rates
    cfg = copy.deepcopy(config)
    del cfg["experiments"]["fig4_sampling_interval_ms"]
    with pytest.raises(ValueError, match="fig4_sampling_interval_ms"):
        fig4_sampling_intervals_ms(cfg)

"""Frozen JEPA → DP-command recoverability probe tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Subset

from ts_jepa.evaluation.command_linear_probe import (
    FeatureCommandDataset,
    _signal_quality,
    fit_linear_ols,
    interpret_probe_table,
    pearson_corr,
    predict_linear_ols,
    probe_metrics,
    train_sgd_probe,
)
from ts_jepa.evaluation.metrics import nmae
from ts_jepa.preprocessing.command_stats import CommandNormalizer


def test_ols_recovers_known_affine_map():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4)).astype(np.float64)
    w = np.array([1.5, -0.5, 0.25, 2.0])
    b = -0.3
    y = x @ w + b
    coef = fit_linear_ols(x, y, ridge=0.0)
    pred = predict_linear_ols(x, coef)
    np.testing.assert_allclose(pred, y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(coef[:-1], w, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(coef[-1], b, rtol=1e-6, atol=1e-6)


def test_pearson_and_nmae_mean_baseline():
    target = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])
    mean = np.full_like(target, target.mean())
    assert pearson_corr(mean, target) == 0.0
    perfect = target.copy()
    assert pearson_corr(perfect, target) == pytest.approx(1.0)
    assert nmae(perfect, target, force_range_n=40.0) == 0.0
    assert nmae(mean, target, force_range_n=40.0) > 0.0


def test_probe_metrics_beats_mean_when_perfect():
    normalizer = CommandNormalizer(mean=0.0, std=4.0)
    y_phys = np.array([-8.0, -4.0, 4.0, 8.0], dtype=np.float32)
    y_norm = normalizer.normalize(y_phys)
    metrics = probe_metrics(
        y_norm,
        y_norm,
        normalizer,
        force_range_n=40.0,
        mean_baseline_phys=np.array([float(y_phys.mean())]),
    )
    assert metrics["nmae_physical"] == 0.0
    assert metrics["pearson_physical"] == pytest.approx(1.0)
    assert metrics["beats_mean_baseline"] is True
    assert metrics["beats_mean_baseline_mse"] is True
    assert metrics["relative_mse_improvement_vs_mean"] > 0.5


def test_interpret_jepa_linear_ok_implies_representation_has_signal():
    good = {
        "pearson_physical": 0.7,
        "relative_mse_improvement_vs_mean": 0.4,
    }
    poor = {
        "pearson_physical": 0.01,
        "relative_mse_improvement_vs_mean": 0.0,
    }
    out = interpret_probe_table(
        jepa_linear=good,
        jepa_mlp=poor,
        state_linear=good,
        state_mlp=good,
    )
    assert out["code"] == "linear_ok_mlp_fail"
    assert _signal_quality(good) == "good"
    assert _signal_quality(poor) == "poor"


def test_interpret_jepa_poor_state_good_is_jepa_bottleneck():
    poor = {"pearson_physical": 0.0, "relative_mse_improvement_vs_mean": 0.0}
    good = {"pearson_physical": 0.8, "relative_mse_improvement_vs_mean": 0.5}
    out = interpret_probe_table(
        jepa_linear=poor,
        jepa_mlp=poor,
        state_linear=good,
        state_mlp=good,
    )
    assert out["code"] == "jepa_is_bottleneck"


def test_interpret_both_poor_is_pairing_or_teacher():
    poor = {"pearson_physical": 0.0, "relative_mse_improvement_vs_mean": 0.0}
    out = interpret_probe_table(
        jepa_linear=poor,
        jepa_mlp=poor,
        state_linear=poor,
        state_mlp=poor,
    )
    assert out["code"] == "target_or_pairing_wrong"


def test_sgd_linear_fits_tiny_dataset():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 3)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 1]).astype(np.float32)
    ds = FeatureCommandDataset(x, y)
    train_ds = Subset(ds, list(range(48)))
    val_ds = Subset(ds, list(range(48, 64)))
    model = nn.Linear(3, 1)
    info = train_sgd_probe(
        model,
        train_ds,
        val_ds,
        device=torch.device("cpu"),
        lr=0.05,
        weight_decay=0.0,
        batch_size=16,
        epochs=40,
        patience=40,
    )
    assert info["best_val_mse_norm"] < 0.05

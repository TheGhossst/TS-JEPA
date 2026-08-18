"""Frozen z → kinematics probe tests (no JEPA training)."""

from __future__ import annotations

import numpy as np
import pytest

from ts_jepa.evaluation.command_linear_probe import fit_linear_ols, predict_linear_ols
from ts_jepa.evaluation.mlp_state_decoder import predict_mlp_state, train_mlp_state_decoder
from ts_jepa.evaluation.state_probe import (
    STATE_DIM_NAMES,
    interpret_state_probes,
    load_state_history_aligned,
    regression_metrics,
)


def test_state_history_repeats_first_state_and_does_not_cross_files(tmp_path):
    a = tmp_path / "00000.npz"
    b = tmp_path / "00001.npz"
    s0 = np.array([[1.0, 0.0, 0.1, 0.0], [2.0, 0.5, 0.2, 0.1]], dtype=np.float32)
    s1 = np.array([[9.0, 9.0, 9.0, 9.0]], dtype=np.float32)
    np.savez(a, states=s0, commands=np.zeros(2))
    np.savez(b, states=s1, commands=np.zeros(1))
    hist = load_state_history_aligned(tmp_path, kappa=2)
    assert hist.shape == (3, 8)
    np.testing.assert_allclose(hist[0], np.concatenate([s0[0], s0[0]]))
    np.testing.assert_allclose(hist[1], np.concatenate([s0[0], s0[1]]))
    np.testing.assert_allclose(hist[2], np.concatenate([s1[0], s1[0]]))


def test_ols_recovers_state_from_embedding():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(400, 8))
    w = np.array([0.5, -1.0, 0.25, 0.0, 0.1, 0.0, 0.0, 2.0])
    y = z @ w + 0.3
    coef = fit_linear_ols(z, y, ridge=0.0)
    pred = predict_linear_ols(z, coef)
    np.testing.assert_allclose(pred, y, rtol=1e-6, atol=1e-6)


def test_regression_metrics_beats_mean_on_perfect_pred():
    y = np.array([-0.2, -0.1, 0.1, 0.3])
    metrics = regression_metrics(y, y, train_mean=float(y.mean()))
    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["beats_mean_baseline_mse"] is True
    assert metrics["beats_mean_baseline_mae"] is True


def test_interpret_missing_state():
    poor = {
        "test": {
            "pearson": 0.01,
            "relative_mse_improvement_vs_mean": 0.0,
            "beats_mean_baseline_mse": False,
        }
    }
    per_dim = {name: poor for name in STATE_DIM_NAMES}
    out = interpret_state_probes(per_dim)
    assert out["code"] == "z_missing_state"


def test_interpret_contains_state():
    good = {
        "test": {
            "pearson": 0.8,
            "relative_mse_improvement_vs_mean": 0.5,
            "beats_mean_baseline_mse": True,
        }
    }
    per_dim = {name: good for name in STATE_DIM_NAMES}
    out = interpret_state_probes(per_dim)
    assert out["code"] == "z_contains_state"


def test_interpret_pose_not_velocity():
    good = {
        "test": {
            "pearson": 0.8,
            "relative_mse_improvement_vs_mean": 0.5,
            "beats_mean_baseline_mse": True,
        }
    }
    poor = {
        "test": {
            "pearson": 0.02,
            "relative_mse_improvement_vs_mean": 0.0,
            "beats_mean_baseline_mse": False,
        }
    }
    per_dim = {
        "cart_position": good,
        "cart_velocity": poor,
        "pole_angle": good,
        "pole_angular_velocity": poor,
    }
    out = interpret_state_probes(per_dim)
    assert out["code"] == "z_has_pose_not_velocity"


def test_mlp_decoder_fits_nonlinear_velocity_like_target():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(800, 16)).astype(np.float32)
    # Pose is linear in z; a velocity-like channel is quadratic — linear OLS cannot get this.
    s = np.zeros((800, 4), dtype=np.float32)
    s[:, 0] = z[:, 0]
    s[:, 2] = z[:, 1]
    s[:, 3] = (z[:, 0] * z[:, 1]) + 0.05 * rng.normal(size=800).astype(np.float32)
    s[:, 1] = z[:, 2] ** 2
    probe = train_mlp_state_decoder(
        z,
        s,
        device=__import__("torch").device("cpu"),
        epochs=25,
        batch_size=64,
        hidden=(64, 32),
        dropout=0.0,
        val_fraction=0.2,
        seed=0,
    )
    pred = predict_mlp_state(z, probe)
    r2_thdot = 1.0 - np.sum((s[:, 3] - pred[:, 3]) ** 2) / max(np.sum((s[:, 3] - s[:, 3].mean()) ** 2), 1e-12)
    assert r2_thdot > 0.7

"""True-state discrete LQR should keep the working cart-pole in the Eq. 28 band."""

from __future__ import annotations

import numpy as np
import pytest

from ts_jepa.config import load_config
from ts_jepa.control.lqr import discrete_linearization, discrete_lqr_gain, default_lqr_weights, lqr_force, lqr_forces, lqr_gain_from_config
from ts_jepa.env.factory import build_inverted_cartpole_env
from ts_jepa.evaluation.evaluate import _observation_stride, control_loop_stride
from ts_jepa.training.probe_lqr_actor import evaluate_true_state_lqr, fit_z_state_probe, predict_state_from_z


def test_true_state_lqr_beats_zero_on_working_seeds():
    config = load_config("configs/ts_jepa_working.yaml")
    gain = lqr_gain_from_config(config)
    out = evaluate_true_state_lqr(config, gain, seeds=[100, 101, 102])
    assert out["mean_control_score"] > 0.3
    assert all(s > 0.1 for s in out["per_seed"])
    for final in out["final_states"]:
        assert abs(final[0]) < 0.05
        assert abs(final[2]) < 0.05


def test_true_state_lqr_beats_zero_on_dp_fixed_with_control_hold():
    """Oracle LQR on paper inits only scores if each decision is held ~20 ms (2 s episode)."""
    config = load_config("configs/ts_jepa_dp_fixed.yaml")
    assert control_loop_stride(config) == 20
    gain = lqr_gain_from_config(config)
    out = evaluate_true_state_lqr(config, gain, seeds=[100, 101, 102])
    assert out["mean_control_score"] > 0.05
    for final in out["final_states"]:
        assert abs(final[0]) < 0.15
        assert abs(final[2]) < 0.15


def test_observer_dt_uses_control_hold_not_dataset_stride():
    from ts_jepa.training.probe_lqr_actor import _dt_obs_from_config

    paper = load_config()
    results = load_config("configs/ts_jepa_dp_fixed.yaml")
    working = load_config("configs/ts_jepa_working.yaml")
    assert _dt_obs_from_config(paper) == pytest.approx(0.001)
    assert _dt_obs_from_config(results) == pytest.approx(0.02)
    assert _dt_obs_from_config(working) == pytest.approx(0.02)


def test_probe_lqr_dirname_follows_actor_family():
    from ts_jepa.config import probe_lqr_run_dirname

    working = load_config("configs/ts_jepa_working.yaml")
    results = load_config("configs/ts_jepa_dp_fixed.yaml")
    assert probe_lqr_run_dirname(working, "linear") == "semantic_actor_working_probe_lqr"
    assert probe_lqr_run_dirname(working, "mlp") == "semantic_actor_working_probe_lqr_mlp"
    assert probe_lqr_run_dirname(results, "linear") == "semantic_actor_dp_fixed_probe_lqr"


def test_linearization_is_stabilizable():
    config = load_config("configs/ts_jepa_working.yaml")
    env = build_inverted_cartpole_env(config)
    a, b = discrete_linearization(env.ode, _observation_stride(config))
    q, r = default_lqr_weights()
    k = discrete_lqr_gain(a, b, q, r)
    eig = np.linalg.eigvals(a - b @ k)
    assert np.max(np.abs(eig)) < 1.0
    states = np.array([[0.1, 0.0, 0.05, 0.0], [0.0, 0.0, 0.0, 0.0]])
    batched = lqr_forces(k, states, -20.0, 20.0)
    assert batched.shape == (2,)
    assert batched[0] == pytest.approx(lqr_force(k, states[0], -20.0, 20.0))
    assert batched[1] == pytest.approx(0.0, abs=1e-6)


def test_z_state_probe_recovers_linear_map():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 8))
    z = rng.normal(size=(200, 8))
    states = z @ w.T + 0.01 * rng.normal(size=(200, 4))
    probe = fit_z_state_probe(z, states)
    pred = predict_state_from_z(z, probe)
    mae = np.mean(np.abs(pred - states), axis=0)
    assert np.all(mae < 0.05)

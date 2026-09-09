"""Observer-LQR ablations, packet-loss masks, and CLI flags."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ts_jepa.config import load_config
from ts_jepa.control.lqr import discrete_linearization, lqr_gain_from_config
from ts_jepa.env.factory import build_inverted_cartpole_env
from ts_jepa.evaluation.evaluate import _observation_stride, evaluate_closed_loop
from ts_jepa.evaluation.working_gates import (
    bernoulli_receive_mask,
    closed_loop_eval_seeds,
    evaluate_closed_loop_packet_loss_sweep,
    make_receive_mask,
    parse_packet_receive_spec,
)
from ts_jepa.training.probe_lqr_actor import (
    FiniteDifferenceLQRRuntimeController,
    TruePositionObserverLQRController,
    alpha_beta_correct,
    evaluate_true_state_lqr,
    finite_difference_state,
)


def _working_lqr():
    config = load_config("configs/ts_jepa_working.yaml")
    env = build_inverted_cartpole_env(config)
    a, b = discrete_linearization(env.ode, _observation_stride(config))
    gain = lqr_gain_from_config(config)
    return config, gain, a, b[:, 0]


def test_closed_loop_eval_seeds_cli_override():
    config = {"evaluation": {"working_gates": {"closed_loop_seeds": [100, 101, 102]}}}
    assert closed_loop_eval_seeds(config) == [100, 101, 102]
    assert closed_loop_eval_seeds(config, [100, 101, 102, 103, 104]) == [100, 101, 102, 103, 104]


def test_alpha_beta_correct_and_finite_difference_helpers():
    corrected = alpha_beta_correct(
        np.zeros(4), 1.0, 0.5, alpha=1.0, beta=1.0, dt_obs=0.02
    )
    assert corrected[0] == pytest.approx(1.0)
    assert corrected[1] == pytest.approx(50.0)
    assert corrected[2] == pytest.approx(0.5)
    assert corrected[3] == pytest.approx(25.0)
    fd = finite_difference_state(0.2, 0.1, 0.0, 0.0, 1, 0.02)
    assert fd[1] == pytest.approx(10.0)
    assert fd[3] == pytest.approx(5.0)
    fd2 = finite_difference_state(0.2, 0.1, 0.0, 0.0, 2, 0.02)
    assert fd2[1] == pytest.approx(5.0)


def test_packet_receive_specs_and_masks():
    periodic = parse_packet_receive_spec("1/15", kp=15)
    assert periodic["kind"] == "periodic_kp"
    assert periodic["receive_rate"] == pytest.approx(1.0 / 15.0)
    bernoulli = parse_packet_receive_spec(0.8, kp=15)
    assert bernoulli["kind"] == "bernoulli"
    assert bernoulli["receive_rate"] == pytest.approx(0.8)
    mask = make_receive_mask(100, periodic, seed=0)
    assert mask[0] is True
    assert mask[1] is False
    assert mask[15] is True
    assert sum(mask) == 7
    always = bernoulli_receive_mask(20, 1.0, seed=1)
    assert always == [True] * 20
    never = bernoulli_receive_mask(20, 0.0, seed=1)
    assert never == [False] * 20
    mid = bernoulli_receive_mask(200, 0.4, seed=7)
    assert 0.25 < (sum(mid) / len(mid)) < 0.55


def test_true_position_observer_lqr_beats_zero_on_working_seeds():
    config, gain, a, b = _working_lqr()
    ctrl = TruePositionObserverLQRController(
        config, gain, a, b, alpha=0.5, beta=0.3, full_information=True
    )
    scores = [
        float(evaluate_closed_loop(config, ctrl, seed=seed)["mean_control_score"])
        for seed in (100, 101, 102)
    ]
    assert float(np.mean(scores)) > 0.3
    assert all(s > 0.1 for s in scores)
    true = evaluate_true_state_lqr(config, gain, seeds=[100, 101, 102])
    assert float(np.mean(scores)) == pytest.approx(true["mean_control_score"], abs=0.15)


def test_true_position_observer_kp_loss_beats_zero_on_dp_fixed():
    """Oracle (x,θ) observer + LQR under the overlay stability receive period."""
    from ts_jepa.evaluation.evaluate import (
        control_loop_stride,
        receive_every_kp_mask,
        stability_receive_period,
    )

    config = load_config("configs/ts_jepa_dp_fixed.yaml")
    env = build_inverted_cartpole_env(config)
    a, b = discrete_linearization(env.ode, control_loop_stride(config))
    gain = lqr_gain_from_config(config)
    ctrl = TruePositionObserverLQRController(
        config, gain, a, b[:, 0], alpha=0.5, beta=0.3, full_information=False
    )
    steps = int(config["simulation"]["trajectory_steps"])
    period = stability_receive_period(config)
    assert period == 8
    mask = receive_every_kp_mask(steps, period)
    scores = [
        float(
            evaluate_closed_loop(
                config, ctrl, seed=seed, packet_receive_mask=mask
            )["mean_control_score"]
        )
        for seed in (100, 102)
    ]
    assert float(np.mean(scores)) > 0.3


def test_true_position_observer_predicts_on_miss():
    config, gain, a, b = _working_lqr()
    ctrl = TruePositionObserverLQRController(
        config, gain, a, b, alpha=0.5, beta=0.3, full_information=False
    )
    plant = np.array([0.01, 0.0, 0.02, 0.0], dtype=np.float64)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert ctrl.step(frame, False, plant) == 0.0
    u0 = ctrl.step(frame, True, plant)
    assert u0 != 0.0
    est_after_recv = ctrl.est.copy()
    ctrl.step(frame, False, plant)
    predicted = a @ est_after_recv + b * float(u0)
    assert np.allclose(ctrl.est, predicted)


def test_finite_difference_controller_uses_decoded_position_deltas():
    config, gain, a, b = _working_lqr()

    class _StubEncoder:
        def __init__(self) -> None:
            self.config = config
            self.frame_buffer: list = []
            self.stats = SimpleNamespace(force_min=-20.0, force_max=20.0)

        def reset_episode(self) -> None:
            self.frame_buffer.clear()

        def observe_frame(self, frame) -> None:
            self.frame_buffer.append(frame)

    class _ScriptedFD(FiniteDifferenceLQRRuntimeController):
        def __init__(self) -> None:
            super().__init__(_StubEncoder(), {}, gain, a, b, full_information=True)
            self.queue = [(0.0, 0.0), (0.02, 0.01)]

        def _measure(self, frame):
            self.encoder.observe_frame(frame)
            return self.queue.pop(0)

    ctrl = _ScriptedFD()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    ctrl.step(frame, True)
    assert ctrl.est[1] == pytest.approx(0.0)
    ctrl.step(frame, True)
    assert ctrl.est[0] == pytest.approx(0.02)
    assert ctrl.est[1] == pytest.approx(1.0)
    assert ctrl.est[2] == pytest.approx(0.01)
    assert ctrl.est[3] == pytest.approx(0.5)


def test_packet_loss_sweep_rate_one_matches_hold_and_zero():
    config, gain, a, b = _working_lqr()
    config = copy.deepcopy(config)
    config["simulation"]["trajectory_steps"] = 8
    ctrl = TruePositionObserverLQRController(
        config, gain, a, b, alpha=0.5, beta=0.3, full_information=False
    )
    sweep = evaluate_closed_loop_packet_loss_sweep(
        config, ctrl, seeds=[100], rate_specs=["1.0", "1/15"]
    )
    assert sweep["rates"][0]["mask"] == "bernoulli"
    assert sweep["rates"][0]["mean_realized_receive_rate"] == pytest.approx(1.0)
    assert sweep["rates"][0]["predict_only_mean_control_score"] == pytest.approx(
        sweep["rates"][0]["hold_last_mean_control_score"]
    )
    assert sweep["rates"][0]["predict_only_mean_control_score"] == pytest.approx(
        sweep["rates"][0]["zero_action_mean_control_score"]
    )
    assert sweep["rates"][1]["mask"] == "periodic_kp"
    assert sweep["rates"][1]["kp"] == 15


def test_eval_observer_lqr_cli_accepts_seeds_out_dir_and_packet_sweep():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "eval_observer_lqr.py"
    spec = importlib.util.spec_from_file_location("eval_observer_lqr_cli", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.build_arg_parser().parse_args(
        [
            "--seeds",
            "100",
            "101",
            "102",
            "103",
            "104",
            "--out-dir",
            "runs/eval/observer_lqr_5seed",
            "--packet-loss-sweep",
            "--packet-receive-rates",
            "1.0",
            "0.8",
            "1/15",
        ]
    )
    assert args.seeds == [100, 101, 102, 103, 104]
    assert args.out_dir == "runs/eval/observer_lqr_5seed"
    assert args.packet_loss_sweep is True
    assert args.packet_receive_rates == ["1.0", "0.8", "1/15"]

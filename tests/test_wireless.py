from __future__ import annotations

import copy

import numpy as np
import pytest

from ts_jepa.config import load_config
from ts_jepa.wireless.channel import WirelessChannelModel
from ts_jepa.wireless.scheduler import ChannelAwareScheduler


def test_scheduler_selects_at_most_j_and_updates_aoi_on_schedule():
    config = load_config()
    sched = ChannelAwareScheduler(config, policy="channel_aware")
    rng = np.random.default_rng(0)
    before_q = {i: sched.states[i].virtual_queue for i in sched.states}
    before_aoi = {i: sched.states[i].aoi for i in sched.states}
    decision = sched.schedule(rng)
    assert sum(decision.alphas.values()) <= config["wireless"]["max_devices_scheduled_J"]
    assert set(decision.successes) == set(decision.alphas)
    for i, alpha in decision.alphas.items():
        # Plan §13 Eq. (17): β←1 iff scheduled (α=1), else β←β+1.
        if alpha == 1:
            assert sched.states[i].aoi == 1.0
            assert decision.powers[i] > 0.0
        else:
            assert sched.states[i].aoi == before_aoi[i] + 1.0
            assert decision.powers[i] == 0.0
        # Paper eq. (18): Q <- max(Q - beta_th, 0) + beta_k
        expected_q = max(before_q[i] - config["wireless"]["aoi_threshold_beta_th"], 0.0) + before_aoi[i]
        assert abs(sched.states[i].virtual_queue - expected_q) < 1e-9
    for i, alpha in decision.alphas.items():
        if alpha == 1 and decision.outages[i] == 1:
            assert decision.successes[i] == 0
            assert sched.states[i].aoi == 1.0


def test_round_robin_and_opportunistic_run():
    config = load_config()
    rng = np.random.default_rng(1)
    for policy in ("round_robin", "opportunistic"):
        sched = ChannelAwareScheduler(config, policy=policy)
        decision = sched.schedule(rng)
        assert set(decision.alphas.keys()) == set(range(config["wireless"]["num_devices"]))
        assert set(decision.successes.keys()) == set(decision.alphas.keys())


def test_policies_at_configured_snrs_use_channel_success():
    config = load_config()
    snrs = list(config["wireless"]["snr_targets_db"])
    assert snrs == [5, 10, 20]
    rng = np.random.default_rng(3)
    for policy in ("channel_aware", "round_robin", "opportunistic"):
        for snr in snrs:
            sched = ChannelAwareScheduler(config, policy=policy)
            sched.set_snr_target(float(snr))
            decision = sched.schedule(rng)
            for i in decision.alphas:
                if decision.successes[i] == 1:
                    assert decision.alphas[i] == 1
                    assert decision.outages[i] == 0
                    assert not sched.channel.is_outage(decision.realized_snr_db[i], float(snr))
                if decision.alphas[i] == 0:
                    assert decision.successes[i] == 0


def test_scheduled_outage_still_resets_aoi(monkeypatch):
    """Plan §13 Eq. (17): AoI follows α, not packet success."""
    config = load_config()
    sched = ChannelAwareScheduler(config, policy="round_robin")
    monkeypatch.setattr(sched.channel, "is_outage", lambda snr_db, gamma_th_db: True)
    before_aoi = {i: sched.states[i].aoi for i in sched.states}
    decision = sched.schedule(np.random.default_rng(0))
    assert any(decision.alphas[i] == 1 for i in decision.alphas)
    for i in decision.alphas:
        if decision.alphas[i] == 1:
            assert decision.successes[i] == 0
            assert decision.outages[i] == 1
            assert sched.states[i].aoi == 1.0
        else:
            assert sched.states[i].aoi == before_aoi[i] + 1.0


def test_outage_reaches_controller_predictive_path(monkeypatch):
    from ts_jepa.evaluation.evaluate import evaluate_with_scheduler

    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 6
    config["wireless"]["p_max_watt"] = 1e-30  # nothing feasible → no successful packets

    received: list[bool] = []
    lost_calls = {"n": 0}

    class _TrackingController:
        device = "cpu"

        def reset_episode(self) -> None:
            return None

        def step(self, frame, packet_received: bool, plant_state=None) -> float:
            received.append(bool(packet_received))
            if not packet_received:
                lost_calls["n"] += 1
            return 0.0

    out = evaluate_with_scheduler(config, _TrackingController(), policy="channel_aware", snr_db=10.0, seed=0)
    assert received
    assert all(flag is False for flag in received)
    assert lost_calls["n"] == len(received)
    assert out["packet_receive_rate"] == pytest.approx(0.0)
    assert out["schedule_rate"] == pytest.approx(0.0)
    assert out["schedule_receive_rate"] == pytest.approx(out["packet_receive_rate"])


def test_controller_packet_flag_matches_channel_successes(monkeypatch):
    from ts_jepa.evaluation.evaluate import evaluate_with_scheduler
    from ts_jepa.wireless.scheduler import ChannelAwareScheduler, ScheduleDecision

    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 4
    flags = [True, False, True, False]
    calls = {"i": 0}

    class _Scripted(ChannelAwareScheduler):
        def schedule(self, rng=None):  # type: ignore[override]
            i = calls["i"]
            calls["i"] += 1
            received = flags[i]
            n = self.num_devices
            return ScheduleDecision(
                alphas={j: int(received) if j == 0 else 0 for j in range(n)},
                powers={j: 0.1 if (j == 0 and received) else 0.0 for j in range(n)},
                successes={j: int(received) if j == 0 else 0 for j in range(n)},
                outages={j: int(not received) if j == 0 else 0 for j in range(n)},
                realized_snr_db={j: 20.0 if (j == 0 and received) else float("-inf") for j in range(n)},
            )

    monkeypatch.setattr("ts_jepa.evaluation.evaluate.ChannelAwareScheduler", _Scripted)

    seen: list[bool] = []

    class _TrackingController:
        device = "cpu"

        def reset_episode(self) -> None:
            return None

        def step(self, frame, packet_received: bool, plant_state=None) -> float:
            seen.append(bool(packet_received))
            # Device always has the RGB frame; packet_received only gates z vs Pφ.
            assert frame is not None
            return 0.0

    out = evaluate_with_scheduler(config, _TrackingController(), policy="round_robin", snr_db=5.0, seed=1)
    assert seen == flags
    assert out["packet_receive_rate"] == pytest.approx(0.5)
    assert out["schedule_receive_rate"] == pytest.approx(0.5)


def test_wireless_closed_loop_policies_at_snrs():
    from ts_jepa.evaluation.evaluate import evaluate_with_scheduler

    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 3

    class _Stub:
        device = "cpu"

        def reset_episode(self) -> None:
            return None

        def step(self, frame, packet_received: bool, plant_state=None) -> float:
            return 0.0

    for policy in ("channel_aware", "round_robin", "opportunistic"):
        for snr in config["wireless"]["snr_targets_db"]:
            out = evaluate_with_scheduler(config, _Stub(), policy=policy, snr_db=float(snr), seed=2)
            assert 0.0 <= out["packet_receive_rate"] <= 1.0
            assert 0.0 <= out["schedule_rate"] <= 1.0
            assert out["packet_receive_rate"] <= out["schedule_rate"] + 1e-12
            assert len(out["delivered_slots"]) == 3
            assert out["schedule_receive_rate"] == out["packet_receive_rate"]



def test_path_loss_and_p_req_paper_forms():
    config = load_config()
    model = WirelessChannelModel(config)
    rng = np.random.default_rng(0)
    sample = model.sample(0, snr_target_db=10.0, rng=rng)
    assert sample.p_req > 0.0
    assert np.isfinite(sample.path_loss_db)
    # p_req = gamma * Nc / |H|^2 * 10^(PL/10)
    gamma = 10 ** (10.0 / 10.0)
    expected = gamma * model.noise_watt / sample.h_complex_power * (10 ** (sample.path_loss_db / 10.0))
    assert abs(sample.p_req - expected) / expected < 1e-9


def test_capacity_matches_plan_section_13_eq9():
    config = load_config()
    model = WirelessChannelModel(config)
    snr_lin = model.snr_linear(0.2, 1.0, 90.0)
    assert model.channel_capacity_bps(snr_lin) == pytest.approx(
        model.bandwidth_hz * np.log2(1.0 + snr_lin)
    )


def test_urgency_is_negative_cost():
    config = load_config()
    sched = ChannelAwareScheduler(config, policy="channel_aware")
    cost = sched.drift_plus_penalty_cost(aoi=3.0, queue=2.0, p_req=0.01)
    urgency = sched.urgency_index(aoi=3.0, queue=2.0, p_req=0.01)
    assert abs(urgency + cost) < 1e-12
    assert sched.drift_plus_penalty_index(3.0, 2.0, 0.01) == pytest.approx(urgency)


def test_path_loss_los_eq4_and_nlos_eq6():
    config = load_config()
    model = WirelessChannelModel(config)
    wc = model.carrier_ghz
    d3d = model.distance_3d
    pl_los = model.path_loss_los_db()
    expected_los = 31.84 + 21.5 * np.log10(d3d) + 19.0 * np.log10(wc)
    assert pl_los == pytest.approx(expected_los)
    rng = np.random.default_rng(0)
    # Shadow is added inside path_loss_general_db; NLoS must be >= LoS (eq. 6).
    for _ in range(20):
        pl, is_los = model.sample_path_loss_db(rng)
        if not is_los:
            assert pl >= pl_los - 1e-9
        else:
            assert pl == pytest.approx(pl_los)
    assert model.shadow_sigma_db == pytest.approx(4.0)


def test_los_probability_inf_sh_eq5():
    config = load_config()
    model = WirelessChannelModel(config)
    den_ln = np.log(1.0 - model.clutter_density)
    height_ratio = (model.bs_height - model.device_height) / (model.clutter_height - model.device_height)
    k = -model.clutter_size / den_ln * height_ratio
    expected = float(np.exp(-model.distance_2d / k))
    assert model.los_probability() == pytest.approx(expected)
    assert 0.0 < model.los_probability() < 1.0


def test_eq10_closed_form_matches_exponential_cdf():
    config = load_config()
    model = WirelessChannelModel(config)
    p_tx, pl, gth = 0.2, 90.0, 10.0
    closed = model.outage_probability_eq10(p_tx, pl, gth)
    gamma_th = 10 ** (gth / 10.0)
    scale = (10 ** (pl / 10.0)) * model.noise_watt / p_tx * gamma_th
    assert closed == pytest.approx(1.0 - np.exp(-scale))


def test_infeasible_index_is_negative_infinity():
    config = load_config()
    config = copy.deepcopy(config)
    config["wireless"]["p_max_watt"] = 1e-30
    sched = ChannelAwareScheduler(config, policy="channel_aware")
    decision = sched.schedule(np.random.default_rng(0))
    for i in decision.indices:
        assert decision.indices[i] == float("-inf")
        assert decision.alphas[i] == 0


def test_aoi_and_queue_initialization():
    config = load_config()
    sched = ChannelAwareScheduler(config)
    for st in sched.states.values():
        assert st.aoi == 1.0
        assert st.virtual_queue == 0.0


def test_predictor_has_no_wireless_inputs():
    config = load_config()
    forbidden = config["ts_jepa"]["predictor"]["forbidden_inputs"]
    assert "virtual_channel_variables" in forbidden
    assert "embedding" in config["ts_jepa"]["predictor"]["inputs"]
    assert "control_command" in config["ts_jepa"]["predictor"]["inputs"]

from __future__ import annotations

import numpy as np

from ts_jepa.config import load_config
from ts_jepa.wireless.channel import WirelessChannelModel
from ts_jepa.wireless.scheduler import ChannelAwareScheduler


def test_scheduler_selects_at_most_j_and_updates_aoi_queue():
    config = load_config()
    sched = ChannelAwareScheduler(config, policy="channel_aware")
    rng = np.random.default_rng(0)
    before_q = {i: sched.states[i].virtual_queue for i in sched.states}
    before_aoi = {i: sched.states[i].aoi for i in sched.states}
    decision = sched.schedule(rng)
    assert sum(decision.alphas.values()) <= config["wireless"]["max_devices_scheduled_J"]
    for i, alpha in decision.alphas.items():
        if alpha == 1:
            assert sched.states[i].aoi == 1.0
        else:
            assert sched.states[i].aoi == before_aoi[i] + 1.0
        # Paper eq. (18): Q <- max(Q - beta_th, 0) + beta_k
        expected_q = max(before_q[i] - config["wireless"]["aoi_threshold_beta_th"], 0.0) + before_aoi[i]
        assert abs(sched.states[i].virtual_queue - expected_q) < 1e-9


def test_round_robin_and_opportunistic_run():
    config = load_config()
    rng = np.random.default_rng(1)
    for policy in ("round_robin", "opportunistic"):
        sched = ChannelAwareScheduler(config, policy=policy)
        decision = sched.schedule(rng)
        assert set(decision.alphas.keys()) == set(range(config["wireless"]["num_devices"]))


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


def test_urgency_is_negative_cost():
    config = load_config()
    sched = ChannelAwareScheduler(config, policy="channel_aware")
    cost = sched.drift_plus_penalty_cost(aoi=3.0, queue=2.0, p_req=0.01)
    urgency = sched.urgency_index(aoi=3.0, queue=2.0, p_req=0.01)
    assert abs(urgency + cost) < 1e-12

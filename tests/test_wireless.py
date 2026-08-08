from __future__ import annotations

import numpy as np

from ts_jepa.config import load_config
from ts_jepa.wireless.scheduler import ChannelAwareScheduler


def test_scheduler_selects_at_most_j_and_updates_aoi():
    config = load_config()
    sched = ChannelAwareScheduler(config, policy="channel_aware")
    rng = np.random.default_rng(0)
    decision = sched.schedule(rng)
    assert sum(decision.alphas.values()) <= config["wireless"]["max_devices_scheduled_J"]
    for i, alpha in decision.alphas.items():
        if alpha == 1:
            assert sched.states[i].aoi == 1.0
        else:
            assert sched.states[i].aoi >= 2.0


def test_round_robin_and_opportunistic_run():
    config = load_config()
    rng = np.random.default_rng(1)
    for policy in ("round_robin", "opportunistic"):
        sched = ChannelAwareScheduler(config, policy=policy)
        decision = sched.schedule(rng)
        assert set(decision.alphas.keys()) == set(range(config["wireless"]["num_devices"]))

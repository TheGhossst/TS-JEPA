"""Raw-state actor diagnostic tests (no JEPA training)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_jepa.config import load_config
from ts_jepa.evaluation.raw_state_actor import (
    ActorFeatureDataset,
    RawStateRuntimeController,
    audit_action_scaling,
    interpret_raw_vs_jepa,
)
from ts_jepa.preprocessing.command_stats import CommandNormalizer


def test_feature_dataset_returns_physical_command_keys():
    x = np.ones((3, 4), dtype=np.float32)
    y = np.array([-4.0, 0.0, 8.0], dtype=np.float32)
    yn = y / 4.0
    ds = ActorFeatureDataset(x, y, yn)
    item = ds[2]
    assert item["embedding"].shape == (4,)
    assert float(item["command"].item()) == pytest.approx(8.0)
    assert float(item["command_norm"].item()) == pytest.approx(2.0)


def test_raw_controller_requires_plant_state_and_holds_on_miss():
    config = load_config("configs/ts_jepa_working.yaml")
    actor = torch.nn.Linear(4, 1)
    with torch.no_grad():
        actor.weight.zero_()
        actor.bias.zero_()
        actor.weight[0, 0] = 1.0
    actor.eval()
    ctrl = RawStateRuntimeController(
        config,
        actor,
        CommandNormalizer(mean=0.0, std=14.0),
        feature="current",
        full_information=False,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="plant_state"):
        ctrl.step(None, True, plant_state=None)
    s1 = np.array([0.1, 0.0, 0.05, 0.0])
    s2 = np.array([1.2, 0.4, 0.25, 0.8])
    f1 = ctrl.step(None, True, plant_state=s1)
    f_hold = ctrl.step(None, False, plant_state=s2)
    assert f1 == pytest.approx(f_hold)
    ctrl_full = RawStateRuntimeController(
        config,
        actor,
        CommandNormalizer(mean=0.0, std=14.0),
        feature="current",
        full_information=True,
        device=torch.device("cpu"),
    )
    g1 = ctrl_full.step(None, True, plant_state=s1)
    g2 = ctrl_full.step(None, False, plant_state=s2)
    assert g1 != pytest.approx(g2)


def test_audit_flags_zscore_head_not_physical():
    config = load_config("configs/ts_jepa_working.yaml")
    normalizer = CommandNormalizer(mean=0.0, std=10.0)
    y = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    zscore_out = normalizer.normalize(y.astype(np.float32))
    flagged = audit_action_scaling(
        config,
        actor_outputs=zscore_out,
        commands_phys=y,
        normalizer=normalizer,
        train_loss_domain="physical",
    )
    assert flagged["suspected_zscore_head"] is True
    assert flagged["train_infer_domain_consistent"] is False

    physical_out = y.copy()
    ok = audit_action_scaling(
        config,
        actor_outputs=physical_out,
        commands_phys=y,
        normalizer=normalizer,
        train_loss_domain="physical",
    )
    assert ok["suspected_zscore_head"] is False
    assert ok["train_infer_domain_consistent"] is True
    assert ok["command_roundtrip_mae"] == pytest.approx(0.0, abs=1e-5)


def test_interpret_z_bottleneck_when_raw_controls():
    z_nmae = {"nmae": 0.31, "beats_mean_command_baseline": True}
    raw = {
        "current": {
            "actor_nmae": {"nmae": 0.05, "beats_mean_command_baseline": True},
            "closed_loop": {"full_information": {"mean_control_score": 0.4}},
        }
    }
    z_loop = {"mean_control_score": 0.0}
    scaling = {"suspected_zscore_head": False, "train_infer_domain_consistent": True}
    out = interpret_raw_vs_jepa(z_nmae=z_nmae, z_loop_full=z_loop, raw_reports=raw, z_scaling=scaling)
    assert out["code"] == "z_is_bottleneck"


def test_interpret_actor_cannot_stabilize():
    z_nmae = {"nmae": 0.31, "beats_mean_command_baseline": True}
    raw = {
        "current": {
            "actor_nmae": {"nmae": 0.08, "beats_mean_command_baseline": True},
            "closed_loop": {"full_information": {"mean_control_score": 0.0}},
        }
    }
    out = interpret_raw_vs_jepa(
        z_nmae=z_nmae,
        z_loop_full={"mean_control_score": 0.0},
        raw_reports=raw,
        z_scaling={"suspected_zscore_head": False, "train_infer_domain_consistent": True},
    )
    assert out["code"] == "actor_imitates_but_cannot_stabilize"

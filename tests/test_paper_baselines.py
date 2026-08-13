from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from ts_jepa.baselines.controllers import NonlinearDPController, ZeroActionController
from ts_jepa.baselines.experiments import (
    apply_embedding_dim,
    evaluate_agent_closed_loop,
    evaluate_fig6_prediction_modes,
    evaluate_with_scheduler_agent,
    fig7_embedding_dims,
)
from ts_jepa.baselines.models import GenerativeAutoencoder, SupervisedRGBToCommand
from ts_jepa.config import load_config
from ts_jepa.data.trajectory_generator import build_env_and_teacher
from ts_jepa.models.encoder import ContextEncoder
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.plan.paper_baselines import (
    PLAN_SUPERVISED_KAPPA,
    PLAN_UNSCHEDULED_ACTION,
    assert_plan_section_16_17_config,
)



def test_paper_section_16_names_not_ic_grids():
    config = load_config()
    assert_plan_section_16_17_config(config)
    assert PLAN_SUPERVISED_KAPPA == (2, 4)
    assert PLAN_UNSCHEDULED_ACTION == "hold_last_command"
    # Fig. 7 numbers live in YAML as IC (plan §18); not a PLAN_* paper tuple.
    assert fig7_embedding_dims(config) == tuple(config["experiments"]["fig7_embedding_dims"])
    assert list(config["wireless"]["snr_targets_db"]) == [5, 10, 20]


def test_fig7_grid_missing_is_ic_error_not_paper_default():
    config = copy.deepcopy(load_config())
    del config["experiments"]["fig7_embedding_dims"]
    with pytest.raises(ValueError, match="fig7_embedding_dims"):
        fig7_embedding_dims(config)
    with pytest.raises(ValueError, match="fig7_embedding_dims"):
        assert_plan_section_16_17_config(config)


def test_encoder_fig7_dim_allowed_only_when_not_strict():
    with pytest.raises(ValueError, match="256"):
        ContextEncoder(embedding_dim=64)
    enc = ContextEncoder(embedding_dim=64, strict_baseline_dim=False)
    x = torch.randn(1, 3, 64, 128)
    assert enc(x).shape == (1, 64)


def test_tsjepa_fig7_config_builds():
    config = apply_embedding_dim(load_config(), 128)
    model = TSJEPA(config)
    z = model.encode_context(torch.randn(2, 3, 64, 128))
    assert z.shape == (2, 128)


def test_dp_holds_last_command_when_packet_lost():
    config = load_config()
    config = copy.deepcopy(config)
    config["simulation"]["trajectory_steps"] = 4
    _, teacher = build_env_and_teacher(config)
    agent = NonlinearDPController(config, teacher)
    mask = [True, False, False, True]
    out = evaluate_agent_closed_loop(config, agent, steps=4, seed=0, packet_receive_mask=mask)
    assert out["miss_behavior"] == "hold_last_command"
    assert out["forces"][1] == pytest.approx(out["forces"][0])
    assert out["forces"][2] == pytest.approx(out["forces"][0])


def test_zero_action_on_miss_is_not_jepa_predict():
    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 3
    _, teacher = build_env_and_teacher(config)
    inner = NonlinearDPController(config, teacher)
    agent = ZeroActionController(config, inner)
    out = evaluate_agent_closed_loop(
        config, agent, steps=3, seed=1, packet_receive_mask=[True, False, False]
    )
    assert agent.miss_behavior == "zero_action"
    assert out["forces"][1] == 0.0
    assert out["forces"][2] == 0.0


def test_supervised_and_ae_forward_shapes():
    config = load_config()
    sup = SupervisedRGBToCommand.from_config(config, kappa=2)
    ae = GenerativeAutoencoder.from_config(config, kappa=2)
    ctx = torch.randn(2, 6, 64, 128)
    assert sup(ctx).shape == (2, 1)
    z, recon, state = ae(ctx)
    assert z.shape == (2, 256)
    assert recon.shape == ctx.shape
    assert state.shape == (2, 4)


def test_fig6_report_separates_no_prediction_and_single_tx():
    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 5
    _, teacher = build_env_and_teacher(config)
    agents = {"dp_hold_last": NonlinearDPController(config, teacher)}
    report = evaluate_fig6_prediction_modes(config, agents, seed=0)
    assert report["figure"] == "6"
    no_pred = report["results"]["dp_hold_last"]["no_prediction"]
    one_tx = report["results"]["dp_hold_last"]["single_initial_transmission"]
    assert no_pred["transmissions"] == 5
    assert one_tx["transmissions"] == 1
    assert no_pred["miss_behavior"] == "hold_last_command"


def test_scheduler_conventional_hold_not_predict():
    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 4
    config["wireless"]["num_devices"] = 2
    config["wireless"]["max_devices_scheduled_J"] = 1
    _, teacher = build_env_and_teacher(config)
    agent = NonlinearDPController(config, teacher)
    out = evaluate_with_scheduler_agent(config, agent, policy="round_robin", snr_db=10.0, seed=0)
    assert out["miss_behavior"] == "hold_last_command"
    assert 0.0 <= out["packet_receive_rate"] <= out["schedule_rate"] + 1e-12


def test_extra_packet_loss_is_labeled_implementation_choice():
    config = copy.deepcopy(load_config())
    config["simulation"]["trajectory_steps"] = 6
    config["wireless"]["num_devices"] = 1
    config["wireless"]["max_devices_scheduled_J"] = 1
    _, teacher = build_env_and_teacher(config)
    agent = NonlinearDPController(config, teacher)
    out = evaluate_with_scheduler_agent(
        config, agent, policy="channel_aware", snr_db=20.0, seed=0, extra_packet_loss=1.0
    )
    assert out["packet_receive_rate"] == pytest.approx(0.0)
    assert out["extra_packet_loss"] == 1.0

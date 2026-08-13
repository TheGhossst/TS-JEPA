"""Plan §9 predictor tests."""

from __future__ import annotations

import copy

import pytest
import torch

from ts_jepa.config import load_config
from ts_jepa.models.predictor import Predictor
from ts_jepa.models.predictor_plan import PLAN_PREDICTOR, assert_plan_predictor_config
from ts_jepa.models.ts_jepa import TSJEPA


def test_plan_predictor_config_matches_baseline_yaml():
    config = load_config()
    assert_plan_predictor_config(config)
    pred = config["ts_jepa"]["predictor"]
    assert pred["hidden_dim"] == PLAN_PREDICTOR["hidden_dim"]
    assert pred["output_dim"] == PLAN_PREDICTOR["output_dim"]
    assert pred["command_source"] == "teacher_dp"


def test_predictor_mlp_layer_sizes():
    pred = Predictor()
    # 257 = concat(z, u) width: implementation choice (plan §18), not paper-specified.
    assert pred.fc_in.in_features == 257
    assert pred.fc_in.out_features == 1024
    assert pred.fc_out.in_features == 1024
    assert pred.fc_out.out_features == 256
    summary = pred.architecture_summary()
    assert summary["stack"] == "Linear-1024-ReLU-Linear-256"
    assert summary["autoregressive"] is True


def test_predictor_rejects_non_plan_dims():
    with pytest.raises(ValueError, match="hidden_dim"):
        Predictor(hidden_dim=512)
    with pytest.raises(ValueError, match="output_dim"):
        Predictor(output_dim=128)


def test_predictor_autoregressive_loop_updates_state():
    pred = Predictor()
    b, kp = 2, 4
    z0 = torch.randn(b, 256)
    commands = torch.randn(b, kp)
    out = pred(z0, commands)
    assert out.shape == (b, kp, 256)

    # Manual loop must match module forward.
    z_current = z0
    manual = []
    for j in range(kp):
        z_next = pred.forward_step(z_current, commands[:, j])
        manual.append(z_next)
        z_current = z_next
    assert torch.allclose(out, torch.stack(manual, dim=1), atol=1e-6)


def test_predictor_input_is_embedding_plus_command_only():
    pred = Predictor()
    z = torch.randn(3, 256)
    u = torch.randn(3, 1)
    x = torch.cat([z, u], dim=-1)
    assert pred.forward_mlp(x).shape == (3, 256)
    with pytest.raises(ValueError, match="input dim"):
        pred.forward_mlp(torch.randn(3, 300))


def test_tsjepa_predictor_wired_from_config():
    config = load_config()
    model = TSJEPA(config)
    assert model.command_source == "teacher_dp"
    assert model.predictor.hidden_dim == 1024
    assert model.predictor.output_dim == 256


def test_plan_predictor_config_rejects_wrong_hidden_dim():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["predictor"]["hidden_dim"] = 512
    with pytest.raises(ValueError, match="hidden_dim"):
        assert_plan_predictor_config(config)


def test_plan_predictor_config_rejects_virtual_channel_input():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["predictor"]["inputs"] = ["embedding", "control_command", "virtual_channel_embeddings"]
    with pytest.raises(ValueError, match="virtual_channel"):
        assert_plan_predictor_config(config)


def test_concat_fusion_is_not_a_plan_requirement():
    """Plan §9: concat vs other fusion is NOT SPECIFIED."""
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["predictor"]["input_tensor_construction"] = "other_fusion"
    assert_plan_predictor_config(config)

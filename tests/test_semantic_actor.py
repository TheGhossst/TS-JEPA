"""Plan §12 semantic actor tests."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from ts_jepa.config import load_config
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.plan.actor import (
    IC_ACTOR_OUTPUT_ACTIVATION,
    PLAN_SEMANTIC_ACTOR,
    PLAN_SEMANTIC_ACTOR_TRAINING,
    assert_plan_semantic_actor_config,
    assert_plan_semantic_actor_training_config,
)
from ts_jepa.plan.enforce import plan_enforced
from ts_jepa.training.train_actor import _assert_encoder_frozen, _assert_optimizer_only_actor


def test_baseline_config_matches_plan_section12():
    config = load_config()
    assert_plan_semantic_actor_config(config)
    assert_plan_semantic_actor_training_config(config)
    arch = config["semantic_actor"]["architecture"]
    opt = config["semantic_actor"]["optimizer"]
    assert arch["hidden_dims"] == PLAN_SEMANTIC_ACTOR["hidden_dims"]
    assert arch["activation"] == "ReLU"
    assert "output_activation" not in PLAN_SEMANTIC_ACTOR
    assert arch["output_activation"] == IC_ACTOR_OUTPUT_ACTIVATION
    assert arch["dropout"] == PLAN_SEMANTIC_ACTOR_TRAINING["dropout"]
    assert config["semantic_actor"]["loss"] == "MSE"
    assert opt["type"] == "AdamW"
    assert opt["learning_rate"] == 0.006
    assert opt["batch_size"] == 200
    assert opt["epochs"] == 300
    assert config["semantic_actor"]["early_stopping"]["enabled"] is True
    assert config["evaluation"]["repetitions"] == 5
    assert config["evaluation"]["reported_result"] == "best"


def test_dp_fixed_overlay_is_results_actor_recipe():
    config = load_config("configs/ts_jepa_dp_fixed.yaml")
    assert plan_enforced(config) is False
    sa = config["semantic_actor"]
    assert sa["loss"] == "Huber"
    assert sa["large_force_weight"] == 10.0
    assert sa["architecture"]["layernorm"] is True
    assert sa["early_stopping"]["split"] == "shuffled_trajectories"
    assert sa["dagger"]["enabled"] is True
    assert sa["dagger"]["expert"] == "lqr"
    # Paper asserts are no-ops when plan.enforce is false.
    assert_plan_semantic_actor_config(config)
    assert_plan_semantic_actor_training_config(config)


def test_semantic_actor_architecture_is_256_1024_256_1():
    actor = SemanticActor(embedding_dim=256, hidden_dims=(1024, 256), dropout=0.2)
    actor.assert_plan_architecture()
    linears = [m for m in actor.net if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [
        (256, 1024),
        (1024, 256),
        (256, 1),
    ]
    relus = [m for m in actor.net if isinstance(m, nn.ReLU)]
    assert len(relus) == 2
    drops = [m for m in actor.net if isinstance(m, nn.Dropout)]
    assert len(drops) == 2
    assert all(d.p == 0.2 for d in drops)


def test_semantic_actor_from_config_matches_plan():
    config = load_config()
    actor = SemanticActor.from_config(config)
    actor.assert_plan_architecture()
    z = torch.randn(4, 256)
    u = actor(z)
    assert u.shape == (4, 1)


def test_results_actor_layernorm_from_dp_fixed_config():
    config = load_config("configs/ts_jepa_dp_fixed.yaml")
    actor = SemanticActor.from_config(config)
    assert any(isinstance(m, nn.LayerNorm) for m in actor.net)
    actor.assert_plan_architecture()
    z = torch.randn(3, 256)
    u = actor(z)
    assert u.shape == (3, 1)
    assert torch.isfinite(u).all()


def test_semantic_actor_rejects_wrong_hidden_depth():
    with pytest.raises(ValueError, match="two hidden layers"):
        SemanticActor(embedding_dim=256, hidden_dims=(1024, 512, 256))


def test_assert_plan_semantic_actor_rejects_wrong_hidden_dims():
    config = copy.deepcopy(load_config())
    config["semantic_actor"]["architecture"]["hidden_dims"] = [512, 128]
    with pytest.raises(ValueError, match="Plan §12|hidden_dims"):
        assert_plan_semantic_actor_config(config)


def test_assert_plan_semantic_actor_rejects_non_mse_loss():
    config = copy.deepcopy(load_config())
    config["semantic_actor"]["loss"] = "L1"
    with pytest.raises(ValueError, match="MSE"):
        assert_plan_semantic_actor_config(config)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda c: c["semantic_actor"]["architecture"].update({"dropout": 0.5}), "dropout"),
        (lambda c: c["semantic_actor"]["optimizer"].update({"type": "Adam"}), "AdamW"),
        (lambda c: c["semantic_actor"]["optimizer"].update({"learning_rate": 0.001}), "0.006"),
        (lambda c: c["semantic_actor"]["optimizer"].update({"batch_size": 64}), "200"),
        (lambda c: c["semantic_actor"]["optimizer"].update({"epochs": 100}), "300"),
        (lambda c: c["semantic_actor"]["early_stopping"].update({"enabled": False}), "early_stopping"),
        (lambda c: c["evaluation"].update({"repetitions": 3}), "repetitions"),
        (lambda c: c["evaluation"].update({"reported_result": "mean"}), "reported_result"),
    ],
)
def test_assert_plan_semantic_actor_training_rejects_mismatches(mutator, match):
    config = copy.deepcopy(load_config())
    mutator(config)
    with pytest.raises(ValueError, match=match):
        assert_plan_semantic_actor_training_config(config)


def test_encoder_frozen_helper():
    config = load_config()
    jepa = TSJEPA(config)
    for p in jepa.parameters():
        p.requires_grad_(False)
    _assert_encoder_frozen(jepa)
    next(jepa.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen"):
        _assert_encoder_frozen(jepa)


def test_optimizer_only_actor_helper():
    config = load_config()
    actor = SemanticActor.from_config(config)
    jepa = TSJEPA(config)
    opt = torch.optim.AdamW(actor.parameters(), lr=0.006)
    _assert_optimizer_only_actor(opt, actor)
    bad = torch.optim.AdamW(
        list(actor.parameters()) + list(jepa.context_encoder.parameters()),
        lr=0.006,
    )
    with pytest.raises(RuntimeError, match="exactly SemanticActor"):
        _assert_optimizer_only_actor(bad, actor)


def test_mse_regression_shape():
    actor = SemanticActor.from_config(load_config())
    z = torch.randn(8, 256)
    target = torch.randn(8, 1)
    pred = actor(z)
    loss = nn.MSELoss()(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in actor.parameters())


def test_actor_linear_output_has_no_clip_or_tanh():
    actor = SemanticActor.from_config(load_config())
    actor.assert_plan_architecture()
    assert isinstance(actor.net[-1], nn.Linear)
    assert actor.net[-1].out_features == 1
    assert not any(isinstance(m, (nn.Tanh, nn.Sigmoid, nn.Hardtanh)) for m in actor.net)
    z = torch.randn(5, 256)
    u = actor(z)
    assert u.shape == (5, 1)


def test_assert_plan_does_not_lock_output_activation_ic():
    """Linear head is a plan §18 IC; changing it must not fail assert_plan_*."""
    config = copy.deepcopy(load_config())
    config["semantic_actor"]["architecture"]["output_activation"] = "tanh"
    assert_plan_semantic_actor_config(config)


def test_assert_plan_rejects_desired_state_actor_input():
    config = copy.deepcopy(load_config())
    config["semantic_actor"]["architecture"]["use_desired_state"] = True
    with pytest.raises(ValueError, match="desired_state"):
        assert_plan_semantic_actor_config(config)
    config = copy.deepcopy(load_config())
    config["semantic_actor"]["inputs"] = ["embedding", "desired_state"]
    with pytest.raises(ValueError, match="desired_state|embedding-only"):
        assert_plan_semantic_actor_config(config)


def test_actor_forward_takes_embedding_only():
    import inspect

    sig = inspect.signature(SemanticActor.forward)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["embedding"]

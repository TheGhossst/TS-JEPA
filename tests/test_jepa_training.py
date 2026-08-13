"""Plan §11 JEPA training hyperparameter tests."""

from __future__ import annotations

import copy

import pytest
import torch

from ts_jepa.config import load_config
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.training.jepa_optimizer import (
    apply_jepa_lr_decay,
    build_jepa_optimizer,
    jepa_learning_rate_at_epoch,
    jepa_trainable_parameters,
    should_apply_jepa_lr_decay,
)
from ts_jepa.training.jepa_training_plan import PLAN_JEPA_TRAINING, assert_plan_jepa_training_config


def test_plan_jepa_training_config_matches_baseline_yaml():
    config = load_config()
    assert_plan_jepa_training_config(config)
    opt = config["ts_jepa"]["optimizer"]
    assert opt["type"] == PLAN_JEPA_TRAINING["optimizer_type"]
    assert opt["learning_rate"] == PLAN_JEPA_TRAINING["learning_rate"]
    assert opt["batch_size"] == PLAN_JEPA_TRAINING["batch_size"]
    assert opt["epochs"] == PLAN_JEPA_TRAINING["epochs"]
    assert opt["weight_decay"] == PLAN_JEPA_TRAINING["weight_decay"]
    assert config["ts_jepa"]["lr_decay"]["factor"] == PLAN_JEPA_TRAINING["lr_decay_factor"]
    assert config["ts_jepa"]["lr_decay"]["interval_epochs"] == PLAN_JEPA_TRAINING["lr_decay_interval_epochs"]
    assert config["ts_jepa"]["target_encoder"]["ema_decay"] == PLAN_JEPA_TRAINING["ema_decay"]
    assert config["ts_jepa"]["early_stopping"]["enabled"] is True
    assert config["evaluation"]["repetitions"] == PLAN_JEPA_TRAINING["repetitions"]
    assert config["evaluation"]["reported_result"] == PLAN_JEPA_TRAINING["reported_result"]
    assert config["input"]["kappa"] == PLAN_JEPA_TRAINING["kappa"]
    assert config["ts_jepa"]["prediction_horizon"]["Kp"] == PLAN_JEPA_TRAINING["Kp"]
    assert config["ts_jepa"]["encoder"]["embedding_dim"] == PLAN_JEPA_TRAINING["embedding_dim"]


def test_build_jepa_optimizer_is_sgd_with_plan_hparams():
    config = load_config()
    model = TSJEPA(config)
    optimizer = build_jepa_optimizer(model, config)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 0.2
    assert optimizer.param_groups[0]["weight_decay"] == 0.0004
    assert optimizer.param_groups[0]["momentum"] == 0.0
    trainable = {id(p) for p in jepa_trainable_parameters(model)}
    opt_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    target_params = {id(p) for p in model.target_encoder.parameters()}
    assert opt_params == trainable
    assert trainable.isdisjoint(target_params)


def test_build_jepa_optimizer_allows_smoke_batch_override():
    """Smoke tests shrink batch_size; optimizer build must not hard-assert §11."""
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["optimizer"]["batch_size"] = 2
    config["ts_jepa"]["prediction_horizon"]["Kp"] = 5
    model = TSJEPA(config)
    optimizer = build_jepa_optimizer(model, config)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 0.2


def test_build_jepa_optimizer_rejects_adam():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["optimizer"]["type"] = "Adam"
    model = TSJEPA(config)
    with pytest.raises(ValueError, match="SGD"):
        build_jepa_optimizer(model, config)


def test_lr_schedule_matches_plan_section11():
    config = load_config()
    base = float(config["ts_jepa"]["optimizer"]["learning_rate"])
    assert jepa_learning_rate_at_epoch(base, 0) == base
    assert jepa_learning_rate_at_epoch(base, 19) == base
    assert jepa_learning_rate_at_epoch(base, 20) == pytest.approx(base * 0.99)
    assert jepa_learning_rate_at_epoch(base, 40) == pytest.approx(base * 0.99**2)
    assert should_apply_jepa_lr_decay(20, config)
    assert not should_apply_jepa_lr_decay(19, config)


def test_apply_jepa_lr_decay_updates_optimizer():
    config = load_config()
    model = TSJEPA(config)
    optimizer = build_jepa_optimizer(model, config)
    new_lr = apply_jepa_lr_decay(optimizer, config)
    assert new_lr == pytest.approx(0.2 * 0.99)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "Adam"),
        ("learning_rate", 0.001),
        ("epochs", 200),
        ("weight_decay", 1e-5),
        ("batch_size", 128),
    ],
)
def test_plan_jepa_training_rejects_wrong_or_forbidden_defaults(field, value):
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["optimizer"][field] = value
    with pytest.raises(ValueError, match="Plan §11"):
        assert_plan_jepa_training_config(config)


def test_plan_jepa_training_rejects_forbidden_ema_decay():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["target_encoder"]["ema_decay"] = 0.996
    with pytest.raises(ValueError, match="0.996|0.99"):
        assert_plan_jepa_training_config(config)


def test_plan_jepa_training_rejects_disabled_early_stopping():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["early_stopping"]["enabled"] = False
    with pytest.raises(ValueError, match="early_stopping"):
        assert_plan_jepa_training_config(config)


def test_plan_jepa_training_rejects_wrong_repetition_count():
    config = copy.deepcopy(load_config())
    config["evaluation"]["repetitions"] = 1
    with pytest.raises(ValueError, match="repetitions"):
        assert_plan_jepa_training_config(config)

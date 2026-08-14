"""Plan §10 Algorithm 1 TS-JEPA training procedure tests."""

from __future__ import annotations

import copy

import pytest
import torch

from ts_jepa.config import load_config
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.training.jepa_optimizer import build_jepa_optimizer, jepa_trainable_parameters
from ts_jepa.training.jepa_procedure import (
    PLAN_JEPA_PROCEDURE,
    assert_plan_jepa_procedure_config,
    jepa_forward_batch,
    jepa_sgd_and_ema_step,
)


def _tiny_batch(config: dict, batch_size: int = 2) -> dict[str, torch.Tensor]:
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    return {
        "context": torch.randn(batch_size, 3, 64, 128),
        "future_frames": torch.randn(batch_size, kp, 3, 64, 128),
        "teacher_commands_norm": torch.randn(batch_size, kp),
    }


def test_plan_jepa_procedure_config_matches_baseline_yaml():
    config = load_config()
    assert_plan_jepa_procedure_config(config)
    proc = config["ts_jepa"]["training_procedure"]
    assert tuple(proc["steps"]) == PLAN_JEPA_PROCEDURE["steps"]
    assert proc["ema_after_optimizer_step"] is True
    assert proc["target_stop_gradient"] is True


def test_jepa_forward_batch_shapes_and_loss():
    config = load_config()
    model = TSJEPA(config)
    batch = _tiny_batch(config)
    result = jepa_forward_batch(model, batch)
    b, kp = batch["context"].shape[0], int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    assert result.z_context.shape == (b, 256)
    assert result.z_target.shape == (b, kp, 256)
    assert result.z_pred.shape == (b, kp, 256)
    assert torch.isfinite(result.loss)
    assert result.commands_norm.shape == (b, kp)


def test_predictor_detach_blocks_bptt_into_encoder():
    """Later-horizon cosine must not backprop through previous Pφ steps into Ψθ."""
    config = load_config()
    model = TSJEPA(config)
    b, kp = 2, int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    context = torch.randn(b, 3, 64, 128, requires_grad=False)
    commands = torch.randn(b, kp)
    z = model.encode_context(context)
    z_pred = model.predict(z, commands)
    z_pred[:, -1].sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.context_encoder.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.predictor.parameters())


def test_jepa_forward_target_has_no_grad_graph():
    config = load_config()
    model = TSJEPA(config)
    model.train()
    batch = _tiny_batch(config)
    result = jepa_forward_batch(model, batch)
    assert result.z_target.requires_grad is False
    result.loss.backward()
    for p in model.target_encoder.parameters():
        assert p.grad is None
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())


def test_jepa_sgd_and_ema_updates_context_not_via_target_grads():
    config = load_config()
    model = TSJEPA(config)
    optimizer = build_jepa_optimizer(model, config)
    batch = _tiny_batch(config)

    before_ctx = [p.detach().clone() for p in model.context_encoder.parameters()]
    before_tgt = [p.detach().clone() for p in model.target_encoder.parameters()]
    before_pred = [p.detach().clone() for p in model.predictor.parameters()]

    result = jepa_forward_batch(model, batch)
    result.loss.backward()
    jepa_sgd_and_ema_step(model, optimizer)

    ctx_changed = any(
        not torch.equal(b, a.detach()) for b, a in zip(before_ctx, model.context_encoder.parameters())
    )
    pred_changed = any(
        not torch.equal(b, a.detach()) for b, a in zip(before_pred, model.predictor.parameters())
    )
    # Target moves via EMA toward (updated) context, not via backprop.
    tgt_changed = any(
        not torch.equal(b, a.detach()) for b, a in zip(before_tgt, model.target_encoder.parameters())
    )
    assert ctx_changed
    assert pred_changed
    assert tgt_changed
    trainable = {id(p) for p in jepa_trainable_parameters(model)}
    assert all(id(p) not in trainable for p in model.target_encoder.parameters())


def test_optimizer_excludes_target_encoder():
    config = load_config()
    model = TSJEPA(config)
    optimizer = build_jepa_optimizer(model, config)
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    for p in model.target_encoder.parameters():
        assert id(p) not in opt_ids


def test_plan_jepa_procedure_rejects_wrong_step_order():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["training_procedure"]["steps"] = list(reversed(PLAN_JEPA_PROCEDURE["steps"]))
    with pytest.raises(ValueError, match="Plan §10"):
        assert_plan_jepa_procedure_config(config)


def test_plan_jepa_procedure_rejects_ema_before_optimizer_flag():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["training_procedure"]["ema_after_optimizer_step"] = False
    with pytest.raises(ValueError, match="ema_after_optimizer_step"):
        assert_plan_jepa_procedure_config(config)

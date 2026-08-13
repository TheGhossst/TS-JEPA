"""Plan §10 JEPA cosine alignment loss tests."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from ts_jepa.config import load_config
from ts_jepa.losses.jepa_loss import cosine_alignment_loss, jepa_cosine_similarity, jepa_loss
from ts_jepa.losses.loss_plan import PLAN_JEPA_LOSS, assert_plan_jepa_loss_config
from ts_jepa.models.ts_jepa import TSJEPA


def test_plan_jepa_loss_config_matches_baseline_yaml():
    config = load_config()
    assert_plan_jepa_loss_config(config)
    loss = config["ts_jepa"]["loss"]
    assert loss["paper_objective"] == PLAN_JEPA_LOSS["paper_objective"]
    assert loss["implementation"] == PLAN_JEPA_LOSS["implementation"]


def test_jepa_cosine_similarity_matches_manual_formula():
    pred = torch.tensor([[[3.0, 4.0], [1.0, 0.0]]], dtype=torch.float64)
    target = torch.tensor([[[6.0, 8.0], [0.0, 2.0]]], dtype=torch.float64)
    cos = jepa_cosine_similarity(pred, target)
    manual = []
    for j in range(pred.shape[1]):
        p = pred[0, j]
        t = target[0, j]
        manual.append(float((p @ t) / (p.norm() * t.norm())))
    assert torch.allclose(cos, torch.tensor([manual], dtype=torch.float64), atol=1e-6)


def test_jepa_loss_is_negative_mean_cosine():
    pred = torch.randn(4, 15, 256)
    target = torch.randn(4, 15, 256)
    cos = jepa_cosine_similarity(pred, target)
    expected = -cos.mean()
    assert torch.allclose(jepa_loss(pred, target), expected)


def test_cosine_alignment_loss_equivalent_gradients_to_jepa_loss():
    pred = torch.randn(2, 15, 256, requires_grad=True)
    target = torch.randn(2, 15, 256)
    loss_a = jepa_loss(pred, target)
    loss_b = cosine_alignment_loss(pred, target)
    assert torch.allclose(loss_a + 1.0, loss_b)
    grad_a = torch.autograd.grad(loss_a, pred, retain_graph=True)[0]
    pred2 = pred.detach().clone().requires_grad_(True)
    grad_b = torch.autograd.grad(cosine_alignment_loss(pred2, target), pred2)[0]
    assert torch.allclose(grad_a, grad_b)


def test_jepa_loss_zero_when_pred_equals_target():
    z = F.normalize(torch.randn(3, 15, 256), dim=-1)
    loss = jepa_loss(z, z)
    align = cosine_alignment_loss(z, z)
    assert torch.allclose(loss, torch.tensor(-1.0), atol=1e-5)
    assert torch.allclose(align, torch.tensor(0.0), atol=1e-5)


def test_target_branch_does_not_receive_gradients():
    pred = torch.randn(2, 15, 256, requires_grad=True)
    target = torch.randn(2, 15, 256, requires_grad=True)
    loss = jepa_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert target.grad is None


def test_tsjepa_training_tensors_use_plan_loss():
    config = load_config()
    model = TSJEPA(config)
    b, kp = 2, int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    context = torch.randn(b, 3, 64, 128)
    future = torch.randn(b, kp, 3, 64, 128)
    commands = torch.randn(b, kp)
    z = model.encode_context(context)
    z_tgt = model.encode_targets(future)
    z_pred = model.predict(z, commands)
    assert z_tgt.shape == (b, kp, 256)
    assert z_pred.shape == (b, kp, 256)
    loss = jepa_loss(z_pred, z_tgt)
    assert torch.isfinite(loss)


def test_plan_jepa_loss_config_rejects_wrong_objective():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["loss"]["paper_objective"] = "mse"
    with pytest.raises(ValueError, match="Plan §10"):
        assert_plan_jepa_loss_config(config)


def test_plan_jepa_loss_config_rejects_vicreg():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["loss"]["paper_objective"] = "vicreg"
    with pytest.raises(ValueError, match="Plan §10"):
        assert_plan_jepa_loss_config(config)


def test_plan_jepa_loss_config_rejects_reconstruction_weight():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["loss"]["reconstruction_weight"] = 1.0
    with pytest.raises(ValueError, match="reconstruction"):
        assert_plan_jepa_loss_config(config)

from __future__ import annotations

import copy

import pytest
import torch

from ts_jepa.config import load_config
from ts_jepa.losses.cosine import cosine_alignment_loss
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ema import assert_target_initialized_from_context, ema_update, initialize_target_from_context
from ts_jepa.models.encoder import ContextEncoder, TargetEncoder
from ts_jepa.models.encoder_plan import PLAN_ENCODER, PLAN_TARGET_ENCODER, assert_plan_encoder_config
from ts_jepa.models.ts_jepa import TSJEPA


def test_plan_encoder_config_matches_baseline_yaml():
    config = load_config()
    assert_plan_encoder_config(config)
    assert config["ts_jepa"]["encoder"]["widths"] == PLAN_ENCODER["widths"]
    assert config["ts_jepa"]["target_encoder"]["ema_decay"] == PLAN_TARGET_ENCODER["ema_decay"]


def test_context_encoder_stem_and_stages():
    """Stem / 4×8 pool / 3-ch RGB input (Algorithm 1). 6-ch concat is supervised/AE only."""
    enc = ContextEncoder(in_channels=3, widths=[64, 128, 256], embedding_dim=256)
    stem_conv = enc.stem[0]
    assert isinstance(stem_conv, torch.nn.Conv2d)
    assert stem_conv.kernel_size == (7, 7)
    assert stem_conv.stride == (2, 2)
    assert stem_conv.in_channels == 3
    assert stem_conv.out_channels == 64
    assert isinstance(enc.stem[3], torch.nn.MaxPool2d)
    summary = enc.architecture_summary()
    assert summary["widths"] == [64, 128, 256]
    assert summary["embedding_dim"] == 256
    assert summary["spatial_pool_hw"] == [4, 8]
    assert summary["head"] == "SpatialPool-Flatten-Linear"
    assert enc.global_pool.output_size == (4, 8)


def test_context_encoder_rejects_non_plan_widths():
    try:
        ContextEncoder(widths=[32, 64, 128], embedding_dim=256)
    except ValueError as exc:
        assert "widths" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-plan widths")


def test_model_forward_shapes_and_loss():
    config = load_config()
    model = TSJEPA(config)
    actor = SemanticActor(embedding_dim=256)
    b, kp = 2, config["ts_jepa"]["prediction_horizon"]["Kp"]
    context = torch.randn(b, 3, 64, 128)
    future = torch.randn(b, kp, 3, 64, 128)
    commands = torch.randn(b, kp)
    z = model.encode_context(context)
    z_tgt = model.encode_targets(future)
    z_pred = model.predict(z, commands)
    loss = cosine_alignment_loss(z_pred, z_tgt)
    assert z.shape == (b, 256)
    assert z_tgt.shape == (b, kp, 256)
    assert z_pred.shape == (b, kp, 256)
    assert torch.isfinite(loss)
    u = actor(z)
    assert u.shape == (b, 1)
    model.ema_step()


def test_target_encoder_same_class_and_init():
    config = load_config()
    context = ContextEncoder(in_channels=3, widths=[64, 128, 256], embedding_dim=256)
    target = initialize_target_from_context(context)
    assert isinstance(target, TargetEncoder)
    assert_target_initialized_from_context(context, target)
    assert all(not p.requires_grad for p in target.parameters())


def test_target_encoder_no_grad_on_backward():
    config = load_config()
    model = TSJEPA(config)
    model.train()
    model.context_encoder.train()
    model.target_encoder.eval()
    b, kp = 2, int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    context = torch.randn(b, 3, 64, 128)
    future = torch.randn(b, kp, 3, 64, 128)
    commands = torch.randn(b, kp)
    z = model.encode_context(context)
    z_tgt = model.encode_targets(future)
    z_pred = model.predict(z, commands)
    loss = cosine_alignment_loss(z_pred, z_tgt)
    loss.backward()
    for param in model.target_encoder.parameters():
        assert param.grad is None
    assert any(p.grad is not None for p in model.context_encoder.parameters())


def test_ema_update_moves_target_toward_context():
    config = load_config()
    model = TSJEPA(config)
    before = [p.detach().clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(0.1)
    model.ema_step()
    after = list(model.target_encoder.parameters())
    changed = any(not torch.equal(b, a.detach()) for b, a in zip(before, after))
    assert changed
    decay = float(config["ts_jepa"]["target_encoder"]["ema_decay"])
    for b, t_param, o_param in zip(before, model.target_encoder.parameters(), model.context_encoder.parameters()):
        expected = decay * b + (1.0 - decay) * o_param.detach()
        assert torch.allclose(t_param.detach(), expected, atol=1e-6)


def test_spatial_pool_is_not_a_plan_requirement():
    """Plan §7 / §18: pooling / 4×8 head is NOT SPECIFIED."""
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["encoder"]["spatial_pool_hw"] = [1, 1]
    assert_plan_encoder_config(config)


def test_resnet_stem_and_block_count_are_not_plan_requirements():
    """Plan §7 / §18: stem and blocks_per_stage are NOT SPECIFIED."""
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["encoder"]["blocks_per_stage"] = 8
    config["ts_jepa"]["encoder"]["stem"] = {"kernel_size": 3, "stride": 1, "pooling": "none"}
    assert_plan_encoder_config(config)


def test_plan_config_rejects_wrong_widths():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["encoder"]["widths"] = [32, 64, 128]
    with pytest.raises(ValueError, match="widths"):
        assert_plan_encoder_config(config)


def test_plan_config_rejects_wrong_ema():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["target_encoder"]["ema_decay"] = 0.996
    with pytest.raises(ValueError, match="ema_decay"):
        assert_plan_encoder_config(config)

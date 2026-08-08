from __future__ import annotations

import torch

from ts_jepa.config import load_config
from ts_jepa.losses.cosine import cosine_alignment_loss
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA


def test_model_forward_shapes_and_loss():
    config = load_config()
    model = TSJEPA(config)
    actor = SemanticActor(embedding_dim=256)
    b, kp = 2, config["ts_jepa"]["prediction_horizon"]["Kp"]
    context = torch.randn(b, 6, 64, 128)
    future = torch.randn(b, kp, 6, 64, 128)
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

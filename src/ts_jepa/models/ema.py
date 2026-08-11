from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn


def initialize_target_from_context(context_encoder: nn.Module) -> nn.Module:
    """
    Plan §7: θ̄ ← θ — clone context encoder weights into the target encoder.

    Target parameters are frozen (no backprop) and the module stays in eval mode.
    """
    target = copy.deepcopy(context_encoder)
    for param in target.parameters():
        param.requires_grad_(False)
    target.eval()
    return target


def clone_encoder(encoder: nn.Module) -> nn.Module:
    """Alias for initialize_target_from_context (backward compatible)."""
    return initialize_target_from_context(encoder)


def assert_target_initialized_from_context(context_encoder: nn.Module, target_encoder: nn.Module) -> None:
    """Verify plan §7 initialization θ̄ ← θ."""
    for t_param, c_param in zip(target_encoder.parameters(), context_encoder.parameters()):
        if not torch.equal(t_param.data, c_param.data):
            raise AssertionError("Target encoder parameters differ from context encoder at initialization")
    for t_param in target_encoder.parameters():
        if t_param.requires_grad:
            raise AssertionError("Target encoder parameters must be frozen (requires_grad=False)")


def ema_update(target: nn.Module, online: nn.Module, decay: float = 0.99) -> None:
    """
    Plan §7: θ̄ ← η θ̄ + (1 - η) θ with η = ema_decay.

    BatchNorm running-stat buffers are copied from the online encoder each step
    (IMPLEMENTATION CHOICE — plan specifies θ only).
    """
    with torch.no_grad():
        for t_param, o_param in zip(target.parameters(), online.parameters()):
            t_param.data.mul_(decay).add_(o_param.data, alpha=1.0 - decay)
        for t_buf, o_buf in zip(target.buffers(), online.buffers()):
            t_buf.data.copy_(o_buf.data)


def trainable_encoder_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Parameters optimized during JEPA training (context encoder + predictor only)."""
    if hasattr(model, "context_encoder") and hasattr(model, "predictor"):
        return list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    raise TypeError("Expected TSJEPA-like module with context_encoder and predictor")

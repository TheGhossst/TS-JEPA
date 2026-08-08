from __future__ import annotations

import copy

import torch.nn as nn


def clone_encoder(encoder: nn.Module) -> nn.Module:
    target = copy.deepcopy(encoder)
    for param in target.parameters():
        param.requires_grad_(False)
    target.eval()
    return target


def ema_update(target: nn.Module, online: nn.Module, decay: float = 0.99) -> None:
    """θ̄ ← η θ̄ + (1 - η) θ with η = ema_decay (PAPER-SPECIFIED)."""
    import torch

    with torch.no_grad():
        for t_param, o_param in zip(target.parameters(), online.parameters()):
            t_param.data.mul_(decay).add_(o_param.data, alpha=1.0 - decay)
        for t_buf, o_buf in zip(target.buffers(), online.buffers()):
            t_buf.data.copy_(o_buf.data)

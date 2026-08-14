"""Plan §11 JEPA optimizer and LR schedule helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.plan.training import IC_SGD_MOMENTUM, PLAN_JEPA_TRAINING

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)


def jepa_trainable_parameters(model: TSJEPA) -> list[torch.nn.Parameter]:
    """Plan §10 Algorithm 1 steps 5–6: optimize θ (context encoder) and ϕ (predictor) only."""
    return list(model.context_encoder.parameters()) + list(model.predictor.parameters())


def jepa_sgd_param_groups(model: TSJEPA, weight_decay: float) -> list[dict[str, Any]]:
    """
    Table II weight decay 0.0004. BatchNorm affine params are excluded (IC):
    cosine JEPA + WD on BN scale is a known oscillation source and is not specified.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in (model.context_encoder, model.predictor):
        for submodule in module.modules():
            bucket = no_decay if isinstance(submodule, _BN_TYPES) else decay
            for param in submodule.parameters(recurse=False):
                if not param.requires_grad or id(param) in seen:
                    continue
                bucket.append(param)
                seen.add(id(param))
    groups = [{"params": decay, "weight_decay": float(weight_decay)}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_jepa_optimizer(model: TSJEPA, config: dict[str, Any]) -> torch.optim.SGD:
    """
    Build SGD optimizer for context encoder + predictor (plan §11).

    Does not call assert_plan_jepa_training_config — smoke tests may override
    batch_size / Kp / epochs. Baseline scripts assert plan values before training.
    Target encoder Ψθ̄ is updated via EMA only (plan §8 / §10 Algorithm 1 step 6).
    """
    opt_cfg = config["ts_jepa"]["optimizer"]
    opt_type = str(opt_cfg.get("type", "SGD"))
    if opt_type != "SGD":
        raise ValueError(
            f"plan §11 requires SGD for TS-JEPA (not Adam/BYOL defaults); got {opt_type!r}"
        )
    momentum = float(opt_cfg.get("momentum", IC_SGD_MOMENTUM))
    return torch.optim.SGD(
        jepa_sgd_param_groups(model, float(opt_cfg["weight_decay"])),
        lr=float(opt_cfg["learning_rate"]),
        momentum=momentum,
    )


def should_apply_jepa_lr_decay(epoch: int, config: dict[str, Any]) -> bool:
    """Plan §11: multiply LR by 0.99 every 20 completed epochs."""
    interval = int(config["ts_jepa"]["lr_decay"]["interval_epochs"])
    return int(epoch) > 0 and int(epoch) % interval == 0


def apply_jepa_lr_decay(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> float:
    """Apply one LR decay step; returns the new learning rate."""
    factor = float(config["ts_jepa"]["lr_decay"]["factor"])
    for group in optimizer.param_groups:
        group["lr"] *= factor
    return float(optimizer.param_groups[0]["lr"])


def jepa_warmup_epochs(config: dict[str, Any]) -> int:
    """Linear warmup length (IC). Table II is silent; 0 disables warmup."""
    return max(0, int(config["ts_jepa"]["optimizer"].get("lr_warmup_epochs", 0)))


def jepa_scheduled_lr(epoch: int, config: dict[str, Any]) -> float:
    """
    Learning rate used *during* 1-indexed training epoch ``epoch``.

    IC linear warmup for the first ``lr_warmup_epochs`` epochs, then Table II
    peak LR 0.2 with ×0.99 every 20 completed epochs.
    Epoch 1 uses ``base / warmup`` (not 0) so the first epoch still takes a step.
    """
    base = float(config["ts_jepa"]["optimizer"]["learning_rate"])
    warmup = jepa_warmup_epochs(config)
    factor = float(config["ts_jepa"]["lr_decay"]["factor"])
    interval = int(config["ts_jepa"]["lr_decay"]["interval_epochs"])
    e = max(1, int(epoch))
    if warmup > 0 and e <= warmup:
        return base * float(e) / float(warmup)
    n_decay = (e - 1) // interval
    return base * (float(factor) ** n_decay)


def apply_jepa_scheduled_lr(
    optimizer: torch.optim.Optimizer, epoch: int, config: dict[str, Any]
) -> float:
    """Write the scheduled LR into every param group. Returns that LR."""
    lr = jepa_scheduled_lr(epoch, config)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def jepa_learning_rate_at_epoch(
    base_lr: float,
    epoch: int,
    *,
    factor: float = PLAN_JEPA_TRAINING["lr_decay_factor"],
    interval_epochs: int = PLAN_JEPA_TRAINING["lr_decay_interval_epochs"],
) -> float:
    """
    Table II decay only: LR after completing `epoch` epochs (decay at 20, 40, ...).

    Does not include the IC warmup. Prefer ``jepa_scheduled_lr`` in the train loop.
    """
    if epoch <= 0:
        return float(base_lr)
    n_decay = int(epoch) // int(interval_epochs)
    return float(base_lr) * (float(factor) ** n_decay)

"""Plan §12 JEPA optimizer and LR schedule helpers."""

from __future__ import annotations

from typing import Any

import torch

from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.training.jepa_training_plan import PLAN_JEPA_TRAINING


def jepa_trainable_parameters(model: TSJEPA) -> list[torch.nn.Parameter]:
    """Plan §13 steps 5–6: optimize θ (context encoder) and ϕ (predictor) only."""
    return list(model.context_encoder.parameters()) + list(model.predictor.parameters())


def build_jepa_optimizer(model: TSJEPA, config: dict[str, Any]) -> torch.optim.SGD:
    """
    Build SGD optimizer for context encoder + predictor (plan §12).

    Does not call assert_plan_jepa_training_config — smoke tests may override
    batch_size / Kp / epochs. Baseline scripts assert plan values before training.
    Target encoder Ψθ̄ is updated via EMA only (plan §7 / §13 step 6).
    """
    opt_cfg = config["ts_jepa"]["optimizer"]
    opt_type = str(opt_cfg.get("type", "SGD"))
    if opt_type != "SGD":
        raise ValueError(
            f"plan §12 requires SGD for TS-JEPA (not Adam/BYOL defaults); got {opt_type!r}"
        )
    momentum = float(opt_cfg.get("momentum", PLAN_JEPA_TRAINING["momentum"]))
    return torch.optim.SGD(
        jepa_trainable_parameters(model),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        momentum=momentum,
    )


def should_apply_jepa_lr_decay(epoch: int, config: dict[str, Any]) -> bool:
    """Plan §12: multiply LR by 0.99 every 20 completed epochs."""
    interval = int(config["ts_jepa"]["lr_decay"]["interval_epochs"])
    return int(epoch) > 0 and int(epoch) % interval == 0


def apply_jepa_lr_decay(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> float:
    """Apply one LR decay step; returns the new learning rate."""
    factor = float(config["ts_jepa"]["lr_decay"]["factor"])
    for group in optimizer.param_groups:
        group["lr"] *= factor
    return float(optimizer.param_groups[0]["lr"])


def jepa_learning_rate_at_epoch(
    base_lr: float,
    epoch: int,
    *,
    factor: float = PLAN_JEPA_TRAINING["lr_decay_factor"],
    interval_epochs: int = PLAN_JEPA_TRAINING["lr_decay_interval_epochs"],
) -> float:
    """
    Learning rate after completing `epoch` epochs (decay at epochs 20, 40, ...).

    Matches `train_jepa` which applies decay when `epoch % interval_epochs == 0`.
    """
    if epoch <= 0:
        return float(base_lr)
    n_decay = int(epoch) // int(interval_epochs)
    return float(base_lr) * (float(factor) ** n_decay)

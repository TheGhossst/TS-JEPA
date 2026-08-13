"""
Plan §10 Algorithm 1 TS-JEPA training procedure (per effective training batch).

Steps:
  1. Context encoding:  z_k = Ψθ(x_k)
  2. Target encoding:   z̄_{k+j} = Ψθ̄(x_{k+j})  with stop-gradient
  3. Autoregressive prediction of z̃_{k+1..k+Kp} using §9 command source
  4. JEPA loss L_JEPA = -mean(cos_sim)
  5. SGD update of θ (context encoder) and ϕ (predictor) only
  6. EMA update θ̄ ← η θ̄ + (1-η) θ
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ts_jepa.losses.jepa_loss import jepa_loss
from ts_jepa.models.predictor_command_resolution import (
    PredictorCommandResolution,
    select_predictor_conditioning_commands,
)
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.plan.procedure import PLAN_JEPA_PROCEDURE, assert_plan_jepa_procedure_config

__all__ = [
    "JEPAForwardResult",
    "PLAN_JEPA_PROCEDURE",
    "assert_plan_jepa_procedure_config",
    "jepa_forward_batch",
    "jepa_sgd_and_ema_step",
]


@dataclass(frozen=True)
class JEPAForwardResult:
    """Outputs of plan §10 Algorithm 1 steps 1–4 for one microbatch / batch."""

    z_context: torch.Tensor
    z_target: torch.Tensor
    z_pred: torch.Tensor
    loss: torch.Tensor
    commands_norm: torch.Tensor


def jepa_forward_batch(
    model: TSJEPA,
    batch: dict[str, torch.Tensor],
    resolution: PredictorCommandResolution | None = None,
) -> JEPAForwardResult:
    """
    Plan §10 Algorithm 1 steps 1–4 for one batch.

    Does not call optimizer or EMA (steps 5–6 belong to the outer train loop /
    accumulation group).
    """
    resolution = resolution or model.command_resolution
    context = batch["context"]
    future = batch["future_frames"]
    commands_norm = select_predictor_conditioning_commands(batch, resolution)

    # Step 1 — context encoding
    z_context = model.encode_context(context)

    # Step 2 — target encoding (stop-gradient)
    with torch.no_grad():
        z_target = model.encode_targets(future)

    # Step 3 — autoregressive prediction (§9 command source)
    z_pred = model.predict(z_context, commands_norm)

    # Step 4 — JEPA loss
    loss = jepa_loss(z_pred, z_target)

    return JEPAForwardResult(
        z_context=z_context,
        z_target=z_target,
        z_pred=z_pred,
        loss=loss,
        commands_norm=commands_norm,
    )


def jepa_sgd_and_ema_step(model: TSJEPA, optimizer: torch.optim.Optimizer) -> None:
    """
    Plan §10 Algorithm 1 steps 5–6: SGD on θ,ϕ then EMA on θ̄.

    Call once per effective batch (after gradient accumulation completes).
    """
    # Step 5 — gradient update (θ, ϕ only; optimizer was built that way)
    optimizer.step()
    # Step 6 — EMA target update
    model.ema_step()
    optimizer.zero_grad(set_to_none=True)

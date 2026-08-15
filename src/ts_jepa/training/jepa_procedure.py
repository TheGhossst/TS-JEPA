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

from ts_jepa.losses.jepa_loss import jepa_loss, vicreg_covariance_loss, vicreg_variance_loss
from ts_jepa.models.predictor_command_resolution import (
    PredictorCommandResolution,
    select_predictor_conditioning_commands,
)
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.plan.procedure import PLAN_JEPA_PROCEDURE, assert_plan_jepa_procedure_config
from ts_jepa.training.jepa_optimizer import jepa_trainable_parameters

__all__ = [
    "JEPAForwardResult",
    "PLAN_JEPA_PROCEDURE",
    "assert_plan_jepa_procedure_config",
    "jepa_forward_batch",
    "jepa_sgd_and_ema_step",
    "vicreg_regularizer",
]


@dataclass(frozen=True)
class JEPAForwardResult:
    """Outputs of plan §10 Algorithm 1 steps 1–4 for one microbatch / batch."""

    z_context: torch.Tensor
    z_target: torch.Tensor
    z_pred: torch.Tensor
    loss: torch.Tensor
    commands_norm: torch.Tensor
    cosine_loss: torch.Tensor
    vicreg_variance: torch.Tensor
    vicreg_covariance: torch.Tensor


def vicreg_regularizer(
    z: torch.Tensor,
    *,
    variance_weight: float,
    covariance_weight: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weighted VICReg on embeddings [B, D].

    Returns (weighted_sum, unweighted_variance_hinge, unweighted_covariance).
    The logged ``vicreg_var`` is the hinge mean(relu(γ - std)), not the std itself.
    """
    zero = z.new_zeros(())
    var_w = float(variance_weight)
    cov_w = float(covariance_weight)
    var_term = vicreg_variance_loss(z, gamma=float(gamma)) if var_w != 0.0 else zero
    cov_term = vicreg_covariance_loss(z) if cov_w != 0.0 else zero
    return var_w * var_term + cov_w * cov_term, var_term, cov_term


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

    # Step 4 — JEPA cosine loss, plus optional VICReg on raw context embeddings.
    cosine = jepa_loss(z_pred, z_target)
    var_w = float(getattr(model, "vicreg_variance_weight", 0.0) or 0.0)
    cov_w = float(getattr(model, "vicreg_covariance_weight", 0.0) or 0.0)
    gamma = float(getattr(model, "vicreg_gamma", 1.0) or 1.0)
    vicreg, var_term, cov_term = vicreg_regularizer(
        z_context,
        variance_weight=var_w,
        covariance_weight=cov_w,
        gamma=gamma,
    )
    loss = cosine + vicreg

    return JEPAForwardResult(
        z_context=z_context,
        z_target=z_target,
        z_pred=z_pred,
        loss=loss,
        commands_norm=commands_norm,
        cosine_loss=cosine,
        vicreg_variance=var_term,
        vicreg_covariance=cov_term,
    )


def jepa_sgd_and_ema_step(
    model: TSJEPA,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float | None = None,
) -> None:
    """
    Plan §10 Algorithm 1 steps 5–6: SGD on θ,ϕ then EMA on θ̄.

    Call once per effective batch (after gradient accumulation completes).
    ``max_grad_norm`` is an implementation choice (Table II is silent).
    """
    # Step 5 — gradient update (θ, ϕ only; optimizer was built that way)
    if max_grad_norm is not None and float(max_grad_norm) > 0.0:
        torch.nn.utils.clip_grad_norm_(jepa_trainable_parameters(model), float(max_grad_norm))
    optimizer.step()
    # Step 6 — EMA target update
    model.ema_step()
    optimizer.zero_grad(set_to_none=True)

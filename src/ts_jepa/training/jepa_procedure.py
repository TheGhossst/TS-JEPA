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

import math
from dataclasses import dataclass
from typing import Any

import torch

from ts_jepa.losses.jepa_loss import (
    command_contrastive_hinge,
    jepa_loss,
    vicreg_covariance_loss,
    vicreg_variance_loss,
)
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
    "attach_command_norm_range",
    "jepa_forward_batch",
    "jepa_sgd_and_ema_step",
    "sample_broad_command_pair",
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
    command_contrast: torch.Tensor
    command_contrast_teacher: torch.Tensor
    command_contrast_broad: torch.Tensor


def attach_command_norm_range(
    model: TSJEPA,
    normalizer: Any,
    config: dict[str, Any],
) -> None:
    """Map config actuator limits through the fitted z-score normalizer."""
    lo_phys = float(config["simulation"]["control_min_N"])
    hi_phys = float(config["simulation"]["control_max_N"])
    mean = float(normalizer.mean)
    std = float(normalizer.std)
    if std == 0.0:
        raise ValueError("command normalizer std must be nonzero")
    model.command_norm_min = (lo_phys - mean) / std
    model.command_norm_max = (hi_phys - mean) / std


def sample_broad_command_pair(
    ref: torch.Tensor,
    low: float,
    high: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform samples in normalized command space, constant over the horizon."""
    batch, kp = int(ref.shape[0]), int(ref.shape[1])
    u_a = ref.new_empty(batch, 1).uniform_(float(low), float(high))
    u_b = ref.new_empty(batch, 1).uniform_(float(low), float(high))
    return u_a.expand(batch, kp).contiguous(), u_b.expand(batch, kp).contiguous()


def _command_norm_range(model: TSJEPA) -> tuple[float, float] | None:
    lo = float(getattr(model, "command_norm_min", float("nan")))
    hi = float(getattr(model, "command_norm_max", float("nan")))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return None
    return lo, hi


def vicreg_regularizer(
    z: torch.Tensor,
    *,
    variance_weight: float,
    covariance_weight: float,
    gamma: float,
    covariance_standardize: bool = False,
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
    cov_term = (
        vicreg_covariance_loss(z, standardize=bool(covariance_standardize)) if cov_w != 0.0 else zero
    )
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

    # Step 4 — JEPA cosine loss, plus optional VICReg and command contrast.
    cosine = jepa_loss(z_pred, z_target)
    var_w = float(getattr(model, "vicreg_variance_weight", 0.0) or 0.0)
    cov_w = float(getattr(model, "vicreg_covariance_weight", 0.0) or 0.0)
    gamma = float(getattr(model, "vicreg_gamma", 1.0) or 1.0)
    contrast_w = float(getattr(model, "command_contrast_weight", 0.0) or 0.0)
    contrast_margin = float(getattr(model, "command_contrast_margin", 0.05) or 0.05)
    sampling = str(getattr(model, "command_contrast_sampling", "teacher_pair") or "teacher_pair")
    vicreg, var_term, cov_term = vicreg_regularizer(
        z_context,
        variance_weight=var_w,
        covariance_weight=cov_w,
        gamma=gamma,
        covariance_standardize=bool(getattr(model, "vicreg_covariance_standardize", False)),
    )
    zero = cosine.new_zeros(())
    teacher_contrast = zero
    broad_contrast = zero
    want_teacher = sampling in {"teacher_pair", "both"}
    want_broad = sampling in {"broad_range", "both"}
    if contrast_w != 0.0 and want_teacher:
        teacher_contrast = command_contrastive_hinge(
            z_pred,
            model.predict(z_context, -commands_norm),
            margin=contrast_margin,
        )
    if contrast_w != 0.0 and want_broad:
        rng = _command_norm_range(model)
        if rng is None:
            if sampling == "broad_range":
                raise ValueError(
                    "command_contrast_sampling='broad_range' requires "
                    "model.command_norm_min/max (attach_command_norm_range)"
                )
        else:
            u_a, u_b = sample_broad_command_pair(commands_norm, rng[0], rng[1])
            broad_contrast = command_contrastive_hinge(
                model.predict(z_context, u_a),
                model.predict(z_context, u_b),
                margin=contrast_margin,
            )
    contrast = teacher_contrast + broad_contrast
    loss = cosine + vicreg + contrast_w * contrast

    return JEPAForwardResult(
        z_context=z_context,
        z_target=z_target,
        z_pred=z_pred,
        loss=loss,
        commands_norm=commands_norm,
        cosine_loss=cosine,
        vicreg_variance=var_term,
        vicreg_covariance=cov_term,
        command_contrast=contrast,
        command_contrast_teacher=teacher_contrast,
        command_contrast_broad=broad_contrast,
    )


def jepa_sgd_and_ema_step(
    model: TSJEPA,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """
    Plan §10 Algorithm 1 steps 5–6: SGD on θ,ϕ then EMA on θ̄.

    Call once per effective batch (after gradient accumulation completes).
    ``max_grad_norm`` is an implementation choice (Table II is silent).
    ``scaler`` is only used for fp16 AMP.
    """
    # Step 5 — gradient update (θ, ϕ only; optimizer was built that way)
    if scaler is not None:
        if max_grad_norm is not None and float(max_grad_norm) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(jepa_trainable_parameters(model), float(max_grad_norm))
        scaler.step(optimizer)
        scaler.update()
    else:
        if max_grad_norm is not None and float(max_grad_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(jepa_trainable_parameters(model), float(max_grad_norm))
        optimizer.step()
    # Step 6 — EMA target update
    model.ema_step()
    optimizer.zero_grad(set_to_none=True)

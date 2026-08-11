"""
Plan §13 TS-JEPA training procedure (per effective training batch).

Steps:
  1. Context encoding:  z_k = Ψθ(x_k)
  2. Target encoding:   z̄_{k+j} = Ψθ̄(x_{k+j})  with stop-gradient
  3. Autoregressive prediction of z̃_{k+1..k+Kp} using §10 command source
  4. JEPA loss L_JEPA = -mean(cos_sim)
  5. SGD update of θ (context encoder) and ϕ (predictor) only
  6. EMA update θ̄ ← η θ̄ + (1-η) θ
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ts_jepa.losses.jepa_loss import jepa_loss
from ts_jepa.models.predictor_command_resolution import (
    PredictorCommandResolution,
    select_predictor_conditioning_commands,
)
from ts_jepa.models.ts_jepa import TSJEPA

PLAN_JEPA_PROCEDURE: dict[str, Any] = {
    "steps": (
        "context_encoding",
        "target_encoding_stop_gradient",
        "autoregressive_prediction",
        "jepa_loss",
        "sgd_update_theta_phi",
        "ema_update_target",
    ),
    "optimized_modules": ("context_encoder", "predictor"),
    "frozen_modules": ("target_encoder",),
    "loss": "jepa_loss",
    "optimizer": "SGD",
    "ema_after_optimizer_step": True,
    "target_stop_gradient": True,
}


@dataclass(frozen=True)
class JEPAForwardResult:
    """Outputs of plan §13 steps 1–4 for one microbatch / batch."""

    z_context: torch.Tensor
    z_target: torch.Tensor
    z_pred: torch.Tensor
    loss: torch.Tensor
    commands_norm: torch.Tensor


def assert_plan_jepa_procedure_config(config: dict[str, Any]) -> None:
    """
    Validate config documents plan §13 training procedure invariants.

    Call from baseline entry scripts. Does not enforce batch/Kp (smoke-safe).
    """
    errors: list[str] = []
    jepa = config.get("ts_jepa", {})
    opt = jepa.get("optimizer", {})
    tgt = jepa.get("target_encoder", {})
    loss = jepa.get("loss", {})
    proc = jepa.get("training_procedure", {})

    if str(opt.get("type", "")) != "SGD":
        errors.append(f"plan §13 step 5 requires SGD; got optimizer.type={opt.get('type')!r}")
    if not bool(tgt.get("stop_gradient", False)):
        errors.append("plan §13 step 2 requires target_encoder.stop_gradient=true")
    if str(tgt.get("update_rule", "")) != "ema":
        errors.append("plan §13 step 6 requires target_encoder.update_rule='ema'")
    if str(loss.get("implementation", "")) not in {
        "negative_mean_cosine_similarity",
        "one_minus_mean_cosine_similarity",
        "one_minus_cosine_similarity",
    }:
        errors.append("plan §13 step 4 requires cosine JEPA loss implementation")

    if proc:
        steps = tuple(proc.get("steps", ()))
        if steps and steps != PLAN_JEPA_PROCEDURE["steps"]:
            errors.append(
                f"ts_jepa.training_procedure.steps must match plan §13 order; got {list(steps)}"
            )
        if proc.get("ema_after_optimizer_step") is False:
            errors.append("plan §13 requires ema_after_optimizer_step=true")
        if proc.get("target_stop_gradient") is False:
            errors.append("plan §13 requires training_procedure.target_stop_gradient=true")

    if errors:
        raise ValueError("Plan §13 JEPA training procedure mismatch:\n  - " + "\n  - ".join(errors))


def jepa_forward_batch(
    model: TSJEPA,
    batch: dict[str, torch.Tensor],
    resolution: PredictorCommandResolution | None = None,
) -> JEPAForwardResult:
    """
    Plan §13 steps 1–4 for one batch.

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

    # Step 3 — autoregressive prediction (§10 command source)
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
    Plan §13 steps 5–6: SGD on θ,ϕ then EMA on θ̄.

    Call once per effective batch (after gradient accumulation completes).
    """
    # Step 5 — gradient update (θ, ϕ only; optimizer was built that way)
    optimizer.step()
    # Step 6 — EMA target update
    model.ema_step()
    optimizer.zero_grad(set_to_none=True)

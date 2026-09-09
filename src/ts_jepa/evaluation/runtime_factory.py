"""Build the closed-loop controller used by baseline eval.

Offline metrics (t-SNE, actor NMAE, horizon NMAE) always read Ψθ / Cε from the
wrapped FrozenRuntimeController. Closed-loop Eq. (28) may use Observer-LQR
instead of Cε + Pφ when ``evaluation.closed_loop_controller: observer_lqr``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.training.probe_lqr_actor import (
    ObserverLQRRuntimeController,
    load_state_decoder_checkpoint,
    observer_lqr_checkpoint_path,
    train_probe_lqr_actor,
)


def closed_loop_controller_kind(config: dict[str, Any]) -> str:
    raw = (config.get("evaluation") or {}).get("closed_loop_controller", "semantic_actor")
    kind = str(raw or "semantic_actor").strip().lower()
    if kind in {"semantic_actor", "frozen", "actor", "ce"}:
        return "semantic_actor"
    if kind in {"observer_lqr", "observer-lqr"}:
        return "observer_lqr"
    raise ValueError(f"unknown evaluation.closed_loop_controller {raw!r}")


def wrap_closed_loop_controller(
    config: dict[str, Any],
    encoder: FrozenRuntimeController,
    *,
    jepa_checkpoint: Path | None = None,
    actor_checkpoint: Path | None = None,
    fit_probe_if_missing: bool = True,
    refit_probe: bool = False,
) -> FrozenRuntimeController | ObserverLQRRuntimeController:
    kind = closed_loop_controller_kind(config)
    if kind == "semantic_actor":
        return encoder
    settings = (config.get("evaluation") or {}).get("observer_lqr") or {}
    decoder = str(settings.get("decoder", "linear"))
    ckpt = observer_lqr_checkpoint_path(config, decoder)
    if refit_probe or (fit_probe_if_missing and not ckpt.is_file()):
        train_probe_lqr_actor(
            config,
            jepa_checkpoint=jepa_checkpoint,
            actor_checkpoint=actor_checkpoint,
            device=encoder.device,
            decoder=decoder,
            run_closed_loop_eval=False,
        )
        ckpt = observer_lqr_checkpoint_path(config, decoder)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Observer-LQR decoder not found at {ckpt}. Fit it with "
            "train_actor_probe_lqr.py or omit --skip-probe."
        )
    loaded = load_state_decoder_checkpoint(ckpt)
    controller = ObserverLQRRuntimeController(
        encoder,
        loaded["probe"],
        loaded["gain"],
        loaded["a"],
        loaded["b"],
        alpha=float(settings.get("alpha", 0.5)),
        beta=float(settings.get("beta", 0.3)),
        full_information=False,
    )
    print(
        f"closed-loop controller=observer_lqr decoder={decoder} checkpoint={ckpt} "
        f"alpha={controller.alpha} beta={controller.beta}",
        flush=True,
    )
    return controller


def build_baseline_controller(
    config: dict[str, Any],
    jepa_ckpt: Path,
    actor_ckpt: Path,
    *,
    device: torch.device | None = None,
    fit_probe_if_missing: bool = True,
    refit_probe: bool = False,
) -> FrozenRuntimeController | ObserverLQRRuntimeController:
    encoder = FrozenRuntimeController.from_checkpoints(config, jepa_ckpt, actor_ckpt, device=device)
    return wrap_closed_loop_controller(
        config,
        encoder,
        jepa_checkpoint=jepa_ckpt,
        actor_checkpoint=actor_ckpt,
        fit_probe_if_missing=fit_probe_if_missing,
        refit_probe=refit_probe,
    )

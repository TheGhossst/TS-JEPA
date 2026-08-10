#!/usr/bin/env python
"""
Closed-loop force-gain diagnostic for Semantic Actor regression-to-mean hypothesis.

Runs frozen JEPA + actor closed-loop control, but multiplies the actor's normalized
force output by --gain BEFORE denormalization and clipping to ±control_max_N.

If higher gain improves control score, the actor may be predicting valid but
conservative (under-scaled) forces relative to what pole balancing requires.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.evaluation.evaluate import evaluate_closed_loop
from ts_jepa.evaluation.metrics import summarize_scores
from ts_jepa.inference.infer import FrozenRuntimeController


class GainScaledRuntimeController(FrozenRuntimeController):
    """Applies scalar gain to normalized actor output before denorm + clip."""

    def __init__(
        self,
        config: dict[str, Any],
        jepa,
        actor,
        normalizer,
        device: torch.device | None = None,
        gain: float = 1.0,
    ) -> None:
        super().__init__(config, jepa, actor, normalizer, device=device)
        self.gain = float(gain)

    def _denorm_with_gain(self, u_norm: torch.Tensor) -> float:
        return self.stats.denormalize_and_clip(u_norm * self.gain)

    @torch.no_grad()
    def step_packet_received(self, frame: np.ndarray) -> float:
        self.observe_frame(frame)
        context = self._context_from_buffer()
        z = self.jepa.encode_context(context)
        u_norm = self.actor(z)
        force = self._denorm_with_gain(u_norm)
        self.latent = z
        self.last_command_norm = float(u_norm.reshape(-1)[0].item())
        return force

    @torch.no_grad()
    def step_packet_lost(self) -> float:
        if self.latent is None:
            return 0.0
        z_next = self.jepa.predictor.forward_step(self.latent)
        u_norm = self.actor(z_next)
        force = self._denorm_with_gain(u_norm)
        self.latent = z_next
        self.last_command_norm = float(u_norm.reshape(-1)[0].item())
        return force


def _load_controller(
    config: dict[str, Any],
    jepa_ckpt: Path,
    actor_ckpt: Path,
    device: torch.device,
    gain: float,
) -> GainScaledRuntimeController:
    base = FrozenRuntimeController.from_checkpoints(config, jepa_ckpt, actor_ckpt, device=device)
    from ts_jepa.preprocessing.command_stats import CommandNormalizer

    normalizer = CommandNormalizer(mean=base.stats.mean, std=base.stats.std)
    return GainScaledRuntimeController(
        config,
        base.jepa,
        base.actor,
        normalizer,
        device=base.device,
        gain=gain,
    )


def run_gain_sweep(
    config: dict[str, Any],
    jepa_ckpt: Path,
    actor_ckpt: Path,
    device: torch.device,
    gains: list[float],
    repetitions: int,
    seed_base: int = 100,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for gain in gains:
        controller = _load_controller(config, jepa_ckpt, actor_ckpt, device, gain=gain)
        scores: list[float] = []
        runs: list[dict[str, float]] = []
        for r in range(repetitions):
            seed = seed_base + r
            out = evaluate_closed_loop(config, controller, seed=seed)
            scores.append(out["mean_control_score"])
            runs.append({"seed": seed, "mean_control_score": out["mean_control_score"]})
        summary = summarize_scores(scores)
        summary["gain"] = gain
        summary["runs"] = runs
        results[str(gain)] = summary
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_35ms.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Checkpoint seed when paths omitted.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Single gain factor applied to normalized force before denorm+clip.",
    )
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 3.0],
        help="Gain sweep values (ignored when --gain is set).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Closed-loop repetitions per gain (default: 3).",
    )
    parser.add_argument("--out", type=str, default="runs/eval/closed_loop_gain_diagnostic.json")
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root(config)
    runs_root = root / config["paths"]["runs_root"]
    device = select_device(args.device)
    print(json.dumps(describe_device(device), indent=2))

    jepa_ckpt = resolve_run_checkpoint(
        runs_root,
        jepa_run_dirname(config),
        explicit=Path(args.jepa_checkpoint) if args.jepa_checkpoint else None,
        seed=args.seed,
    )
    actor_ckpt = resolve_run_checkpoint(
        runs_root,
        actor_run_dirname(config),
        explicit=Path(args.actor_checkpoint) if args.actor_checkpoint else None,
        seed=args.seed,
    )
    print(f"jepa_checkpoint={jepa_ckpt}")
    print(f"actor_checkpoint={actor_ckpt}")

    gains = [args.gain] if args.gain is not None else list(args.gains)
    report = {
        "config": args.config,
        "jepa_checkpoint": str(jepa_ckpt),
        "actor_checkpoint": str(actor_ckpt),
        "repetitions": args.repetitions,
        "denorm_formula": "force = clip(gain * u_norm * std + mean, control_min_N, control_max_N)",
        "command_stats": {
            "mean": None,
            "std": None,
            "control_min_N": float(config["simulation"]["control_min_N"]),
            "control_max_N": float(config["simulation"]["control_max_N"]),
        },
        "results_by_gain": run_gain_sweep(
            config,
            jepa_ckpt,
            actor_ckpt,
            device,
            gains=gains,
            repetitions=args.repetitions,
        ),
    }

    base_ctrl = FrozenRuntimeController.from_checkpoints(config, jepa_ckpt, actor_ckpt, device=device)
    report["command_stats"]["mean"] = base_ctrl.stats.mean
    report["command_stats"]["std"] = base_ctrl.stats.std

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=== Closed-loop force-gain diagnostic ===")
    print(
        f"denorm: u_phys = clip(gain * u_norm * {report['command_stats']['std']:.4f} "
        f"+ {report['command_stats']['mean']:.4f}, "
        f"{report['command_stats']['control_min_N']}, {report['command_stats']['control_max_N']})"
    )
    for gain_key, summary in report["results_by_gain"].items():
        print(
            f"gain={gain_key}: mean_control_score={summary['mean']:.4f} "
            f"(best={summary['best']:.4f}, worst={summary['worst']:.4f}, n={summary['repetitions']})"
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

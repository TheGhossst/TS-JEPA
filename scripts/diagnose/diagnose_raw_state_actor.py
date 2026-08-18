#!/usr/bin/env python
"""
Train the plan §12 SemanticActor MLP on raw CartPole state (and κ-history)
instead of frozen z. Compare offline NMAE and closed-loop scores to the
existing z-actor. Also audits train vs inference action scaling.

Does not train or modify TS-JEPA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.raw_state_actor import FeatureKind, run_raw_state_actor_diagnostic


def _print_nmae(label: str, payload: dict) -> None:
    print(
        f"{label:28s}  NMAE/40={payload.get('nmae', float('nan')):.4f}  "
        f"MAE={payload.get('mean_abs_error', float('nan')):.3f} N  "
        f"mean_NMAE={payload.get('mean_command_baseline_nmae', float('nan')):.4f}  "
        f"beats_mean={payload.get('beats_mean_command_baseline')}"
    )


def _print_loop(label: str, payload: dict | None) -> None:
    if not payload:
        print(f"{label:28s}  (skipped)")
        return
    print(
        f"{label:28s}  mean_score={payload['mean_control_score']:.4f}  "
        f"per_seed={payload['per_seed']}"
    )


def _print_report(report: dict) -> None:
    print("=== Raw-state actor vs frozen-z actor ===")
    print(f"jepa_checkpoint: {report['jepa_checkpoint']}")
    print(f"z_actor_checkpoint: {report['z_actor_checkpoint']}")
    print(f"jepa_modified: {report['jepa_modified']}")
    print()
    z = report["z_actor"]
    print("OFFLINE NMAE (untouched actor test)")
    _print_nmae("JEPA z → Cε", z["actor_nmae"])
    for kind, block in report["raw_state_actor"].items():
        _print_nmae(f"raw {kind} → Cε", block["actor_nmae"])
    print()
    print("CLOSED LOOP (Eq. 28)")
    _print_loop("z full-info", z.get("closed_loop_full_information"))
    _print_loop("z working-gate", z.get("closed_loop_working_gate"))
    for kind, block in report["raw_state_actor"].items():
        loops = block.get("closed_loop") or {}
        _print_loop(f"raw {kind} full-info", loops.get("full_information"))
        _print_loop(f"raw {kind} hold-last", loops.get("working_gate_hold_last"))
    print()
    scaling = z["action_scaling"]
    print("ACTION SCALING (z-actor)")
    print(
        f"  train_domain={scaling['train_loss_domain']}  "
        f"infer={scaling['inference_treats_output_as']}"
    )
    print(
        f"  actor_out std={scaling['actor_output_std']:.3f} N  "
        f"command std={scaling['command_phys_std']:.3f} N  "
        f"suspected_zscore_head={scaling['suspected_zscore_head']}  "
        f"consistent={scaling['train_infer_domain_consistent']}"
    )
    interp = report["interpretation"]
    print()
    print(f"decision: {interp['code']}")
    print(interp["honest_conclusion"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_working.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override semantic_actor.optimizer.epochs (default: same as z-actor).",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        choices=("current", "history"),
        default=("current", "history"),
        help="Raw features: current 4-D state and/or κ-stacked state history.",
    )
    parser.add_argument("--skip-closed-loop", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root(config)
    runs_root = root / config["paths"]["runs_root"]
    device = select_device(args.device)
    print(describe_device(device))

    jepa_ckpt = Path(args.jepa_checkpoint) if args.jepa_checkpoint else None
    if jepa_ckpt is not None and not jepa_ckpt.is_absolute():
        jepa_ckpt = root / jepa_ckpt
    actor_ckpt = Path(args.actor_checkpoint) if args.actor_checkpoint else None
    if actor_ckpt is not None and not actor_ckpt.is_absolute():
        actor_ckpt = root / actor_ckpt

    kinds: tuple[FeatureKind, ...] = tuple(args.inputs)  # type: ignore[assignment]
    report = run_raw_state_actor_diagnostic(
        config,
        device=device,
        jepa_ckpt=jepa_ckpt,
        actor_ckpt=actor_ckpt,
        max_epochs=args.epochs,
        kinds=kinds,
        skip_closed_loop=args.skip_closed_loop,
        seed=args.seed,
    )
    _print_report(report)
    out = Path(args.out) if args.out else runs_root / "eval" / "raw_state_actor_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Evaluate a frozen JEPA + Semantic Actor pair.

Modes:
  all          — NMAE (Kp=1..15) + closed-loop + wireless (default)
  nmae         — prediction-horizon NMAE only (+ plot)
  closed_loop  — frozen closed-loop control scores only
  wireless     — channel-aware / RR / opportunistic vs SNR only

Checkpoint resolution (unless overridden):
  runs/<family>/best.pt  else  runs/<family>/seed_0/best.pt
  or --seed N → runs/<family>/seed_N/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.evaluation.evaluate import (
    baseline_report,
    evaluate_closed_loop,
    evaluate_prediction_horizon_nmae,
    evaluate_with_scheduler,
    write_evaluation_artifacts,
)
from ts_jepa.evaluation.metrics import summarize_scores
from ts_jepa.evaluation.plotting import plot_nmae_by_horizon, plot_wireless_control_scores
from ts_jepa.inference.infer import FrozenRuntimeController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Prefer runs/*/seed_<seed>/best.pt when explicit checkpoints are omitted.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=("all", "nmae", "closed_loop", "wireless"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Artifact directory (default: runs/eval).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root(config)
    runs_root = root / config["paths"]["runs_root"]
    out_dir = Path(args.out_dir) if args.out_dir else runs_root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    device = select_device(args.device)
    print(json.dumps(describe_device(device), indent=2))
    print(f"jepa_checkpoint={jepa_ckpt}")
    print(f"actor_checkpoint={actor_ckpt}")
    print(f"mode={args.mode} out_dir={out_dir}")

    controller = FrozenRuntimeController.from_checkpoints(config, jepa_ckpt, actor_ckpt, device=device)

    if args.mode == "all":
        report = baseline_report(config, controller)
        paths = write_evaluation_artifacts(report, out_dir)
        print(json.dumps({"artifacts": paths, "report": report}, indent=2, default=str))
        return

    if args.mode == "nmae":
        nmae_report = evaluate_prediction_horizon_nmae(config, controller)
        nmae_json = out_dir / "nmae_report.json"
        with nmae_json.open("w", encoding="utf-8") as handle:
            json.dump(nmae_report, handle, indent=2)
        plot_path = plot_nmae_by_horizon(nmae_report, out_dir / "nmae_by_horizon.png")
        print(json.dumps({"nmae_report": nmae_report, "plot": str(plot_path)}, indent=2))
        return

    if args.mode == "closed_loop":
        reps = int(config["evaluation"]["repetitions"])
        scores = []
        details = []
        for r in range(reps):
            out = evaluate_closed_loop(config, controller, seed=100 + r)
            scores.append(out["mean_control_score"])
            details.append({"seed": 100 + r, "mean_control_score": out["mean_control_score"]})
        control = summarize_scores(scores)
        control["runs"] = details
        path = out_dir / "closed_loop_report.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(control, handle, indent=2)
        print(json.dumps(control, indent=2))
        return

    # wireless
    wireless: dict = {}
    for policy in ("channel_aware", "round_robin", "opportunistic"):
        wireless[policy] = {}
        for snr in config["wireless"]["snr_targets_db"]:
            out = evaluate_with_scheduler(
                config, controller, policy=policy, snr_db=float(snr), seed=7
            )
            wireless[policy][str(snr)] = {
                "mean_control_score": out["mean_control_score"],
                "schedule_receive_rate": out["schedule_receive_rate"],
            }
    wireless_json = out_dir / "wireless_report.json"
    with wireless_json.open("w", encoding="utf-8") as handle:
        json.dump(wireless, handle, indent=2)
    plot_path = plot_wireless_control_scores(wireless, out_dir / "wireless_control_scores.png")
    print(json.dumps({"wireless": wireless, "plot": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()

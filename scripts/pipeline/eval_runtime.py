#!/usr/bin/env python
"""
Evaluate a frozen JEPA + Semantic Actor pair.

Modes:
  baseline     — plan §16 checks: NMAE + closed-loop + t-SNE + validation gate (default)
  all          — alias for baseline
  nmae         — prediction-horizon NMAE only (+ plot)
  closed_loop  — frozen closed-loop control scores only
  tsne         — embedding t-SNE diagnostic only
  wireless     — channel-aware / RR / opportunistic vs SNR (requires --force-wireless)

Wireless evaluation is gated behind baseline validation (plan §16–§17).
Use --include-wireless with baseline/all after validation passes.
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
    evaluate_embedding_tsne,
    evaluate_prediction_horizon_nmae,
    evaluate_with_scheduler,
    validate_baseline,
    write_evaluation_artifacts,
)
from ts_jepa.evaluation.metrics import summarize_scores
from ts_jepa.evaluation.plotting import plot_embedding_tsne, plot_nmae_by_horizon, plot_wireless_control_scores
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
        default="baseline",
        choices=("baseline", "all", "nmae", "closed_loop", "tsne", "wireless"),
    )
    parser.add_argument(
        "--include-wireless",
        action="store_true",
        help="With baseline/all: run wireless eval only if baseline validation passes.",
    )
    parser.add_argument(
        "--force-wireless",
        action="store_true",
        help="With wireless mode: skip baseline validation gate (debug only).",
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

    if args.mode in ("baseline", "all"):
        report = baseline_report(
            config,
            controller,
            include_wireless=args.include_wireless,
        )
        paths = write_evaluation_artifacts(report, out_dir)
        print(json.dumps({"artifacts": paths, "baseline_validation": report.get("baseline_validation")}, indent=2, default=str))
        return

    if args.mode == "tsne":
        tsne_report = evaluate_embedding_tsne(config, controller)
        tsne_json = out_dir / "embedding_tsne.json"
        with tsne_json.open("w", encoding="utf-8") as handle:
            json.dump(tsne_report, handle, indent=2)
        plot_path = plot_embedding_tsne(tsne_report, out_dir / "embedding_tsne.png")
        print(json.dumps({"embedding_tsne": tsne_report, "plot": str(plot_path)}, indent=2))
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
    if not args.force_wireless:
        probe = baseline_report(config, controller, include_wireless=False)
        validation = validate_baseline(probe, config)
        if not validation.get("wireless_allowed"):
            print(json.dumps({"error": "baseline_validation_failed", "checks": validation.get("checks")}, indent=2))
            return
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

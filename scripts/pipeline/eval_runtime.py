#!/usr/bin/env python
"""
Evaluate a frozen JEPA + Semantic Actor pair.

Modes:
  baseline     — plan §15 checks: MAPE, NMAE, closed-loop, t-SNE, comm bits, validation gate (default)
  all          — alias for baseline
  nmae         — prediction-horizon NMAE only (+ plot)
  closed_loop  — frozen closed-loop control scores only
  closed_loop_diagnostic — full-info RGB closed loop + actor vs DP teacher (debug)
  tsne         — embedding t-SNE diagnostic only
  fig4         — Fig. 4 consecutive-frame MAPE at IC sampling rates (no checkpoints)
  wireless     — TS-JEPA under channel-aware / RR / opportunistic vs SNR (not §16 conventional control)

Wireless evaluation is gated behind baseline validation (plan §15).
§16 round-robin / opportunistic with hold-last conventional control: scripts/pipeline/run_paper_experiments.py
Use --include-wireless with baseline/all after validation passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import actor_run_dirname, load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.checkpoints import resolve_jepa_checkpoint_from_actor, resolve_run_checkpoint
from ts_jepa.evaluation.evaluate import (
    baseline_report,
    evaluate_closed_loop,
    evaluate_closed_loop_full_information_diagnostic,
    evaluate_embedding_tsne,
    evaluate_fig4_sampling_rate_mape,
    evaluate_prediction_horizon_nmae,
    evaluate_with_scheduler,
    validate_baseline,
    write_evaluation_artifacts,
)
from ts_jepa.evaluation.metrics import summarize_scores
from ts_jepa.evaluation.plotting import plot_embedding_tsne, plot_nmae_by_horizon, plot_wireless_control_scores
from ts_jepa.evaluation.runtime_factory import wrap_closed_loop_controller
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.plan.baseline_validation import assert_plan_baseline_validation_config
from ts_jepa.plan.wireless import assert_plan_wireless_config


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
        choices=("baseline", "all", "nmae", "closed_loop", "closed_loop_diagnostic", "tsne", "fig4", "wireless"),
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
    parser.add_argument(
        "--refit-probe",
        action="store_true",
        help="Refit the z→state Observer-LQR decoder even if a checkpoint exists.",
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
    assert_plan_baseline_validation_config(config)
    assert_plan_wireless_config(config)
    root = project_root(config)
    runs_root = root / config["paths"]["runs_root"]
    out_dir = Path(args.out_dir) if args.out_dir else runs_root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "fig4":
        report = evaluate_fig4_sampling_rate_mape(config)
        fig4_json = out_dir / "fig4_mape.json"
        with fig4_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        from ts_jepa.evaluation.plotting import plot_fig4_mape

        plot_path = plot_fig4_mape(report, out_dir / "fig4_mape.png")
        print(json.dumps({"fig4_mape": report, "plot": str(plot_path)}, indent=2, default=str))
        return

    actor_ckpt = resolve_run_checkpoint(
        runs_root,
        actor_run_dirname(config),
        explicit=Path(args.actor_checkpoint) if args.actor_checkpoint else None,
        seed=args.seed,
    )
    if args.jepa_checkpoint is not None:
        jepa_ckpt = Path(args.jepa_checkpoint).resolve()
        if not jepa_ckpt.exists():
            raise FileNotFoundError(f"JEPA checkpoint not found: {jepa_ckpt}")
    else:
        jepa_ckpt = resolve_jepa_checkpoint_from_actor(actor_ckpt, project_dir=root)
    device = select_device(args.device)
    print(json.dumps(describe_device(device), indent=2))
    print(f"jepa_checkpoint={jepa_ckpt}")
    print(f"actor_checkpoint={actor_ckpt}")
    print(f"mode={args.mode} out_dir={out_dir}")

    encoder = FrozenRuntimeController.from_checkpoints(config, jepa_ckpt, actor_ckpt, device=device)
    if args.mode == "closed_loop_diagnostic":
        controller = encoder
    else:
        controller = wrap_closed_loop_controller(
            config,
            encoder,
            jepa_checkpoint=jepa_ckpt,
            actor_checkpoint=actor_ckpt,
            refit_probe=args.refit_probe,
        )

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
        control = summarize_scores(scores, reported_result=str(config.get("evaluation", {}).get("reported_result", "best")))
        control["runs"] = details
        path = out_dir / "closed_loop_report.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(control, handle, indent=2)
        print(json.dumps(control, indent=2))
        return

    if args.mode == "closed_loop_diagnostic":
        report = evaluate_closed_loop_full_information_diagnostic(config, controller)
        path = out_dir / "closed_loop_full_info_diagnostic.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))
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
                "schedule_rate": out["schedule_rate"],
                "packet_receive_rate": out["packet_receive_rate"],
                "scheduled_outage_rate": out["scheduled_outage_rate"],
                "schedule_receive_rate": out["packet_receive_rate"],
            }
    wireless_json = out_dir / "wireless_report.json"
    with wireless_json.open("w", encoding="utf-8") as handle:
        json.dump(wireless, handle, indent=2)
    plot_path = plot_wireless_control_scores(wireless, out_dir / "wireless_control_scores.png")
    print(json.dumps({"wireless": wireless, "plot": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()

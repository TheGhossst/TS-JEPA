#!/usr/bin/env python
"""
Plan §16–§17: paper control/scheduling baselines and Figs. 6–11 protocols.

Conventional baselines hold the last command when unscheduled / packet lost.
Do not attach TS-JEPA Pφ to DP, supervised, or autoencoder controllers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.baselines.controllers import ZeroActionController
from ts_jepa.baselines.experiments import (
    apply_embedding_dim,
    evaluate_agent_closed_loop,
    evaluate_fig6_prediction_modes,
    evaluate_with_scheduler_agent,
    fig10_device_counts,
    fig11_packet_losses,
    fig7_embedding_dims,
    fig8_train_sizes,
    fig9_snr_targets,
    max_devices_for_score_band,
)
from ts_jepa.baselines.load import build_dp_controller, load_autoencoder_controller, load_supervised_controller
from ts_jepa.config import actor_run_dirname, jepa_run_dirname, load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.checkpoints import resolve_jepa_checkpoint_from_actor, resolve_run_checkpoint
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.training.train_actor import train_semantic_actor_repetitions
from ts_jepa.training.train_jepa import train_ts_jepa_repetitions
from ts_jepa.plan.paper_baselines import assert_plan_section_16_17_config


def _jepa_controller(config, args, device) -> FrozenRuntimeController:
    root = project_root(config)
    runs = root / config["paths"]["runs_root"]
    actor = resolve_run_checkpoint(
        runs,
        actor_run_dirname(config),
        explicit=Path(args.actor_checkpoint) if args.actor_checkpoint else None,
    )
    if args.jepa_checkpoint:
        jepa = Path(args.jepa_checkpoint).resolve()
    else:
        jepa = resolve_jepa_checkpoint_from_actor(actor, project_dir=root)
    return FrozenRuntimeController.from_checkpoints(config, jepa, actor, device=device)


def _write(out_dir: Path, name: str, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"wrote": str(path)}, indent=2))


def cmd_fig6(config, args, device) -> None:
    agents = {}
    agents["dp_hold_last"] = build_dp_controller(config)
    agents["dp_zero_action"] = ZeroActionController(config, build_dp_controller(config))
    if args.supervised_kappa2:
        agents["supervised_kappa2_hold_last"] = load_supervised_controller(
            config, Path(args.supervised_kappa2), kappa=2, device=device
        )
        agents["supervised_kappa2_zero_action"] = ZeroActionController(
            config, load_supervised_controller(config, Path(args.supervised_kappa2), kappa=2, device=device)
        )
    if args.supervised_kappa4:
        agents["supervised_kappa4_hold_last"] = load_supervised_controller(
            config, Path(args.supervised_kappa4), kappa=4, device=device
        )
    if args.autoencoder_checkpoint:
        agents["autoencoder_hold_last"] = load_autoencoder_controller(
            config, Path(args.autoencoder_checkpoint), kappa=2, device=device
        )
        agents["autoencoder_zero_action"] = ZeroActionController(
            config, load_autoencoder_controller(config, Path(args.autoencoder_checkpoint), kappa=2, device=device)
        )
    if args.jepa_checkpoint or args.actor_checkpoint:
        agents["ts_jepa_predict"] = _jepa_controller(config, args, device)
    report = evaluate_fig6_prediction_modes(config, agents, seed=int(args.seed))
    _write(Path(args.out_dir), "fig6_report.json", report)


def cmd_fig9(config, args, device) -> None:
    agent = _jepa_controller(config, args, device)
    cfg = dict(config)
    cfg = {**config, "wireless": {**config["wireless"], "num_devices": 1, "max_devices_scheduled_J": 1}}
    results = {}
    for snr in fig9_snr_targets(config):
        results[str(snr)] = evaluate_with_scheduler_agent(
            cfg, agent, policy="channel_aware", snr_db=float(snr), seed=int(args.seed)
        )
        results[str(snr)].pop("forces", None)
        results[str(snr)].pop("scores", None)
    _write(
        Path(args.out_dir),
        "fig9_report.json",
        {
            "plan_section": "17",
            "figure": "9",
            "snr_db": list(fig9_snr_targets(config)),
            "note": "Eval-time SNR. Paper also discusses SNR during training; that filter is not specified numerically.",
            "results": results,
        },
    )


def cmd_fig10(config, args, device) -> None:
    counts = fig10_device_counts(config)
    rows = []
    ts_jepa = _jepa_controller(config, args, device)
    for policy in ("channel_aware", "round_robin", "opportunistic"):
        for snr in fig9_snr_targets(config):
            rows.append(
                {
                    "control": "ts_jepa_predict",
                    **max_devices_for_score_band(
                        config, ts_jepa, policy=policy, snr_db=float(snr), device_counts=counts, seed=int(args.seed)
                    ),
                }
            )
    if args.supervised_kappa2:
        sup = load_supervised_controller(config, Path(args.supervised_kappa2), kappa=2, device=device)
        for policy in ("round_robin", "opportunistic", "channel_aware"):
            for snr in fig9_snr_targets(config):
                rows.append(
                    {
                        "control": "supervised_hold_last",
                        **max_devices_for_score_band(
                            config, sup, policy=policy, snr_db=float(snr), device_counts=counts, seed=int(args.seed)
                        ),
                    }
                )
    _write(Path(args.out_dir), "fig10_report.json", {"plan_section": "17", "figure": "10", "rows": rows})


def cmd_fig11(config, args, device) -> None:
    counts = fig10_device_counts(config)
    snr = float(args.snr)
    rows = []
    ts_jepa = _jepa_controller(config, args, device)
    sup = None
    if args.supervised_kappa2:
        sup = load_supervised_controller(config, Path(args.supervised_kappa2), kappa=2, device=device)
    for p_loss in fig11_packet_losses(config):
        rows.append(
            {
                "control": "ts_jepa_predict",
                "policy": "channel_aware",
                **max_devices_for_score_band(
                    config,
                    ts_jepa,
                    policy="channel_aware",
                    snr_db=snr,
                    device_counts=counts,
                    extra_packet_loss=float(p_loss),
                    seed=int(args.seed),
                ),
            }
        )
        if sup is not None:
            for policy in ("round_robin", "opportunistic"):
                rows.append(
                    {
                        "control": "supervised_hold_last",
                        "policy": policy,
                        **max_devices_for_score_band(
                            config,
                            sup,
                            policy=policy,
                            snr_db=snr,
                            device_counts=counts,
                            extra_packet_loss=float(p_loss),
                            seed=int(args.seed),
                        ),
                    }
                )
    _write(
        Path(args.out_dir),
        "fig11_report.json",
        {
            "plan_section": "17",
            "figure": "11",
            "snr_db": snr,
            "packet_loss_implementation_choice": list(fig11_packet_losses(config)),
            "rows": rows,
        },
    )


def cmd_dp(config, args, device) -> None:
    del device
    agent = build_dp_controller(config)
    out = evaluate_agent_closed_loop(config, agent, seed=int(args.seed))
    out.pop("forces", None)
    out.pop("teacher_forces", None)
    _write(Path(args.out_dir), "dp_closed_loop.json", {"plan_section": "16", "baseline": "optimal_nonlinear_dp", **out})


def cmd_train_fig7(config, args, device) -> None:
    summaries = []
    for dim in fig7_embedding_dims(config):
        cfg = apply_embedding_dim(config, dim)
        jepa = train_ts_jepa_repetitions(cfg, device=device, max_epochs=args.jepa_epochs)
        actor = train_semantic_actor_repetitions(
            cfg,
            jepa_checkpoint=project_root(cfg) / cfg["paths"]["runs_root"] / jepa_run_dirname(cfg) / "best.pt",
            device=device,
            max_epochs=args.actor_epochs,
        )
        summaries.append({"embedding_dim": dim, "jepa": jepa, "actor": actor})
    _write(Path(args.out_dir), "fig7_train_summary.json", {"plan_section": "17", "figure": "7", "runs": summaries})


def cmd_train_fig8(config, args, device) -> None:
    summaries = []
    for n in fig8_train_sizes(config):
        cfg = dict(config)
        cfg = {**config, "experiments": {**config.get("experiments", {}), "max_jepa_train_trajectories": int(n)}}
        cfg["paths"] = {**config["paths"], "ts_jepa_dirname": f"{jepa_run_dirname(config)}_n{n}"}
        cfg["paths"]["semantic_actor_dirname"] = f"{actor_run_dirname(config)}_n{n}"
        jepa = train_ts_jepa_repetitions(cfg, device=device, max_epochs=args.jepa_epochs)
        actor = train_semantic_actor_repetitions(
            cfg,
            jepa_checkpoint=project_root(cfg) / cfg["paths"]["runs_root"] / jepa_run_dirname(cfg) / "best.pt",
            device=device,
            max_epochs=args.actor_epochs,
        )
        summaries.append({"train_trajectories": n, "size_grid": "implementation_choice", "jepa": jepa, "actor": actor})
    _write(Path(args.out_dir), "fig8_train_summary.json", {"plan_section": "17", "figure": "8", "runs": summaries})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=("dp", "fig6", "fig9", "fig10", "fig11", "train_fig7", "train_fig8"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="runs/eval/paper_experiments")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
    parser.add_argument("--supervised-kappa2", type=str, default=None)
    parser.add_argument("--supervised-kappa4", type=str, default=None)
    parser.add_argument("--autoencoder-checkpoint", type=str, default=None)
    parser.add_argument("--snr", type=float, default=10.0, help="Fig. 11 SNR (dB)")
    parser.add_argument("--jepa-epochs", type=int, default=None)
    parser.add_argument("--actor-epochs", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    assert_plan_section_16_17_config(config)
    device = select_device(args.device)
    print(json.dumps(describe_device(device), indent=2))
    dispatch = {
        "dp": cmd_dp,
        "fig6": cmd_fig6,
        "fig9": cmd_fig9,
        "fig10": cmd_fig10,
        "fig11": cmd_fig11,
        "train_fig7": cmd_train_fig7,
        "train_fig8": cmd_train_fig8,
    }
    dispatch[args.experiment](config, args, device)


if __name__ == "__main__":
    main()

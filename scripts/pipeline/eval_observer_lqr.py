#!/usr/bin/env python
"""Closed-loop eval: frozen-z decoded (x, theta) + alpha-beta observer + discrete LQR.

Uses an existing z->state decoder checkpoint (train_actor_probe_lqr.py output).
No training and no JEPA updates; this isolates whether temporal filtering of
the position decodes recovers the velocity feedback that memoryless actors
lose to MSE attenuation.

Also reports (unless skipped):
  true-state LQR, true-position + observer, frozen-z + finite-difference
  velocities, memoryless probe-LQR, and the receive-every-Kp working gate.
Optional --packet-loss-sweep compares predict-only / hold-last / zero-action
at receive rates 1.0, 0.8, 0.6, 0.4, 0.2 (Bernoulli) and 1/15 (periodic Kp).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import apply_cli_path_overrides, load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.working_gates import DEFAULT_OBSERVER_PACKET_RECEIVE_RATES
from ts_jepa.training.probe_lqr_actor import evaluate_observer_lqr


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_working.yaml")
    parser.add_argument(
        "--decoder-checkpoint",
        type=str,
        default="runs/semantic_actor_working_probe_lqr_mlp/decoder.pt",
    )
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--runs-root", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Closed-loop eval seeds (default: evaluation.working_gates.closed_loop_seeds).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for metrics.json (default: runs/semantic_actor_working_observer_lqr).",
    )
    parser.add_argument(
        "--packet-loss-sweep",
        action="store_true",
        help="Sweep Observer-LQR miss behaviors at default receive rates "
        f"{list(DEFAULT_OBSERVER_PACKET_RECEIVE_RATES)}.",
    )
    parser.add_argument(
        "--packet-receive-rates",
        nargs="+",
        default=None,
        help="Rate tokens for the packet-loss sweep, e.g. 1.0 0.8 0.6 0.4 0.2 1/15. "
        "Implies --packet-loss-sweep. 1/15 is receive-every-Kp, not Bernoulli.",
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Skip true-position observer and frozen-z finite-difference controllers.",
    )
    parser.add_argument(
        "--skip-working-gate",
        action="store_true",
        help="Skip the receive-every-Kp hold/zero working-gate block.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    apply_cli_path_overrides(config, data_root=args.data_root, runs_root=args.runs_root)
    device = select_device(args.device)
    print(describe_device(device), flush=True)
    print({"data_root": str(project_root(config) / config["paths"]["data_root"])}, flush=True)
    rates = args.packet_receive_rates
    result = evaluate_observer_lqr(
        config,
        decoder_checkpoint=Path(args.decoder_checkpoint),
        jepa_checkpoint=Path(args.jepa_checkpoint) if args.jepa_checkpoint else None,
        actor_checkpoint=Path(args.actor_checkpoint) if args.actor_checkpoint else None,
        device=device,
        alpha=args.alpha,
        beta=args.beta,
        seeds=args.seeds,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        packet_receive_rates=rates,
        include_ablations=not args.skip_ablations,
        include_working_gate=not args.skip_working_gate,
        include_packet_loss_sweep=bool(args.packet_loss_sweep),
    )
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

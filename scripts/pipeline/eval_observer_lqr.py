#!/usr/bin/env python
"""Closed-loop eval: frozen-z decoded (x, θ) + α-β observer + discrete LQR.

Uses an existing z→state decoder checkpoint (train_actor_probe_lqr.py output).
No training and no JEPA updates; this isolates whether temporal filtering of
the position decodes recovers the velocity feedback that memoryless actors
lose to MSE attenuation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import apply_cli_path_overrides, load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.training.probe_lqr_actor import evaluate_observer_lqr


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    args = parser.parse_args()
    config = load_config(args.config)
    apply_cli_path_overrides(config, data_root=args.data_root)
    device = select_device(args.device)
    print(describe_device(device), flush=True)
    print({"data_root": str(project_root(config) / config["paths"]["data_root"])}, flush=True)
    result = evaluate_observer_lqr(
        config,
        decoder_checkpoint=Path(args.decoder_checkpoint),
        jepa_checkpoint=Path(args.jepa_checkpoint) if args.jepa_checkpoint else None,
        actor_checkpoint=Path(args.actor_checkpoint) if args.actor_checkpoint else None,
        device=device,
        alpha=args.alpha,
        beta=args.beta,
    )
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

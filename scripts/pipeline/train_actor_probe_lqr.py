#!/usr/bin/env python
"""Fit frozen-z → state probe + discrete LQR (working overlay; no JEPA updates)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import apply_cli_path_overrides, load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.training.probe_lqr_actor import train_probe_lqr_actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ts_jepa_working.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--decoder", type=str, default="linear", choices=["linear", "mlp"])
    args = parser.parse_args()
    config = load_config(args.config)
    apply_cli_path_overrides(config, data_root=args.data_root, runs_root=args.runs_root)
    device = select_device(args.device)
    print(describe_device(device), flush=True)
    print({"data_root": str(project_root(config) / config["paths"]["data_root"])}, flush=True)
    result = train_probe_lqr_actor(
        config,
        jepa_checkpoint=Path(args.jepa_checkpoint) if args.jepa_checkpoint else None,
        actor_checkpoint=Path(args.actor_checkpoint) if args.actor_checkpoint else None,
        device=device,
        decoder=args.decoder,
    )
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

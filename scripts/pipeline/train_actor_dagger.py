#!/usr/bin/env python
"""Actor-only DAgger on frozen JEPA embeddings (working overlay).

Does not retrain Ψθ. Writes runs/<semantic_actor.dagger.run_dirname>/best.pt
and leaves the behavior-cloned actor checkpoint untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import apply_cli_path_overrides, load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.training.dagger_actor import train_actor_dagger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ts_jepa_working.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--runs-root", type=str, default=None)
    parser.add_argument("--feature", type=str, default=None, choices=["z", "z_pair", "z_delta"])
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--run-dirname", type=str, default=None)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip per-round and final closed-loop / NMAE evaluation.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    apply_cli_path_overrides(config, data_root=args.data_root, runs_root=args.runs_root)
    device = select_device(args.device)
    print(describe_device(device), flush=True)
    print({"data_root": str(project_root(config) / config["paths"]["data_root"])}, flush=True)
    overrides = {}
    if args.feature is not None:
        overrides["feature"] = args.feature
    if args.rounds is not None:
        overrides["rounds"] = int(args.rounds)
    if args.run_dirname is not None:
        overrides["run_dirname"] = args.run_dirname
    result = train_actor_dagger(
        config,
        jepa_checkpoint=Path(args.jepa_checkpoint) if args.jepa_checkpoint else None,
        actor_checkpoint=Path(args.actor_checkpoint) if args.actor_checkpoint else None,
        device=device,
        skip_eval=bool(args.skip_eval),
        settings_overrides=overrides or None,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

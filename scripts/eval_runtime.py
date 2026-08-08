#!/usr/bin/env python
"""Evaluate frozen runtime + wireless scheduler baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ts_jepa.config import load_config, project_root
from ts_jepa.evaluation.evaluate import baseline_report
from ts_jepa.inference.infer import FrozenRuntimeController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root(config)
    jepa_ckpt = Path(args.jepa_checkpoint) if args.jepa_checkpoint else root / "runs" / "ts_jepa" / "best.pt"
    actor_ckpt = (
        Path(args.actor_checkpoint) if args.actor_checkpoint else root / "runs" / "semantic_actor" / "best.pt"
    )
    device = torch.device(args.device) if args.device else None
    controller = FrozenRuntimeController.from_checkpoints(config, jepa_ckpt, actor_ckpt, device=device)
    report = baseline_report(config, controller)
    out = root / "runs" / "eval" / "baseline_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

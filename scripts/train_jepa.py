#!/usr/bin/env python
"""Train TS-JEPA with the paper 5-repetition protocol (best validation run)."""

from __future__ import annotations

import argparse
import json

import torch

from ts_jepa.config import load_config
from ts_jepa.training.train_jepa import train_ts_jepa, train_ts_jepa_repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--single-seed",
        type=int,
        default=None,
        help="If set, train only this seed instead of the full repetition protocol.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device) if args.device else None
    if args.single_seed is not None:
        result = train_ts_jepa(config, device=device, max_epochs=args.epochs, seed=args.single_seed)
    else:
        result = train_ts_jepa_repetitions(config, device=device, max_epochs=args.epochs)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

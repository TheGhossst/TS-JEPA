#!/usr/bin/env python
"""Train TS-JEPA context encoder + predictor with cosine alignment."""

from __future__ import annotations

import argparse

import torch

from ts_jepa.config import load_config
from ts_jepa.training.train_jepa import train_ts_jepa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device) if args.device else None
    result = train_ts_jepa(config, device=device, max_epochs=args.epochs)
    print(result)


if __name__ == "__main__":
    main()

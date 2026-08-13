#!/usr/bin/env python
"""Train plan §16 generative autoencoder baseline (κ=2)."""

from __future__ import annotations

import argparse
import json

from ts_jepa.baselines.train import train_autoencoder_baseline
from ts_jepa.config import load_config
from ts_jepa.device import describe_device, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--kappa", type=int, default=2, help="κ for AE context (IC; plan §16 does not set AE κ)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-trajectories", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    device = select_device(args.device)
    print(json.dumps(describe_device(device), indent=2))
    summary = train_autoencoder_baseline(
        config,
        kappa=int(args.kappa),
        device=device,
        seed=args.seed,
        max_epochs=args.epochs,
        max_train_trajectories=args.max_train_trajectories,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

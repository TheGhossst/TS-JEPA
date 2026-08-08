#!/usr/bin/env python
"""Train semantic actor on frozen TS-JEPA embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ts_jepa.config import load_config
from ts_jepa.training.train_actor import train_semantic_actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device) if args.device else None
    ckpt = Path(args.jepa_checkpoint) if args.jepa_checkpoint else None
    result = train_semantic_actor(config, jepa_checkpoint=ckpt, device=device, max_epochs=args.epochs)
    print(result)


if __name__ == "__main__":
    main()

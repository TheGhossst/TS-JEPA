#!/usr/bin/env python
"""Generate paper-count cart-pole RGB trajectories with DP teacher."""

from __future__ import annotations

import argparse

from ts_jepa.config import load_config
from ts_jepa.data.trajectory_generator import generate_all_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    root = generate_all_trajectories(config)
    print(f"Wrote trajectories under {root}")


if __name__ == "__main__":
    main()

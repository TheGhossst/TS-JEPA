#!/usr/bin/env python
"""Generate paper-count cart-pole RGB trajectories with DP teacher."""

from __future__ import annotations

import argparse
from pathlib import Path

from ts_jepa.config import load_config, project_root
from ts_jepa.data.trajectory_generator import generate_all_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=(
            "Output dataset root (contains trajectories/). "
            "Default: config paths.data_root under the project. "
            "Use a separate directory (e.g. data_dp_fixed) to avoid overwriting existing data."
        ),
    )
    parser.add_argument(
        "--pilot-count",
        type=int,
        default=None,
        help="Generate only this many trajectories per split (all four splits) for cheap validation.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = Path(args.data_root) if args.data_root else (project_root(config) / config["paths"]["data_root"])
    root = generate_all_trajectories(
        config,
        data_root=data_root,
        pilot_count_per_split=args.pilot_count,
    )
    print(f"Wrote trajectories under {root}")


if __name__ == "__main__":
    main()

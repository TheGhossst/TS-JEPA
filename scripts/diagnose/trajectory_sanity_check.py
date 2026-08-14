#!/usr/bin/env python
"""Verify an on-disk trajectory dataset (no generation, no training).

Checks split counts, native frame size, finite values, train/test and D_s/D_a
index disjointness, command diversity, and that commands are not i.i.d. uniform.
Writes runs/eval/trajectory_sanity.json. Exits 1 if overall_pass is false.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import load_config, project_root
from ts_jepa.data.dataset_sanity import sanity_check_trajectory_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_fixed.yaml")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset root containing trajectories/ (default: config paths.data_root).",
    )
    parser.add_argument(
        "--fit-normalizer",
        action="store_true",
        help="Rewrite stats/command_norm.json from JEPA train (off by default; verify-only).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="JSON report path (default: runs/eval/trajectory_sanity.json).",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(args.data_root) if args.data_root else (project_root(config) / config["paths"]["data_root"])
    report = sanity_check_trajectory_root(
        config,
        root,
        fit_normalizer=bool(args.fit_normalizer),
    )
    out = Path(args.out) if args.out else (project_root(config) / "runs" / "eval" / "trajectory_sanity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Plot NMAE-by-horizon from an existing nmae_report.json or baseline_report.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.evaluation.plotting import plot_nmae_by_horizon, plot_wireless_control_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to nmae_report.json or baseline_report.json",
    )
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    parser.add_argument(
        "--kind",
        type=str,
        default="nmae",
        choices=("nmae", "wireless"),
    )
    args = parser.parse_args()
    path = Path(args.report)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if args.kind == "nmae":
        report = payload.get("prediction_horizon_nmae", payload)
        out = Path(args.out) if args.out else path.with_name("nmae_by_horizon.png")
        written = plot_nmae_by_horizon(report, out)
    else:
        report = payload.get("wireless", payload)
        out = Path(args.out) if args.out else path.with_name("wireless_control_scores.png")
        written = plot_wireless_control_scores(report, out)
    print(written)


if __name__ == "__main__":
    main()

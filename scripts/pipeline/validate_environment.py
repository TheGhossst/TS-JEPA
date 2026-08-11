#!/usr/bin/env python
"""Run plan §3.3 inverted cart-pole environment validation."""

from __future__ import annotations

import argparse
import json

from ts_jepa.config import load_config, project_root
from ts_jepa.env.env_validation import assert_plan_environment_validated, validate_plan_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--write-report", action="store_true", help="Write JSON report under runs/eval/")
    args = parser.parse_args()
    config = load_config(args.config)
    report = validate_plan_environment(config)
    if args.write_report:
        out = project_root(config) / "runs" / "eval" / "environment_validation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Wrote {out}")
    print(json.dumps(report, indent=2))
    if not report["overall_pass"]:
        raise SystemExit(1)
    assert_plan_environment_validated(config)


if __name__ == "__main__":
    main()

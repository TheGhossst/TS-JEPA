#!/usr/bin/env python
"""
Frozen JEPA kinematic probe: z → (x, ẋ, θ, θ̇).

Does not train or modify TS-JEPA. Uses actor train/test trajectories and the
existing frozen context encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.state_probe import resolve_jepa_ckpt, run_state_probe


def _fmt_dim(name: str, split: dict) -> str:
    return (
        f"{name:24s}  MSE={split['mse']:.5f}  MAE={split['mae']:.5f}  "
        f"Pearson={split['pearson']:+.4f}  "
        f"mean_MSE={split['mean_baseline_mse']:.5f}  "
        f"rel_impr={split['relative_mse_improvement_vs_mean']:+.1%}  "
        f"beats_mean={split['beats_mean_baseline_mse']}"
    )


def _print_report(report: dict) -> None:
    print("=== Frozen JEPA state probe (z -> kinematics) ===")
    print(f"checkpoint: {report['jepa_checkpoint']}")
    print(
        f"samples: train={report['splits']['actor_train_samples']} "
        f"test={report['splits']['actor_test_samples']} "
        f"z_dim={report['splits']['embedding_dim']}"
    )
    print()
    print("TEST SET (ridge OLS, z standardized on train)")
    for name in report["state_dim_names"]:
        print(_fmt_dim(name, report["per_dim"][name]["test"]))
    mv = report["multivariate"]
    print(
        f"{'multivariate R^2 mean':24s}  "
        f"R2={mv['test_mean_r2']:.4f}  per_dim={mv['test_r2_per_dim']}"
    )
    interp = report["interpretation"]
    print()
    print(f"decision: {interp['code']}")
    print(interp["honest_conclusion"])
    print(f"per-dim quality: {interp['per_dim_quality']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_working.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--jepa-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root(config)
    runs_root = root / config["paths"]["runs_root"]
    device = select_device(args.device)
    print(describe_device(device))

    if args.jepa_checkpoint is not None:
        ckpt = Path(args.jepa_checkpoint)
        if not ckpt.is_absolute():
            ckpt = root / ckpt
    else:
        ckpt = resolve_jepa_ckpt(config, explicit=None, seed=args.jepa_seed)
    print(f"jepa_checkpoint={ckpt}")

    report = run_state_probe(config, jepa_ckpt=ckpt, device=device)
    _print_report(report)
    out = Path(args.out) if args.out else runs_root / "eval" / "frozen_jepa_state_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

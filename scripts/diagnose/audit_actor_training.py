#!/usr/bin/env python
"""
Semantic Actor training-path audit (read-only).

Runs the library audit in ts_jepa.evaluation.actor_training_audit and writes JSON.
Does not modify architecture, hyperparameters, checkpoints, or datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.actor_training_audit import audit_actor_training_from_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_fixed.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap embedding samples per split")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = select_device(args.device)
    print(describe_device(device))

    report = audit_actor_training_from_runs(
        config,
        jepa_checkpoint=args.jepa_checkpoint,
        actor_checkpoint=args.actor_checkpoint,
        seed=args.seed,
        device=device,
        max_embedding_samples=args.max_samples,
    )

    runs_root = project_root(config) / config["paths"]["runs_root"]
    out = (
        Path(args.out)
        if args.out
        else runs_root / "eval" / f"actor_training_audit_seed{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=== Actor training audit ===")
    print(f"jepa: {report['jepa_checkpoint']}")
    print(f"actor: {report['actor_checkpoint']}")
    jepa_ok = report["1_jepa_checkpoint_consistency"]["uses_same_frozen_encoder_as_actor_training"]
    print(f"same JEPA as training: {jepa_ok}")
    test_pred = report["4_actor_prediction_distributions"]["test_physical_N"]
    print(
        f"test pred physical: mean={test_pred['mean']:.4g} std={test_pred['std']:.4g} "
        f"unique={test_pred['num_unique_rounded']}"
    )
    corr = report["5_correlation"]["test"]["pred_vs_target_pearson"]
    print(f"test pred-target pearson: {corr:.4g}")
    const = report["7_constant_mean_solution"]["test"]["detected_near_constant_mean_solution"]
    print(f"near-constant-mean detected (test): {const}")
    for line in report["findings"]:
        print(f"  - {line}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

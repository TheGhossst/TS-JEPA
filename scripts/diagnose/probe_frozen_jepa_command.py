#!/usr/bin/env python
"""
Can the DP command be predicted from a frozen JEPA embedding?

Probes (same actor train/test split, embedding extraction, command pairing, z-score):
  1) z[256] → Linear  (closed-form OLS + AdamW Linear)
  2) z[256] → SemanticActor MLP
  3) raw state[4] → Linear / MLP  (learnability control)

Reports physical NMAE/40, Pearson, prediction std, and mean-baseline comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.command_linear_probe import resolve_jepa_ckpt, run_command_recoverability_probe


def _fmt_row(name: str, metrics: dict) -> str:
    pearson = metrics.get("pearson_physical", float("nan"))
    nmae = metrics.get("nmae_physical", float("nan"))
    std = metrics.get("pred_phys_std_N", float("nan"))
    mean = metrics.get("pred_phys_mean_N", float("nan"))
    rel = metrics.get("relative_nmae_improvement_vs_mean", float("nan"))
    mse = metrics.get("mse_norm", float("nan"))
    return (
        f"{name:22s}  NMAE/40={nmae:.4f}  Pearson={pearson:+.4f}  "
        f"pred_std={std:.4f} N  pred_mean={mean:+.4f} N  "
        f"rel_impr vs mean={rel:+.1%}  mse_norm={mse:.4f}"
    )


def _print_report(report: dict) -> None:
    print("=== Frozen JEPA command recoverability ===")
    print(f"checkpoint: {report['jepa_checkpoint']}")
    print(
        f"samples: train={report['splits']['actor_train_samples']} "
        f"test={report['splits']['actor_test_samples']} "
        f"z_dim={report['splits']['embedding_dim']} "
        f"state_dim={report['splits']['state_dim']}"
    )
    pair = report["pairing_check"]
    print(
        f"pairing MAE (denorm vs npz commands): "
        f"train={pair['train_denorm_vs_npz_mae_N']:.3e} N  "
        f"test={pair['test_denorm_vs_npz_mae_N']:.3e} N"
    )
    print()
    print("TEST SET")
    print(_fmt_row("mean baseline", report["mean_baseline_test"]))
    print(_fmt_row("JEPA OLS linear", report["jepa_ols_linear"]["test"]))
    print(_fmt_row("state OLS linear", report["state_ols_linear"]["test"]))
    sgd = report["sgd"]
    for key, label in (
        ("jepa_sgd_linear", "JEPA SGD linear"),
        ("jepa_mlp", "JEPA MLP actor"),
        ("state_sgd_linear", "state SGD linear"),
        ("state_mlp", "state MLP"),
    ):
        if key in sgd:
            print(_fmt_row(label, sgd[key]["test"]))
    interp = report["interpretation"]
    print()
    print(f"decision: {interp['code']}")
    print(interp["honest_conclusion"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_fixed.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--jepa-seed", type=int, default=None, help="JEPA seed if family best.pt is missing.")
    parser.add_argument(
        "--jepa-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run the probe for each JEPA seed (overrides --jepa-seed).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mlp-epochs", type=int, default=None)
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root(config)
    runs_root = root / config["paths"]["runs_root"]
    device = select_device(args.device)
    print(describe_device(device))

    seeds = args.jepa_seeds
    if seeds is None and args.jepa_checkpoint is None and args.jepa_seed is None:
        seeds = [0]
    elif seeds is None and args.jepa_seed is not None:
        seeds = [args.jepa_seed]

    reports = []
    if args.jepa_checkpoint is not None:
        ckpt = Path(args.jepa_checkpoint)
        if not ckpt.is_absolute():
            ckpt = root / ckpt
        report = run_command_recoverability_probe(
            config,
            jepa_ckpt=ckpt,
            device=device,
            mlp_epochs=args.mlp_epochs,
            skip_mlp=args.skip_mlp,
        )
        reports.append(report)
        _print_report(report)
        print()
    else:
        assert seeds is not None
        for seed in seeds:
            ckpt = resolve_jepa_ckpt(config, explicit=None, seed=seed)
            print(f"--- JEPA seed {seed}: {ckpt} ---")
            report = run_command_recoverability_probe(
                config,
                jepa_ckpt=ckpt,
                device=device,
                mlp_epochs=args.mlp_epochs,
                skip_mlp=args.skip_mlp,
            )
            report["jepa_seed"] = int(seed)
            reports.append(report)
            out_seed = runs_root / "eval" / f"frozen_jepa_command_probe_seed{seed}.json"
            out_seed.parent.mkdir(parents=True, exist_ok=True)
            with out_seed.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
            _print_report(report)
            print(f"Wrote {out_seed}")
            print()

    payload = reports[0] if len(reports) == 1 else {"runs": reports}
    out = (
        Path(args.out)
        if args.out
        else runs_root / "eval" / "frozen_jepa_command_probe.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

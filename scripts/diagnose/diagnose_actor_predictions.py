#!/usr/bin/env python
"""
Report Semantic Actor prediction statistics on actor-test embeddings.

Use after training on data_dp_fixed to verify the actor is not a near-constant /
mean predictor (the previous degenerate failure mode).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ts_jepa.config import actor_run_dirname, load_config, project_root
from ts_jepa.data.datasets import ActorEmbeddingDataset, load_command_normalizer
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.checkpoints import resolve_jepa_checkpoint_from_actor, resolve_run_checkpoint
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA


def _stats(name: str, values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    uniq = np.unique(np.round(v, 6))
    return {
        "name": name,
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "num_unique_rounded_6dp": int(uniq.size),
        "zero_fraction": float(np.mean(np.abs(v) < 1e-8)),
        "positive_fraction": float(np.mean(v > 0.0)),
        "negative_fraction": float(np.mean(v < 0.0)),
        "n": int(v.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_fixed.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root(config)
    data_root = root / config["paths"]["data_root"]
    runs_root = root / config["paths"]["runs_root"]
    device = select_device(args.device)
    print(describe_device(device))

    actor_ckpt = resolve_run_checkpoint(
        runs_root,
        actor_run_dirname(config),
        explicit=Path(args.actor_checkpoint) if args.actor_checkpoint else None,
        seed=args.seed,
    )
    jepa_ckpt = resolve_jepa_checkpoint_from_actor(actor_ckpt, project_dir=root)

    jepa_payload = torch.load(jepa_ckpt, map_location="cpu", weights_only=False)
    actor_payload = torch.load(actor_ckpt, map_location="cpu", weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(jepa_payload["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    actor = SemanticActor(
        embedding_dim=int(config["ts_jepa"]["encoder"]["embedding_dim"]),
        hidden_dims=tuple(config["semantic_actor"]["architecture"]["hidden_dims"]),
        dropout=float(config["semantic_actor"]["architecture"]["dropout"]),
    ).to(device)
    actor.load_state_dict(actor_payload["actor"])
    actor.eval()

    normalizer = load_command_normalizer(config, data_root=data_root)
    test_ds = ActorEmbeddingDataset(
        data_root / "trajectories" / "actor" / "test",
        config,
        normalizer,
        jepa.context_encoder,
        device,
        training=False,
    )
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    preds_phys = []
    tgts_phys = []
    with torch.no_grad():
        for batch in loader:
            emb = batch["embedding"].to(device)
            pred = actor(emb).reshape(-1).cpu().numpy()
            tgt = batch["command"].reshape(-1).cpu().numpy()
            preds_phys.append(pred)
            tgts_phys.append(tgt)
    preds_phys = np.concatenate(preds_phys)
    tgts_phys = np.concatenate(tgts_phys)
    preds_norm = normalizer.normalize(preds_phys.astype(np.float32))
    tgts_norm = normalizer.normalize(tgts_phys.astype(np.float32))

    metrics = actor_payload.get("test_loss")
    if metrics is None and "val_loss" in actor_payload:
        metrics = {"checkpoint_val_loss": actor_payload["val_loss"]}
    seed_metrics_path = Path(actor_ckpt).parent / "metrics.json"
    seed_metrics = None
    if seed_metrics_path.exists():
        seed_metrics = json.loads(seed_metrics_path.read_text(encoding="utf-8"))

    report = {
        "jepa_checkpoint": str(jepa_ckpt),
        "actor_checkpoint": str(actor_ckpt),
        "data_root": str(data_root),
        "seed_metrics": seed_metrics,
        "predictions_norm": _stats("pred_norm", preds_norm),
        "targets_norm": _stats("tgt_norm", tgts_norm),
        "predictions_phys_N": _stats("pred_phys", preds_phys),
        "targets_phys_N": _stats("tgt_phys", tgts_phys),
        "mse_norm": float(np.mean((preds_norm - tgts_norm) ** 2)),
        "collapse_warnings": [],
    }
    # Heuristics for the previous zero/mean-collapse failure mode.
    if report["predictions_phys_N"]["std"] < 0.5:
        report["collapse_warnings"].append("pred_phys_std < 0.5 (near-constant)")
    if report["predictions_phys_N"]["num_unique_rounded_6dp"] < 3:
        report["collapse_warnings"].append("fewer than 3 unique physical predictions")
    if report["predictions_phys_N"]["positive_fraction"] == 0.0 or report["predictions_phys_N"][
        "negative_fraction"
    ] == 0.0:
        report["collapse_warnings"].append("predictions lack both signs")

    out = (
        Path(args.out)
        if args.out
        else runs_root / "eval" / f"actor_prediction_stats_seed{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

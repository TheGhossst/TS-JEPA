from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ts_jepa.config import project_root
from ts_jepa.data.datasets import ActorEmbeddingDataset, load_command_normalizer
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_train_val_actor(dataset: ActorEmbeddingDataset, val_fraction: float) -> tuple[Subset, Subset]:
    """
    Hold out a fraction of the actor TRAIN embeddings for early stopping.

    Count/fraction is IMPLEMENTATION CHOICE. Untouched actor test trajectories
    are never used for model selection.
    """
    n = len(dataset)
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, max(1, n - 1)) if n > 1 else 1
    indices = list(range(n))
    val_idx = indices[-n_val:]
    train_idx = indices[:-n_val] if n > n_val else indices
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


@torch.no_grad()
def evaluate_mse(actor: SemanticActor, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    actor.eval()
    total = 0.0
    n_batches = 0
    for batch in loader:
        emb = batch["embedding"].to(device)
        target = batch["command_norm"].to(device)
        pred = actor(emb)
        total += float(criterion(pred, target).item())
        n_batches += 1
    return total / max(1, n_batches)


def train_semantic_actor(
    config: dict[str, Any],
    jepa_checkpoint: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int | None = None,
    data_root: Path | None = None,
    seed: int = 0,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Train one semantic-actor seed.

    Early stopping / checkpoint selection uses a holdout from actor train only.
    Untouched actor test trajectories are evaluated after training only.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(seed)
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    runs = run_dir or (project_root(config) / config["paths"]["runs_root"] / "semantic_actor" / f"seed_{seed}")
    runs.mkdir(parents=True, exist_ok=True)

    ckpt_path = jepa_checkpoint or (project_root(config) / config["paths"]["runs_root"] / "ts_jepa" / "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    normalizer = load_command_normalizer(config, data_root=root)
    train_dir = root / "trajectories" / "actor" / "train"
    test_dir = root / "trajectories" / "actor" / "test"
    train_full = ActorEmbeddingDataset(train_dir, config, normalizer, jepa.context_encoder, device, training=True)
    test_ds = ActorEmbeddingDataset(test_dir, config, normalizer, jepa.context_encoder, device, training=False)

    val_fraction = float(config["semantic_actor"]["early_stopping"].get("val_fraction", 0.2))
    train_ds, val_ds = _split_train_val_actor(train_full, val_fraction)

    opt_cfg = config["semantic_actor"]["optimizer"]
    actor_cfg = config["semantic_actor"]["architecture"]
    actor = SemanticActor(
        embedding_dim=int(config["ts_jepa"]["encoder"]["embedding_dim"]),
        hidden_dims=tuple(actor_cfg["hidden_dims"]),
        dropout=float(actor_cfg["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
    )
    criterion = nn.MSELoss()
    batch_size = int(opt_cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, max(1, len(train_ds))), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, max(1, len(val_ds))), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=min(batch_size, max(1, len(test_ds))), shuffle=False, num_workers=0)

    epochs = int(max_epochs if max_epochs is not None else opt_cfg["epochs"])
    patience = int(config["semantic_actor"]["early_stopping"]["patience"])
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        actor.train()
        train_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"actor seed {seed} epoch {epoch}", leave=False):
            emb = batch["embedding"].to(device)
            target = batch["command_norm"].to(device)
            pred = actor(emb)
            loss = criterion(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1
        train_loss /= max(1, n_batches)

        val_loss = evaluate_mse(actor, val_loader, device, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(actor.state_dict())
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "actor": best_state,
                    "jepa_checkpoint": str(ckpt_path),
                    "normalizer": normalizer.to_dict(),
                    "config": config,
                    "val_loss": best_val,
                    "seed": seed,
                    "selection_split": "actor_train_holdout_validation",
                },
                runs / "best.pt",
            )
        else:
            stale += 1

        if config["semantic_actor"]["early_stopping"]["enabled"] and stale >= patience:
            break

    if best_state is not None:
        actor.load_state_dict(best_state)

    test_loss = evaluate_mse(actor, test_loader, device, criterion)
    with (runs / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": seed,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "test_loss": test_loss,
                "history": history,
            },
            handle,
            indent=2,
        )
    return {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "history": history,
        "runs_dir": str(runs),
        "seed": seed,
        "checkpoint": str(runs / "best.pt"),
    }


def train_semantic_actor_repetitions(
    config: dict[str, Any],
    jepa_checkpoint: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Paper protocol: repeat actor training, select best validation run."""
    reps = int(config["evaluation"]["repetitions"])
    seeds = list(config["evaluation"].get("seeds", list(range(reps))))[:reps]
    root_runs = project_root(config) / config["paths"]["runs_root"] / "semantic_actor"
    root_runs.mkdir(parents=True, exist_ok=True)
    ckpt_path = jepa_checkpoint or (project_root(config) / config["paths"]["runs_root"] / "ts_jepa" / "best.pt")

    seed_results = []
    for seed in seeds:
        result = train_semantic_actor(
            config,
            jepa_checkpoint=ckpt_path,
            device=device,
            max_epochs=max_epochs,
            data_root=data_root,
            seed=int(seed),
            run_dir=root_runs / f"seed_{seed}",
        )
        seed_results.append(result)

    best = min(seed_results, key=lambda r: r["best_val"])
    best_ckpt = Path(best["checkpoint"])
    selected = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    selected["selected_from_seeds"] = seeds
    selected["selection_criterion"] = "best_validation_mse"
    selected["seed_results"] = [
        {"seed": r["seed"], "best_val": r["best_val"], "test_loss": r["test_loss"], "checkpoint": r["checkpoint"]}
        for r in seed_results
    ]
    torch.save(selected, root_runs / "best.pt")
    summary = {
        "repetitions": reps,
        "seeds": seeds,
        "selection_criterion": "best_validation_mse",
        "best_seed": best["seed"],
        "best_val": best["best_val"],
        "best_test_loss": best["test_loss"],
        "seed_results": selected["seed_results"],
        "best_checkpoint": str(root_runs / "best.pt"),
    }
    with (root_runs / "repetition_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary

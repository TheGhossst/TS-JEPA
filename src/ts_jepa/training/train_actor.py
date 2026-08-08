from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ts_jepa.config import project_root
from ts_jepa.data.datasets import ActorEmbeddingDataset, load_command_normalizer
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA


def train_semantic_actor(
    config: dict[str, Any],
    jepa_checkpoint: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    runs = project_root(config) / config["paths"]["runs_root"] / "semantic_actor"
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
    train_ds = ActorEmbeddingDataset(train_dir, config, normalizer, jepa.context_encoder, device, training=True)
    test_ds = ActorEmbeddingDataset(test_dir, config, normalizer, jepa.context_encoder, device, training=False)

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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    epochs = int(max_epochs if max_epochs is not None else opt_cfg["epochs"])
    patience = int(config["semantic_actor"]["early_stopping"]["patience"])
    best_val = float("inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        actor.train()
        train_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"actor epoch {epoch}", leave=False):
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

        actor.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in test_loader:
                emb = batch["embedding"].to(device)
                target = batch["command_norm"].to(device)
                pred = actor(emb)
                loss = criterion(pred, target)
                val_loss += float(loss.item())
                n_val += 1
        val_loss /= max(1, n_val)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(actor.state_dict())
            stale = 0
            torch.save(
                {
                    "actor": best_state,
                    "jepa_checkpoint": str(ckpt_path),
                    "normalizer": normalizer.to_dict(),
                    "config": config,
                    "val_loss": best_val,
                },
                runs / "best.pt",
            )
        else:
            stale += 1

        if config["semantic_actor"]["early_stopping"]["enabled"] and stale >= patience:
            break

    if best_state is not None:
        actor.load_state_dict(best_state)
    return {"best_val": best_val, "history": history, "runs_dir": str(runs)}

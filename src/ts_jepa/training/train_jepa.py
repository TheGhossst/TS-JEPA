from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ts_jepa.config import project_root
from ts_jepa.data.datasets import TrajectoryDataset, fit_command_normalizer, load_command_normalizer
from ts_jepa.losses.cosine import cosine_alignment_loss
from ts_jepa.models.ts_jepa import TSJEPA


def _split_train_val(dataset: TrajectoryDataset, val_traj_count: int) -> tuple[Subset, Subset]:
    """Carve validation indices from train trajectories (count is IMPLEMENTATION CHOICE)."""
    n_files = len(dataset.files)
    val_files = set(range(max(0, n_files - val_traj_count), n_files))
    train_idx = []
    val_idx = []
    for i, (file_idx, _) in enumerate(dataset.index_map):
        if file_idx in val_files:
            val_idx.append(i)
        else:
            train_idx.append(i)
    if not train_idx:
        train_idx = list(range(len(dataset)))
        val_idx = train_idx[: max(1, len(dataset) // 10)]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train_ts_jepa(
    config: dict[str, Any],
    device: torch.device | None = None,
    max_epochs: int | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    runs = project_root(config) / config["paths"]["runs_root"] / "ts_jepa"
    runs.mkdir(parents=True, exist_ok=True)

    normalizer = fit_command_normalizer(config, data_root=root)
    train_dir = root / "trajectories" / "jepa" / "train"
    dataset = TrajectoryDataset(train_dir, config, normalizer, training=True)
    val_count = int(config["ts_jepa"]["early_stopping"]["validation_trajectory_count"])
    train_set, val_set = _split_train_val(dataset, val_count)

    opt_cfg = config["ts_jepa"]["optimizer"]
    batch_size = int(opt_cfg["batch_size"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TSJEPA(config).to(device)
    optimizer = torch.optim.SGD(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        momentum=0.0,
    )

    epochs = int(max_epochs if max_epochs is not None else opt_cfg["epochs"])
    lr_factor = float(config["ts_jepa"]["lr_decay"]["factor"])
    lr_interval = int(config["ts_jepa"]["lr_decay"]["interval_epochs"])
    patience = int(config["ts_jepa"]["early_stopping"]["patience"])
    best_val = float("inf")
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.context_encoder.train()
        model.predictor.train()
        model.target_encoder.eval()
        train_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"jepa epoch {epoch}", leave=False):
            context = batch["context"].to(device)
            future = batch["future_frames"].to(device)
            commands_norm = batch["commands_norm"].to(device)

            z = model.encode_context(context)
            with torch.no_grad():
                z_tgt = model.encode_targets(future)
            z_pred = model.predict(z, commands_norm)
            loss = cosine_alignment_loss(z_pred, z_tgt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.ema_step()

            train_loss += float(loss.item())
            n_batches += 1

        train_loss /= max(1, n_batches)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                context = batch["context"].to(device)
                future = batch["future_frames"].to(device)
                commands_norm = batch["commands_norm"].to(device)
                z = model.encode_context(context)
                z_tgt = model.encode_targets(future)
                z_pred = model.predict(z, commands_norm)
                loss = cosine_alignment_loss(z_pred, z_tgt)
                val_loss += float(loss.item())
                n_val += 1
        val_loss /= max(1, n_val)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            torch.save(
                {
                    "model": best_state,
                    "config": config,
                    "normalizer": normalizer.to_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                runs / "best.pt",
            )
        else:
            stale += 1

        if epoch % lr_interval == 0:
            for group in optimizer.param_groups:
                group["lr"] *= lr_factor

        if config["ts_jepa"]["early_stopping"]["enabled"] and stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "normalizer": load_command_normalizer(config, data_root=root).to_dict(),
            "history": history,
            "best_val": best_val,
        },
        runs / "last.pt",
    )
    return {"best_val": best_val, "history": history, "runs_dir": str(runs)}

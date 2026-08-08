from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ts_jepa.config import project_root
from ts_jepa.data.datasets import TrajectoryDataset, fit_command_normalizer, load_command_normalizer
from ts_jepa.losses.cosine import cosine_alignment_loss
from ts_jepa.models.ts_jepa import TSJEPA


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_train_val(dataset: TrajectoryDataset, val_traj_count: int) -> tuple[Subset, Subset]:
    """Carve validation indices from train trajectories only (count is IMPLEMENTATION CHOICE)."""
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


@torch.no_grad()
def evaluate_cosine_loss(
    model: TSJEPA,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    for batch in loader:
        context = batch["context"].to(device)
        future = batch["future_frames"].to(device)
        # Teacher/trajectory control sequence (not Semantic Actor predictions).
        teacher_commands_norm = batch["teacher_commands_norm"].to(device)
        z = model.encode_context(context)
        z_tgt = model.encode_targets(future)
        z_pred = model.predict(z, teacher_commands_norm)
        total += float(cosine_alignment_loss(z_pred, z_tgt).item())
        n_batches += 1
    return total / max(1, n_batches)


def train_ts_jepa(
    config: dict[str, Any],
    device: torch.device | None = None,
    max_epochs: int | None = None,
    data_root: Path | None = None,
    seed: int = 0,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Train one TS-JEPA seed.

    Model selection uses validation carved from the train split only.
    The untouched JEPA test set is evaluated after training and never used for selection.
    Predictor conditioning uses the trajectory/teacher control sequence.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(seed)
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    runs = run_dir or (project_root(config) / config["paths"]["runs_root"] / "ts_jepa" / f"seed_{seed}")
    runs.mkdir(parents=True, exist_ok=True)

    normalizer = fit_command_normalizer(config, data_root=root)
    train_dir = root / "trajectories" / "jepa" / "train"
    test_dir = root / "trajectories" / "jepa" / "test"
    train_dataset = TrajectoryDataset(train_dir, config, normalizer, training=True)
    test_dataset = TrajectoryDataset(test_dir, config, normalizer, training=False)
    val_count = int(config["ts_jepa"]["early_stopping"]["validation_trajectory_count"])
    train_set, val_set = _split_train_val(train_dataset, val_count)

    opt_cfg = config["ts_jepa"]["optimizer"]
    batch_size = int(opt_cfg["batch_size"])
    # drop_last only when the split is large enough; smoke tests may be tiny.
    drop_last = len(train_set) >= batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=min(batch_size, max(1, len(val_set))), shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(batch_size, max(1, len(test_dataset))),
        shuffle=False,
        num_workers=0,
    )

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
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.context_encoder.train()
        model.predictor.train()
        model.target_encoder.eval()
        train_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"jepa seed {seed} epoch {epoch}", leave=False):
            context = batch["context"].to(device)
            future = batch["future_frames"].to(device)
            teacher_commands_norm = batch["teacher_commands_norm"].to(device)

            z = model.encode_context(context)
            with torch.no_grad():
                z_tgt = model.encode_targets(future)
            z_pred = model.predict(z, teacher_commands_norm)
            loss = cosine_alignment_loss(z_pred, z_tgt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.ema_step()

            train_loss += float(loss.item())
            n_batches += 1

        train_loss /= max(1, n_batches)
        val_loss = evaluate_cosine_loss(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": best_state,
                    "config": config,
                    "normalizer": normalizer.to_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "seed": seed,
                    "selection_split": "train_holdout_validation",
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

    # Untouched test set: report only; never used for checkpoint selection.
    test_loss = evaluate_cosine_loss(model, test_loader, device)
    payload = {
        "model": model.state_dict(),
        "config": config,
        "normalizer": load_command_normalizer(config, data_root=root).to_dict(),
        "history": history,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "seed": seed,
        "selection_split": "train_holdout_validation",
        "test_split": "jepa_test_untouched",
    }
    torch.save(payload, runs / "last.pt")
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


def train_ts_jepa_repetitions(
    config: dict[str, Any],
    device: torch.device | None = None,
    max_epochs: int | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """
    Paper protocol: repeat the experiment, save each seed, report the best validation run.
    """
    reps = int(config["evaluation"]["repetitions"])
    seeds = list(config["evaluation"].get("seeds", list(range(reps))))[:reps]
    root_runs = project_root(config) / config["paths"]["runs_root"] / "ts_jepa"
    root_runs.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in seeds:
        result = train_ts_jepa(
            config,
            device=device,
            max_epochs=max_epochs,
            data_root=data_root,
            seed=int(seed),
            run_dir=root_runs / f"seed_{seed}",
        )
        seed_results.append(result)

    # Paper: report best result. Selection criterion = validation cosine-alignment loss.
    best = min(seed_results, key=lambda r: r["best_val"])
    best_ckpt = Path(best["checkpoint"])
    selected = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    selected["selected_from_seeds"] = seeds
    selected["selection_criterion"] = "best_validation_cosine_alignment_loss"
    selected["seed_results"] = [
        {"seed": r["seed"], "best_val": r["best_val"], "test_loss": r["test_loss"], "checkpoint": r["checkpoint"]}
        for r in seed_results
    ]
    torch.save(selected, root_runs / "best.pt")
    summary = {
        "repetitions": reps,
        "seeds": seeds,
        "selection_criterion": "best_validation_cosine_alignment_loss",
        "best_seed": best["seed"],
        "best_val": best["best_val"],
        "best_test_loss": best["test_loss"],
        "seed_results": selected["seed_results"],
        "best_checkpoint": str(root_runs / "best.pt"),
    }
    with (root_runs / "repetition_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary

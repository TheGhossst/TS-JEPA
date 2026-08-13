"""Train plan §16 supervised and generative autoencoder baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ts_jepa.baselines.dataset import FrameCommandStateDataset
from ts_jepa.baselines.models import GenerativeAutoencoder, SupervisedRGBToCommand
from ts_jepa.config import project_root
from ts_jepa.data.datasets import load_command_normalizer
from ts_jepa.device import select_device
from ts_jepa.runtime import make_dataloader, save_checkpoint, state_dict_to_cpu


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_split_files(train_dir: Path, max_trajectories: int | None) -> list[Path]:
    files = sorted(train_dir.glob("*.npz"))
    if max_trajectories is not None:
        files = files[: int(max_trajectories)]
    return files


def train_supervised_baseline(
    config: dict[str, Any],
    *,
    kappa: int,
    device: torch.device | None = None,
    seed: int = 0,
    max_epochs: int | None = None,
    max_train_trajectories: int | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Supervised RGB(κ) → u. Optimizer / epochs / batch are IMPLEMENTATION CHOICE
    (paper does not publish Table II/III for this baseline).
    """
    device = select_device(device)
    _set_seed(seed)
    ic = config.get("experiments", {}).get("supervised_training", {})
    epochs = int(max_epochs if max_epochs is not None else ic.get("epochs", 50))
    batch = int(ic.get("batch_size", 64))
    lr = float(ic.get("learning_rate", 1e-3))
    root = project_root(config) / config["paths"]["data_root"]
    train_dir = root / "trajectories" / "jepa" / "train"
    files = _train_split_files(train_dir, max_train_trajectories)
    normalizer = load_command_normalizer(config, data_root=root)
    dataset = FrameCommandStateDataset(
        train_dir, config, normalizer, kappa=kappa, training=True, file_list=files
    )
    loader = make_dataloader(dataset, batch_size=batch, shuffle=True, device=device, config=config)
    model = SupervisedRGBToCommand.from_config(config, kappa=kappa).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    last_loss = float("nan")
    for _ in range(epochs):
        losses = []
        for batch_data in loader:
            context = batch_data["context"].to(device)
            target = batch_data["command_norm"].to(device)
            pred = model(context)
            loss = criterion(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        last_loss = float(np.mean(losses)) if losses else float("nan")
    out = run_dir or (
        project_root(config)
        / config["paths"]["runs_root"]
        / "baselines"
        / f"supervised_kappa{kappa}"
        / f"seed_{seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "best.pt"
    save_checkpoint(
        ckpt,
        {
            "model": state_dict_to_cpu(model.state_dict()),
            "kappa": int(kappa),
            "normalizer": normalizer.to_dict(),
            "seed": seed,
        },
    )
    summary = {
        "kappa": int(kappa),
        "seed": seed,
        "epochs": epochs,
        "train_loss": last_loss,
        "num_train_trajectories": len(files),
        "checkpoint": str(ckpt),
        "plan_section": "16",
        "architecture": "implementation_choice",
        "optimizer": "AdamW",
        "learning_rate": lr,
        "batch_size": batch,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _fit_state_stats(dataset: FrameCommandStateDataset) -> tuple[np.ndarray, np.ndarray]:
    states = np.concatenate([s for s in dataset.states], axis=0)
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def train_autoencoder_baseline(
    config: dict[str, Any],
    *,
    kappa: int = 2,
    device: torch.device | None = None,
    seed: int = 0,
    max_epochs: int | None = None,
    max_train_trajectories: int | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Generative AE. Loss weights / optimizer are IMPLEMENTATION CHOICE."""
    device = select_device(device)
    _set_seed(seed)
    ic = config.get("experiments", {}).get("autoencoder_training", {})
    epochs = int(max_epochs if max_epochs is not None else ic.get("epochs", 50))
    batch = int(ic.get("batch_size", 32))
    lr = float(ic.get("learning_rate", 1e-3))
    rgb_w = float(ic.get("rgb_recon_weight", 1.0))
    state_w = float(ic.get("state_recon_weight", 1.0))
    root = project_root(config) / config["paths"]["data_root"]
    train_dir = root / "trajectories" / "jepa" / "train"
    files = _train_split_files(train_dir, max_train_trajectories)
    normalizer = load_command_normalizer(config, data_root=root)
    dataset = FrameCommandStateDataset(
        train_dir, config, normalizer, kappa=kappa, training=True, file_list=files
    )
    state_mean, state_std = _fit_state_stats(dataset)
    loader = make_dataloader(dataset, batch_size=batch, shuffle=True, device=device, config=config)
    model = GenerativeAutoencoder.from_config(config, kappa=kappa).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    last_loss = float("nan")
    model.train()
    mean_t = torch.tensor(state_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(state_std, dtype=torch.float32, device=device)
    for _ in range(epochs):
        losses = []
        for batch_data in loader:
            context = batch_data["context"].to(device)
            state = batch_data["state"].to(device)
            state_n = (state - mean_t) / std_t
            _, recon, state_hat = model(context)
            loss = rgb_w * mse(recon, context) + state_w * mse(state_hat, state_n)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        last_loss = float(np.mean(losses)) if losses else float("nan")
    out = run_dir or (
        project_root(config)
        / config["paths"]["runs_root"]
        / "baselines"
        / f"autoencoder_kappa{kappa}"
        / f"seed_{seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "best.pt"
    save_checkpoint(
        ckpt,
        {
            "model": state_dict_to_cpu(model.state_dict()),
            "kappa": int(kappa),
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "seed": seed,
        },
    )
    summary = {
        "kappa": int(kappa),
        "seed": seed,
        "epochs": epochs,
        "train_loss": last_loss,
        "num_train_trajectories": len(files),
        "checkpoint": str(ckpt),
        "plan_section": "16",
        "architecture": "implementation_choice",
        "rgb_recon_weight": rgb_w,
        "state_recon_weight": state_w,
        "state_head": "4D plant state for nonlinear DP (implementation choice)",
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

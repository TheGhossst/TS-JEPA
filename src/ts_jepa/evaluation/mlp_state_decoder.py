"""Nonlinear frozen-z → (x, ẋ, θ, θ̇) decoder (JEPA weights untouched)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ts_jepa.evaluation.command_linear_probe import _standardize_apply, _standardize_fit
from ts_jepa.evaluation.state_probe import STATE_DIM_NAMES, regression_metrics


class StateDecoderMLP(nn.Module):
    """256 → 256 → 128 → 4. Trained on standardized z and standardized state."""

    def __init__(self, z_dim: int = 256, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(z_dim)
        for h in hidden:
            layers.extend([nn.Linear(dim, int(h)), nn.ReLU(inplace=True), nn.Dropout(float(dropout))])
            dim = int(h)
        layers.append(nn.Linear(dim, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _r2_per_dim(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    sst = np.sum((tgt - tgt.mean(axis=0)) ** 2, axis=0)
    sse = np.sum((tgt - pred) ** 2, axis=0)
    return 1.0 - sse / np.maximum(sst, 1e-12)


def train_mlp_state_decoder(
    z_train: np.ndarray,
    s_train: np.ndarray,
    *,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden: tuple[int, ...] = (256, 128),
    dropout: float = 0.1,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    z = np.asarray(z_train, dtype=np.float32)
    s = np.asarray(s_train, dtype=np.float32).reshape(z.shape[0], -1)[:, :4]
    z_mean, z_std = _standardize_fit(z)
    y_mean, y_std = _standardize_fit(s)
    zs = _standardize_apply(z, z_mean, z_std)
    ys = _standardize_apply(s, y_mean, y_std)
    n = zs.shape[0]
    n_val = max(1, int(round(n * float(val_fraction))))
    n_val = min(n_val, n - 1) if n > 1 else 1
    z_tr, y_tr = zs[:-n_val], ys[:-n_val]
    z_va, y_va = zs[-n_val:], ys[-n_val:]
    model = StateDecoderMLP(z_dim=int(z.shape[1]), hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    crit = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(z_tr), torch.from_numpy(y_tr)),
        batch_size=min(int(batch_size), max(1, z_tr.shape[0])),
        shuffle=True,
    )
    best_state = None
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.detach().item())
            n_batches += 1
        train_loss /= max(1, n_batches)
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(z_va).to(device))
            val_loss = float(crit(val_pred, torch.from_numpy(y_va).to(device)).item())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"mlp-decoder epoch={epoch} train={train_loss:.5f} val={val_loss:.5f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {
        "kind": "mlp",
        "module": model.cpu(),
        "z_mean": np.asarray(z_mean),
        "z_std": np.asarray(z_std),
        "y_mean": np.asarray(y_mean),
        "y_std": np.asarray(y_std),
        "hidden": list(hidden),
        "best_val": float(best_val),
        "history": history,
    }


@torch.no_grad()
def predict_mlp_state(z: np.ndarray, probe: dict[str, Any]) -> np.ndarray:
    model: StateDecoderMLP = probe["module"]
    model.eval()
    z = np.asarray(z, dtype=np.float32)
    single = z.ndim == 1
    z2 = z.reshape(1, -1) if single else z
    zs = _standardize_apply(z2, probe["z_mean"], probe["z_std"])
    pred_std = model(torch.from_numpy(np.asarray(zs, dtype=np.float32))).numpy()
    pred = pred_std * np.asarray(probe["y_std"]) + np.asarray(probe["y_mean"])
    return pred.reshape(4) if single else pred.astype(np.float64)


def mlp_decoder_metrics(
    z: np.ndarray,
    states: np.ndarray,
    probe: dict[str, Any],
    *,
    train_means: np.ndarray,
) -> dict[str, Any]:
    pred = predict_mlp_state(z, probe)
    tgt = np.asarray(states, dtype=np.float64).reshape(pred.shape[0], -1)[:, :4]
    per_dim = {}
    for i, name in enumerate(STATE_DIM_NAMES):
        per_dim[name] = regression_metrics(pred[:, i], tgt[:, i], train_mean=float(train_means[i]))
    r2 = _r2_per_dim(pred, tgt)
    return {
        "per_dim": per_dim,
        "test_r2_per_dim": [float(x) for x in r2],
        "test_mean_r2": float(np.mean(r2)),
        "test_mae_per_dim": [float(x) for x in np.mean(np.abs(pred - tgt), axis=0)],
    }

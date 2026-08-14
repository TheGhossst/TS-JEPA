"""
Frozen-representation command recoverability probes.

Answers: can DP teacher commands be predicted from frozen JEPA embeddings
at all? Controls: the same pairing from raw simulator state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ts_jepa.config import jepa_run_dirname, project_root
from ts_jepa.data.datasets import ActorEmbeddingDataset, load_command_normalizer
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.evaluation.metrics import nmae, physical_force_range_n
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.training.actor_helpers import split_train_val_actor

# #region agent log
def _agent_dbg(hypothesis_id: str, location: str, message: str, data: dict[str, Any], run_id: str = "pre-fix") -> None:
    import json as _json
    import time

    payload = {
        "sessionId": "1367dc",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(r"c:\code\TS-JEPA\debug-1367dc.log", "a", encoding="utf-8") as handle:
        handle.write(_json.dumps(payload) + "\n")
# #endregion


PEARSON_GOOD = 0.20
PEARSON_POOR = 0.05
MSE_REL_IMPROVE_GOOD = 0.05
MSE_REL_CLOSE_TO_MEAN = 0.02


class FeatureCommandDataset(Dataset):
    """Actor-compatible (embedding, command_norm) pairs from numpy arrays."""

    def __init__(self, features: np.ndarray, command_norm: np.ndarray) -> None:
        self.x = torch.as_tensor(np.asarray(features, dtype=np.float32))
        self.y = torch.as_tensor(np.asarray(command_norm, dtype=np.float32).reshape(-1, 1))
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"feature/command length mismatch: {self.x.shape[0]} vs {self.y.shape[0]}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"embedding": self.x[index], "command_norm": self.y[index]}


def pearson_corr(pred: np.ndarray, target: np.ndarray) -> float:
    a = np.asarray(pred, dtype=np.float64).reshape(-1)
    b = np.asarray(target, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(value):
        return 0.0
    return value


def load_states_aligned(trajectory_dir: Path) -> np.ndarray:
    """Load states in ActorEmbeddingDataset order: sorted npz, then every timestep."""
    chunks: list[np.ndarray] = []
    for file_path in sorted(Path(trajectory_dir).glob("*.npz")):
        with np.load(file_path) as data:
            chunks.append(np.asarray(data["states"], dtype=np.float32))
    if not chunks:
        raise FileNotFoundError(f"no trajectory npz files in {trajectory_dir}")
    return np.concatenate(chunks, axis=0)


def load_commands_aligned(trajectory_dir: Path) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for file_path in sorted(Path(trajectory_dir).glob("*.npz")):
        with np.load(file_path) as data:
            chunks.append(np.asarray(data["commands"], dtype=np.float32).reshape(-1))
    if not chunks:
        raise FileNotFoundError(f"no trajectory npz files in {trajectory_dir}")
    return np.concatenate(chunks, axis=0)


def collect_actor_arrays(
    dataset: ActorEmbeddingDataset,
    trajectory_dir: Path,
) -> dict[str, np.ndarray]:
    z_list: list[np.ndarray] = []
    y_norm_list: list[float] = []
    for i in range(len(dataset)):
        item = dataset[i]
        z_list.append(item["embedding"].numpy())
        y_norm_list.append(float(item["command_norm"].reshape(-1)[0].item()))
    z = np.stack(z_list, axis=0).astype(np.float32)
    y_norm = np.asarray(y_norm_list, dtype=np.float32)
    states = load_states_aligned(trajectory_dir)
    commands_phys = load_commands_aligned(trajectory_dir)
    if states.shape[0] != z.shape[0] or commands_phys.shape[0] != z.shape[0]:
        raise ValueError(
            "aligned lengths disagree: "
            f"z={z.shape[0]} states={states.shape[0]} commands={commands_phys.shape[0]}"
        )
    y_phys_from_norm = dataset.normalizer.denormalize(y_norm)
    pairing_mae = float(np.mean(np.abs(y_phys_from_norm.reshape(-1) - commands_phys.reshape(-1))))
    return {
        "z": z,
        "state": states,
        "command_norm": y_norm,
        "command_phys": commands_phys.astype(np.float32),
        "pairing_mae_N": np.asarray([pairing_mae], dtype=np.float64),
    }


def fit_linear_ols(x: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Affine least squares: y ≈ X w + b. Returns concat([w, b])."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n = x.shape[0]
    a = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)
    if ridge and ridge > 0.0:
        d = a.shape[1]
        reg = ridge * np.eye(d, dtype=np.float64)
        reg[-1, -1] = 0.0
        coef = np.linalg.solve(a.T @ a + reg, a.T @ y)
        return coef.reshape(-1)
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return coef.reshape(-1)


def predict_linear_ols(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64).reshape(-1)
    return (x @ coef[:-1] + coef[-1]).astype(np.float64)


def probe_metrics(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    normalizer: CommandNormalizer,
    *,
    force_range_n: float,
    mean_baseline_phys: np.ndarray | None = None,
) -> dict[str, Any]:
    pred_norm = np.asarray(pred_norm, dtype=np.float64).reshape(-1)
    target_norm = np.asarray(target_norm, dtype=np.float64).reshape(-1)
    pred_phys = np.asarray(normalizer.denormalize(pred_norm.astype(np.float32)), dtype=np.float64).reshape(-1)
    target_phys = np.asarray(normalizer.denormalize(target_norm.astype(np.float32)), dtype=np.float64).reshape(-1)
    nmae_value = nmae(pred_phys, target_phys, force_range_n=force_range_n)
    out: dict[str, Any] = {
        "n": int(pred_phys.size),
        "mse_norm": float(np.mean((pred_norm - target_norm) ** 2)),
        "nmae_physical": float(nmae_value),
        "nmae_formula": "mean(abs(u_pred - u_true)) / 40",
        "pearson_physical": pearson_corr(pred_phys, target_phys),
        "pred_phys_mean_N": float(pred_phys.mean()),
        "pred_phys_std_N": float(pred_phys.std()),
        "target_phys_mean_N": float(target_phys.mean()),
        "target_phys_std_N": float(target_phys.std()),
    }
    if mean_baseline_phys is not None:
        base = np.asarray(mean_baseline_phys, dtype=np.float64).reshape(-1)
        if base.size == 1:
            base = np.full_like(target_phys, float(base[0]))
        base_nmae = nmae(base, target_phys, force_range_n=force_range_n)
        base_norm = np.asarray(normalizer.normalize(base.astype(np.float32)), dtype=np.float64).reshape(-1)
        base_mse = float(np.mean((base_norm - target_norm) ** 2))
        mse_norm = float(out["mse_norm"])
        out["mean_baseline_nmae_physical"] = float(base_nmae)
        out["mean_baseline_mse_norm"] = base_mse
        out["nmae_vs_mean_baseline"] = float(nmae_value - base_nmae)
        out["relative_nmae_improvement_vs_mean"] = float(
            (base_nmae - nmae_value) / max(base_nmae, 1e-12)
        )
        out["relative_mse_improvement_vs_mean"] = float((base_mse - mse_norm) / max(base_mse, 1e-12))
        out["beats_mean_baseline_nmae"] = bool(nmae_value < base_nmae - 1e-12)
        out["beats_mean_baseline_mse"] = bool(mse_norm < base_mse - 1e-12)
        out["beats_mean_baseline"] = bool(out["beats_mean_baseline_mse"])
    return out


def _signal_quality(metrics: dict[str, Any]) -> str:
    """Pearson + MSE vs mean; NMAE is reported but not the fork criterion.

    On this DP dataset the command distribution is concentrated near 0, so a
    constant mean predictor already has low NMAE and is hard to beat on MAE.
    Recoverability is decided by held-out Pearson and whether MSE beats the mean.
    """
    pearson = abs(float(metrics.get("pearson_physical", 0.0)))
    rel_mse = metrics.get("relative_mse_improvement_vs_mean")
    if rel_mse is None:
        rel_mse = float(metrics.get("relative_nmae_improvement_vs_mean", 0.0))
    rel_mse = float(rel_mse)
    pred_std = float(metrics.get("pred_phys_std_N", float("nan")))
    tgt_std = float(metrics.get("target_phys_std_N", float("nan")))
    collapsed = (
        np.isfinite(pred_std)
        and np.isfinite(tgt_std)
        and tgt_std > 1e-8
        and pred_std / tgt_std < 0.05
    )
    if collapsed and pearson < PEARSON_GOOD:
        return "poor"
    if pearson >= PEARSON_GOOD and rel_mse >= MSE_REL_IMPROVE_GOOD:
        return "good"
    if pearson >= PEARSON_GOOD:
        return "weak"
    if pearson < PEARSON_POOR and rel_mse < MSE_REL_CLOSE_TO_MEAN:
        return "poor"
    return "weak"


def interpret_probe_table(
    *,
    jepa_linear: dict[str, Any],
    jepa_mlp: dict[str, Any] | None,
    state_linear: dict[str, Any],
    state_mlp: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map probe outcomes to the diagnostic fork in the recoverability protocol."""
    jepa_lin_q = _signal_quality(jepa_linear)
    state_lin_q = _signal_quality(state_linear)
    jepa_mlp_q = _signal_quality(jepa_mlp) if jepa_mlp is not None else "skipped"
    state_mlp_q = _signal_quality(state_mlp) if state_mlp is not None else "skipped"

    jepa_any_good = jepa_lin_q == "good" or jepa_mlp_q == "good"
    jepa_both_poor = jepa_lin_q == "poor" and (jepa_mlp_q in {"poor", "skipped"})
    state_any_good = state_lin_q == "good" or state_mlp_q == "good"
    state_both_poor = state_lin_q == "poor" and (state_mlp_q in {"poor", "skipped"})

    if jepa_lin_q == "good" and jepa_mlp_q == "poor":
        code = "linear_ok_mlp_fail"
        conclusion = (
            "JEPA contains linearly recoverable control information. "
            "Semantic-actor MLP failure is likely an actor training/optimization bug."
        )
    elif jepa_any_good:
        code = "jepa_has_control_information"
        conclusion = (
            "JEPA embeddings contain usable control information. "
            "Closed-loop/actor collapse is likely in semantic-actor training or runtime, not in the frozen encoder."
        )
    elif jepa_both_poor and state_any_good:
        code = "jepa_is_bottleneck"
        conclusion = (
            "The DP command is learnable from raw simulator state, but not from the frozen JEPA embedding. "
            "JEPA is the bottleneck, not the dataset or the DP teacher pairing."
        )
    elif jepa_both_poor and state_both_poor:
        code = "target_or_pairing_wrong"
        conclusion = (
            "Even raw simulator state cannot recover the DP command. "
            "Dataset/target pairing or the teacher-control formulation is likely wrong."
        )
    else:
        code = "inconclusive"
        conclusion = (
            "Signals are weak or mixed. Frozen JEPA is not a clearly usable control representation "
            "under the linear/MLP probes, but the controls are also not decisive."
        )
    return {
        "code": code,
        "honest_conclusion": conclusion,
        "jepa_linear_quality": jepa_lin_q,
        "jepa_mlp_quality": jepa_mlp_q,
        "state_linear_quality": state_lin_q,
        "state_mlp_quality": state_mlp_q,
        "thresholds": {
            "pearson_good": PEARSON_GOOD,
            "pearson_poor": PEARSON_POOR,
            "mse_relative_improvement_good": MSE_REL_IMPROVE_GOOD,
            "mse_relative_close_to_mean": MSE_REL_CLOSE_TO_MEAN,
            "fork_criterion": (
                "held-out Pearson and MSE vs mean baseline; "
                "NMAE is reported but MAE is mean-dominated on this command distribution"
            ),
        },
    }


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float64) - mean) / std).astype(np.float32)


@torch.no_grad()
def _predict_module(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    for start in range(0, tensor.shape[0], batch_size):
        chunk = tensor[start : start + batch_size].to(device)
        outs.append(model(chunk).detach().cpu().numpy().reshape(-1))
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,), dtype=np.float32)


def train_sgd_probe(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    device: torch.device,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
) -> dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, max(1, len(train_ds))),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, max(1, len(val_ds))),
        shuffle=False,
        num_workers=0,
    )
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0
        for batch in train_loader:
            emb = batch["embedding"].to(device)
            target = batch["command_norm"].to(device)
            pred = model(emb)
            loss = criterion(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item())
            n_batches += 1
        train_loss = total / max(1, n_batches)
        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                emb = batch["embedding"].to(device)
                target = batch["command_norm"].to(device)
                val_total += float(criterion(model(emb), target).item())
                val_n += 1
        val_loss = val_total / max(1, val_n)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val_mse_norm": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(history[-1]["epoch"]) if history else 0,
        "history_tail": history[-5:],
    }


def _ols_eval(
    x_train: np.ndarray,
    y_train_norm: np.ndarray,
    x_test: np.ndarray,
    y_test_norm: np.ndarray,
    normalizer: CommandNormalizer,
    force_range_n: float,
    mean_baseline_phys: np.ndarray,
    *,
    ridge: float = 1e-4,
) -> dict[str, Any]:
    mean, std = _standardize_fit(x_train)
    xtr = _standardize_apply(x_train, mean, std)
    xte = _standardize_apply(x_test, mean, std)
    coef = fit_linear_ols(xtr, y_train_norm, ridge=ridge)
    pred_test = predict_linear_ols(xte, coef)
    pred_train = predict_linear_ols(xtr, coef)
    return {
        "train": probe_metrics(
            pred_train,
            y_train_norm,
            normalizer,
            force_range_n=force_range_n,
            mean_baseline_phys=mean_baseline_phys,
        ),
        "test": probe_metrics(
            pred_test,
            y_test_norm,
            normalizer,
            force_range_n=force_range_n,
            mean_baseline_phys=mean_baseline_phys,
        ),
        "ridge": float(ridge),
        "input_standardized": True,
        "note": "X is z-scored on train only; this is affine-equivalent to raw-X OLS up to numerics.",
    }


def run_command_recoverability_probe(
    config: dict[str, Any],
    *,
    jepa_ckpt: Path,
    device: torch.device,
    data_root: Path | None = None,
    mlp_epochs: int | None = None,
    skip_mlp: bool = False,
) -> dict[str, Any]:
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    payload = torch.load(jepa_ckpt, map_location="cpu", weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(payload["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    normalizer = load_command_normalizer(config, data_root=root)
    train_dir = root / "trajectories" / "actor" / "train"
    test_dir = root / "trajectories" / "actor" / "test"
    print(f"encoding frozen JEPA embeddings from {train_dir} and {test_dir} ...")
    train_full = ActorEmbeddingDataset(
        train_dir, config, normalizer, jepa.context_encoder, device, training=True
    )
    test_ds = ActorEmbeddingDataset(
        test_dir, config, normalizer, jepa.context_encoder, device, training=False
    )
    print(f"encoded train={len(train_full)} test={len(test_ds)}")
    train_arrays = collect_actor_arrays(train_full, train_dir)
    test_arrays = collect_actor_arrays(test_ds, test_dir)

    force_range = physical_force_range_n(config)
    mean_phys = float(np.mean(train_arrays["command_phys"]))
    mean_norm = float(np.mean(train_arrays["command_norm"]))
    mean_baseline_test = probe_metrics(
        np.full(test_arrays["command_norm"].shape, mean_norm, dtype=np.float64),
        test_arrays["command_norm"],
        normalizer,
        force_range_n=force_range,
        mean_baseline_phys=np.array([mean_phys]),
    )

    jepa_ols = _ols_eval(
        train_arrays["z"],
        train_arrays["command_norm"],
        test_arrays["z"],
        test_arrays["command_norm"],
        normalizer,
        force_range,
        np.array([mean_phys]),
    )
    state_ols = _ols_eval(
        train_arrays["state"],
        train_arrays["command_norm"],
        test_arrays["state"],
        test_arrays["command_norm"],
        normalizer,
        force_range,
        np.array([mean_phys]),
    )

    opt_cfg = config["semantic_actor"]["optimizer"]
    es_cfg = config["semantic_actor"]["early_stopping"]
    val_fraction = float(es_cfg.get("val_fraction", 0.2))
    epochs = int(mlp_epochs if mlp_epochs is not None else opt_cfg["epochs"])
    patience = int(es_cfg["patience"])
    lr = float(opt_cfg["learning_rate"])
    wd = float(opt_cfg.get("weight_decay", 0.01))
    batch_size = int(opt_cfg["batch_size"])

    sgd_reports: dict[str, Any] = {}

    def _run_sgd(x_train: np.ndarray, x_test: np.ndarray, factory: Callable[[], nn.Module]) -> dict[str, Any]:
        print("training SGD probe ...")
        full = FeatureCommandDataset(x_train, train_arrays["command_norm"])
        train_split, val_split = split_train_val_actor(full, val_fraction)  # type: ignore[arg-type]
        model = factory()
        train_info = train_sgd_probe(
            model,
            train_split,
            val_split,
            device=device,
            lr=lr,
            weight_decay=wd,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
        )
        pred_test = _predict_module(model, x_test, device)
        pred_train = _predict_module(model, x_train, device)
        return {
            "train_info": train_info,
            "train": probe_metrics(
                pred_train,
                train_arrays["command_norm"],
                normalizer,
                force_range_n=force_range,
                mean_baseline_phys=np.array([mean_phys]),
            ),
            "test": probe_metrics(
                pred_test,
                test_arrays["command_norm"],
                normalizer,
                force_range_n=force_range,
                mean_baseline_phys=np.array([mean_phys]),
            ),
        }

    emb_dim = int(config["ts_jepa"]["encoder"]["embedding_dim"])
    hidden = tuple(config["semantic_actor"]["architecture"]["hidden_dims"])
    dropout = float(config["semantic_actor"]["architecture"]["dropout"])
    state_dim = int(train_arrays["state"].shape[1])

    sgd_reports["jepa_sgd_linear"] = _run_sgd(
        train_arrays["z"],
        test_arrays["z"],
        lambda: nn.Linear(emb_dim, 1),
    )
    sgd_reports["state_sgd_linear"] = _run_sgd(
        train_arrays["state"],
        test_arrays["state"],
        lambda: nn.Linear(state_dim, 1),
    )

    jepa_mlp = None
    state_mlp = None
    if not skip_mlp:
        sgd_reports["jepa_mlp"] = _run_sgd(
            train_arrays["z"],
            test_arrays["z"],
            lambda: SemanticActor(embedding_dim=emb_dim, hidden_dims=hidden, dropout=dropout),
        )
        sgd_reports["state_mlp"] = _run_sgd(
            train_arrays["state"],
            test_arrays["state"],
            lambda: SemanticActor(embedding_dim=state_dim, hidden_dims=hidden, dropout=dropout),
        )
        jepa_mlp = sgd_reports["jepa_mlp"]["test"]
        state_mlp = sgd_reports["state_mlp"]["test"]

    interpretation = interpret_probe_table(
        jepa_linear=jepa_ols["test"],
        jepa_mlp=jepa_mlp,
        state_linear=state_ols["test"],
        state_mlp=state_mlp,
    )

    # #region agent log
    z_std = float(train_arrays["z"].std())
    z_rank = None
    try:
        zc = train_arrays["z"].astype(np.float64)
        zc = zc - zc.mean(axis=0, keepdims=True)
        s = np.linalg.svd(zc, compute_uv=False)
        p = (s ** 2) / max(float((s ** 2).sum()), 1e-12)
        z_rank = float(1.0 / np.sum(p ** 2))
        top1 = float(p[0])
    except Exception:
        top1 = float("nan")
    cmd_zero = float(np.mean(np.abs(train_arrays["command_phys"]) <= 1e-6))
    _agent_dbg(
        "H2",
        "command_linear_probe.py:jepa_ols",
        "frozen JEPA linear probe vs mean baseline",
        {
            "jepa_ols_pearson": jepa_ols["test"]["pearson_physical"],
            "jepa_ols_nmae": jepa_ols["test"]["nmae_physical"],
            "jepa_ols_rel_mse": jepa_ols["test"].get("relative_mse_improvement_vs_mean"),
            "jepa_ols_pred_std": jepa_ols["test"]["pred_phys_std_N"],
            "mean_baseline_nmae": mean_baseline_test["nmae_physical"],
            "z_train_global_std": z_std,
            "z_train_effective_rank": z_rank,
            "z_train_top1_var_frac": top1,
        },
    )
    _agent_dbg(
        "H4",
        "command_linear_probe.py:state_ols",
        "raw-state linear probe vs mean baseline",
        {
            "state_ols_pearson": state_ols["test"]["pearson_physical"],
            "state_ols_nmae": state_ols["test"]["nmae_physical"],
            "state_ols_rel_mse": state_ols["test"].get("relative_mse_improvement_vs_mean"),
            "state_ols_pred_std": state_ols["test"]["pred_phys_std_N"],
            "train_command_zero_fraction": cmd_zero,
            "target_phys_std": jepa_ols["test"]["target_phys_std_N"],
        },
    )
    _agent_dbg(
        "H2",
        "command_linear_probe.py:interpretation",
        "recoverability fork",
        {
            "code": interpretation["code"],
            "jepa_linear_quality": interpretation["jepa_linear_quality"],
            "state_linear_quality": interpretation["state_linear_quality"],
            "jepa_mlp_quality": interpretation["jepa_mlp_quality"],
            "state_mlp_quality": interpretation["state_mlp_quality"],
        },
    )
    # #endregion

    return {
        "jepa_checkpoint": str(jepa_ckpt),
        "data_root": str(root),
        "splits": {
            "actor_train_samples": int(train_arrays["z"].shape[0]),
            "actor_test_samples": int(test_arrays["z"].shape[0]),
            "embedding_dim": int(train_arrays["z"].shape[1]),
            "state_dim": state_dim,
            "protocol": "exact ActorEmbeddingDataset extraction, actor train/test trajectories, command z-score from command_norm.json",
        },
        "pairing_check": {
            "train_denorm_vs_npz_mae_N": float(train_arrays["pairing_mae_N"][0]),
            "test_denorm_vs_npz_mae_N": float(test_arrays["pairing_mae_N"][0]),
        },
        "mean_baseline_test": mean_baseline_test,
        "jepa_ols_linear": jepa_ols,
        "state_ols_linear": state_ols,
        "sgd": sgd_reports,
        "interpretation": interpretation,
    }


def resolve_jepa_ckpt(config: dict[str, Any], *, explicit: Path | None, seed: int | None) -> Path:
    runs_root = project_root(config) / config["paths"]["runs_root"]
    return resolve_run_checkpoint(
        runs_root,
        jepa_run_dirname(config),
        explicit=explicit,
        seed=seed,
    )

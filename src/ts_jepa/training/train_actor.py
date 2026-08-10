from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, project_root
from ts_jepa.data.datasets import ActorEmbeddingDataset, ActorStateDataset, load_command_normalizer
from ts_jepa.device import select_device
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.runtime import (
    CUDAPrefetcher,
    DataLoaderStallError,
    TrainProgressWatchdog,
    configure_train_logging,
    configure_training_runtime,
    format_exception,
    gpu_mem_str,
    make_dataloader,
    reraise_cuda_context,
    save_checkpoint,
    state_dict_to_cpu,
)

VALID_ACTOR_INPUT_MODES = ("embedding", "state")


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_input_mode(input_mode: str) -> str:
    mode = str(input_mode).strip().lower()
    if mode not in VALID_ACTOR_INPUT_MODES:
        raise ValueError(
            f"Unsupported actor input_mode={input_mode!r}; expected one of {VALID_ACTOR_INPUT_MODES}"
        )
    return mode


def _actor_runs_dirname(config: dict[str, Any], input_mode: str) -> str:
    """Keep embedding runs at the default path; isolate state runs with a suffix."""
    base = actor_run_dirname(config)
    if input_mode == "embedding":
        return base
    return f"{base}_{input_mode}"


def _split_train_val_actor(dataset: Dataset, val_fraction: float) -> tuple[Subset, Subset]:
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
def evaluate_mse(actor: SemanticActor, loader, device: torch.device, criterion: nn.Module) -> float:
    actor.eval()
    total = 0.0
    n_batches = 0
    try:
        for batch in CUDAPrefetcher(loader, device):
            emb = batch["embedding"]
            target = batch["command_norm"]
            pred = actor(emb)
            total += float(criterion(pred, target).item())
            n_batches += 1
    except DataLoaderStallError:
        raise
    except Exception as exc:
        reraise_cuda_context(exc, where=f"evaluate_mse after {n_batches} batches", device=device)
    return total / max(1, n_batches)


def train_semantic_actor(
    config: dict[str, Any],
    jepa_checkpoint: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int | None = None,
    data_root: Path | None = None,
    seed: int = 0,
    run_dir: Path | None = None,
    input_mode: str = "embedding",
) -> dict[str, Any]:
    """
    Train one semantic-actor seed.

    Early stopping / checkpoint selection uses a holdout from actor train only.
    Untouched actor test trajectories are evaluated after training only.

    input_mode:
      - "embedding" (default): frozen JEPA context_encoder embeddings
      - "state": κ-window raw physical state features (bypasses JEPA encoder)
    """
    input_mode = _validate_input_mode(input_mode)
    device = select_device(device)
    _set_seed(seed)
    configure_train_logging()
    configure_training_runtime(
        device,
        cudnn_benchmark=bool(config.get("runtime", {}).get("cudnn_benchmark", True)),
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    runs = run_dir or (
        project_root(config)
        / config["paths"]["runs_root"]
        / _actor_runs_dirname(config, input_mode)
        / f"seed_{seed}"
    )
    runs.mkdir(parents=True, exist_ok=True)
    runtime_cfg = config.get("runtime", {})
    watchdog = TrainProgressWatchdog(
        device=device,
        name=f"actor-{input_mode}-seed{seed}",
        log_path=runs / "train.log",
        heartbeat_s=float(runtime_cfg.get("heartbeat_s", 30.0)),
        stall_timeout_s=float(runtime_cfg.get("stall_timeout_s", 180.0)),
    )
    watchdog.log(f"start device={device} input_mode={input_mode} GPU={gpu_mem_str(device)}")

    try:
        return _train_semantic_actor_body(
            config=config,
            device=device,
            max_epochs=max_epochs,
            data_root=root,
            seed=seed,
            runs=runs,
            jepa_checkpoint=jepa_checkpoint,
            watchdog=watchdog,
            input_mode=input_mode,
        )
    except Exception as exc:
        watchdog.log(f"FATAL {type(exc).__name__}: {exc}")
        watchdog.log(format_exception(exc))
        raise
    finally:
        watchdog.set_stage("done")
        watchdog.stop()


def _train_semantic_actor_body(
    *,
    config: dict[str, Any],
    device: torch.device,
    max_epochs: int | None,
    data_root: Path,
    seed: int,
    runs: Path,
    jepa_checkpoint: Path | None,
    watchdog: TrainProgressWatchdog,
    input_mode: str,
) -> dict[str, Any]:
    root = data_root
    normalizer = load_command_normalizer(config, data_root=root)
    train_dir = root / "trajectories" / "actor" / "train"
    test_dir = root / "trajectories" / "actor" / "test"

    ckpt_path: Path | None = None
    if input_mode == "embedding":
        if jepa_checkpoint is not None:
            ckpt_path = Path(jepa_checkpoint)
        else:
            ckpt_path = resolve_run_checkpoint(
                project_root(config) / config["paths"]["runs_root"],
                jepa_run_dirname(config),
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        jepa = TSJEPA(config).to(device)
        jepa.load_state_dict(ckpt["model"])
        jepa.eval()
        for p in jepa.parameters():
            p.requires_grad_(False)
        train_full: Dataset = ActorEmbeddingDataset(
            train_dir, config, normalizer, jepa.context_encoder, device, training=True
        )
        test_ds: Dataset = ActorEmbeddingDataset(
            test_dir, config, normalizer, jepa.context_encoder, device, training=False
        )
        feature_dim = int(
            getattr(train_full, "feature_dim", None)
            or config["ts_jepa"]["encoder"]["embedding_dim"]
        )
    else:
        train_full = ActorStateDataset(train_dir, config, normalizer, training=True)
        test_ds = ActorStateDataset(test_dir, config, normalizer, training=False)
        feature_dim = int(train_full.feature_dim)  # type: ignore[attr-defined]
        watchdog.log(
            f"state features: kappa={config['input']['kappa']} "
            f"state_dim={train_full.state_dim} feature_dim={feature_dim}"  # type: ignore[attr-defined]
        )

    val_fraction = float(config["semantic_actor"]["early_stopping"].get("val_fraction", 0.2))
    train_ds, val_ds = _split_train_val_actor(train_full, val_fraction)

    opt_cfg = config["semantic_actor"]["optimizer"]
    actor_cfg = config["semantic_actor"]["architecture"]
    actor = SemanticActor(
        embedding_dim=feature_dim,
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
    train_loader = make_dataloader(
        train_ds,
        batch_size=min(batch_size, max(1, len(train_ds))),
        shuffle=True,
        device=device,
        config=config,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=min(batch_size, max(1, len(val_ds))),
        shuffle=False,
        device=device,
        config=config,
    )
    test_loader = make_dataloader(
        test_ds,
        batch_size=min(batch_size, max(1, len(test_ds))),
        shuffle=False,
        device=device,
        config=config,
    )

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
        watchdog.begin_epoch(epoch, len(train_loader))
        prefetcher = CUDAPrefetcher(train_loader, device)
        try:
            for batch in tqdm(
                prefetcher,
                desc=f"actor[{input_mode}] seed {seed} epoch {epoch}",
                leave=False,
                total=len(train_loader),
                mininterval=1.0,
            ):
                watchdog.touch(micro=n_batches + 1, stage="train")
                try:
                    emb = batch["embedding"]
                    target = batch["command_norm"]
                    pred = actor(emb)
                    loss = criterion(pred, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    step_loss = float(loss.detach().item())
                except Exception as exc:
                    reraise_cuda_context(
                        exc,
                        where=f"actor epoch {epoch} batch={n_batches}",
                        device=device,
                    )
                train_loss += step_loss
                n_batches += 1
                watchdog.touch(micro=n_batches, effective_step=n_batches, loss=step_loss)
        except DataLoaderStallError:
            watchdog.log(f"DataLoader stall at actor epoch={epoch} batch={n_batches}")
            raise
        train_loss /= max(1, n_batches)

        watchdog.set_stage("validate")
        try:
            val_loss = evaluate_mse(actor, val_loader, device, criterion)
        except Exception as exc:
            reraise_cuda_context(exc, where=f"actor epoch {epoch} validation", device=device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        watchdog.log(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = state_dict_to_cpu(actor.state_dict())
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                runs / "best.pt",
                {
                    "actor": best_state,
                    "jepa_checkpoint": str(ckpt_path) if ckpt_path is not None else None,
                    "normalizer": normalizer.to_dict(),
                    "config": config,
                    "val_loss": best_val,
                    "seed": seed,
                    "input_mode": input_mode,
                    "feature_dim": feature_dim,
                    "selection_split": "actor_train_holdout_validation",
                },
            )
            watchdog.log(f"saved best.pt epoch={epoch} val_loss={best_val:.6f}")
        else:
            stale += 1

        if config["semantic_actor"]["early_stopping"]["enabled"] and stale >= patience:
            watchdog.log(f"early stop epoch={epoch} stale={stale}")
            break

    if best_state is not None:
        actor.load_state_dict(best_state)

    watchdog.set_stage("test")
    try:
        test_loss = evaluate_mse(actor, test_loader, device, criterion)
    except Exception as exc:
        reraise_cuda_context(exc, where="actor untouched test", device=device)
    save_checkpoint(
        runs / "last.pt",
        {
            "actor": state_dict_to_cpu(actor.state_dict()),
            "jepa_checkpoint": str(ckpt_path) if ckpt_path is not None else None,
            "normalizer": normalizer.to_dict(),
            "config": config,
            "val_loss": best_val,
            "test_loss": test_loss,
            "seed": seed,
            "history": history,
            "input_mode": input_mode,
            "feature_dim": feature_dim,
            "selection_split": "actor_train_holdout_validation",
        },
    )
    with (runs / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": seed,
                "input_mode": input_mode,
                "feature_dim": feature_dim,
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
        "input_mode": input_mode,
        "feature_dim": feature_dim,
        "checkpoint": str(runs / "best.pt"),
    }


def train_semantic_actor_repetitions(
    config: dict[str, Any],
    jepa_checkpoint: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int | None = None,
    data_root: Path | None = None,
    input_mode: str = "embedding",
) -> dict[str, Any]:
    """Paper protocol: repeat actor training, select best validation run."""
    input_mode = _validate_input_mode(input_mode)
    reps = int(config["evaluation"]["repetitions"])
    seeds = list(config["evaluation"].get("seeds", list(range(reps))))[:reps]
    root_runs = (
        project_root(config) / config["paths"]["runs_root"] / _actor_runs_dirname(config, input_mode)
    )
    root_runs.mkdir(parents=True, exist_ok=True)
    if input_mode == "embedding":
        if jepa_checkpoint is not None:
            ckpt_path: Path | None = Path(jepa_checkpoint)
        else:
            ckpt_path = resolve_run_checkpoint(
                project_root(config) / config["paths"]["runs_root"],
                jepa_run_dirname(config),
            )
    else:
        ckpt_path = None

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
            input_mode=input_mode,
        )
        seed_results.append(result)

    best = min(seed_results, key=lambda r: r["best_val"])
    best_ckpt = Path(best["checkpoint"])
    selected = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    selected["selected_from_seeds"] = seeds
    selected["selection_criterion"] = "best_validation_mse"
    selected["input_mode"] = input_mode
    selected["seed_results"] = [
        {
            "seed": r["seed"],
            "best_val": r["best_val"],
            "test_loss": r["test_loss"],
            "checkpoint": r["checkpoint"],
            "input_mode": r["input_mode"],
        }
        for r in seed_results
    ]
    torch.save(selected, root_runs / "best.pt")
    summary = {
        "repetitions": reps,
        "seeds": seeds,
        "input_mode": input_mode,
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

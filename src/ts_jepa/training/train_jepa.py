from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ts_jepa.config import jepa_run_dirname, project_root
from ts_jepa.data.datasets import TrajectoryDataset, fit_command_normalizer, load_command_normalizer
from ts_jepa.device import select_device
from ts_jepa.models.predictor_command_resolution import load_predictor_command_resolution
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.training.jepa_optimizer import (
    apply_jepa_lr_decay,
    build_jepa_optimizer,
    should_apply_jepa_lr_decay,
)
from ts_jepa.training.jepa_procedure import jepa_forward_batch, jepa_sgd_and_ema_step
from ts_jepa.runtime import (
    CUDAPrefetcher,
    DataLoaderStallError,
    TrainProgressWatchdog,
    configure_train_logging,
    configure_training_runtime,
    format_exception,
    gpu_mem_str,
    load_checkpoint,
    make_dataloader,
    reraise_cuda_context,
    save_checkpoint,
    state_dict_to_cpu,
)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_ENCODER_ARCH_KEYS = ("type", "widths", "embedding_dim", "blocks_per_stage", "batch_norm", "activation")
_PREDICTOR_ARCH_KEYS = ("type", "hidden_dim", "output_dim", "activation", "autoregressive")
_PREDICTOR_COMMAND_RESOLUTION_KEYS = ("selected_source", "paper_exact")


def _optimizer_momentum(opt_cfg: dict[str, Any]) -> float:
    return float(opt_cfg.get("momentum", 0.0))


def _optimizer_batch_sizes(opt_cfg: dict[str, Any]) -> tuple[int, int]:
    effective = int(opt_cfg["batch_size"])
    micro = int(opt_cfg.get("microbatch_size", effective))
    return effective, micro


def validate_checkpoint_config_compatibility(
    checkpoint_config: dict[str, Any],
    current_config: dict[str, Any],
) -> None:
    """Fail clearly when a checkpoint cannot be resumed with the current config."""
    ckpt_ts = checkpoint_config["ts_jepa"]
    cur_ts = current_config["ts_jepa"]
    ckpt_opt = ckpt_ts["optimizer"]
    cur_opt = cur_ts["optimizer"]
    ckpt_effective_bs, ckpt_micro_bs = _optimizer_batch_sizes(ckpt_opt)
    cur_effective_bs, cur_micro_bs = _optimizer_batch_sizes(cur_opt)
    ckpt_cmd = ckpt_ts.get("predictor_command_resolution", {})
    cur_cmd = cur_ts.get("predictor_command_resolution", {})
    scalar_checks = [
        ("Kp", ckpt_ts["prediction_horizon"]["Kp"], cur_ts["prediction_horizon"]["Kp"]),
        ("kappa", checkpoint_config["input"]["kappa"], current_config["input"]["kappa"]),
        ("embedding_dim", ckpt_ts["encoder"]["embedding_dim"], cur_ts["encoder"]["embedding_dim"]),
        ("optimizer type", ckpt_opt["type"], cur_opt["type"]),
        ("learning_rate", ckpt_opt["learning_rate"], cur_opt["learning_rate"]),
        ("weight_decay", ckpt_opt["weight_decay"], cur_opt["weight_decay"]),
        ("optimizer momentum", _optimizer_momentum(ckpt_opt), _optimizer_momentum(cur_opt)),
        ("lr_decay.factor", ckpt_ts["lr_decay"]["factor"], cur_ts["lr_decay"]["factor"]),
        (
            "lr_decay.interval_epochs",
            ckpt_ts["lr_decay"]["interval_epochs"],
            cur_ts["lr_decay"]["interval_epochs"],
        ),
        ("batch_size", ckpt_effective_bs, cur_effective_bs),
        ("microbatch_size", ckpt_micro_bs, cur_micro_bs),
        ("ema_decay", ckpt_ts["target_encoder"]["ema_decay"], cur_ts["target_encoder"]["ema_decay"]),
    ]
    mismatches = [f"{name}: checkpoint={a!r} current={b!r}" for name, a, b in scalar_checks if a != b]
    for key in _PREDICTOR_COMMAND_RESOLUTION_KEYS:
        ckpt_val = ckpt_cmd.get(key)
        cur_val = cur_cmd.get(key)
        if ckpt_val != cur_val:
            mismatches.append(
                f"predictor_command_resolution.{key}: checkpoint={ckpt_val!r} current={cur_val!r}"
            )
    for key in _ENCODER_ARCH_KEYS:
        ckpt_val = ckpt_ts["encoder"].get(key)
        cur_val = cur_ts["encoder"].get(key)
        if ckpt_val != cur_val:
            mismatches.append(f"encoder.{key}: checkpoint={ckpt_val!r} current={cur_val!r}")
    for key in _PREDICTOR_ARCH_KEYS:
        ckpt_val = ckpt_ts["predictor"].get(key)
        cur_val = cur_ts["predictor"].get(key)
        if ckpt_val != cur_val:
            mismatches.append(f"predictor.{key}: checkpoint={ckpt_val!r} current={cur_val!r}")
    if mismatches:
        detail = "; ".join(mismatches)
        raise ValueError(f"Checkpoint config incompatible with current config: {detail}")


def _resumable_checkpoint_payload(
    *,
    model: TSJEPA,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    normalizer: Any,
    resolution: Any,
    epoch: int,
    best_val: float,
    best_epoch: int,
    stale: int,
    history: list[dict[str, float]],
    seed: int,
    best_state: dict[str, torch.Tensor] | None,
    test_loss: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": state_dict_to_cpu(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "normalizer": normalizer.to_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "stale": stale,
        "history": history,
        "seed": seed,
        "selection_split": "train_holdout_validation",
        "predictor_command_resolution": resolution.to_dict(),
    }
    if best_state is not None:
        payload["best_model"] = best_state
    if test_loss is not None:
        payload["test_loss"] = test_loss
        payload["test_split"] = "jepa_test_untouched"
    return payload


def _seed_training_complete(run_dir: Path) -> bool:
    """True when a seed finished training and wrote a final checkpoint."""
    metrics_path = run_dir / "metrics.json"
    last_path = run_dir / "last.pt"
    if not metrics_path.is_file() or not last_path.is_file():
        return False
    checkpoint = load_checkpoint(last_path)
    return "test_loss" in checkpoint


def _load_completed_seed_result(run_dir: Path) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "best_val": metrics["best_val"],
        "best_epoch": metrics["best_epoch"],
        "test_loss": metrics["test_loss"],
        "history": metrics["history"],
        "runs_dir": str(run_dir),
        "seed": metrics["seed"],
        "checkpoint": str(run_dir / "best.pt"),
    }


def resolve_jepa_batching(opt_cfg: dict[str, Any]) -> tuple[int, int, int]:
    """
    Paper effective batch vs GPU microbatch.

    Returns (effective_batch_size, microbatch_size, accumulation_steps).
    Gradients are accumulated so one optimizer/EMA step covers `effective_batch_size`
    samples. If `microbatch_size` is omitted, it defaults to `batch_size` (no accum).
    If `microbatch_size` exceeds `batch_size` (tiny smoke overrides), it is clamped.
    """
    effective = int(opt_cfg["batch_size"])
    micro = int(opt_cfg.get("microbatch_size", effective))
    if effective < 1:
        raise ValueError(f"batch_size must be >= 1, got {effective}")
    if micro < 1:
        raise ValueError(f"microbatch_size must be >= 1, got {micro}")
    if micro > effective:
        micro = effective
    if effective % micro != 0:
        raise ValueError(
            f"batch_size ({effective}) must be divisible by microbatch_size ({micro})"
        )
    return effective, micro, effective // micro


def accumulation_plan(num_microbatches: int, accum_steps: int) -> tuple[int, int, int]:
    """
    Map microbatches to complete effective steps only.

    Returns (num_effective_steps, num_microbatches_used, num_leftover_microbatches).
    Leftover micros must not call optimizer.step() and should not be trained on.
    """
    accum = int(accum_steps)
    if accum < 1:
        raise ValueError(f"accumulation_steps must be >= 1, got {accum}")
    n_micro = max(0, int(num_microbatches))
    n_effective = n_micro // accum
    n_used = n_effective * accum
    return n_effective, n_used, n_micro - n_used


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
    resolution = model.command_resolution
    total = 0.0
    n_batches = 0
    try:
        for batch in CUDAPrefetcher(loader, device):
            # Plan §13 steps 1–4 (eval; no SGD/EMA).
            result = jepa_forward_batch(model, batch, resolution)
            total += float(result.loss.item())
            n_batches += 1
    except DataLoaderStallError:
        raise
    except Exception as exc:
        reraise_cuda_context(exc, where=f"evaluate_cosine_loss after {n_batches} batches", device=device)
    return total / max(1, n_batches)


def train_ts_jepa(
    config: dict[str, Any],
    device: torch.device | None = None,
    max_epochs: int | None = None,
    data_root: Path | None = None,
    seed: int = 0,
    run_dir: Path | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """
    Train one TS-JEPA seed.

    Model selection uses validation carved from the train split only.
    The untouched JEPA test set is evaluated after training and never used for selection.
    Predictor conditioning uses the plan §10 documented candidate (see command_resolution).

    When ``resume_from`` points to a resumable ``last.pt``, training continues from the
    next epoch in the existing run directory without resetting optimizer or best metrics.
    """
    device = select_device(device)
    if resume_from is not None:
        resume_from = Path(resume_from)
        checkpoint = load_checkpoint(resume_from)
        validate_checkpoint_config_compatibility(checkpoint["config"], config)
        ckpt_seed = int(checkpoint["seed"])
        if seed != ckpt_seed:
            raise ValueError(
                f"Requested seed {seed} does not match checkpoint seed {ckpt_seed} in {resume_from}"
            )
        seed = ckpt_seed
        run_dir = resume_from.parent
    configure_train_logging()
    _set_seed(seed)
    configure_training_runtime(
        device,
        cudnn_benchmark=bool(config.get("runtime", {}).get("cudnn_benchmark", True)),
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    runs = run_dir or (
        project_root(config) / config["paths"]["runs_root"] / jepa_run_dirname(config) / f"seed_{seed}"
    )
    runs.mkdir(parents=True, exist_ok=True)
    runtime_cfg = config.get("runtime", {})
    watchdog = TrainProgressWatchdog(
        device=device,
        name=f"jepa-seed{seed}",
        log_path=runs / "train.log",
        heartbeat_s=float(runtime_cfg.get("heartbeat_s", 30.0)),
        stall_timeout_s=float(runtime_cfg.get("stall_timeout_s", 180.0)),
    )
    watchdog.log(
        f"start device={device} GPU={gpu_mem_str(device)} "
        f"workers={runtime_cfg.get('num_workers', 'default')} "
        f"dataloader_timeout_s={runtime_cfg.get('dataloader_timeout_s', 120.0)}"
        + (f" resume_from={resume_from}" if resume_from is not None else "")
    )

    try:
        return _train_ts_jepa_body(
            config=config,
            device=device,
            max_epochs=max_epochs,
            data_root=root,
            seed=seed,
            runs=runs,
            watchdog=watchdog,
            resume_from=resume_from,
        )
    except Exception as exc:
        watchdog.log(f"FATAL {type(exc).__name__}: {exc}")
        watchdog.log(format_exception(exc))
        raise
    finally:
        watchdog.set_stage("done")
        watchdog.stop()


def _train_ts_jepa_body(
    *,
    config: dict[str, Any],
    device: torch.device,
    max_epochs: int | None,
    data_root: Path,
    seed: int,
    runs: Path,
    watchdog: TrainProgressWatchdog,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    root = data_root
    normalizer = fit_command_normalizer(config, data_root=root)
    train_dir = root / "trajectories" / "jepa" / "train"
    test_dir = root / "trajectories" / "jepa" / "test"
    train_dataset = TrajectoryDataset(train_dir, config, normalizer, training=True)
    test_dataset = TrajectoryDataset(test_dir, config, normalizer, training=False)
    val_count = int(config["ts_jepa"]["early_stopping"]["validation_trajectory_count"])
    train_set, val_set = _split_train_val(train_dataset, val_count)

    opt_cfg = config["ts_jepa"]["optimizer"]
    effective_bs, micro_bs, accum_steps = resolve_jepa_batching(opt_cfg)
    # drop_last only when the split is large enough; smoke tests may be tiny.
    drop_last = len(train_set) >= effective_bs
    train_loader = make_dataloader(
        train_set,
        batch_size=micro_bs,
        shuffle=True,
        device=device,
        config=config,
        drop_last=drop_last,
    )
    # Eval uses microbatches for VRAM; protocol (holdout / untouched test) unchanged.
    val_loader = make_dataloader(
        val_set,
        batch_size=min(micro_bs, max(1, len(val_set))),
        shuffle=False,
        device=device,
        config=config,
        drop_last=False,
    )
    test_loader = make_dataloader(
        test_dataset,
        batch_size=min(micro_bs, max(1, len(test_dataset))),
        shuffle=False,
        device=device,
        config=config,
        drop_last=False,
    )
    n_effective, n_micros_used, n_leftover = accumulation_plan(len(train_loader), accum_steps)
    watchdog.log(
        f"data train={len(train_set)} val={len(val_set)} test={len(test_dataset)} "
        f"effective_bs={effective_bs} micro={micro_bs} accum={accum_steps} "
        f"micros/epoch={n_micros_used} steps/epoch={n_effective} leftover={n_leftover}"
    )

    model = TSJEPA(config).to(device)
    resolution = model.command_resolution
    watchdog.log(
        "predictor_command_resolution "
        + json.dumps(resolution.to_dict(), separators=(",", ":"))
    )
    optimizer = build_jepa_optimizer(model, config)

    epochs = int(max_epochs if max_epochs is not None else opt_cfg["epochs"])
    patience = int(config["ts_jepa"]["early_stopping"]["patience"])
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    start_epoch = 1

    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from)
        if "optimizer" not in checkpoint:
            raise ValueError(
                f"Checkpoint at {resume_from} is not resumable (missing optimizer state). "
                "Use a per-epoch last.pt written by the current trainer."
            )
        required = ("model", "epoch", "best_val", "best_epoch", "stale", "history", "config", "normalizer")
        missing = [key for key in required if key not in checkpoint]
        if missing:
            raise ValueError(f"Checkpoint at {resume_from} is missing required resume fields: {missing}")
        if int(checkpoint["seed"]) != seed:
            raise ValueError(
                f"Checkpoint seed {checkpoint['seed']!r} does not match requested seed {seed!r}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val"])
        best_epoch = int(checkpoint["best_epoch"])
        stale = int(checkpoint["stale"])
        history = list(checkpoint["history"])
        best_state = checkpoint.get("best_model")
        if best_state is None and (runs / "best.pt").is_file():
            best_ckpt = load_checkpoint(runs / "best.pt")
            best_state = best_ckpt["model"]
        watchdog.log(
            f"resumed from epoch={checkpoint['epoch']} -> start_epoch={start_epoch} "
            f"best_val={best_val:.6f} best_epoch={best_epoch} stale={stale} "
            f"lr={optimizer.param_groups[0]['lr']}"
        )
        if start_epoch > epochs:
            watchdog.log(
                f"checkpoint already completed epoch {checkpoint['epoch']} >= max_epochs={epochs}; "
                "running final evaluation only"
            )

    for epoch in range(start_epoch, epochs + 1):
        model.context_encoder.train()
        model.predictor.train()
        model.target_encoder.eval()
        train_loss = 0.0
        n_steps = 0
        micro_in_group = 0
        micros_seen = 0
        group_loss_sum: torch.Tensor | None = None
        optimizer.zero_grad(set_to_none=True)
        watchdog.begin_epoch(epoch, n_micros_used)
        prefetcher = CUDAPrefetcher(train_loader, device)
        try:
            batch_iter = iter(
                tqdm(
                    prefetcher,
                    desc=f"jepa seed {seed} epoch {epoch}",
                    leave=False,
                    total=n_micros_used,
                    mininterval=1.0,
                )
            )
            while micros_seen < n_micros_used:
                watchdog.set_stage("fetch_batch")
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    break
                except DataLoaderStallError:
                    raise
                except Exception as exc:
                    reraise_cuda_context(
                        exc,
                        where=f"jepa epoch {epoch} fetch micro={micros_seen}",
                        device=device,
                    )

                micros_seen += 1
                watchdog.touch(micro=micros_seen, stage="forward")
                try:
                    # Plan §13 steps 1–4 (context → target stop-grad → predict → loss).
                    result = jepa_forward_batch(model, batch, resolution)
                    loss = result.loss
                    # Scale so accumulated grads match mean loss over the effective batch.
                    watchdog.touch(micro=micros_seen, stage="backward")
                    (loss / accum_steps).backward()
                except Exception as exc:
                    reraise_cuda_context(
                        exc,
                        where=f"jepa epoch {epoch} micro={micros_seen}",
                        device=device,
                    )

                # Keep loss on GPU until the effective step ends (avoid per-micro .item() sync).
                det = loss.detach()
                group_loss_sum = det if group_loss_sum is None else (group_loss_sum + det)
                micro_in_group += 1
                if micro_in_group < accum_steps:
                    watchdog.touch(micro=micros_seen, stage="accumulating")
                    continue

                try:
                    watchdog.touch(micro=micros_seen, stage="optimizer")
                    # Plan §13 steps 5–6: SGD(θ,ϕ) then EMA(θ̄).
                    jepa_sgd_and_ema_step(model, optimizer)
                    assert group_loss_sum is not None
                    step_loss = float(group_loss_sum.item()) / accum_steps
                except Exception as exc:
                    reraise_cuda_context(
                        exc,
                        where=f"jepa epoch {epoch} optimizer step={n_steps + 1}",
                        device=device,
                    )
                train_loss += step_loss
                n_steps += 1
                group_loss_sum = None
                micro_in_group = 0
                watchdog.touch(micro=micros_seen, effective_step=n_steps, loss=step_loss, stage="train")
        except DataLoaderStallError as exc:
            watchdog.log(
                f"DataLoader stall at epoch={epoch} micro={micros_seen}/{n_micros_used} "
                f"last_fetch_s={prefetcher.last_fetch_s} GPU={gpu_mem_str(device)}"
            )
            raise

        # Incomplete trailing microbatches must never produce an optimizer step.
        if micro_in_group > 0:
            optimizer.zero_grad(set_to_none=True)
            group_loss_sum = None
            micro_in_group = 0
        if n_steps != n_effective or micros_seen != n_micros_used:
            raise RuntimeError(
                f"Effective step count mismatch: optimizer.step()={n_steps}, "
                f"planned={n_effective}, micros_seen={micros_seen}, used={n_micros_used}, "
                f"leftover={n_leftover}, accum={accum_steps}, loader_micros={len(train_loader)}"
            )

        train_loss /= max(1, n_steps)
        watchdog.set_stage("validate")
        watchdog.log(f"epoch={epoch} train_loss={train_loss:.6f} starting val GPU={gpu_mem_str(device)}")
        try:
            val_loss = evaluate_cosine_loss(model, val_loader, device)
        except Exception as exc:
            reraise_cuda_context(exc, where=f"jepa epoch {epoch} validation", device=device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        watchdog.log(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = state_dict_to_cpu(model.state_dict())
            best_epoch = epoch
            stale = 0
            watchdog.set_stage("checkpoint")
            save_checkpoint(
                runs / "best.pt",
                {
                    "model": best_state,
                    "config": config,
                    "normalizer": normalizer.to_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "seed": seed,
                    "selection_split": "train_holdout_validation",
                    "predictor_command_resolution": resolution.to_dict(),
                },
            )
            watchdog.log(f"saved best.pt epoch={epoch} val_loss={best_val:.6f}")
        else:
            stale += 1

        if should_apply_jepa_lr_decay(epoch, config):
            new_lr = apply_jepa_lr_decay(optimizer, config)
            watchdog.log(f"lr decayed to {new_lr}")

        if config["ts_jepa"]["early_stopping"]["enabled"] and stale >= patience:
            watchdog.log(f"early stop epoch={epoch} stale={stale}")
            break

        watchdog.set_stage("checkpoint")
        save_checkpoint(
            runs / "last.pt",
            _resumable_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                normalizer=normalizer,
                resolution=resolution,
                epoch=epoch,
                best_val=best_val,
                best_epoch=best_epoch,
                stale=stale,
                history=history,
                seed=seed,
                best_state=best_state,
            ),
        )
        watchdog.log(f"saved last.pt epoch={epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Untouched test set: report only; never used for checkpoint selection.
    watchdog.set_stage("test")
    try:
        test_loss = evaluate_cosine_loss(model, test_loader, device)
    except Exception as exc:
        reraise_cuda_context(exc, where="jepa untouched test", device=device)
    payload = _resumable_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        normalizer=load_command_normalizer(config, data_root=root),
        resolution=resolution,
        epoch=history[-1]["epoch"] if history else 0,
        best_val=best_val,
        best_epoch=best_epoch,
        stale=stale,
        history=history,
        seed=seed,
        best_state=best_state,
        test_loss=test_loss,
    )
    save_checkpoint(runs / "last.pt", payload)
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
    watchdog.log(
        f"done best_epoch={best_epoch} best_val={best_val:.6f} test_loss={test_loss:.6f} "
        f"ckpt={runs / 'best.pt'}"
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

    Seeds that already finished training (final ``last.pt`` with ``test_loss``) are skipped.
    """
    reps = int(config["evaluation"]["repetitions"])
    seeds = list(config["evaluation"].get("seeds", list(range(reps))))[:reps]
    root_runs = project_root(config) / config["paths"]["runs_root"] / jepa_run_dirname(config)
    root_runs.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in seeds:
        seed_dir = root_runs / f"seed_{seed}"
        if _seed_training_complete(seed_dir):
            seed_results.append(_load_completed_seed_result(seed_dir))
            continue
        result = train_ts_jepa(
            config,
            device=device,
            max_epochs=max_epochs,
            data_root=data_root,
            seed=int(seed),
            run_dir=seed_dir,
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
        "predictor_command_resolution": load_predictor_command_resolution(config).to_dict(),
    }
    with (root_runs / "repetition_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary

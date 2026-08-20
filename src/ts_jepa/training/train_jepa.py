from __future__ import annotations

import json
from contextlib import nullcontext
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
    apply_jepa_scheduled_lr,
    build_jepa_optimizer,
)
from ts_jepa.training.jepa_procedure import (
    attach_command_norm_range,
    jepa_forward_batch,
    jepa_sgd_and_ema_step,
    vicreg_regularizer,
)
from ts_jepa.runtime import (
    CUDAPrefetcher,
    DataLoaderStallError,
    TrainProgressWatchdog,
    inprocess_dataloader,
    is_dataloader_spawn_error,
    channels_last_enabled,
    configure_train_logging,
    configure_training_runtime,
    format_exception,
    gpu_mem_str,
    load_checkpoint,
    make_dataloader,
    recommended_jepa_microbatch,
    release_cuda_cache,
    reraise_cuda_context,
    resolve_amp_dtype,
    save_checkpoint,
    should_release_cuda_cache,
    state_dict_to_cpu,
)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_ENCODER_ARCH_KEYS = ("type", "widths", "embedding_dim", "blocks_per_stage", "batch_norm", "activation")
_PREDICTOR_ARCH_KEYS = (
    "type",
    "hidden_dim",
    "output_dim",
    "activation",
    "autoregressive",
    "hidden_batch_norm",
    "l2_normalize_output",
    "command_scale",
    "conditioning",
)
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
        (
            "lr_warmup_epochs",
            int(ckpt_opt.get("lr_warmup_epochs", 0)),
            int(cur_opt.get("lr_warmup_epochs", 0)),
        ),
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


def seed_training_complete(run_dir: Path, expected_epochs: int) -> bool:
    """
    True when a seed finished the requested epoch budget and wrote a final checkpoint.

    ``test_loss`` alone is not enough: a 2-epoch smoke run also writes test_loss
    and must not skip a later 150-epoch protocol.
    """
    metrics_path = run_dir / "metrics.json"
    last_path = run_dir / "last.pt"
    if not metrics_path.is_file() or not last_path.is_file():
        return False
    checkpoint = load_checkpoint(last_path)
    if "test_loss" not in checkpoint:
        return False
    epoch = int(checkpoint.get("epoch") or 0)
    history = checkpoint.get("history") or []
    history_epochs = len(history)
    reached = max(epoch, history_epochs)
    return reached >= int(expected_epochs)


_seed_training_complete = seed_training_complete


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


def resolve_jepa_batching(
    opt_cfg: dict[str, Any],
    *,
    device: torch.device | None = None,
    runtime_cfg: dict[str, Any] | None = None,
) -> tuple[int, int, int]:
    """
    Paper effective batch vs GPU microbatch.

    Returns (effective_batch_size, microbatch_size, accumulation_steps).
    Gradients are accumulated so one optimizer/EMA step covers `effective_batch_size`
    samples. If `microbatch_size` is omitted, it defaults to `batch_size` (no accum).
    If `microbatch_size` exceeds `batch_size` (tiny smoke overrides), it is clamped.

    When ``runtime.auto_tune_microbatch`` is on (default) and ``device`` is CUDA,
    microbatch is raised to the largest divisor of the paper batch that fits VRAM
    (256 on RTX 6000 Ada-class 48 GB cards).
    """
    effective = int(opt_cfg["batch_size"])
    raw_micro = opt_cfg.get("microbatch_size", effective)
    if effective < 1:
        raise ValueError(f"batch_size must be >= 1, got {effective}")
    runtime = runtime_cfg or {}
    auto = bool(runtime.get("auto_tune_microbatch", True))
    if device is not None and auto and str(raw_micro).strip().lower() != "off":
        micro = recommended_jepa_microbatch(effective, device)
    else:
        micro = int(raw_micro)
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


def _cuda_autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate_cosine_loss(
    model: TSJEPA,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None = None,
) -> float:
    model.eval()
    resolution = model.command_resolution
    total = 0.0
    n_batches = 0
    prefetcher = CUDAPrefetcher(loader, device)
    try:
        for batch in prefetcher:
            # Plan §10 Algorithm 1 steps 1–4 (eval; no SGD/EMA).
            with _cuda_autocast(device, amp_dtype):
                result = jepa_forward_batch(model, batch, resolution)
            contrast_w = float(getattr(model, "command_contrast_weight", 0.0) or 0.0)
            total += float(result.cosine_loss.item()) + contrast_w * float(result.command_contrast.item())
            n_batches += 1
            del result
    except DataLoaderStallError as exc:
        if int(getattr(loader, "num_workers", 0) or 0) > 0 and is_dataloader_spawn_error(exc):
            return evaluate_cosine_loss(model, inprocess_dataloader(loader), device, amp_dtype=amp_dtype)
        raise
    except Exception as exc:
        if int(getattr(loader, "num_workers", 0) or 0) > 0 and is_dataloader_spawn_error(exc):
            return evaluate_cosine_loss(model, inprocess_dataloader(loader), device, amp_dtype=amp_dtype)
        reraise_cuda_context(exc, where=f"evaluate_cosine_loss after {n_batches} batches", device=device)
    finally:
        prefetcher.close()
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
    Predictor conditioning uses the plan §9 documented candidate (see command_resolution).

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
    train_files = sorted(train_dir.glob("*.npz"))
    max_train = config.get("experiments", {}).get("max_jepa_train_trajectories")
    if max_train is not None:
        train_files = train_files[: int(max_train)]
    train_dataset = TrajectoryDataset(train_dir, config, normalizer, training=True, files=train_files)
    test_dataset = TrajectoryDataset(test_dir, config, normalizer, training=False)
    val_count = int(config["ts_jepa"]["early_stopping"]["validation_trajectory_count"])
    train_set, val_set = _split_train_val(train_dataset, val_count)

    opt_cfg = config["ts_jepa"]["optimizer"]
    runtime_cfg = config.get("runtime", {})
    effective_bs, micro_bs, accum_steps = resolve_jepa_batching(
        opt_cfg, device=device, runtime_cfg=runtime_cfg
    )
    amp_dtype = resolve_amp_dtype(device, runtime_cfg)
    use_channels_last = channels_last_enabled(device, runtime_cfg)
    scaler = None
    if amp_dtype is torch.float16:
        scaler = torch.amp.GradScaler("cuda")
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
    # Eval must not spawn workers: Windows cannot pickle the in-memory RGB
    # TrajectoryDataset a second time while train persistent_workers are alive
    # (OSError 22 / truncated pickle). Val/test stay in-process.
    val_loader = make_dataloader(
        val_set,
        batch_size=min(micro_bs, max(1, len(val_set))),
        shuffle=False,
        device=device,
        config=config,
        drop_last=False,
        num_workers=0,
    )
    test_loader = make_dataloader(
        test_dataset,
        batch_size=min(micro_bs, max(1, len(test_dataset))),
        shuffle=False,
        device=device,
        config=config,
        drop_last=False,
        num_workers=0,
    )
    n_effective, n_micros_used, n_leftover = accumulation_plan(len(train_loader), accum_steps)
    watchdog.log(
        f"data train={len(train_set)} val={len(val_set)} test={len(test_dataset)} "
        f"effective_bs={effective_bs} micro={micro_bs} accum={accum_steps} "
        f"micros/epoch={n_micros_used} steps/epoch={n_effective} leftover={n_leftover} "
        f"amp={str(amp_dtype).replace('torch.', '') if amp_dtype is not None else 'off'} "
        f"channels_last={use_channels_last}"
    )

    model = TSJEPA(config).to(device)
    if use_channels_last:
        model.context_encoder.to(memory_format=torch.channels_last)
        model.target_encoder.to(memory_format=torch.channels_last)
    chunk_cfg = runtime_cfg.get("target_encode_chunk_size", "auto")
    if chunk_cfg in {"auto", None} or (
        isinstance(chunk_cfg, str) and chunk_cfg.strip().lower() == "auto"
    ):
        model.target_encode_chunk_size = 0
    else:
        model.target_encode_chunk_size = int(chunk_cfg)
    attach_command_norm_range(model, normalizer, config)
    resolution = model.command_resolution
    watchdog.log(
        "predictor_command_resolution "
        + json.dumps(resolution.to_dict(), separators=(",", ":"))
    )
    watchdog.log(
        f"command_contrast_sampling={model.command_contrast_sampling} "
        f"command_norm_range=[{model.command_norm_min:.6g}, {model.command_norm_max:.6g}]"
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
        epoch_lr = apply_jepa_scheduled_lr(optimizer, epoch, config)
        train_loss = 0.0
        train_cosine = 0.0
        train_vicreg_var = 0.0
        train_vicreg_cov = 0.0
        train_contrast = 0.0
        train_contrast_teacher = 0.0
        train_contrast_broad = 0.0
        n_steps = 0
        micro_in_group = 0
        micros_seen = 0
        group_loss_sum: torch.Tensor | None = None
        group_components_sum: torch.Tensor | None = None
        group_contexts: list[torch.Tensor] = []
        var_w = float(getattr(model, "vicreg_variance_weight", 0.0) or 0.0)
        cov_w = float(getattr(model, "vicreg_covariance_weight", 0.0) or 0.0)
        gamma = float(getattr(model, "vicreg_gamma", 1.0) or 1.0)
        cov_stdize = bool(getattr(model, "vicreg_covariance_standardize", False))
        contrast_w = float(getattr(model, "command_contrast_weight", 0.0) or 0.0)
        # Covariance on microbatch 16×256 is rank-deficient; apply VICReg on the
        # concatenated effective batch after cosine grads are accumulated.
        vicreg_on_effective = (var_w != 0.0 or cov_w != 0.0) and accum_steps > 1
        optimizer.zero_grad(set_to_none=True)
        watchdog.begin_epoch(epoch, n_micros_used)
        watchdog.log(f"epoch={epoch} lr={epoch_lr:.6g} GPU={gpu_mem_str(device)}")
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
                    # Plan §10 Algorithm 1 steps 1–4 (context → target stop-grad → predict → loss).
                    with _cuda_autocast(device, amp_dtype):
                        result = jepa_forward_batch(model, batch, resolution)
                        task_loss = result.cosine_loss + contrast_w * result.command_contrast
                        loss_bwd = task_loss if vicreg_on_effective else result.loss
                    # Scale so accumulated grads match mean loss over the effective batch.
                    watchdog.touch(micro=micros_seen, stage="backward")
                    scaled = loss_bwd / accum_steps
                    if scaler is not None:
                        scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                except Exception as exc:
                    reraise_cuda_context(
                        exc,
                        where=f"jepa epoch {epoch} micro={micros_seen}",
                        device=device,
                    )

                # Keep loss on GPU until the effective step ends (avoid per-micro .item() sync).
                det = task_loss.detach() if vicreg_on_effective else result.loss.detach()
                group_loss_sum = det if group_loss_sum is None else (group_loss_sum + det)
                components = torch.stack(
                    (
                        result.cosine_loss.detach(),
                        result.vicreg_variance.detach(),
                        result.vicreg_covariance.detach(),
                        result.command_contrast.detach(),
                        result.command_contrast_teacher.detach(),
                        result.command_contrast_broad.detach(),
                    )
                )
                group_components_sum = (
                    components if group_components_sum is None else (group_components_sum + components)
                )
                if vicreg_on_effective:
                    group_contexts.append(batch["context"])
                micro_in_group += 1
                if micro_in_group < accum_steps:
                    watchdog.touch(micro=micros_seen, stage="accumulating")
                    continue

                try:
                    watchdog.touch(micro=micros_seen, stage="optimizer")
                    assert group_loss_sum is not None
                    assert group_components_sum is not None
                    cosine_v, var_v, cov_v, contrast_v, contrast_teacher_v, contrast_broad_v = (
                        group_components_sum / accum_steps
                    ).tolist()
                    step_loss = float(group_loss_sum.item()) / accum_steps
                    if vicreg_on_effective:
                        z_all = model.encode_context(torch.cat(group_contexts, dim=0))
                        with _cuda_autocast(device, amp_dtype):
                            vicreg, var_term, cov_term = vicreg_regularizer(
                                z_all,
                                variance_weight=var_w,
                                covariance_weight=cov_w,
                                gamma=gamma,
                                covariance_standardize=cov_stdize,
                            )
                        if scaler is not None:
                            scaler.scale(vicreg).backward()
                        else:
                            vicreg.backward()
                        var_v = float(var_term.detach())
                        cov_v = float(cov_term.detach())
                        step_loss = float(cosine_v) + contrast_w * float(contrast_v) + float(vicreg.detach())
                    # Plan §10 Algorithm 1 steps 5–6: SGD(θ,ϕ) then EMA(θ̄).
                    jepa_sgd_and_ema_step(
                        model,
                        optimizer,
                        max_grad_norm=float(opt_cfg.get("grad_clip_norm", 0.0)) or None,
                        scaler=scaler,
                    )
                except Exception as exc:
                    reraise_cuda_context(
                        exc,
                        where=f"jepa epoch {epoch} optimizer step={n_steps + 1}",
                        device=device,
                    )
                train_loss += step_loss
                train_cosine += float(cosine_v)
                train_vicreg_var += float(var_v)
                train_vicreg_cov += float(cov_v)
                train_contrast += float(contrast_v)
                train_contrast_teacher += float(contrast_teacher_v)
                train_contrast_broad += float(contrast_broad_v)
                n_steps += 1
                group_loss_sum = None
                group_components_sum = None
                group_contexts = []
                micro_in_group = 0
                watchdog.touch(micro=micros_seen, effective_step=n_steps, loss=step_loss, stage="train")
        except DataLoaderStallError as exc:
            watchdog.log(
                f"DataLoader stall at epoch={epoch} micro={micros_seen}/{n_micros_used} "
                f"last_fetch_s={prefetcher.last_fetch_s} GPU={gpu_mem_str(device)}"
            )
            raise
        finally:
            prefetcher.close()
            group_contexts = []
            group_loss_sum = None
            group_components_sum = None

        # Incomplete trailing microbatches must never produce an optimizer step.
        if micro_in_group > 0:
            optimizer.zero_grad(set_to_none=True)
            group_loss_sum = None
            group_components_sum = None
            group_contexts = []
            micro_in_group = 0
        if n_steps != n_effective or micros_seen != n_micros_used:
            raise RuntimeError(
                f"Effective step count mismatch: optimizer.step()={n_steps}, "
                f"planned={n_effective}, micros_seen={micros_seen}, used={n_micros_used}, "
                f"leftover={n_leftover}, accum={accum_steps}, loader_micros={len(train_loader)}"
            )

        train_loss /= max(1, n_steps)
        train_cosine /= max(1, n_steps)
        train_vicreg_var /= max(1, n_steps)
        train_vicreg_cov /= max(1, n_steps)
        train_contrast /= max(1, n_steps)
        train_contrast_teacher /= max(1, n_steps)
        train_contrast_broad /= max(1, n_steps)
        watchdog.set_stage("validate")
        watchdog.log(
            f"epoch={epoch} train_loss={train_loss:.6f} cosine={train_cosine:.6f} "
            f"vicreg_var={train_vicreg_var:.6f} vicreg_cov={train_vicreg_cov:.6f} "
            f"cmd_contrast={train_contrast:.6f} "
            f"cmd_contrast_teacher={train_contrast_teacher:.6f} "
            f"cmd_contrast_broad={train_contrast_broad:.6f} "
            f"starting val GPU={gpu_mem_str(device)}"
        )
        try:
            val_loss = evaluate_cosine_loss(model, val_loader, device, amp_dtype=amp_dtype)
        except Exception as exc:
            reraise_cuda_context(exc, where=f"jepa epoch {epoch} validation", device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_cosine": train_cosine,
                "train_vicreg_variance": train_vicreg_var,
                "train_vicreg_covariance": train_vicreg_cov,
                "train_command_contrast": train_contrast,
                "train_command_contrast_teacher": train_contrast_teacher,
                "train_command_contrast_broad": train_contrast_broad,
                "val_loss": val_loss,
            }
        )
        watchdog.log(
            f"epoch={epoch} train_loss={train_loss:.6f} cosine={train_cosine:.6f} "
            f"vicreg_var={train_vicreg_var:.6f} vicreg_cov={train_vicreg_cov:.6f} "
            f"cmd_contrast={train_contrast:.6f} "
            f"cmd_contrast_teacher={train_contrast_teacher:.6f} "
            f"cmd_contrast_broad={train_contrast_broad:.6f} "
            f"val_loss={val_loss:.6f}"
        )

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
        if should_release_cuda_cache(device, runtime_cfg):
            release_cuda_cache(device)
            watchdog.log(f"epoch={epoch} cache_released GPU={gpu_mem_str(device)}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Untouched test set: report only; never used for checkpoint selection.
    watchdog.set_stage("test")
    try:
        test_loss = evaluate_cosine_loss(model, test_loader, device, amp_dtype=amp_dtype)
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

    Seeds that already finished the requested epoch budget (final ``last.pt``
    with ``test_loss`` and history/epoch covering ``expected_epochs``) are skipped.
    """
    reps = int(config["evaluation"]["repetitions"])
    seeds = list(config["evaluation"].get("seeds", list(range(reps))))[:reps]
    root_runs = project_root(config) / config["paths"]["runs_root"] / jepa_run_dirname(config)
    root_runs.mkdir(parents=True, exist_ok=True)
    expected_epochs = int(max_epochs if max_epochs is not None else config["ts_jepa"]["optimizer"]["epochs"])

    seed_results = []
    for seed in seeds:
        seed_dir = root_runs / f"seed_{seed}"
        if seed_training_complete(seed_dir, expected_epochs):
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

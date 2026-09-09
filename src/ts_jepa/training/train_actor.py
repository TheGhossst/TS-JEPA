from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, project_root
from ts_jepa.data.datasets import ActorEmbeddingDataset, load_command_normalizer
from ts_jepa.device import select_device
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.plan.actor import PLAN_SEMANTIC_ACTOR
from ts_jepa.plan.enforce import plan_enforced
from ts_jepa.training.actor_helpers import (
    actor_loss_kwargs,
    actor_regression_loss,
    actor_train_recipe_id,
    evaluate_actor_objective,
    evaluate_mse,
    mean_command_baseline_mse,
    split_train_val_actor,
)
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


def _assert_encoder_frozen(jepa: TSJEPA) -> None:
    """Plan §12: TS-JEPA encoder is no longer updated; only Cε is optimized."""
    trainable = [n for n, p in jepa.named_parameters() if p.requires_grad]
    if trainable:
        raise RuntimeError(
            "plan §12 requires frozen TS-JEPA during actor training; "
            f"trainable JEPA params remain: {trainable[:8]}"
        )


def _set_actor_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    schedule: str,
) -> float:
    kind = str(schedule or "constant").strip().lower()
    if kind in {"", "constant", "none"}:
        lr = float(base_lr)
    elif kind == "cosine":
        warm = max(int(warmup_epochs), 0)
        if warm > 0 and epoch <= warm:
            lr = float(base_lr) * float(epoch) / float(max(warm, 1))
        else:
            t = (epoch - warm) / max(int(epochs) - warm, 1)
            lr = float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))
    else:
        raise ValueError(f"unknown actor lr_schedule {schedule!r}")
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _assert_optimizer_only_actor(optimizer: torch.optim.Optimizer, actor: SemanticActor) -> None:
    """Plan §12: optimize only Cε."""
    actor_ids = {id(p) for p in actor.parameters()}
    opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if opt_ids != actor_ids:
        raise RuntimeError(
            "plan §12 requires optimizer to cover exactly SemanticActor parameters "
            f"(actor={len(actor_ids)}, optimizer={len(opt_ids)})"
        )


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
        / actor_run_dirname(config)
        / f"seed_{seed}"
    )
    runs.mkdir(parents=True, exist_ok=True)
    runtime_cfg = config.get("runtime", {})
    watchdog = TrainProgressWatchdog(
        device=device,
        name=f"actor-seed{seed}",
        log_path=runs / "train.log",
        heartbeat_s=float(runtime_cfg.get("heartbeat_s", 30.0)),
        stall_timeout_s=float(runtime_cfg.get("stall_timeout_s", 180.0)),
    )
    watchdog.log(f"start device={device} GPU={gpu_mem_str(device)}")

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
) -> dict[str, Any]:
    root = data_root
    if jepa_checkpoint is not None:
        ckpt_path = Path(jepa_checkpoint)
    else:
        ckpt_path = resolve_run_checkpoint(
            project_root(config) / config["paths"]["runs_root"],
            jepa_run_dirname(config),
        )
    ckpt_path = ckpt_path.resolve()
    runtime_cfg = config.get("runtime", {})
    amp_dtype = resolve_amp_dtype(device, runtime_cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    _assert_encoder_frozen(jepa)

    normalizer = load_command_normalizer(config, data_root=root)
    train_dir = root / "trajectories" / "actor" / "train"
    test_dir = root / "trajectories" / "actor" / "test"
    # Plan §12: D_a from trained Ψθ (no further encoder updates); cache embeddings then drop JEPA.
    train_full = ActorEmbeddingDataset(train_dir, config, normalizer, jepa.context_encoder, device, training=True)
    test_ds = ActorEmbeddingDataset(test_dir, config, normalizer, jepa.context_encoder, device, training=False)
    del jepa
    if device.type == "cuda":
        torch.cuda.empty_cache()

    val_fraction = float(config["semantic_actor"]["early_stopping"].get("val_fraction", 0.2))
    split_mode = str(config["semantic_actor"]["early_stopping"].get("split", "contiguous"))
    train_ds, val_ds = split_train_val_actor(
        train_full, val_fraction, seed=seed, mode=split_mode
    )

    opt_cfg = config["semantic_actor"]["optimizer"]
    actor = SemanticActor.from_config(config).to(device)
    if plan_enforced(config) and (
        int(config["ts_jepa"]["encoder"]["embedding_dim"]) == PLAN_SEMANTIC_ACTOR["input_dim"]
        and list(config["semantic_actor"]["architecture"]["hidden_dims"])
        == PLAN_SEMANTIC_ACTOR["hidden_dims"]
    ):
        actor.assert_plan_architecture()

    if plan_enforced(config) and str(opt_cfg.get("type", "AdamW")) != "AdamW":
        raise ValueError(
            f"plan §12 requires AdamW for the semantic actor; got {opt_cfg.get('type')!r}"
        )
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
    )
    _assert_optimizer_only_actor(optimizer, actor)
    loss_name = str(config["semantic_actor"].get("loss", "MSE"))
    if plan_enforced(config) and loss_name.upper() != "MSE":
        raise ValueError(
            f"plan §12 requires MSE actor loss; got {config['semantic_actor'].get('loss')!r}"
        )
    loss_kwargs = actor_loss_kwargs(config)
    report_criterion = nn.MSELoss()
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
        num_workers=0,
    )
    test_loader = make_dataloader(
        test_ds,
        batch_size=min(batch_size, max(1, len(test_ds))),
        shuffle=False,
        device=device,
        config=config,
        num_workers=0,
    )

    epochs = int(max_epochs if max_epochs is not None else opt_cfg["epochs"])
    patience = int(config["semantic_actor"]["early_stopping"]["patience"])
    min_epochs = int(config["semantic_actor"]["early_stopping"].get("min_epochs", 0))
    grad_clip = float(config["semantic_actor"].get("grad_clip_norm", 0.0) or 0.0)
    schedule = str(config["semantic_actor"].get("lr_schedule", "constant"))
    warmup_epochs = int(config["semantic_actor"].get("lr_warmup_epochs", 0))
    base_lr = float(opt_cfg["learning_rate"])
    scaler = (
        torch.amp.GradScaler("cuda")
        if amp_dtype is torch.float16 and device.type == "cuda"
        else None
    )
    recipe = actor_train_recipe_id(config)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    last_epoch = 0

    watchdog.log(
        f"actor recipe={recipe} loss={loss_name} split={split_mode} "
        f"large_force_weight={loss_kwargs['large_force_weight']} "
        f"std_match_weight={loss_kwargs['std_match_weight']} min_epochs={min_epochs}"
    )

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        lr_now = _set_actor_lr(
            optimizer,
            base_lr=base_lr,
            epoch=epoch,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            schedule=schedule,
        )
        actor.train()
        train_loss = 0.0
        n_batches = 0
        watchdog.begin_epoch(epoch, len(train_loader))
        prefetcher = CUDAPrefetcher(train_loader, device)
        try:
            for batch in tqdm(
                prefetcher,
                desc=f"actor seed {seed} epoch {epoch}",
                leave=False,
                total=len(train_loader),
                mininterval=1.0,
            ):
                watchdog.touch(micro=n_batches + 1, stage="train")
                try:
                    emb = batch["embedding"]
                    target = batch["command"]
                    optimizer.zero_grad(set_to_none=True)
                    if amp_dtype is not None and device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            pred = actor(emb)
                            loss = actor_regression_loss(pred, target, **loss_kwargs)
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            if grad_clip > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            if grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
                            optimizer.step()
                    else:
                        pred = actor(emb)
                        loss = actor_regression_loss(pred, target, **loss_kwargs)
                        loss.backward()
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
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
        finally:
            prefetcher.close()
        train_loss /= max(1, n_batches)

        watchdog.set_stage("validate")
        try:
            val_loss = evaluate_actor_objective(
                actor, val_loader, device, loss_kwargs=loss_kwargs
            )
        except Exception as exc:
            reraise_cuda_context(exc, where=f"actor epoch {epoch} validation", device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr_now,
            }
        )
        watchdog.log(
            f"epoch={epoch} lr={lr_now:.6g} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        if should_release_cuda_cache(device, runtime_cfg):
            release_cuda_cache(device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = state_dict_to_cpu(actor.state_dict())
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                runs / "best.pt",
                {
                    "actor": best_state,
                    "jepa_checkpoint": str(ckpt_path.resolve()),
                    "normalizer": normalizer.to_dict(),
                    "config": config,
                    "val_loss": best_val,
                    "seed": seed,
                    "selection_split": "actor_train_holdout_validation",
                    "train_recipe": recipe,
                },
            )
            watchdog.log(f"saved best.pt epoch={epoch} val_loss={best_val:.6f}")
        elif epoch >= min_epochs:
            stale += 1

        if (
            config["semantic_actor"]["early_stopping"]["enabled"]
            and epoch >= min_epochs
            and stale >= patience
        ):
            watchdog.log(f"early stop epoch={epoch} stale={stale}")
            break

    if best_state is not None:
        actor.load_state_dict(best_state)

    watchdog.set_stage("test")
    try:
        test_loss = evaluate_mse(actor, test_loader, device, report_criterion)
    except Exception as exc:
        reraise_cuda_context(exc, where="actor untouched test", device=device)

    train_targets = np.asarray([float(s[1]) for s in train_full.samples], dtype=np.float64)
    test_targets = np.asarray([float(s[1]) for s in test_ds.samples], dtype=np.float64)
    train_mean_norm = float(train_targets.mean()) if train_targets.size else float("nan")
    mean_baseline_train = mean_command_baseline_mse(train_targets, train_mean_norm)
    mean_baseline_test = mean_command_baseline_mse(test_targets, train_mean_norm)
    beats_mean_baseline = bool(
        np.isfinite(test_loss) and np.isfinite(mean_baseline_test) and test_loss < mean_baseline_test
    )

    save_checkpoint(
        runs / "last.pt",
        {
            "actor": state_dict_to_cpu(actor.state_dict()),
            "jepa_checkpoint": str(ckpt_path.resolve()),
            "normalizer": normalizer.to_dict(),
            "config": config,
            "val_loss": best_val,
            "test_loss": test_loss,
            "seed": seed,
            "epoch": int(last_epoch),
            "best_epoch": int(best_epoch),
            "history": history,
            "selection_split": "actor_train_holdout_validation",
            "mean_command_baseline_mse_train": mean_baseline_train,
            "mean_command_baseline_mse_test": mean_baseline_test,
            "beats_mean_command_baseline": beats_mean_baseline,
            "train_recipe": recipe,
        },
    )
    with (runs / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": seed,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "test_loss": test_loss,
                "history": history,
                "mean_command_baseline_mse_train": mean_baseline_train,
                "mean_command_baseline_mse_test": mean_baseline_test,
                "beats_mean_command_baseline": beats_mean_baseline,
                "train_recipe": recipe,
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
        "mean_command_baseline_mse_test": mean_baseline_test,
        "beats_mean_command_baseline": beats_mean_baseline,
        "train_recipe": recipe,
    }


def seed_actor_training_complete(
    run_dir: Path,
    expected_epochs: int,
    *,
    recipe: str | None = None,
) -> bool:
    """True when an actor seed finished and wrote last.pt + test_loss for this recipe."""
    del expected_epochs  # early-stop is allowed; recipe + test_loss mark a finished seed
    metrics_path = run_dir / "metrics.json"
    last_path = run_dir / "last.pt"
    if not metrics_path.is_file() or not last_path.is_file() or not (run_dir / "best.pt").is_file():
        return False
    checkpoint = load_checkpoint(last_path)
    if "test_loss" not in checkpoint:
        return False
    if recipe is not None:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        stored = str(metrics.get("train_recipe") or checkpoint.get("train_recipe") or "")
        if stored != str(recipe):
            return False
    return True


def _load_completed_actor_seed_result(run_dir: Path) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "best_val": metrics["best_val"],
        "best_epoch": metrics["best_epoch"],
        "test_loss": metrics["test_loss"],
        "history": metrics["history"],
        "runs_dir": str(run_dir),
        "seed": metrics["seed"],
        "checkpoint": str(run_dir / "best.pt"),
        "mean_command_baseline_mse_test": metrics.get("mean_command_baseline_mse_test"),
        "beats_mean_command_baseline": metrics.get("beats_mean_command_baseline"),
        "train_recipe": metrics.get("train_recipe"),
    }


def _maybe_refine_actor_dagger(
    config: dict[str, Any],
    *,
    jepa_checkpoint: Path,
    actor_checkpoint: Path,
    device: torch.device,
    data_root: Path | None,
    max_epochs: int | None,
) -> dict[str, Any] | None:
    dagger_cfg = config.get("semantic_actor", {}).get("dagger") or {}
    if not bool(dagger_cfg.get("enabled", False)):
        return None
    if max_epochs is not None and int(max_epochs) <= 3:
        return None
    from ts_jepa.training.dagger_actor import train_actor_dagger

    overrides = {k: v for k, v in dagger_cfg.items() if k != "enabled"}
    print("actor: LQR DAgger refinement (closed-loop recovery data)", flush=True)
    return train_actor_dagger(
        config,
        jepa_checkpoint=jepa_checkpoint,
        actor_checkpoint=actor_checkpoint,
        device=device,
        data_root=data_root,
        skip_eval=False,
        settings_overrides=overrides,
    )


def train_semantic_actor_repetitions(
    config: dict[str, Any],
    jepa_checkpoint: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Paper protocol (plan §12): repeat actor training, select best validation run."""
    reps = int(config["evaluation"]["repetitions"])
    seeds = list(config["evaluation"].get("seeds", list(range(reps))))[:reps]
    root_runs = project_root(config) / config["paths"]["runs_root"] / actor_run_dirname(config)
    root_runs.mkdir(parents=True, exist_ok=True)
    if jepa_checkpoint is not None:
        ckpt_path = Path(jepa_checkpoint)
    else:
        ckpt_path = resolve_run_checkpoint(
            project_root(config) / config["paths"]["runs_root"],
            jepa_run_dirname(config),
        )
    expected_epochs = int(
        max_epochs if max_epochs is not None else config["semantic_actor"]["optimizer"]["epochs"]
    )
    recipe = actor_train_recipe_id(config)

    seed_results = []
    for seed in seeds:
        seed_dir = root_runs / f"seed_{seed}"
        if seed_actor_training_complete(seed_dir, expected_epochs, recipe=recipe):
            seed_results.append(_load_completed_actor_seed_result(seed_dir))
            continue
        result = train_semantic_actor(
            config,
            jepa_checkpoint=ckpt_path,
            device=device,
            max_epochs=max_epochs,
            data_root=data_root,
            seed=int(seed),
            run_dir=seed_dir,
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
    dagger_report = _maybe_refine_actor_dagger(
        config,
        jepa_checkpoint=ckpt_path,
        actor_checkpoint=root_runs / "best.pt",
        device=select_device(device),
        data_root=data_root,
        max_epochs=max_epochs,
    )
    if dagger_report and dagger_report.get("checkpoint"):
        dagger_path = Path(dagger_report["checkpoint"])
        if dagger_path.is_file():
            shutil.copy2(dagger_path, root_runs / "best.pt")
            selected = torch.load(root_runs / "best.pt", map_location="cpu", weights_only=False)
            selected["selection_criterion"] = "lqr_dagger_after_best_bc"
            selected["bc_seed"] = best["seed"]
            torch.save(selected, root_runs / "best.pt")
    summary = {
        "repetitions": reps,
        "seeds": seeds,
        "selection_criterion": selected.get("selection_criterion", "best_validation_mse"),
        "best_seed": best["seed"],
        "best_val": best["best_val"],
        "best_test_loss": best["test_loss"],
        "seed_results": selected.get("seed_results")
        or [
            {
                "seed": r["seed"],
                "best_val": r["best_val"],
                "test_loss": r["test_loss"],
                "checkpoint": r["checkpoint"],
            }
            for r in seed_results
        ],
        "best_checkpoint": str(root_runs / "best.pt"),
        "train_recipe": recipe,
        "dagger": dagger_report,
    }
    with (root_runs / "repetition_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary

#!/usr/bin/env python
"""One full-config JEPA epoch on CUDA with gradient accumulation (effective batch 256)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ts_jepa.config import load_config, project_root
from ts_jepa.data.datasets import TrajectoryDataset, fit_command_normalizer
from ts_jepa.device import describe_device, select_device
from ts_jepa.losses.cosine import cosine_alignment_loss
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.training.train_jepa import _set_seed, _split_train_val, evaluate_cosine_loss, resolve_jepa_batching


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_mem_str(device: torch.device) -> str:
    if device.type != "cuda":
        return "n/a"
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free_b, total_b = torch.cuda.mem_get_info()
    used = (total_b - free_b) / (1024**3)
    return f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB used={used:.2f}/{total_b/(1024**3):.2f}GB"


def run_effective_step(
    model: TSJEPA,
    optimizer: torch.optim.Optimizer,
    loader_iter,
    device: torch.device,
    accum_steps: int,
) -> tuple[float, int]:
    """Accumulate `accum_steps` microbatches; one optimizer + EMA step. Returns (mean_loss, samples)."""
    group_loss = 0.0
    samples = 0
    for _ in range(accum_steps):
        batch = next(loader_iter)
        context = batch["context"].to(device, non_blocking=True)
        future = batch["future_frames"].to(device, non_blocking=True)
        cmds = batch["teacher_commands_norm"].to(device, non_blocking=True)
        z = model.encode_context(context)
        with torch.no_grad():
            zt = model.encode_targets(future)
        zp = model.predict(z)
        loss = cosine_alignment_loss(zp, zt)
        (loss / accum_steps).backward()
        group_loss += float(loss.detach().cpu())
        samples += int(context.shape[0])
    optimizer.step()
    model.ema_step()
    optimizer.zero_grad(set_to_none=True)
    return group_loss / accum_steps, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=None,
        help="Override config microbatch_size (effective batch stays config batch_size).",
    )
    args = parser.parse_args()

    config = load_config()
    if args.microbatch_size is not None:
        config["ts_jepa"]["optimizer"]["microbatch_size"] = int(args.microbatch_size)

    device = select_device()
    root = project_root(config)
    data_root = root / config["paths"]["data_root"]
    train_dir = data_root / "trajectories" / "jepa" / "train"
    n_train_files = len(list(train_dir.glob("*.npz")))
    expected = int(config["ts_jepa"]["dataset"]["train_trajectories"])
    if n_train_files != expected:
        raise RuntimeError(f"Expected {expected} JEPA train trajectories, found {n_train_files} in {train_dir}")

    effective_bs, micro_bs, accum_steps = resolve_jepa_batching(config["ts_jepa"]["optimizer"])
    log(f"device={device} | {json.dumps(describe_device(device))}")
    log(
        f"batching: effective={effective_bs} microbatch={micro_bs} "
        f"accumulation={accum_steps} (optimizer.step once per {effective_bs} samples)"
    )
    log(f"train trajectories on disk: {n_train_files}")

    _set_seed(0)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    log("Fitting command normalizer on JEPA train commands...")
    t0 = time.perf_counter()
    normalizer = fit_command_normalizer(config, data_root=data_root)
    log(f"Normalizer ready in {time.perf_counter() - t0:.1f}s (mu={normalizer.mean:.4f}, sigma={normalizer.std:.4f})")

    log("Loading TrajectoryDataset into memory (may take a bit)...")
    t0 = time.perf_counter()
    dataset = TrajectoryDataset(train_dir, config, normalizer, training=True)
    log(f"Dataset loaded in {time.perf_counter() - t0:.1f}s | samples={len(dataset)} files={len(dataset.files)}")

    val_count = int(config["ts_jepa"]["early_stopping"]["validation_trajectory_count"])
    train_set, val_set = _split_train_val(dataset, val_count)
    log(f"Split: train_samples={len(train_set)} val_samples={len(val_set)} (val traj count IC={val_count})")

    opt_cfg = config["ts_jepa"]["optimizer"]
    drop_last = len(train_set) >= effective_bs
    train_loader = DataLoader(
        train_set, batch_size=micro_bs, shuffle=True, num_workers=0, drop_last=drop_last
    )
    val_loader = DataLoader(
        val_set, batch_size=min(micro_bs, max(1, len(val_set))), shuffle=False, num_workers=0
    )
    n_micro = len(train_loader)
    n_effective = n_micro // accum_steps
    log(
        f"Loaders ready | microbatch={micro_bs} micro_batches={n_micro} "
        f"effective_steps={n_effective} lr={opt_cfg['learning_rate']} "
        f"Kp={config['ts_jepa']['prediction_horizon']['Kp']}"
    )

    log("Building TSJEPA model on device...")
    t0 = time.perf_counter()
    model = TSJEPA(config).to(device)
    optimizer = torch.optim.SGD(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        momentum=0.0,
    )
    log(f"Model ready in {time.perf_counter() - t0:.1f}s | gpu={gpu_mem_str(device)}")

    model.context_encoder.train()
    model.predictor.train()
    model.target_encoder.eval()
    train_iter = iter(train_loader)

    log("Warmup effective step (CUDA init; excluded from reported train time)...")
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    warm_loss, _ = run_effective_step(model, optimizer, train_iter, device, accum_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    log(
        f"Warmup done in {time.perf_counter() - t0:.1f}s | loss={warm_loss:.6f} | "
        f"gpu={gpu_mem_str(device)}"
    )

    # Rebuild iterator so timed epoch sees a full pass (warmup consumed one effective step).
    train_iter = iter(train_loader)
    train_loss_sum = 0.0
    n_steps = 0
    samples_seen = 0
    step_times: list[float] = []

    log(f"Starting timed train epoch ({n_effective} effective steps)...")
    train_t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, n_effective + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        bt0 = time.perf_counter()
        step_loss, step_samples = run_effective_step(
            model, optimizer, train_iter, device, accum_steps
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - bt0
        step_times.append(dt)
        train_loss_sum += step_loss
        n_steps += 1
        samples_seen += step_samples
        elapsed = time.perf_counter() - train_t0
        eta = (elapsed / n_steps) * (n_effective - n_steps) if n_steps else 0.0
        log(
            f"effective step {n_steps}/{n_effective} | "
            f"loss={step_loss:.6f} avg={train_loss_sum/n_steps:.6f} | "
            f"step_s={dt:.2f} elapsed={elapsed:.1f}s eta={eta:.1f}s | "
            f"gpu={gpu_mem_str(device)}"
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - train_t0
    train_loss = train_loss_sum / max(1, n_steps)
    log(
        f"Train epoch done | time={train_time:.2f}s loss={train_loss:.6f} "
        f"throughput={samples_seen/max(train_time,1e-9):.2f} samples/s"
    )

    model.eval()
    log(f"Starting validation ({len(val_loader)} microbatches)...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    val_t0 = time.perf_counter()
    val_loss = evaluate_cosine_loss(model, val_loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    val_time = time.perf_counter() - val_t0
    log(f"Validation done | time={val_time:.2f}s loss={val_loss:.6f}")

    info = describe_device(device)
    report = {
        **info,
        "config": {
            "effective_batch_size": effective_bs,
            "microbatch_size": micro_bs,
            "accumulation_steps": accum_steps,
            "learning_rate": opt_cfg["learning_rate"],
            "weight_decay": opt_cfg["weight_decay"],
            "Kp": config["ts_jepa"]["prediction_horizon"]["Kp"],
            "train_trajectories_on_disk": n_train_files,
            "val_trajectory_count_ic": val_count,
            "train_samples": len(train_set),
            "val_samples": len(val_set),
        },
        "train_time_s": round(train_time, 4),
        "validation_time_s": round(val_time, 4),
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "num_effective_steps": n_steps,
        "samples_seen": samples_seen,
        "effective_step_time_s_mean": round(sum(step_times) / max(1, len(step_times)), 4),
        "throughput_samples_per_s": round(samples_seen / max(train_time, 1e-9), 4),
        "throughput_effective_steps_per_s": round(n_steps / max(train_time, 1e-9), 4),
    }
    if device.type == "cuda":
        report["peak_gpu_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        report["peak_gpu_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved() / (1024**3), 3)
        free_b, total_b = torch.cuda.mem_get_info()
        report["gpu_memory_used_gb"] = round((total_b - free_b) / (1024**3), 3)

    out = root / "runs" / "eval" / "one_epoch_jepa_cuda.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    log("Final report:")
    print(json.dumps(report, indent=2), flush=True)
    log(f"Wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FAILED with exception:")
        raise

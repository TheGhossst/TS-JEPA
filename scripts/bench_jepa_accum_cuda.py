#!/usr/bin/env python
"""Short CUDA JEPA gradient-accumulation benchmark (effective batch 256).

Does NOT run full training. Benchmarks microbatch 16 (accum 16) and 32 (accum 8).
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ts_jepa.config import load_config, project_root
from ts_jepa.data.datasets import TrajectoryDataset, fit_command_normalizer
from ts_jepa.device import describe_device, select_device
from ts_jepa.losses.cosine import cosine_alignment_loss
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.training.train_jepa import _set_seed, _split_train_val, resolve_jepa_batching


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_effective_step(
    model: TSJEPA,
    optimizer: torch.optim.Optimizer,
    loader_iter,
    device: torch.device,
    accum_steps: int,
) -> float:
    group_loss = 0.0
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
    optimizer.step()
    model.ema_step()
    optimizer.zero_grad(set_to_none=True)
    return group_loss / accum_steps


def bench_one(
    config: dict,
    device: torch.device,
    train_set,
    microbatch_size: int,
    timed_steps: int = 4,
) -> dict:
    cfg = copy.deepcopy(config)
    cfg["ts_jepa"]["optimizer"]["microbatch_size"] = int(microbatch_size)
    effective_bs, micro_bs, accum_steps = resolve_jepa_batching(cfg["ts_jepa"]["optimizer"])
    assert effective_bs == 256
    assert micro_bs * accum_steps == effective_bs

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    drop_last = len(train_set) >= effective_bs
    loader = DataLoader(
        train_set, batch_size=micro_bs, shuffle=True, num_workers=0, drop_last=drop_last
    )
    n_effective_per_epoch = len(loader) // accum_steps

    model = TSJEPA(cfg).to(device)
    optimizer = torch.optim.SGD(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=float(cfg["ts_jepa"]["optimizer"]["learning_rate"]),
        weight_decay=float(cfg["ts_jepa"]["optimizer"]["weight_decay"]),
        momentum=0.0,
    )
    model.context_encoder.train()
    model.predictor.train()
    model.target_encoder.eval()

    log(f"=== microbatch={micro_bs} accumulation={accum_steps} ===")
    it = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    warm_loss = run_effective_step(model, optimizer, it, device, accum_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    warm_s = time.perf_counter() - t0
    log(f"warmup effective step: {warm_s:.2f}s loss={warm_loss:.6f}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    it = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    step_times: list[float] = []
    losses: list[float] = []
    for i in range(1, timed_steps + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = run_effective_step(model, optimizer, it, device, accum_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        losses.append(loss)
        log(f"timed effective step {i}/{timed_steps}: {dt:.2f}s loss={loss:.6f}")

    mean_step = sum(step_times) / len(step_times)
    epoch_est = mean_step * n_effective_per_epoch
    report = {
        "microbatch_size": micro_bs,
        "accumulation_steps": accum_steps,
        "effective_batch_size": effective_bs,
        "timed_effective_steps": timed_steps,
        "time_per_effective_batch_s_mean": round(mean_step, 4),
        "time_per_effective_batch_s_min": round(min(step_times), 4),
        "time_per_effective_batch_s_max": round(max(step_times), 4),
        "effective_steps_per_epoch": n_effective_per_epoch,
        "epoch_time_estimate_s": round(epoch_est, 2),
        "epoch_time_estimate_min": round(epoch_est / 60.0, 2),
        "mean_train_loss": round(sum(losses) / len(losses), 6),
        "warmup_effective_step_s": round(warm_s, 4),
    }
    if device.type == "cuda":
        report["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        report["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / (1024**3), 3)
        free_b, total_b = torch.cuda.mem_get_info()
        report["gpu_memory_used_gb"] = round((total_b - free_b) / (1024**3), 3)
        report["gpu_memory_total_gb"] = round(total_b / (1024**3), 3)

    # Free before next config.
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return report


def main() -> None:
    device = select_device()
    config = load_config()
    root = project_root(config)
    data_root = root / config["paths"]["data_root"]
    train_dir = data_root / "trajectories" / "jepa" / "train"
    n_files = len(list(train_dir.glob("*.npz")))
    expected = int(config["ts_jepa"]["dataset"]["train_trajectories"])
    if n_files != expected:
        raise RuntimeError(f"Expected {expected} JEPA train trajectories, found {n_files}")

    _set_seed(0)
    log(f"device={device} | {json.dumps(describe_device(device))}")
    normalizer = fit_command_normalizer(config, data_root=data_root)
    dataset = TrajectoryDataset(train_dir, config, normalizer, training=True)
    val_count = int(config["ts_jepa"]["early_stopping"]["validation_trajectory_count"])
    train_set, _ = _split_train_val(dataset, val_count)
    log(f"train_samples={len(train_set)} (val traj holdout={val_count})")

    results = []
    for micro in (16, 32):
        results.append(bench_one(config, device, train_set, microbatch_size=micro, timed_steps=4))

    payload = {
        "device": describe_device(device),
        "paper_settings_unchanged": {
            "Kp": config["ts_jepa"]["prediction_horizon"]["Kp"],
            "learning_rate": config["ts_jepa"]["optimizer"]["learning_rate"],
            "effective_batch_size": config["ts_jepa"]["optimizer"]["batch_size"],
            "weight_decay": config["ts_jepa"]["optimizer"]["weight_decay"],
            "epochs": config["ts_jepa"]["optimizer"]["epochs"],
        },
        "benchmarks": results,
    }
    out = root / "runs" / "eval" / "jepa_accum_cuda_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n=== SUMMARY ===", flush=True)
    print(
        f"{'micro':>6} {'accum':>6} {'peak_alloc_GB':>14} {'peak_res_GB':>12} "
        f"{'s/eff_batch':>12} {'epoch_est_min':>14}",
        flush=True,
    )
    for r in results:
        print(
            f"{r['microbatch_size']:6d} {r['accumulation_steps']:6d} "
            f"{r.get('peak_allocated_gb', float('nan')):14.3f} "
            f"{r.get('peak_reserved_gb', float('nan')):12.3f} "
            f"{r['time_per_effective_batch_s_mean']:12.2f} "
            f"{r['epoch_time_estimate_min']:14.2f}",
            flush=True,
        )
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()

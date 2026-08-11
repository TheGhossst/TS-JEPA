#!/usr/bin/env python
"""Tiny 1-epoch CUDA JEPA smoke: report device, memory, batch/epoch timing."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ts_jepa.config import load_config
from ts_jepa.data.datasets import TrajectoryDataset, fit_command_normalizer
from ts_jepa.data.trajectory_generator import build_env_and_teacher, generate_dataset_split
from ts_jepa.device import describe_device, select_device
from ts_jepa.losses.cosine import cosine_alignment_loss
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.training.train_jepa import _set_seed, _split_train_val


def main() -> None:
    device = select_device()
    config = copy.deepcopy(load_config())
    tmp = Path("runs") / "cuda_smoke"
    data_root = tmp / "data"
    config["paths"]["data_root"] = str(data_root)
    config["paths"]["runs_root"] = str(tmp / "runs")
    config["simulation"]["trajectory_steps"] = 40
    config["ts_jepa"]["dataset"]["train_trajectories"] = 8
    config["ts_jepa"]["dataset"]["test_trajectories"] = 2
    config["ts_jepa"]["prediction_horizon"]["Kp"] = 5
    config["ts_jepa"]["optimizer"]["batch_size"] = 16
    config["ts_jepa"]["early_stopping"]["validation_trajectory_count"] = 2
    config["control_teacher"]["value_iteration_iters"] = 2
    config["control_teacher"]["force_bins"] = 5
    config["control_teacher"]["grid"] = {
        "x": [-0.4, 0.4, 5],
        "x_dot": [-1.0, 1.0, 5],
        "theta": [-0.2, 0.2, 5],
        "theta_dot": [-1.0, 1.0, 5],
    }

    env, teacher = build_env_and_teacher(config)
    generate_dataset_split(config, "jepa_train", 8, 0, data_root / "trajectories" / "jepa" / "train", env, teacher)
    generate_dataset_split(config, "jepa_test", 2, 8, data_root / "trajectories" / "jepa" / "test", env, teacher)
    normalizer = fit_command_normalizer(config, data_root=data_root)

    _set_seed(0)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    model = TSJEPA(config).to(device)
    opt = torch.optim.SGD(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=float(config["ts_jepa"]["optimizer"]["learning_rate"]),
        weight_decay=float(config["ts_jepa"]["optimizer"]["weight_decay"]),
        momentum=0.0,
    )
    ds = TrajectoryDataset(data_root / "trajectories" / "jepa" / "train", config, normalizer, training=True)
    train_set, _ = _split_train_val(ds, 2)
    batch_size = int(config["ts_jepa"]["optimizer"]["batch_size"])
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Warmup one batch (exclude from timing stats).
    batch = next(iter(loader))
    context = batch["context"].to(device)
    future = batch["future_frames"].to(device)
    cmds = batch["teacher_commands_norm"].to(device)
    z = model.encode_context(context)
    with torch.no_grad():
        zt = model.encode_targets(future)
    zp = model.predict(z, cmds)
    loss = cosine_alignment_loss(zp, zt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    model.ema_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    batch_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    epoch_t0 = time.perf_counter()
    model.context_encoder.train()
    model.predictor.train()
    model.target_encoder.eval()
    n_batches = 0
    for batch in loader:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        context = batch["context"].to(device, non_blocking=True)
        future = batch["future_frames"].to(device, non_blocking=True)
        cmds = batch["teacher_commands_norm"].to(device, non_blocking=True)
        z = model.encode_context(context)
        with torch.no_grad():
            zt = model.encode_targets(future)
        zp = model.predict(z, cmds)
        loss = cosine_alignment_loss(zp, zt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.ema_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_times.append(time.perf_counter() - t0)
        n_batches += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    epoch_time = time.perf_counter() - epoch_t0

    info = describe_device(device)
    report = {
        **info,
        "batch_size": batch_size,
        "num_batches": n_batches,
        "batch_time_s_mean": round(sum(batch_times) / max(1, len(batch_times)), 4),
        "batch_time_s_min": round(min(batch_times), 4) if batch_times else None,
        "batch_time_s_max": round(max(batch_times), 4) if batch_times else None,
        "epoch_time_s": round(epoch_time, 4),
        "final_loss": float(loss.detach().cpu()),
    }
    if device.type == "cuda":
        report["gpu_memory_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        report["gpu_memory_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / (1024**3), 3)
        free_b, total_b = torch.cuda.mem_get_info()
        report["gpu_memory_used_gb"] = round((total_b - free_b) / (1024**3), 3)

    out = tmp / "cuda_smoke_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

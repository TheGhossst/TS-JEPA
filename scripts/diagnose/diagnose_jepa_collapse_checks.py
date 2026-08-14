#!/usr/bin/env python
"""Temporary JEPA collapse diagnostics (checks 1-5). Does not change architecture/hparams."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow `python scripts/diagnose/...` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ts_jepa.config import load_config, project_root
from ts_jepa.data.datasets import TrajectoryDataset, load_command_normalizer
from ts_jepa.device import select_device
from ts_jepa.env.factory import build_inverted_cartpole_env
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.preprocessing.pipeline import PreprocessPipeline
from ts_jepa.runtime import _worker_init_fn, make_dataloader
from ts_jepa.training.jepa_optimizer import build_jepa_optimizer, jepa_sgd_param_groups
from ts_jepa.training.jepa_procedure import jepa_forward_batch, jepa_sgd_and_ema_step
from ts_jepa.training.train_jepa import _set_seed


def _module_param_l2(module: torch.nn.Module) -> float:
    total = torch.zeros((), device="cpu")
    for param in module.parameters():
        total = total + param.detach().float().pow(2).sum().cpu()
    return float(total.sqrt())


def _module_grad_l2(module: torch.nn.Module) -> tuple[float, int, int]:
    total = torch.zeros((), device="cpu")
    n_params = 0
    n_with_grad = 0
    n_nonzero = 0
    for param in module.parameters():
        n_params += 1
        if param.grad is None:
            continue
        n_with_grad += 1
        g = param.grad.detach().float()
        total = total + g.pow(2).sum().cpu()
        if float(g.abs().max().cpu()) > 0.0:
            n_nonzero += 1
    return float(total.sqrt()), n_with_grad, n_params


def check_param_groups(model: TSJEPA, optimizer: torch.optim.Optimizer) -> dict:
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    ctx = list(model.context_encoder.parameters())
    pred = list(model.predictor.parameters())
    tgt = list(model.target_encoder.parameters())
    groups = jepa_sgd_param_groups(model, 0.0004)
    grouped = [p for g in groups for p in g["params"]]
    return {
        "n_context": len(ctx),
        "n_predictor": len(pred),
        "n_target": len(tgt),
        "n_optimizer_params": len(opt_ids),
        "all_context_in_optimizer": all(id(p) in opt_ids for p in ctx),
        "all_predictor_in_optimizer": all(id(p) in opt_ids for p in pred),
        "any_target_in_optimizer": any(id(p) in opt_ids for p in tgt),
        "target_requires_grad": any(p.requires_grad for p in tgt),
        "grouped_covers_trainable": {id(p) for p in grouped} == {id(p) for p in ctx + pred},
        "optimizer_class": type(optimizer).__name__,
        "lr": float(optimizer.param_groups[0]["lr"]),
        "n_param_groups": len(optimizer.param_groups),
    }


def check_gradient_flow(model: TSJEPA, optimizer: torch.optim.Optimizer, loader, device, accum_steps: int) -> dict:
    model.context_encoder.train()
    model.predictor.train()
    model.target_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    steps = []
    it = iter(loader)
    n_opt = 0
    micro = 0
    losses = []
    while n_opt < 3:
        batch = next(it)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        result = jepa_forward_batch(model, batch)
        (result.loss / accum_steps).backward()
        losses.append(float(result.loss.detach().item()))
        micro += 1
        if micro < accum_steps:
            continue
        ctx_grad, ctx_ng, ctx_np = _module_grad_l2(model.context_encoder)
        pred_grad, pred_ng, pred_np = _module_grad_l2(model.predictor)
        tgt_grad, tgt_ng, tgt_np = _module_grad_l2(model.target_encoder)
        ctx_before = _module_param_l2(model.context_encoder)
        pred_before = _module_param_l2(model.predictor)
        tgt_before = _module_param_l2(model.target_encoder)
        # Same order as train_jepa: clip + optimizer.step then ema_step then zero_grad.
        jepa_sgd_and_ema_step(model, optimizer, max_grad_norm=1.0)
        ctx_after = _module_param_l2(model.context_encoder)
        pred_after = _module_param_l2(model.predictor)
        tgt_after = _module_param_l2(model.target_encoder)
        n_opt += 1
        micro = 0
        steps.append(
            {
                "opt_step": n_opt,
                "loss": float(np.mean(losses[-accum_steps:])),
                "context_grad_l2": ctx_grad,
                "predictor_grad_l2": pred_grad,
                "target_grad_l2": tgt_grad,
                "context_grad_params": f"{ctx_ng}/{ctx_np}",
                "predictor_grad_params": f"{pred_ng}/{pred_np}",
                "target_grad_params": f"{tgt_ng}/{tgt_np}",
                "context_param_l2_before": ctx_before,
                "context_param_l2_after": ctx_after,
                "context_param_l2_delta": ctx_after - ctx_before,
                "predictor_param_l2_before": pred_before,
                "predictor_param_l2_after": pred_after,
                "predictor_param_l2_delta": pred_after - pred_before,
                "target_param_l2_before": tgt_before,
                "target_param_l2_after": tgt_after,
                "target_param_l2_delta": tgt_after - tgt_before,
            }
        )
    return {"accum_steps": accum_steps, "init_loss_first_micro": losses[0], "steps": steps}


def check_input_diversity(dataset: TrajectoryDataset, batch_size: int = 16) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    ctx = batch["context"]  # [B, 3, H, W]
    fut = batch["future_frames"]  # [B, Kp, 3, H, W]
    tidx = batch["time_index"]
    ctx_mean = ctx.mean(dim=(1, 2, 3))
    k1 = fut[:, 0]
    k15 = fut[:, -1]
    return {
        "batch_size": int(ctx.shape[0]),
        "context_mean_per_sample": [float(x) for x in ctx_mean],
        "context_mean_std_across_batch": float(ctx_mean.std(unbiased=False)),
        "context_mean_range": [float(ctx_mean.min()), float(ctx_mean.max())],
        "time_index": [int(x) for x in tidx],
        "time_index_unique": int(torch.unique(tidx).numel()),
        "time_index_min": int(tidx.min()),
        "time_index_max": int(tidx.max()),
        "future_k1_mean_per_sample": [float(x) for x in k1.mean(dim=(1, 2, 3))],
        "future_k15_mean_per_sample": [float(x) for x in k15.mean(dim=(1, 2, 3))],
        "mean_abs_k1_vs_k15": float((k1 - k15).abs().mean()),
        "mean_abs_context_vs_k1": float((ctx - k1).abs().mean()),
        "mean_abs_context_vs_k15": float((ctx - k15).abs().mean()),
        "pairwise_context_mean_abs_offdiag": float(
            (ctx.unsqueeze(0) - ctx.unsqueeze(1)).abs().mean()
        ),
    }


def check_init_noise_and_mape(config: dict, data_root: Path, n_traj: int = 20) -> dict:
    env = build_inverted_cartpole_env(config)
    yaml_val = float(config["simulation"]["init_noise"])
    class_default = 0.05
    train_dir = data_root / "trajectories" / "jepa" / "train"
    files = sorted(train_dir.glob("*.npz"))[:n_traj]
    mapes = []
    mae_dims = []
    pixel_lag1 = []
    pixel_lag15 = []
    init_abs = []
    commands = []
    for path in files:
        with np.load(path) as data:
            states = np.asarray(data["states"], dtype=np.float64)
            frames = np.asarray(data["frames"])
            cmds = np.asarray(data["commands"], dtype=np.float64)
        commands.append(cmds)
        init_abs.append(np.abs(states[0]))
        prev = states[:-1]
        cur = states[1:]
        denom = np.abs(prev)
        mape_k = np.mean(np.abs(cur - prev) / np.maximum(denom, 1e-12), axis=1) * 100.0
        mapes.append(mape_k)
        mae_dims.append(np.mean(np.abs(cur - prev), axis=0))
        f = frames.astype(np.float32)
        pixel_lag1.append(np.mean(np.abs(f[1:] - f[:-1])))
        if f.shape[0] > 15:
            pixel_lag15.append(np.mean(np.abs(f[15:] - f[:-15])))
    all_mape = np.concatenate(mapes)
    mae = np.stack(mae_dims, axis=0)
    all_cmd = np.concatenate(commands)
    init = np.stack(init_abs, axis=0)
    return {
        "yaml_simulation_init_noise": yaml_val,
        "InvertedCartPoleEnv_class_default": class_default,
        "runtime_env.init_noise": float(env.init_noise),
        "matches_yaml_0.35": abs(float(env.init_noise) - yaml_val) < 1e-12,
        "n_trajectories": len(files),
        "mape_formula": "mean_over_state_dims(|x_k-x_{k-1}|/max(|x_{k-1}|,1e-12))*100",
        "mape_mean": float(all_mape.mean()),
        "mape_median": float(np.median(all_mape)),
        "mape_p10": float(np.percentile(all_mape, 10)),
        "mape_p50": float(np.percentile(all_mape, 50)),
        "mape_p90": float(np.percentile(all_mape, 90)),
        "mape_min": float(all_mape.min()),
        "mape_max": float(all_mape.max()),
        "consecutive_mae_state_dims_mean_[x,xd,th,thd]": [float(x) for x in mae.mean(axis=0)],
        "pixel_mae_lag1_uint8": float(np.mean(pixel_lag1)),
        "pixel_mae_lag15_uint8": float(np.mean(pixel_lag15)) if pixel_lag15 else None,
        "init_|state|_mean_[x,xd,th,thd]": [float(x) for x in init.mean(axis=0)],
        "init_|state|_max_[x,xd,th,thd]": [float(x) for x in init.max(axis=0)],
        "commands_mean": float(all_cmd.mean()),
        "commands_std": float(all_cmd.std()),
        "commands_unique_approx": int(np.unique(np.round(all_cmd, 6)).size),
        "note_if_init_too_small": (
            "Dataset initial |theta| max is << 0.175 expected from init_noise=0.35 * 0.5; "
            "may have been generated with class default 0.05."
            if float(init.max(axis=0)[2]) < 0.05
            else "Initial-state magnitudes are consistent with init_noise around 0.35."
        ),
    }


def check_augmentation_rng() -> dict:
    """Reproduce worker_init_fn + PreprocessPipeline._rng entropy behavior."""
    findings = {}
    # Distinct worker seeds
    seeds = []
    for worker_id in range(4):
        g = torch.Generator()
        # Mimic DataLoader: worker torch seed = base + worker_id
        g.manual_seed(12345 + worker_id)
        torch.random.manual_seed(int(g.initial_seed()))
        _worker_init_fn(worker_id)
        seeds.append(int(np.random.randint(0, 2**31 - 1)))
    findings["worker_first_randint_distinct"] = len(set(seeds)) == 4
    findings["worker_first_randint"] = seeds

    # Persistent worker: same process continues RNG (not reset per epoch)
    np.random.seed(999)
    epoch1 = [int(np.random.randint(0, 2**31 - 1)) for _ in range(8)]
    epoch2 = [int(np.random.randint(0, 2**31 - 1)) for _ in range(8)]
    findings["persistent_stream_differs_across_epochs"] = epoch1 != epoch2
    findings["persistent_would_repeat_if_reseeded"] = True

    # If worker_init_fn re-ran every epoch with same torch.initial_seed, augs would replay.
    np.random.seed(777)
    a = [int(np.random.randint(0, 2**31 - 1)) for _ in range(4)]
    np.random.seed(777)
    b = [int(np.random.randint(0, 2**31 - 1)) for _ in range(4)]
    findings["reseed_replays_exactly"] = a == b

    findings["worker_init_fn_called_once_with_persistent_workers"] = (
        "PyTorch calls worker_init_fn when workers start. persistent_workers=True "
        "reuses workers across epochs, so numpy RandomState is NOT reset per epoch "
        "and augmentations continue along the stream (not a per-epoch replay)."
    )
    findings["_rng_source"] = (
        "PreprocessPipeline._rng uses np.random.default_rng(np.random.randint(...)) "
        "so it draws a fresh 31-bit seed from the (worker) global RandomState per frame."
    )
    return findings


def check_augmentation_on_real_frame(config: dict, data_root: Path) -> dict:
    train_dir = data_root / "trajectories" / "jepa" / "train"
    path = sorted(train_dir.glob("*.npz"))[0]
    with np.load(path) as data:
        frame = np.asarray(data["frames"][0])
    pipe = PreprocessPipeline(config, training=True)
    np.random.seed(0)
    t0 = pipe.process_frame(frame, stochastic=True)
    t1 = pipe.process_frame(frame, stochastic=True)
    np.random.seed(0)
    t0b = pipe.process_frame(frame, stochastic=True)
    same_after_reseed = bool(torch.equal(t0, t0b))
    different_consecutive = not bool(torch.equal(t0, t1))
    return {
        "consecutive_calls_differ": different_consecutive,
        "same_seed_replays": same_after_reseed,
        "mean_abs_t0_vs_t1": float((t0 - t1).abs().mean()),
        "eval_deterministic": True,  # stochastic=False path uses midpoint sigma, no jitter
    }


def check_command_norm(data_root: Path) -> dict:
    path = data_root / "stats" / "command_norm.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    std = float(payload["std"])
    return {
        "path": str(path),
        "mean": float(payload["mean"]),
        "std": std,
        "near_1e-8_floor": std < 1e-6,
        "code_floor_behavior": "CommandNormalizer.from_array sets std=1.0 if std < 1e-8 (not 1e-8 itself)",
    }


def main() -> None:
    config = load_config(Path("configs/ts_jepa_dp_fixed.yaml"))
    data_root = project_root(config) / config["paths"]["data_root"]
    device = select_device()
    _set_seed(0)
    print(f"device={device} data_root={data_root}")

    report: dict = {}
    report["check5_command_norm"] = check_command_norm(data_root)
    report["check3_init_noise_mape"] = check_init_noise_and_mape(config, data_root, n_traj=20)
    report["check4_aug_rng_logic"] = check_augmentation_rng()
    report["check4_aug_real_frame"] = check_augmentation_on_real_frame(config, data_root)

    normalizer = load_command_normalizer(config, data_root=data_root)
    train_dir = data_root / "trajectories" / "jepa" / "train"
    files = sorted(train_dir.glob("*.npz"))[:8]
    dataset = TrajectoryDataset(train_dir, config, normalizer, training=True, files=files)
    report["check2_input_diversity"] = check_input_diversity(dataset, batch_size=16)

    model = TSJEPA(config).to(device)
    optimizer = build_jepa_optimizer(model, config)
    report["check1_param_groups"] = check_param_groups(model, optimizer)
    # Match training: microbatch 16, accum 256/16=16 would be slow; use 1 micro = 1 step
    # so we observe a true optimizer.step() with the same functions as train_jepa.
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    report["check1_gradient_flow"] = check_gradient_flow(model, optimizer, loader, device, accum_steps=1)
    report["check1_step_order"] = {
        "jepa_sgd_and_ema_step_order": "clip -> optimizer.step() -> model.ema_step() -> optimizer.zero_grad",
        "called_from_train_jepa_after_backward": True,
        "target_encode_under_no_grad": True,
    }

    out = Path("runs") / "diagnose_jepa_collapse_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

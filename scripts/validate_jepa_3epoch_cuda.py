#!/usr/bin/env python
"""3-epoch JEPA production validation on CUDA (no 5-seed / 150-epoch run).

Uses exact baseline config: 200 train traj, 20-traj val holdout, effective batch 256,
microbatch 16 / accum 16, Kp=15, LR=0.2, WD=4e-4, EMA=0.99.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ts_jepa.config import load_config, project_root
from ts_jepa.data.datasets import TrajectoryDataset, fit_command_normalizer
from ts_jepa.device import describe_device, select_device
from ts_jepa.losses.cosine import cosine_alignment_loss
from ts_jepa.models.ema import ema_update
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.runtime import configure_training_runtime, make_dataloader, save_checkpoint, state_dict_to_cpu
from ts_jepa.training.train_jepa import (
    _set_seed,
    _split_train_val,
    accumulation_plan,
    evaluate_cosine_loss,
    resolve_jepa_batching,
)

HEARTBEAT_INTERVAL_S = 30.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_finite_number(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def gpu_mem_str(device: torch.device) -> str:
    if device.type != "cuda":
        return "n/a"
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free_b, total_b = torch.cuda.mem_get_info()
    used = (total_b - free_b) / (1024**3)
    return f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB used={used:.2f}/{total_b/(1024**3):.2f}GB"


def peak_mem_gb(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0, "used_gb": 0.0, "total_gb": 0.0}
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024**3), 3),
        "used_gb": round((total_b - free_b) / (1024**3), 3),
        "total_gb": round(total_b / (1024**3), 3),
    }


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


class ProgressTracker:
    """Shared progress state + ≥30s heartbeat while a long stage is in flight."""

    def __init__(self, device: torch.device, accum_steps: int, total_effective_steps: int) -> None:
        self.device = device
        self.accum_steps = accum_steps
        self.total_effective_steps = total_effective_steps
        self.lock = threading.Lock()
        self.epoch = 0
        self.effective_step = 0  # completed steps
        self.micro_in_group = 0
        self.last_loss: float | None = None
        self.stage = "idle"
        self.epoch_t0 = time.perf_counter()
        self._last_emit = time.perf_counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat_loop, name="jepa-progress-hb", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def set_stage(self, stage: str) -> None:
        with self.lock:
            self.stage = stage

    def begin_epoch(self, epoch: int) -> None:
        with self.lock:
            self.epoch = epoch
            self.effective_step = 0
            self.micro_in_group = 0
            self.last_loss = None
            self.stage = "train_epoch_start"
            self.epoch_t0 = time.perf_counter()
            self._last_emit = time.perf_counter()

    def note_micro(self, micro_in_group: int, loss: float | None = None) -> None:
        with self.lock:
            self.micro_in_group = micro_in_group
            if loss is not None:
                self.last_loss = loss

    def note_effective_step(self, completed_steps: int, loss: float | None = None) -> None:
        with self.lock:
            self.effective_step = completed_steps
            self.micro_in_group = 0
            if loss is not None:
                self.last_loss = loss

    def _snapshot(self) -> dict:
        with self.lock:
            elapsed = time.perf_counter() - self.epoch_t0
            done = min(self.effective_step, self.total_effective_steps)
            total = max(1, self.total_effective_steps)
            eta = (elapsed / done) * (total - done) if done > 0 else float("nan")
            return {
                "epoch": self.epoch,
                "effective_step": done,
                "total_steps": self.total_effective_steps,
                "micro_in_group": self.micro_in_group,
                "accum_steps": self.accum_steps,
                "last_loss": self.last_loss,
                "elapsed": elapsed,
                "eta": eta,
                "stage": self.stage,
            }

    def emit(self, prefix: str = "progress") -> None:
        s = self._snapshot()
        loss_s = f"{s['last_loss']:.6f}" if s["last_loss"] is not None else "n/a"
        eta_s = f"{s['eta']:.1f}s" if math.isfinite(s["eta"]) else "n/a"
        log(
            f"{prefix} | epoch={s['epoch']} step={s['effective_step']}/{s['total_steps']} "
            f"micro={s['micro_in_group']}/{s['accum_steps']} loss={loss_s} "
            f"elapsed={s['elapsed']:.1f}s ETA={eta_s} stage={s['stage']} "
            f"GPU={gpu_mem_str(self.device)}"
        )
        with self.lock:
            self._last_emit = time.perf_counter()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(5.0):
            with self.lock:
                silent = time.perf_counter() - self._last_emit
                stage = self.stage
            if silent >= HEARTBEAT_INTERVAL_S and stage not in ("idle", "done"):
                self.emit(prefix="heartbeat")


def tensors_close(a: torch.Tensor, b: torch.Tensor, rtol: float = 0.0, atol: float = 0.0) -> bool:
    return torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol, equal_nan=False)


def state_dicts_equal(a: dict, b: dict) -> tuple[bool, str]:
    if a.keys() != b.keys():
        return False, f"key mismatch: {set(a.keys()) ^ set(b.keys())}"
    for k in a:
        ta, tb = a[k], b[k]
        if not torch.is_floating_point(ta):
            if not torch.equal(ta.cpu(), tb.cpu()):
                return False, f"non-float mismatch at {k}"
            continue
        if not tensors_close(ta, tb, rtol=0.0, atol=0.0):
            max_diff = (ta.cpu().float() - tb.cpu().float()).abs().max().item()
            return False, f"mismatch at {k} max_diff={max_diff}"
    return True, "ok"


def verify_ema_update(model: TSJEPA, decay: float = 0.99) -> dict:
    """One synthetic EMA step on a clone: target must become decay*old + (1-decay)*online."""
    probe = copy.deepcopy(model)
    ctx = probe.context_encoder
    tgt = probe.target_encoder
    name = None
    t_before = None
    for (n, o_p), (_, t_p) in zip(ctx.named_parameters(), tgt.named_parameters()):
        if o_p.numel() > 0 and o_p.is_floating_point():
            name = n
            t_before = t_p.detach().float().cpu().clone()
            break
    assert name is not None and t_before is not None

    with torch.no_grad():
        for o_p in ctx.parameters():
            if o_p.is_floating_point():
                o_p.add_(1e-3)
                break

    o_pert = None
    for n, o_p in ctx.named_parameters():
        if n == name:
            o_pert = o_p.detach().float().cpu().clone()
            break
    assert o_pert is not None

    expected = decay * t_before + (1.0 - decay) * o_pert
    ema_update(tgt, ctx, decay=decay)
    t_after = None
    for n, t_p in tgt.named_parameters():
        if n == name:
            t_after = t_p.detach().float().cpu().clone()
            break
    assert t_after is not None

    max_err = (t_after - expected).abs().max().item()
    return {
        "param": name,
        "max_abs_error_vs_formula": max_err,
        "formula_ok": max_err < 1e-5,
        "target_moved": not torch.allclose(t_after, t_before, atol=0.0, rtol=0.0),
    }


def train_one_epoch(
    model: TSJEPA,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    accum_steps: int,
    seed: int,
    epoch: int,
    progress: ProgressTracker | None = None,
) -> tuple[float, dict]:
    """
    Same training math as production JEPA; logging-only instrumentation.

    CUDA synchronize is used at microbatch / effective-step boundaries (and heartbeats),
    not after every tiny op — otherwise the GPU looks idle and throughput collapses.
    """
    model.context_encoder.train()
    model.predictor.train()
    model.target_encoder.eval()
    train_loss = 0.0
    n_steps = 0
    micro_in_group = 0
    micros_seen = 0
    group_loss_sum: torch.Tensor | None = None
    saw_nonfinite = False
    total_effective, n_micros_used, n_leftover = accumulation_plan(len(train_loader), accum_steps)
    if progress is None:
        progress = ProgressTracker(device, accum_steps, total_effective)
        owns_progress = True
    else:
        owns_progress = False
        # Keep tracker denominator aligned with this loader/accum plan.
        progress.total_effective_steps = total_effective
        progress.accum_steps = accum_steps
    progress.begin_epoch(epoch)
    if n_leftover:
        log(
            f"accumulation plan | epoch={epoch} micro_batches={len(train_loader)} "
            f"effective_steps={total_effective} used_micros={n_micros_used} "
            f"drop_leftover_micros={n_leftover}"
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loader_iter = iter(train_loader)
    step_t0 = time.perf_counter()
    while micros_seen < n_micros_used:
        next_step_idx = min(n_steps + 1, max(1, total_effective))
        next_micro_idx = micro_in_group + 1

        progress.set_stage("data_loading")
        log(
            f"stage | epoch={epoch} step={next_step_idx}/{total_effective} "
            f"micro={next_micro_idx}/{accum_steps} op=data_loading start "
            f"GPU={gpu_mem_str(device)}"
        )
        t_micro0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        micros_seen += 1

        context = batch["context"].to(device, non_blocking=True)
        future = batch["future_frames"].to(device, non_blocking=True)
        cmds = batch["teacher_commands_norm"].to(device, non_blocking=True)

        progress.set_stage("forward_context")
        log(
            f"stage | epoch={epoch} step={next_step_idx}/{total_effective} "
            f"micro={next_micro_idx}/{accum_steps} op=context_encoder start"
        )
        z = model.encode_context(context)

        progress.set_stage("forward_target")
        log(
            f"stage | epoch={epoch} step={next_step_idx}/{total_effective} "
            f"micro={next_micro_idx}/{accum_steps} op=target_encoder start"
        )
        with torch.no_grad():
            zt = model.encode_targets(future)

        progress.set_stage("forward_predictor")
        log(
            f"stage | epoch={epoch} step={next_step_idx}/{total_effective} "
            f"micro={next_micro_idx}/{accum_steps} op=predictor start"
        )
        zp = model.predict(z, cmds)
        loss = cosine_alignment_loss(zp, zt)

        progress.set_stage("backward")
        log(
            f"stage | epoch={epoch} step={next_step_idx}/{total_effective} "
            f"micro={next_micro_idx}/{accum_steps} op=backward start"
        )
        (loss / accum_steps).backward()

        # One sync per microbatch for meaningful wall time + loss readout.
        _cuda_sync(device)
        micro_s = time.perf_counter() - t_micro0
        loss_v = float(loss.detach().item())
        if not is_finite_number(loss_v):
            saw_nonfinite = True
        det = loss.detach()
        group_loss_sum = det if group_loss_sum is None else (group_loss_sum + det)
        micro_in_group += 1
        progress.note_micro(micro_in_group, loss_v)
        log(
            f"micro | epoch={epoch} step={next_step_idx}/{total_effective} "
            f"micro={micro_in_group}/{accum_steps} loss={loss_v:.6f} "
            f"micro_time={micro_s:.3f}s elapsed={time.perf_counter() - t0:.1f}s "
            f"GPU={gpu_mem_str(device)}"
        )

        if micro_in_group < accum_steps:
            continue

        progress.set_stage("optimizer")
        log(
            f"stage | epoch={epoch} step={next_step_idx}/{total_effective} "
            f"micro={accum_steps}/{accum_steps} op=optimizer.step start"
        )
        optimizer.step()
        progress.set_stage("ema")
        log(
            f"stage | epoch={epoch} step={next_step_idx}/{total_effective} "
            f"micro={accum_steps}/{accum_steps} op=ema_update start"
        )
        model.ema_step()
        optimizer.zero_grad(set_to_none=True)

        _cuda_sync(device)
        assert group_loss_sum is not None
        step_loss = float(group_loss_sum.item()) / accum_steps
        train_loss += step_loss
        n_steps += 1
        group_loss_sum = None
        micro_in_group = 0
        progress.note_effective_step(n_steps, step_loss)

        step_time = time.perf_counter() - step_t0
        elapsed = time.perf_counter() - t0
        eta = (elapsed / n_steps) * (total_effective - n_steps) if n_steps else 0.0
        log(
            f"epoch={epoch} step={n_steps}/{total_effective} micro={accum_steps}/{accum_steps} "
            f"loss={step_loss:.6f} step_time={step_time:.2f}s "
            f"elapsed={elapsed:.1f}s ETA={eta:.1f}s GPU={gpu_mem_str(device)}"
        )
        progress.emit(prefix="progress")
        step_t0 = time.perf_counter()

    if micro_in_group > 0:
        optimizer.zero_grad(set_to_none=True)
        group_loss_sum = None
        micro_in_group = 0
    if n_steps != total_effective or micros_seen != n_micros_used:
        raise RuntimeError(
            f"Effective step count mismatch: optimizer.step()={n_steps}, "
            f"planned={total_effective}, micros_seen={micros_seen}/{n_micros_used}, "
            f"leftover_plan={n_leftover}"
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    train_loss /= max(1, n_steps)
    mem = peak_mem_gb(device)
    progress.set_stage("done")
    if owns_progress:
        progress.stop()
    return train_loss, {
        "epoch": epoch,
        "seed": seed,
        "train_loss": train_loss,
        "num_effective_steps": n_steps,
        "planned_effective_steps": total_effective,
        "microbatches_used": micros_seen,
        "microbatches_leftover_dropped": n_leftover,
        "epoch_time_s": round(elapsed, 3),
        "nonfinite_train_loss": saw_nonfinite,
        **mem,
    }


def main() -> None:
    config = load_config()
    device = select_device()
    if device.type != "cuda":
        raise RuntimeError("This validation must run on CUDA")

    root = project_root(config)
    data_root = root / config["paths"]["data_root"]
    run_dir = root / "runs" / "eval" / "jepa_3epoch_validation"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Exact production settings (assert, do not silently alter).
    opt = config["ts_jepa"]["optimizer"]
    effective_bs, micro_bs, accum_steps = resolve_jepa_batching(opt)
    checks_cfg = {
        "train_trajectories": int(config["ts_jepa"]["dataset"]["train_trajectories"]),
        "val_trajectory_count": int(config["ts_jepa"]["early_stopping"]["validation_trajectory_count"]),
        "effective_batch_size": effective_bs,
        "microbatch_size": micro_bs,
        "accumulation_steps": accum_steps,
        "Kp": int(config["ts_jepa"]["prediction_horizon"]["Kp"]),
        "learning_rate": float(opt["learning_rate"]),
        "weight_decay": float(opt["weight_decay"]),
        "ema_decay": float(config["ts_jepa"]["target_encoder"]["ema_decay"]),
    }
    expected = {
        "train_trajectories": 200,
        "val_trajectory_count": 20,
        "effective_batch_size": 256,
        "microbatch_size": 16,
        "accumulation_steps": 16,
        "Kp": 15,
        "learning_rate": 0.2,
        "weight_decay": 0.0004,
        "ema_decay": 0.99,
    }
    for k, v in expected.items():
        if checks_cfg[k] != v:
            raise RuntimeError(f"Config mismatch for {k}: got {checks_cfg[k]}, expected {v}")

    train_dir = data_root / "trajectories" / "jepa" / "train"
    n_files = len(list(train_dir.glob("*.npz")))
    if n_files != 200:
        raise RuntimeError(f"Expected 200 JEPA train trajectories, found {n_files}")

    log(f"device={device} | {json.dumps(describe_device(device))}")
    log(f"production config OK: {json.dumps(checks_cfg)}")
    log(f"run_dir={run_dir}")

    _set_seed(0)
    configure_training_runtime(
        device,
        cudnn_benchmark=bool(config.get("runtime", {}).get("cudnn_benchmark", True)),
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    normalizer = fit_command_normalizer(config, data_root=data_root)
    dataset = TrajectoryDataset(train_dir, config, normalizer, training=True)
    train_set, val_set = _split_train_val(dataset, expected["val_trajectory_count"])
    log(f"samples: train={len(train_set)} val={len(val_set)} files={len(dataset.files)}")

    train_loader = make_dataloader(
        train_set,
        batch_size=micro_bs,
        shuffle=True,
        device=device,
        config=config,
        drop_last=True,
    )
    val_loader = make_dataloader(
        val_set,
        batch_size=min(micro_bs, max(1, len(val_set))),
        shuffle=False,
        device=device,
        config=config,
        drop_last=False,
    )
    n_effective, n_micros_used, n_leftover = accumulation_plan(len(train_loader), accum_steps)
    log(
        f"loader: micro_batches={len(train_loader)} effective_steps/epoch={n_effective} "
        f"used_micros={n_micros_used} drop_leftover_micros={n_leftover}"
    )

    model = TSJEPA(config).to(device)
    optimizer = torch.optim.SGD(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=float(opt["learning_rate"]),
        weight_decay=float(opt["weight_decay"]),
        momentum=0.0,
    )

    history = []
    epoch_reports = []
    best_val = float("inf")
    best_epoch = 0
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    ema_report: dict = {}
    progress = ProgressTracker(device, accum_steps, n_effective)

    try:
        # --- Epochs 1-2 (fresh) ---
        for epoch in (1, 2):
            log(f"=== epoch {epoch}/3 (fresh) ===")
            train_loss, ep = train_one_epoch(
                model,
                optimizer,
                train_loader,
                device,
                accum_steps,
                seed=0,
                epoch=epoch,
                progress=progress,
            )

            progress.set_stage("validation")
            progress.emit(prefix="progress")
            log(f"stage | epoch={epoch} op=validation start GPU={gpu_mem_str(device)}")
            _cuda_sync(device)
            t_val0 = time.perf_counter()
            val_loss = evaluate_cosine_loss(model, val_loader, device)
            _cuda_sync(device)
            val_s = time.perf_counter() - t_val0
            log(
                f"stage | epoch={epoch} op=validation done={val_s:.2f}s "
                f"val_loss={val_loss:.6f} GPU={gpu_mem_str(device)}"
            )

            ep["val_loss"] = val_loss
            ep["val_finite"] = is_finite_number(val_loss)
            ep["train_finite"] = is_finite_number(train_loss)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            epoch_reports.append(ep)
            log(
                f"epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f} "
                f"time={ep['epoch_time_s']:.1f}s peak_alloc={ep['peak_allocated_gb']}GB "
                f"peak_res={ep['peak_reserved_gb']}GB"
            )

            progress.set_stage("checkpointing")
            log(f"stage | epoch={epoch} op=checkpointing start GPU={gpu_mem_str(device)}")
            ckpt = {
                "model": state_dict_to_cpu(model.state_dict()),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "normalizer": normalizer.to_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "history": history,
                "seed": 0,
                "selection_split": "train_holdout_validation",
            }
            save_checkpoint(run_dir / "last.pt", ckpt)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                save_checkpoint(run_dir / "best.pt", ckpt)
                log(f"saved best.pt (epoch={epoch}, val={val_loss:.6f})")
            log(f"stage | epoch={epoch} op=checkpointing done GPU={gpu_mem_str(device)}")

        # --- Checkpoint save / load / resume ---
        assert last_path.exists() and best_path.exists()
        progress.set_stage("checkpointing")
        log("stage | op=checkpoint_load_resume start")
        loaded = torch.load(last_path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 2
        assert "model" in loaded and "optimizer" in loaded

        # Round-trip: fresh model loads last.pt exactly.
        model_rt = TSJEPA(config).to(device)
        model_rt.load_state_dict(loaded["model"])
        ok, detail = state_dicts_equal(model.state_dict(), model_rt.state_dict())
        if not ok:
            raise RuntimeError(f"checkpoint round-trip failed: {detail}")
        log(f"checkpoint round-trip OK ({detail})")

        # Resume epoch 3 from checkpoint into a NEW model/optimizer (simulates restart).
        log("=== epoch 3/3 (resumed from last.pt) ===")
        model2 = TSJEPA(config).to(device)
        model2.load_state_dict(loaded["model"])
        optimizer2 = torch.optim.SGD(
            list(model2.context_encoder.parameters()) + list(model2.predictor.parameters()),
            lr=float(opt["learning_rate"]),
            weight_decay=float(opt["weight_decay"]),
            momentum=0.0,
        )
        optimizer2.load_state_dict(loaded["optimizer"])
        log("stage | op=checkpoint_load_resume done")

        train_loss3, ep3 = train_one_epoch(
            model2,
            optimizer2,
            train_loader,
            device,
            accum_steps,
            seed=0,
            epoch=3,
            progress=progress,
        )

        progress.set_stage("validation")
        progress.emit(prefix="progress")
        log(f"stage | epoch=3 op=validation start GPU={gpu_mem_str(device)}")
        _cuda_sync(device)
        t_val0 = time.perf_counter()
        val_loss3 = evaluate_cosine_loss(model2, val_loader, device)
        _cuda_sync(device)
        val_s = time.perf_counter() - t_val0
        log(
            f"stage | epoch=3 op=validation done={val_s:.2f}s "
            f"val_loss={val_loss3:.6f} GPU={gpu_mem_str(device)}"
        )

        ep3["val_loss"] = val_loss3
        ep3["val_finite"] = is_finite_number(val_loss3)
        ep3["train_finite"] = is_finite_number(train_loss3)
        ep3["resumed_from"] = str(last_path)
        history.append({"epoch": 3, "train_loss": train_loss3, "val_loss": val_loss3})
        epoch_reports.append(ep3)
        log(
            f"epoch 3 (resume): train={train_loss3:.6f} val={val_loss3:.6f} "
            f"time={ep3['epoch_time_s']:.1f}s peak_alloc={ep3['peak_allocated_gb']}GB "
            f"peak_res={ep3['peak_reserved_gb']}GB"
        )

        progress.set_stage("checkpointing")
        log("stage | epoch=3 op=checkpointing start")
        ckpt3 = {
            "model": state_dict_to_cpu(model2.state_dict()),
            "optimizer": optimizer2.state_dict(),
            "config": config,
            "normalizer": normalizer.to_dict(),
            "epoch": 3,
            "val_loss": val_loss3,
            "train_loss": train_loss3,
            "history": history,
            "seed": 0,
            "selection_split": "train_holdout_validation",
            "resumed_from_epoch": 2,
        }
        save_checkpoint(run_dir / "last.pt", ckpt3)
        if val_loss3 < best_val:
            best_val = val_loss3
            best_epoch = 3
            save_checkpoint(run_dir / "best.pt", ckpt3)
            log(f"saved best.pt (epoch=3, val={val_loss3:.6f})")
        log("stage | epoch=3 op=checkpointing done")

        # EMA formula check on the trained resumed model.
        progress.set_stage("ema_verify")
        ema_report = verify_ema_update(model2, decay=float(checks_cfg["ema_decay"]))
        # Confirm after normal training, target != context on at least one weight.
        diverged = False
        for (n, o_p), (_, t_p) in zip(
            model2.context_encoder.named_parameters(), model2.target_encoder.named_parameters()
        ):
            if o_p.is_floating_point() and not torch.allclose(o_p.detach().cpu(), t_p.detach().cpu()):
                diverged = True
                break
        ema_report["context_target_diverged_after_training"] = diverged
        log(f"EMA check: {json.dumps(ema_report)}")
    finally:
        progress.set_stage("idle")
        progress.stop()

    # Aggregate verifications
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    peaks_alloc = [e["peak_allocated_gb"] for e in epoch_reports]
    peaks_res = [e["peak_reserved_gb"] for e in epoch_reports]
    times = [e["epoch_time_s"] for e in epoch_reports]

    train_decreases = all(train_losses[i] > train_losses[i + 1] for i in range(len(train_losses) - 1))
    # Allow non-strict if first two decrease OR overall epoch1 > epoch3 (monotonic preferred).
    train_trend_ok = train_decreases or (train_losses[0] > train_losses[-1])
    all_finite = all(is_finite_number(x) for x in train_losses + val_losses)
    no_nan_flags = not any(e["nonfinite_train_loss"] for e in epoch_reports)
    # Memory stable: peak reserved does not climb >25% after epoch 1.
    mem_stable = max(peaks_res) <= peaks_res[0] * 1.25 + 0.5
    epoch_time_ok = all(t < 30 * 60 for t in times)  # < 30 min/epoch
    mean_epoch_min = (sum(times) / len(times)) / 60.0

    verifications = {
        "1_train_loss_decreases": {
            "pass": train_trend_ok,
            "train_losses": train_losses,
            "strictly_monotonic": train_decreases,
        },
        "2_validation_loss_finite": {
            "pass": all(is_finite_number(v) for v in val_losses),
            "val_losses": val_losses,
        },
        "3_no_nan_inf": {"pass": all_finite and no_nan_flags},
        "4_gpu_memory_stable": {
            "pass": mem_stable,
            "peak_allocated_gb_per_epoch": peaks_alloc,
            "peak_reserved_gb_per_epoch": peaks_res,
        },
        "5_checkpoints_save": {
            "pass": best_path.exists() and last_path.exists(),
            "best_pt": str(best_path),
            "last_pt": str(last_path),
            "best_epoch": best_epoch,
            "best_val": best_val,
        },
        "6_resume_checkpoint": {
            "pass": True,  # reached epoch 3 via load; round-trip already asserted
            "detail": "loaded last.pt after epoch 2 into fresh model+optimizer; completed epoch 3",
            "round_trip_ok": True,
        },
        "7_ema_updates": {
            "pass": bool(ema_report["formula_ok"])
            and bool(ema_report["target_moved"])
            and bool(ema_report["context_target_diverged_after_training"]),
            **ema_report,
        },
        "8_epoch_time_reasonable": {
            "pass": epoch_time_ok,
            "epoch_times_s": times,
            "mean_epoch_min": round(mean_epoch_min, 2),
            "threshold_min": 30.0,
        },
    }

    all_pass = all(v["pass"] for v in verifications.values())
    report = {
        "status": "PASS" if all_pass else "FAIL",
        "device": describe_device(device),
        "config_verified": checks_cfg,
        "history": history,
        "epoch_reports": epoch_reports,
        "verifications": verifications,
        "run_dir": str(run_dir),
    }
    out = run_dir / "validation_report.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"history": history, "best_val": best_val, "best_epoch": best_epoch}, handle, indent=2)

    log("=== VERIFICATION SUMMARY ===")
    for name, v in verifications.items():
        log(f"{'PASS' if v['pass'] else 'FAIL'}: {name}")
    log(f"overall={report['status']} | wrote {out}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

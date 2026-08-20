"""Laptop / Windows training runtime helpers (IMPLEMENTATION CHOICE — not paper)."""

from __future__ import annotations

import copy
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger("ts_jepa.runtime")


class DataLoaderStallError(RuntimeError):
    """Raised when a DataLoader/worker fetch exceeds the configured timeout."""


class TrainingStallError(RuntimeError):
    """Raised when the training loop makes no progress for too long."""


def gpu_total_memory_gb(device: torch.device) -> float | None:
    """Installed VRAM in GiB, or None when CUDA is not usable."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    idx = int(device.index) if device.index is not None else 0
    props = torch.cuda.get_device_properties(idx)
    return float(props.total_memory) / float(1024**3)


def gpu_compute_capability(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    idx = int(device.index) if device.index is not None else 0
    props = torch.cuda.get_device_properties(idx)
    return int(props.major), int(props.minor)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "off", "none", "no"}
    return bool(value)


def recommended_jepa_microbatch(effective: int, device: torch.device) -> int:
    """
    Largest divisor of the paper effective batch that the installed GPU can hold.

    Effective batch (Table II: 256) is unchanged; this only drops gradient
    accumulation on large cards (RTX 6000 Ada 48 GB → microbatch 256).
    """
    effective = int(effective)
    if effective < 1:
        raise ValueError(f"batch_size must be >= 1, got {effective}")
    total_gb = gpu_total_memory_gb(device)
    if total_gb is None:
        target = min(16, effective)
    elif total_gb >= 36:
        target = effective  # 40–48 GB Ada / A6000-class: one full paper batch
    elif total_gb >= 22:
        target = min(128, effective)
    elif total_gb >= 14:
        target = min(64, effective)
    elif total_gb >= 10:
        target = min(32, effective)
    else:
        target = min(16, effective)
    chosen = 1
    for size in range(1, effective + 1):
        if effective % size == 0 and size <= target:
            chosen = size
    return chosen


def resolve_amp_dtype(device: torch.device, runtime: Mapping[str, Any] | None = None) -> torch.dtype | None:
    """
    Mixed precision for JEPA/actor forwards.

    ``auto``: bfloat16 on Ampere/Ada/Hopper (sm>=8), else fp32.
    Paper math is fp32; AMP is an IC throughput toggle (runtime.amp).
    """
    spec = (runtime or {}).get("amp", "auto")
    if spec is None or spec is False:
        return None
    if isinstance(spec, str) and spec.strip().lower() in {"", "off", "fp32", "none", "false", "no"}:
        return None
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    key = spec if not isinstance(spec, str) else spec.strip().lower()
    if key in {True, "auto", "on", "true"}:
        cc = gpu_compute_capability(device)
        if cc is not None and cc[0] >= 8:
            return torch.bfloat16
        return None
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"runtime.amp must be auto|bf16|fp16|off, got {spec!r}")


def channels_last_enabled(device: torch.device, runtime: Mapping[str, Any] | None = None) -> bool:
    flag = (runtime or {}).get("channels_last", True)
    return device.type == "cuda" and torch.cuda.is_available() and _truthy(flag)


def should_release_cuda_cache(device: torch.device, runtime: Mapping[str, Any] | None = None) -> bool:
    """empty_cache each epoch only on small WDDM cards; it stalls large GPUs."""
    flag = (runtime or {}).get("release_cuda_cache_each_epoch", "auto")
    if isinstance(flag, str) and flag.strip().lower() == "auto":
        total = gpu_total_memory_gb(device)
        return total is not None and total < 12.0
    if flag is None:
        return False
    return _truthy(flag)


def resolve_encoder_batch_size(device: torch.device, runtime: Mapping[str, Any] | None = None) -> int:
    raw = (runtime or {}).get("encoder_batch_size", "auto")
    if raw not in {"auto", None} and not (isinstance(raw, str) and raw.strip().lower() == "auto"):
        return max(1, int(raw))
    if device.type != "cuda":
        return 64
    total = gpu_total_memory_gb(device) or 8.0
    if total >= 36:
        return 2048
    if total >= 16:
        return 1024
    return 256


def configure_training_runtime(
    device: torch.device,
    *,
    cudnn_benchmark: bool = True,
    allow_tf32: bool = True,
    cpu_threads: int | None = None,
) -> dict[str, Any]:
    """
    Tune PyTorch for CUDA training without changing paper hyperparameters.

    Keeps the GPU fed (TF32, cuDNN autotune, NHWC-ready alloc) and leaves
    host cores for DataLoader workers.
    """
    info: dict[str, Any] = {"device": str(device)}
    ncpu = os.cpu_count() or 8
    if cpu_threads is None:
        cpu_threads = max(4, min(16, max(4, ncpu // 2)))
    torch.set_num_threads(int(cpu_threads))
    info["torch_num_threads"] = int(cpu_threads)

    if device.type != "cuda" or not torch.cuda.is_available():
        return info

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    info["cudnn_benchmark"] = bool(cudnn_benchmark)
    cc = gpu_compute_capability(device)
    info["gpu_name"] = torch.cuda.get_device_name(device)
    info["compute_capability"] = None if cc is None else f"{cc[0]}.{cc[1]}"
    info["gpu_memory_total_gb"] = gpu_total_memory_gb(device)
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    info["allow_tf32"] = bool(allow_tf32)
    # Do not set_per_process_memory_fraction: on 8 GB WDDM the caching allocator
    # then keeps almost the whole card reserved while live alloc stays small.
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            (alloc_conf + "," if alloc_conf else "") + "expandable_segments:True"
        )
    return info


def resolve_num_workers(config: dict[str, Any] | None, device: torch.device) -> int:
    """CUDA training may use workers; CPU/tests always stay single-process."""
    if device.type != "cuda":
        return 0
    runtime = (config or {}).get("runtime", {})
    raw = runtime.get("num_workers", "auto")
    if raw in {"auto", None} or (isinstance(raw, str) and raw.strip().lower() == "auto"):
        ncpu = os.cpu_count() or 8
        cap = 8 if sys.platform == "win32" else 12
        return max(2, min(cap, ncpu - 2))
    return max(0, int(raw))


def is_dataloader_spawn_error(exc: BaseException) -> bool:
    """True when Windows spawn failed while pickling a DataLoader worker payload."""
    cur: BaseException | None = exc
    seen = 0
    needles = (
        "invalid argument",
        "pickle data was truncated",
        "unpicklingerror",
        "too many open files",
        "cannot pickle",
    )
    while cur is not None and seen < 8:
        errno = getattr(cur, "errno", None)
        if isinstance(cur, OSError) and errno in {12, 22, 24}:
            return True
        text = f"{type(cur).__name__}: {cur}".lower()
        if any(needle in text for needle in needles):
            return True
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return False


def inprocess_dataloader(loader: DataLoader) -> DataLoader:
    """Rebuild a loader that never spawns workers (Windows-safe eval)."""
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=bool(loader.drop_last),
    )


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    config: dict[str, Any] | None = None,
    drop_last: bool = False,
    num_workers: int | None = None,
) -> DataLoader:
    """DataLoader tuned for overlapping CPU preprocess with GPU compute."""
    runtime = (config or {}).get("runtime", {})
    workers = resolve_num_workers(config, device) if num_workers is None else max(0, int(num_workers))
    pin_memory = bool(runtime.get("pin_memory", device.type == "cuda"))
    prefetch = int(runtime.get("prefetch_factor", 4))
    # Avoid infinite hangs when a Windows spawn worker dies/deadlocks mid-epoch.
    timeout_s = float(runtime.get("dataloader_timeout_s", 120.0))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "drop_last": drop_last,
        "pin_memory": pin_memory and device.type == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(runtime.get("persistent_workers", True))
        kwargs["prefetch_factor"] = max(2, prefetch)
        kwargs["worker_init_fn"] = _worker_init_fn
        if timeout_s > 0:
            kwargs["timeout"] = timeout_s
    return DataLoader(**kwargs)


def _worker_init_fn(worker_id: int) -> None:
    import numpy as np

    # Distinct RNG streams per worker; trainers still seed the main process.
    base = torch.initial_seed() % (2**31 - 1)
    np.random.seed(base + worker_id)


def batch_to_device(batch: Mapping[str, Any], device: torch.device, non_blocking: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    use_non_blocking = bool(non_blocking and device.type == "cuda")
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=use_non_blocking)
        else:
            out[key] = value
    return out


def gpu_mem_str(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "n/a"
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        return (
            f"alloc={alloc / 1e9:.2f}GB "
            f"reserved={reserved / 1e9:.2f}GB "
            f"free={free_b / 1e9:.2f}/{total_b / 1e9:.2f}GB"
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"err={type(exc).__name__}"


def release_cuda_cache(device: torch.device) -> None:
    """Drop unused caching-allocator blocks back to the driver (WDDM)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    import gc

    gc.collect()
    torch.cuda.empty_cache()


_PREFETCH_STREAMS: dict[int, torch.cuda.Stream] = {}


def _shared_prefetch_stream(device: torch.device) -> torch.cuda.Stream:
    idx = int(device.index) if device.index is not None else int(torch.cuda.current_device())
    stream = _PREFETCH_STREAMS.get(idx)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _PREFETCH_STREAMS[idx] = stream
    return stream


class CUDAPrefetcher:
    """Overlap H2D copies with the previous GPU step (CUDA only)."""

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self._use_cuda = device.type == "cuda" and torch.cuda.is_available()
        self.batches_yielded = 0
        self.last_fetch_s: float | None = None
        self.last_error: BaseException | None = None
        self._it: Iterator[Any] | None = None
        self._next: dict[str, Any] | None = None
        self._stream = _shared_prefetch_stream(device) if self._use_cuda else None

    def __iter__(self) -> CUDAPrefetcher:
        self.close()
        try:
            self._it = iter(self.loader)
        except Exception as exc:
            self.last_error = exc
            raise DataLoaderStallError(
                f"DataLoader iterator failed to start after {self.batches_yielded} batches "
                f"({type(exc).__name__}: {exc}). "
                "On Windows, spawning val/test workers while train persistent_workers still "
                "hold TrajectoryDataset RGB arrays often raises OSError 22 / truncated pickle. "
                "Use num_workers=0 for eval loaders."
            ) from exc
        if self._use_cuda:
            self._preload()
        return self

    def __next__(self) -> dict[str, Any]:
        if self._it is None:
            raise StopIteration
        if not self._use_cuda:
            raw = next(self._it)
            self.batches_yielded += 1
            return batch_to_device(raw, self.device, non_blocking=False)
        if self._next is None:
            self.close()
            raise StopIteration
        try:
            torch.cuda.current_stream().wait_stream(self._stream)
        except Exception as exc:
            self.last_error = exc
            raise RuntimeError(
                f"CUDA stream sync failed after {self.batches_yielded} batches "
                f"({type(exc).__name__}: {exc}); GPU={gpu_mem_str(self.device)}"
            ) from exc
        current = self._next
        self._preload()
        self.batches_yielded += 1
        return current

    def _preload(self) -> None:
        assert self._it is not None
        t0 = time.perf_counter()
        try:
            raw = next(self._it)
        except StopIteration:
            self._next = None
            return
        except Exception as exc:
            self.last_error = exc
            raise DataLoaderStallError(
                f"DataLoader fetch failed after {self.batches_yielded} batches "
                f"({type(exc).__name__}: {exc}). "
                "On Windows this is often a dead worker or pin_memory hang; "
                "retry with runtime.num_workers=0 or lower prefetch_factor."
            ) from exc
        try:
            with torch.cuda.stream(self._stream):
                self._next = batch_to_device(raw, self.device, non_blocking=True)
        except Exception as exc:
            self.last_error = exc
            raise RuntimeError(
                f"H2D prefetch failed after {self.batches_yielded} batches "
                f"({type(exc).__name__}: {exc}); GPU={gpu_mem_str(self.device)}"
            ) from exc
        self.last_fetch_s = time.perf_counter() - t0

    def close(self) -> None:
        """Drop the preloaded GPU batch and DataLoader iterator (safe if stopped early)."""
        self._next = None
        self._it = None


class TrainProgressWatchdog:
    """
    Heartbeat + stall detector for long training loops.

    A daemon thread logs progress periodically. If `touch()` is not called for
    `stall_timeout_s`, the next `touch()` / `check()` raises TrainingStallError
    (and heartbeats print STALL so silent hangs are visible even while blocked).
    """

    def __init__(
        self,
        *,
        device: torch.device,
        name: str,
        log_path: Path | None = None,
        heartbeat_s: float = 30.0,
        stall_timeout_s: float = 180.0,
    ) -> None:
        self.device = device
        self.name = name
        self.heartbeat_s = float(heartbeat_s)
        self.stall_timeout_s = float(stall_timeout_s)
        self.lock = threading.Lock()
        self.epoch = 0
        self.micro = 0
        self.total_micro = 0
        self.effective_step = 0
        self.stage = "init"
        self.last_loss: float | None = None
        self.last_touch = time.perf_counter()
        self._last_heartbeat = 0.0
        self._stall_announced = False
        self._stop = threading.Event()
        self._log_path = Path(log_path) if log_path is not None else None
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._loop,
            name=f"{name}-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {self.name} | {msg}"
        print(line, flush=True)
        if self._log_path is not None:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    def begin_epoch(self, epoch: int, total_micro: int) -> None:
        with self.lock:
            self.epoch = epoch
            self.micro = 0
            self.total_micro = total_micro
            self.effective_step = 0
            self.stage = "train"
            self.last_loss = None
            self.last_touch = time.perf_counter()
            self._stall_announced = False
        self.log(f"epoch={epoch} start total_micro={total_micro} GPU={gpu_mem_str(self.device)}")

    def set_stage(self, stage: str) -> None:
        with self.lock:
            self.stage = stage
            self.last_touch = time.perf_counter()
            self._stall_announced = False

    def touch(
        self,
        *,
        micro: int | None = None,
        effective_step: int | None = None,
        loss: float | None = None,
        stage: str | None = None,
    ) -> None:
        with self.lock:
            if micro is not None:
                self.micro = micro
            if effective_step is not None:
                self.effective_step = effective_step
            if loss is not None:
                self.last_loss = loss
            if stage is not None:
                self.stage = stage
            self.last_touch = time.perf_counter()
            self._stall_announced = False

    def _snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "epoch": self.epoch,
                "micro": self.micro,
                "total_micro": self.total_micro,
                "effective_step": self.effective_step,
                "stage": self.stage,
                "last_loss": self.last_loss,
                "silent_s": time.perf_counter() - self.last_touch,
            }

    def _loop(self) -> None:
        while not self._stop.wait(min(5.0, max(1.0, self.heartbeat_s / 2.0))):
            snap = self._snapshot()
            if snap["stage"] in ("init", "done"):
                continue
            loss_s = f"{snap['last_loss']:.6f}" if snap["last_loss"] is not None else "n/a"
            now = time.perf_counter()
            if snap["silent_s"] >= self.stall_timeout_s:
                with self.lock:
                    announce = not self._stall_announced
                    self._stall_announced = True
                if announce or (now - self._last_heartbeat) >= self.heartbeat_s:
                    self._last_heartbeat = now
                    self.log(
                        f"STALL | no progress for {snap['silent_s']:.1f}s "
                        f"(DataLoader timeout should raise soon) epoch={snap['epoch']} "
                        f"micro={snap['micro']}/{snap['total_micro']} "
                        f"step={snap['effective_step']} stage={snap['stage']} "
                        f"loss={loss_s} GPU={gpu_mem_str(self.device)}"
                    )
                continue
            if snap["silent_s"] >= self.heartbeat_s and (now - self._last_heartbeat) >= self.heartbeat_s:
                self._last_heartbeat = now
                self.log(
                    f"heartbeat | epoch={snap['epoch']} "
                    f"micro={snap['micro']}/{snap['total_micro']} "
                    f"step={snap['effective_step']} stage={snap['stage']} "
                    f"loss={loss_s} silent={snap['silent_s']:.1f}s "
                    f"GPU={gpu_mem_str(self.device)}"
                )


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def reraise_cuda_context(exc: BaseException, *, where: str, device: torch.device) -> None:
    """Attach GPU memory + location context to CUDA / training failures, then re-raise."""
    detail = (
        f"{where} failed: {type(exc).__name__}: {exc}; "
        f"GPU={gpu_mem_str(device)}"
    )
    if isinstance(exc, (DataLoaderStallError, TrainingStallError)):
        raise
    msg = str(exc).lower()
    cudaish = "cuda" in type(exc).__name__.lower() or "cuda" in msg or "cublas" in msg or "cudnn" in msg
    if cudaish:
        raise RuntimeError(detail) from exc
    # Non-CUDA errors: keep original type but make the site obvious in logs.
    LOGGER.error(detail)
    raise


def state_dict_to_cpu(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().contiguous() for k, v in state.items()}


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    """Load a training or inference checkpoint from disk."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def save_checkpoint(path: Path | str, payload: dict[str, Any]) -> None:
    """
    Windows/WDDM-safe checkpoint write.

    Synchronize CUDA and move tensors to CPU before torch.save to avoid
    intermittent access-violation crashes seen on laptop NVIDIA GPUs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(
                f"cuda.synchronize before checkpoint failed ({type(exc).__name__}: {exc}); "
                f"path={path}"
            ) from exc

    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"model", "actor"} and isinstance(value, dict):
            # Heuristic: tensor-valued state_dicts.
            if value and all(torch.is_tensor(v) for v in value.values()):
                safe[key] = state_dict_to_cpu(value)
            else:
                safe[key] = value
        elif key == "optimizer" and isinstance(value, dict):
            safe[key] = _optimizer_state_to_cpu(value)
        elif torch.is_tensor(value):
            safe[key] = value.detach().cpu()
        else:
            safe[key] = value
    try:
        torch.save(safe, path)
    except Exception as exc:
        raise RuntimeError(f"torch.save failed for {path} ({type(exc).__name__}: {exc})") from exc


def _optimizer_state_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    """CPU-copy optimizer.state_dict() tensors (WDDM-safe) without requiring a live Optimizer."""
    out: dict[str, Any] = {"state": {}, "param_groups": copy.deepcopy(state.get("param_groups", []))}
    for pid, entry in state.get("state", {}).items():
        if not isinstance(entry, dict):
            out["state"][pid] = entry
            continue
        cpu_entry: dict[str, Any] = {}
        for k, v in entry.items():
            if torch.is_tensor(v):
                cpu_entry[k] = v.detach().cpu().contiguous()
            else:
                cpu_entry[k] = copy.deepcopy(v)
        out["state"][pid] = cpu_entry
    return out


def configure_train_logging(level: int = logging.INFO) -> None:
    """Ensure runtime/trainer logs are visible even under PowerShell Tee-Object."""
    root = logging.getLogger("ts_jepa")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(level)

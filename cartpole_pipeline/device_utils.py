"""CUDA / device helpers for training on NVIDIA GPUs."""

from __future__ import annotations

import os

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """
    Resolve the training device.

    ``auto`` prefers the first available CUDA GPU (e.g. RTX 5070 laptop GPU).
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{requested}' but CUDA is not available. "
            "Install a CUDA-enabled PyTorch build and NVIDIA drivers."
        )
    return device


def configure_cuda(device: torch.device) -> None:
    """Enable common NVIDIA performance settings for laptop/desktop training."""
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def describe_device(device: torch.device) -> str:
    """Return a human-readable description of the active compute device."""
    if device.type != "cuda":
        return f"{device} (CPU)"

    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    memory_gb = props.total_memory / (1024**3)
    return f"cuda:{index} ({props.name}, {memory_gb:.1f} GB VRAM)"


def dataloader_kwargs(device: torch.device, num_workers: int | None = None) -> dict:
    """DataLoader settings tuned for GPU training."""
    if num_workers is None:
        num_workers = 4 if device.type == "cuda" else 0

    kwargs: dict = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return kwargs


def set_dataloader_env() -> None:
    """Avoid oversubscribing CPU threads when the GPU does the heavy work."""
    if torch.cuda.is_available():
        os.environ.setdefault("OMP_NUM_THREADS", "4")

from __future__ import annotations

import torch


def select_device(requested: str | torch.device | None = None) -> torch.device:
    """
    Resolve the compute device.

    - If `requested` is set, use it.
    - Otherwise prefer CUDA when available, else CPU.
    """
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def describe_device(device: torch.device | None = None) -> dict[str, object]:
    device = select_device(device)
    info: dict[str, object] = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_index"] = idx
        free_b, total_b = torch.cuda.mem_get_info(idx)
        info["gpu_memory_total_gb"] = round(total_b / (1024**3), 3)
        info["gpu_memory_free_gb"] = round(free_b / (1024**3), 3)
        info["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated(idx) / (1024**3), 3)
        info["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved(idx) / (1024**3), 3)
    else:
        info["gpu_name"] = None
    return info

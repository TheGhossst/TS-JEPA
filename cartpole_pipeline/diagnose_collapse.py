"""Diagnostic script for detecting representation collapse in ContextEncoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cartpole_pipeline.device_utils import describe_device, resolve_device
from cartpole_pipeline.models import ContextEncoder
from cartpole_pipeline.transforms import TSJEPATransform


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert an HWC uint8 frame to a batched CHW float tensor in [0, 1]."""
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def load_encoder(checkpoint_path: Path | None, device: torch.device) -> ContextEncoder:
    encoder = ContextEncoder().to(device)
    encoder.eval()

    if checkpoint_path is None:
        print("Using untrained ContextEncoder.")
        return encoder

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "context_encoder" in checkpoint:
        state_dict = checkpoint["context_encoder"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    encoder.load_state_dict(state_dict)
    print(f"Loaded ContextEncoder weights from {checkpoint_path}")
    return encoder


def diagnose(
    data_path: Path,
    trajectory_idx: int = 0,
    frame_a: int = 0,
    frame_b: int = 50,
    checkpoint_path: Path | None = None,
    device: str = "auto",
) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    with np.load(data_path) as archive:
        frames = archive["frames"]

    if trajectory_idx >= frames.shape[0]:
        raise ValueError(f"trajectory_idx {trajectory_idx} out of range (max {frames.shape[0] - 1})")
    if frame_a >= frames.shape[1] or frame_b >= frames.shape[1]:
        raise ValueError(f"frame indices must be < episode length ({frames.shape[1]})")

    torch_device = resolve_device(device)
    encoder = load_encoder(checkpoint_path, torch_device)
    transform = TSJEPATransform().to(torch_device)
    transform.eval()

    frame_a_np = frames[trajectory_idx, frame_a]
    frame_b_np = frames[trajectory_idx, frame_b]

    batch = torch.cat(
        [frame_to_tensor(frame_a_np), frame_to_tensor(frame_b_np)],
        dim=0,
    ).to(torch_device)

    with torch.no_grad():
        normalized_batch = transform(batch)
        embeddings = encoder(normalized_batch)

    embedding_a = embeddings[0].cpu()
    embedding_b = embeddings[1].cpu()
    cosine_sim = F.cosine_similarity(embedding_a.unsqueeze(0), embedding_b.unsqueeze(0)).item()

    print(f"Device: {describe_device(torch_device)}")
    print(f"Trajectory: {trajectory_idx} | frames: {frame_a} vs {frame_b}")
    print()
    print(f"Embedding A (frame {frame_a}):")
    print(embedding_a.numpy())
    print()
    print(f"Embedding B (frame {frame_b}):")
    print(embedding_b.numpy())
    print()
    print(f"Cosine similarity: {cosine_sim:.6f}")

    if abs(cosine_sim - 1.0) < 1e-4 or torch.allclose(embedding_a, embedding_b, atol=1e-5):
        print("WARNING: Embeddings appear collapsed (near-identical outputs).")
    elif cosine_sim > 0.99:
        print("WARNING: Cosine similarity is very high; encoder may be partially collapsed.")
    else:
        print("Embeddings differ; no obvious collapse detected.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether ContextEncoder produces distinct embeddings for different frames."
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/train_data.npz"))
    parser.add_argument("--trajectory-idx", type=int, default=0)
    parser.add_argument("--frame-a", type=int, default=0)
    parser.add_argument("--frame-b", type=int, default=50)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint with ContextEncoder weights for trained-model diagnostics.",
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    diagnose(
        data_path=args.data_path,
        trajectory_idx=args.trajectory_idx,
        frame_a=args.frame_a,
        frame_b=args.frame_b,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()

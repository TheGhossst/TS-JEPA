"""TS-JEPA pretraining loop."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cartpole_pipeline.data.sequence_dataset import TSJEPADataset
from cartpole_pipeline.device_utils import (
    configure_cuda,
    dataloader_kwargs,
    describe_device,
    resolve_device,
    set_dataloader_env,
)
from cartpole_pipeline.models import ContextEncoder, Predictor


def update_ema(
    target_model: torch.nn.Module,
    context_model: torch.nn.Module,
    eta: float = 0.99,
) -> None:
    """
    Exponential moving average update for the target encoder.

    Parameters are EMA'd. BatchNorm running statistics (``running_mean`` /
    ``running_var``) are copied from the context encoder so the target branch
    does not freeze at the initial deepcopy buffers.
    """
    target_params = dict(target_model.named_parameters())
    for name, context_param in context_model.named_parameters():
        target_params[name].data.mul_(eta).add_(context_param.data, alpha=1.0 - eta)

    target_buffers = dict(target_model.named_buffers())
    for name, context_buffer in context_model.named_buffers():
        if "running" in name and name in target_buffers:
            target_buffers[name].data.copy_(context_buffer.data)


def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style variance hinge: push per-dimension std toward >= 1."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.relu(1.0 - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg-style covariance: penalize off-diagonal correlations."""
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (z.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / z.shape[1]


def train_one_epoch(
    context_encoder: ContextEncoder,
    target_encoder: ContextEncoder,
    predictor: Predictor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema_decay: float = 0.99,
    log_every: int = 10,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run one training epoch and return the average loss."""
    context_encoder.train()
    predictor.train()
    target_encoder.eval()

    total_loss = 0.0
    num_batches = 0
    amp_enabled = use_amp and device.type == "cuda"
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for batch_idx, (current_frames, future_frames, future_actions) in enumerate(dataloader):
        current_frames = current_frames.to(device, non_blocking=True)
        future_frames = future_frames.to(device, non_blocking=True)
        future_actions = future_actions.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(autocast_device, enabled=amp_enabled):
            z_context = context_encoder(current_frames)
            z_predicted = predictor(z_context, future_actions)

            batch_size, kp = future_frames.shape[:2]
            future_flat = future_frames.reshape(batch_size * kp, *future_frames.shape[2:])
            with torch.no_grad():
                z_target_flat = target_encoder(future_flat)
            z_target = z_target_flat.view(batch_size, kp, -1)

            cosine_loss = 1.0 - F.cosine_similarity(z_predicted, z_target, dim=-1).mean()

            # Regularize the encoder's own embedding, not the predictor's output.
            # Target embeddings are detached (EMA branch); variance on z_context is
            # what actually prevents collapse of the trainable encoder.
            var_loss = variance_loss(z_context)
            cov_loss = covariance_loss(z_context)
            loss = cosine_loss + 0.5 * var_loss + 0.1 * cov_loss

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        update_ema(target_encoder, context_encoder, eta=ema_decay)

        batch_loss = loss.item()
        batch_cosine_loss = cosine_loss.item()
        batch_var_loss = var_loss.item()
        batch_cov_loss = cov_loss.item()
        action_std = future_actions.std().item()
        context_std = z_context.detach().std(dim=0).mean().item()
        target_std = z_target.detach().std(dim=0).mean().item()
        total_loss += batch_loss
        num_batches += 1

        if (batch_idx + 1) % log_every == 0:
            print(
                f"Batch {batch_idx + 1}/{len(dataloader)} | "
                f"loss: {batch_loss:.6f} | cosine_loss: {batch_cosine_loss:.6f} | "
                f"var_loss: {batch_var_loss:.6f} | cov_loss: {batch_cov_loss:.6f} | "
                f"action_std: {action_std:.6f} | "
                f"z_ctx_std: {context_std:.4f} | z_tgt_std: {target_std:.4f}"
            )

    return total_loss / max(num_batches, 1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TS-JEPA on CartPole trajectories.")
    parser.add_argument("--data-path", type=Path, default=Path("data/train_data.npz"))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Default 64 is a good starting point for RTX 5070 laptop GPUs.",
    )
    parser.add_argument("--kp", type=int, default=5, help="Prediction horizon in steps.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: 4 on CUDA, 0 on CPU).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mixed-precision training on CUDA (recommended for RTX 5070).",
    )
    return parser


def main() -> None:
    set_dataloader_env()
    args = build_arg_parser().parse_args()

    device = resolve_device(args.device)
    configure_cuda(device)

    if device.type == "cuda" and not args.amp:
        print("Note: --no-amp disables mixed precision; AMP is faster on NVIDIA GPUs.")

    dataset = TSJEPADataset(data_path=args.data_path, kp=args.kp, training=True)
    loader_kwargs = dataloader_kwargs(device, num_workers=args.num_workers)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    context_encoder = ContextEncoder().to(device)
    target_encoder = copy.deepcopy(context_encoder)
    target_encoder.eval()
    for param in target_encoder.parameters():
        param.requires_grad = False

    predictor = Predictor(action_horizon=args.kp).to(device)

    optimizer = torch.optim.Adam(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Device: {describe_device(device)}")
    print(
        f"Training TS-JEPA for 1 epoch | "
        f"samples={len(dataset)} | batch_size={args.batch_size} | Kp={args.kp} | "
        f"amp={use_amp} | workers={loader_kwargs['num_workers']}"
    )

    avg_loss = train_one_epoch(
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        ema_decay=args.ema_decay,
        use_amp=use_amp,
        scaler=scaler,
    )

    print(f"Epoch complete | average loss: {avg_loss:.6f}")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = Path("checkpoints/ts_jepa_epoch_1.pt")
    torch.save(context_encoder.state_dict(), checkpoint_path)
    print(f"Saved context_encoder weights to {checkpoint_path}")


if __name__ == "__main__":
    main()

"""Plan §6–§7 encoder specification and config verification."""

from __future__ import annotations

from typing import Any

PLAN_ENCODER: dict[str, Any] = {
    "type": "deep_convolutional_resnet",
    "widths": [64, 128, 256],
    "embedding_dim": 256,
    "batch_norm": True,
    "activation": "ReLU",
    "stem": {
        "kernel_size": 7,
        "stride": 2,
        "pool": "max",
    },
}

PLAN_TARGET_ENCODER: dict[str, Any] = {
    "architecture": "same_as_context_encoder",
    "initialization": "copy_context_weights",  # θ̄ ← θ
    "trainable": False,
    "stop_gradient": True,
    "update_rule": "ema",
    "ema_decay": 0.99,
}


def assert_plan_encoder_config(config: dict[str, Any]) -> None:
    """Raise ValueError when encoder / target-encoder config deviates from plan §6–§7."""
    enc = config.get("ts_jepa", {}).get("encoder", {})
    tgt = config.get("ts_jepa", {}).get("target_encoder", {})
    errors: list[str] = []

    widths = enc.get("widths")
    if widths != PLAN_ENCODER["widths"]:
        errors.append(f"ts_jepa.encoder.widths: expected {PLAN_ENCODER['widths']}, got {widths}")

    emb = enc.get("embedding_dim")
    if emb != PLAN_ENCODER["embedding_dim"]:
        errors.append(f"ts_jepa.encoder.embedding_dim: expected {PLAN_ENCODER['embedding_dim']}, got {emb}")

    if not enc.get("batch_norm"):
        errors.append("ts_jepa.encoder.batch_norm must be true per plan §6.1")

    if str(enc.get("activation")) != PLAN_ENCODER["activation"]:
        errors.append(f"ts_jepa.encoder.activation: expected ReLU, got {enc.get('activation')}")

    ema = tgt.get("ema_decay")
    if ema is None or abs(float(ema) - PLAN_TARGET_ENCODER["ema_decay"]) > 1e-9:
        errors.append(
            f"ts_jepa.target_encoder.ema_decay: expected {PLAN_TARGET_ENCODER['ema_decay']}, got {ema}"
        )

    kappa = int(config.get("input", {}).get("kappa", 2))
    channels = int(config.get("input", {}).get("channels_per_rgb_frame", 3))
    expected_in = channels * kappa
    if expected_in not in (3, 6):
        errors.append(f"input channels for κ={kappa}: expected 3 or 6, got {expected_in}")

    if errors:
        raise ValueError("Plan §6–§7 encoder config mismatch:\n  - " + "\n  - ".join(errors))

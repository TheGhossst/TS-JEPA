"""Plan §7 context encoder and §8 target encoder (docs/plan.md)."""

from __future__ import annotations

from typing import Any

# PAPER-SPECIFIED: ResNet widths 64/128/256 with BN and ReLU after each.
# Stem, pooling, block counts, and the embedding head are NOT SPECIFIED.
PLAN_ENCODER: dict[str, Any] = {
    "type": "deep_convolutional_resnet",
    "widths": [64, 128, 256],
    "batch_norm": True,
    "activation": "ReLU",
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
    """Raise ValueError when encoder / target-encoder config deviates from plan §7–§8."""
    enc = config.get("ts_jepa", {}).get("encoder", {})
    tgt = config.get("ts_jepa", {}).get("target_encoder", {})
    errors: list[str] = []

    if str(enc.get("type", "deep_convolutional_resnet")) != PLAN_ENCODER["type"]:
        errors.append(f"ts_jepa.encoder.type: expected {PLAN_ENCODER['type']!r}, got {enc.get('type')!r}")

    widths = enc.get("widths")
    if widths != PLAN_ENCODER["widths"]:
        errors.append(f"ts_jepa.encoder.widths: expected {PLAN_ENCODER['widths']}, got {widths}")

    if not enc.get("batch_norm"):
        errors.append("ts_jepa.encoder.batch_norm must be true per plan §7")

    if str(enc.get("activation")) != PLAN_ENCODER["activation"]:
        errors.append(f"ts_jepa.encoder.activation: expected ReLU, got {enc.get('activation')}")

    # 256 is paper-implied (plan §6 / predictor output §9), not a §7 sentence.
    emb = enc.get("embedding_dim")
    pred_out = config.get("ts_jepa", {}).get("predictor", {}).get("output_dim")
    if emb is not None and pred_out is not None and int(emb) != int(pred_out):
        errors.append(f"encoder.embedding_dim ({emb}) must match predictor.output_dim ({pred_out})")

    if str(tgt.get("architecture")) != PLAN_TARGET_ENCODER["architecture"]:
        errors.append(
            f"ts_jepa.target_encoder.architecture: expected "
            f"{PLAN_TARGET_ENCODER['architecture']!r}, got {tgt.get('architecture')!r}"
        )
    if str(tgt.get("initialization")) != PLAN_TARGET_ENCODER["initialization"]:
        errors.append(
            f"ts_jepa.target_encoder.initialization: expected "
            f"{PLAN_TARGET_ENCODER['initialization']!r}, got {tgt.get('initialization')!r}"
        )
    if tgt.get("trainable") is not False:
        errors.append("ts_jepa.target_encoder.trainable must be false (plan §8)")
    if tgt.get("stop_gradient") is not True:
        errors.append("ts_jepa.target_encoder.stop_gradient must be true (plan §8)")
    if str(tgt.get("update_rule")) != PLAN_TARGET_ENCODER["update_rule"]:
        errors.append(
            f"ts_jepa.target_encoder.update_rule: expected "
            f"{PLAN_TARGET_ENCODER['update_rule']!r}, got {tgt.get('update_rule')!r}"
        )

    ema = tgt.get("ema_decay")
    if ema is None or abs(float(ema) - PLAN_TARGET_ENCODER["ema_decay"]) > 1e-9:
        errors.append(
            f"ts_jepa.target_encoder.ema_decay: expected {PLAN_TARGET_ENCODER['ema_decay']}, got {ema}"
        )

    if errors:
        raise ValueError("Plan §7–§8 encoder config mismatch:\n  - " + "\n  - ".join(errors))

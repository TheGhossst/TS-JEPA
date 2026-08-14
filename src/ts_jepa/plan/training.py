"""Plan §11 exact TS-JEPA training hyperparameters (Table II)."""

from __future__ import annotations

from typing import Any

PLAN_JEPA_TRAINING: dict[str, Any] = {
    "optimizer_type": "SGD",
    "learning_rate": 0.2,
    "batch_size": 256,
    "epochs": 150,
    "weight_decay": 0.0004,
    "ema_decay": 0.99,
    "lr_decay_factor": 0.99,
    "lr_decay_interval_epochs": 20,
    "early_stopping_enabled": True,
    "repetitions": 5,
    "reported_result": "best",
    "Kp": 15,
    "kappa": 2,
    "embedding_dim": 256,
}

# Table II does not specify SGD momentum. Baseline uses 0 (IC).
IC_SGD_MOMENTUM = 0.0
# Table II does not specify warmup. Linear warmup over this many epochs (IC).
IC_LR_WARMUP_EPOCHS = 10

# Plan §11 explicitly forbids generic BYOL/JEPA draft defaults.
FORBIDDEN_JEPA_TRAINING_DEFAULTS: dict[str, Any] = {
    "optimizer_types": frozenset({"Adam", "AdamW", "adam", "adamw"}),
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "epochs": 200,
    "ema_decay": 0.996,
}


def assert_plan_jepa_training_config(config: dict[str, Any]) -> None:
    """
    Raise ValueError when JEPA training hyperparameters deviate from plan §11.

    Call from baseline entry scripts only. Smoke tests may override batch_size / Kp /
    epochs; do not call this from TSJEPA.__init__ or the train loop body.
    """
    errors: list[str] = []
    opt = config.get("ts_jepa", {}).get("optimizer", {})
    lr_decay = config.get("ts_jepa", {}).get("lr_decay", {})
    target_enc = config.get("ts_jepa", {}).get("target_encoder", {})
    early = config.get("ts_jepa", {}).get("early_stopping", {})
    eval_cfg = config.get("evaluation", {})
    inp = config.get("input", {})
    enc = config.get("ts_jepa", {}).get("encoder", {})
    kp = int(config.get("ts_jepa", {}).get("prediction_horizon", {}).get("Kp", -1))

    opt_type = str(opt.get("type", ""))
    if opt_type != PLAN_JEPA_TRAINING["optimizer_type"]:
        errors.append(
            f"ts_jepa.optimizer.type: expected {PLAN_JEPA_TRAINING['optimizer_type']!r}, got {opt_type!r}"
        )
    if opt_type in FORBIDDEN_JEPA_TRAINING_DEFAULTS["optimizer_types"]:
        errors.append(
            f"ts_jepa.optimizer.type {opt_type!r} is a forbidden BYOL/JEPA draft default (plan §11)"
        )

    lr = float(opt.get("learning_rate", float("nan")))
    if abs(lr - PLAN_JEPA_TRAINING["learning_rate"]) > 1e-12:
        errors.append(f"ts_jepa.optimizer.learning_rate: expected 0.2, got {lr}")

    batch = int(opt.get("batch_size", -1))
    if batch != PLAN_JEPA_TRAINING["batch_size"]:
        errors.append(f"ts_jepa.optimizer.batch_size: expected 256, got {batch}")

    epochs = int(opt.get("epochs", -1))
    if epochs != PLAN_JEPA_TRAINING["epochs"]:
        errors.append(f"ts_jepa.optimizer.epochs: expected 150, got {epochs}")

    wd = float(opt.get("weight_decay", float("nan")))
    if abs(wd - PLAN_JEPA_TRAINING["weight_decay"]) > 1e-12:
        errors.append(f"ts_jepa.optimizer.weight_decay: expected 0.0004, got {wd}")

    ema = float(target_enc.get("ema_decay", float("nan")))
    if abs(ema - PLAN_JEPA_TRAINING["ema_decay"]) > 1e-12:
        errors.append(f"ts_jepa.target_encoder.ema_decay: expected 0.99, got {ema}")
    if abs(ema - FORBIDDEN_JEPA_TRAINING_DEFAULTS["ema_decay"]) < 1e-12:
        errors.append("ts_jepa.target_encoder.ema_decay 0.996 is forbidden (plan §11)")

    factor = float(lr_decay.get("factor", float("nan")))
    interval = int(lr_decay.get("interval_epochs", -1))
    if abs(factor - PLAN_JEPA_TRAINING["lr_decay_factor"]) > 1e-12:
        errors.append(f"ts_jepa.lr_decay.factor: expected 0.99, got {factor}")
    if interval != PLAN_JEPA_TRAINING["lr_decay_interval_epochs"]:
        errors.append(f"ts_jepa.lr_decay.interval_epochs: expected 20, got {interval}")

    if not bool(early.get("enabled", False)):
        errors.append("ts_jepa.early_stopping.enabled must be true per plan §11 (Section IV.A)")

    reps = int(eval_cfg.get("repetitions", -1))
    if reps != PLAN_JEPA_TRAINING["repetitions"]:
        errors.append(f"evaluation.repetitions: expected 5, got {reps}")
    reported = str(eval_cfg.get("reported_result", ""))
    if reported != PLAN_JEPA_TRAINING["reported_result"]:
        errors.append(f"evaluation.reported_result: expected 'best', got {reported!r}")
    seeds = list(eval_cfg.get("seeds", []))
    if len(seeds) < PLAN_JEPA_TRAINING["repetitions"]:
        errors.append(f"evaluation.seeds must provide at least 5 seeds for plan §11; got {seeds}")

    if int(inp.get("kappa", -1)) != PLAN_JEPA_TRAINING["kappa"]:
        errors.append(f"input.kappa: expected 2, got {inp.get('kappa')}")
    if kp != PLAN_JEPA_TRAINING["Kp"]:
        errors.append(f"ts_jepa.prediction_horizon.Kp: expected 15, got {kp}")
    if int(enc.get("embedding_dim", -1)) != PLAN_JEPA_TRAINING["embedding_dim"] and not bool(
        config.get("experiments", {}).get("allow_non_baseline_embedding_dim", False)
    ):
        errors.append(f"ts_jepa.encoder.embedding_dim: expected 256, got {enc.get('embedding_dim')}")

    if errors:
        raise ValueError("Plan §11 JEPA training config mismatch:\n  - " + "\n  - ".join(errors))

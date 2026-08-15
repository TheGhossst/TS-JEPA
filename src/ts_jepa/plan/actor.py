"""Plan §12 semantic actor Cε architecture and Table III training specification."""

from __future__ import annotations

from typing import Any

from ts_jepa.plan.enforce import plan_enforced

# Plan §12 architecture: embedding → Linear 1024 → ReLU → Linear 256 → ReLU → Linear 1
# Output nonlinearity is NOT SPECIFIED (plan §18); see IC_ACTOR_OUTPUT_ACTIVATION.
PLAN_SEMANTIC_ACTOR: dict[str, Any] = {
    "type": "MLP",
    "input_dim": 256,  # matches encoder embedding_dim
    "hidden_dims": [1024, 256],
    "output_dim": 1,
    "activation": "ReLU",
    "loss": "MSE",
    "loss_domain": "physical",
    "inputs": ("embedding",),
    "encoder_updated_during_training": False,
    "optimized_module": "semantic_actor",
}

# Plan §18 IC: linear head; Eq. 15 is physical Newtons (plant clips after).
IC_ACTOR_OUTPUT_ACTIVATION = "linear"

# Plan §12 Table III training hyperparameters.
PLAN_SEMANTIC_ACTOR_TRAINING: dict[str, Any] = {
    "architecture": "256 → 1024 → 256 → 1",
    "hidden_activation": "ReLU",
    "dropout": 0.2,
    "optimizer_type": "AdamW",
    "learning_rate": 0.006,
    "batch_size": 200,
    "epochs": 300,
    "early_stopping_enabled": True,
    "repetitions": 5,
    "reported_result": "best",
    "selection": "best_validation_result",
}

FORBIDDEN_ACTOR_INPUTS = frozenset({"desired_state", "x_d", "xd", "state", "rgb", "frame"})


def assert_plan_semantic_actor_config(config: dict[str, Any]) -> None:
    """
    Raise ValueError when semantic-actor architecture / loss deviate from plan §12.

    Also checks that the actor input dim matches the JEPA embedding dim.
    Output activation is an implementation choice (plan §18) and is not
    asserted here. Call from baseline entry scripts.
    Smoke tests may shrink dims; do not call this from SemanticActor.__init__
    when overrides are intentional.
    """
    if not plan_enforced(config):
        return
    errors: list[str] = []
    actor = config.get("semantic_actor", {})
    arch = actor.get("architecture", {})
    emb = int(config.get("ts_jepa", {}).get("encoder", {}).get("embedding_dim", -1))

    if str(arch.get("type", "MLP")) != PLAN_SEMANTIC_ACTOR["type"]:
        errors.append(f"semantic_actor.architecture.type: expected MLP, got {arch.get('type')!r}")

    hidden = arch.get("hidden_dims")
    if list(hidden or []) != PLAN_SEMANTIC_ACTOR["hidden_dims"]:
        errors.append(
            f"semantic_actor.architecture.hidden_dims: expected "
            f"{PLAN_SEMANTIC_ACTOR['hidden_dims']}, got {hidden}"
        )

    if str(arch.get("activation")) != PLAN_SEMANTIC_ACTOR["activation"]:
        errors.append(
            f"semantic_actor.architecture.activation: expected ReLU, got {arch.get('activation')!r}"
        )

    extra_inputs = arch.get("inputs") or actor.get("inputs")
    if extra_inputs is not None:
        names = [str(x).lower() for x in extra_inputs]
        bad = [n for n in names if n in FORBIDDEN_ACTOR_INPUTS]
        if bad:
            errors.append(
                f"semantic_actor inputs {bad} are forbidden (plan §12: Cε maps embeddings only; "
                "x_d is scoring, not an actor input)"
            )
        if names and names != ["embedding"] and names != ["z"]:
            errors.append(
                f"semantic_actor inputs must be embedding-only per plan §12; got {list(extra_inputs)}"
            )

    if bool(arch.get("use_desired_state") or actor.get("use_desired_state")):
        errors.append("semantic_actor must not take desired_state / x_d as input (plan §12)")

    if emb != PLAN_SEMANTIC_ACTOR["input_dim"] and not bool(
        config.get("experiments", {}).get("allow_non_baseline_embedding_dim", False)
    ):
        errors.append(
            f"ts_jepa.encoder.embedding_dim must be {PLAN_SEMANTIC_ACTOR['input_dim']} "
            f"for plan §12 actor input; got {emb}"
        )

    loss = actor.get("loss")
    if str(loss) != PLAN_SEMANTIC_ACTOR["loss"]:
        errors.append(f"semantic_actor.loss: expected MSE, got {loss!r}")

    domain = actor.get("loss_domain", "physical")
    if str(domain) != PLAN_SEMANTIC_ACTOR["loss_domain"]:
        errors.append(
            f"semantic_actor.loss_domain: expected {PLAN_SEMANTIC_ACTOR['loss_domain']!r} (Eq. 15), got {domain!r}"
        )

    if errors:
        raise ValueError("Plan §12 semantic actor config mismatch:\n  - " + "\n  - ".join(errors))


def assert_plan_semantic_actor_training_config(config: dict[str, Any]) -> None:
    """
    Raise ValueError when actor training hyperparameters deviate from plan §12 Table III.

    Call from baseline entry scripts only. Smoke tests may override batch_size /
    epochs / early_stopping; do not call this from the train loop body.
    """
    if not plan_enforced(config):
        return
    errors: list[str] = []
    actor = config.get("semantic_actor", {})
    arch = actor.get("architecture", {})
    opt = actor.get("optimizer", {})
    early = actor.get("early_stopping", {})
    eval_cfg = config.get("evaluation", {})

    assert_plan_semantic_actor_config(config)

    dropout = float(arch.get("dropout", float("nan")))
    if abs(dropout - PLAN_SEMANTIC_ACTOR_TRAINING["dropout"]) > 1e-12:
        errors.append(f"semantic_actor.architecture.dropout: expected 0.2, got {dropout}")

    opt_type = str(opt.get("type", ""))
    if opt_type != PLAN_SEMANTIC_ACTOR_TRAINING["optimizer_type"]:
        errors.append(
            f"semantic_actor.optimizer.type: expected "
            f"{PLAN_SEMANTIC_ACTOR_TRAINING['optimizer_type']!r}, got {opt_type!r}"
        )

    lr = float(opt.get("learning_rate", float("nan")))
    if abs(lr - PLAN_SEMANTIC_ACTOR_TRAINING["learning_rate"]) > 1e-12:
        errors.append(f"semantic_actor.optimizer.learning_rate: expected 0.006, got {lr}")

    batch = int(opt.get("batch_size", -1))
    if batch != PLAN_SEMANTIC_ACTOR_TRAINING["batch_size"]:
        errors.append(f"semantic_actor.optimizer.batch_size: expected 200, got {batch}")

    epochs = int(opt.get("epochs", -1))
    if epochs != PLAN_SEMANTIC_ACTOR_TRAINING["epochs"]:
        errors.append(f"semantic_actor.optimizer.epochs: expected 300, got {epochs}")

    if not bool(early.get("enabled", False)):
        errors.append("semantic_actor.early_stopping.enabled must be true per plan §12")

    reps = int(eval_cfg.get("repetitions", -1))
    if reps != PLAN_SEMANTIC_ACTOR_TRAINING["repetitions"]:
        errors.append(f"evaluation.repetitions: expected 5, got {reps}")

    reported = str(eval_cfg.get("reported_result", ""))
    if reported != PLAN_SEMANTIC_ACTOR_TRAINING["reported_result"]:
        errors.append(f"evaluation.reported_result: expected 'best', got {reported!r}")

    seeds = list(eval_cfg.get("seeds", []))
    if len(seeds) < PLAN_SEMANTIC_ACTOR_TRAINING["repetitions"]:
        errors.append(
            f"evaluation.seeds must provide at least 5 seeds for plan §12; got {seeds}"
        )

    if errors:
        raise ValueError(
            "Plan §12 semantic actor training config mismatch:\n  - " + "\n  - ".join(errors)
        )

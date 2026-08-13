"""Plan §5 preprocessing specification and config verification."""

from __future__ import annotations

from typing import Any

# Paper-faithful values from docs/plan.md §5. Follow the numbered training list;
# do not insert extra stages.
PLAN_PREPROCESSING: dict[str, Any] = {
    "color_jitter": {
        "brightness": 0.05,
        "contrast": 0.10,
        "saturation": 0.10,
        "hue": 0.05,
    },
    "color_drop_probability": 0.05,
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "gaussian_kernel": [5, 5],
    "gaussian_sigma_range": [0.1, 0.2],
    "resize": [64, 128],
    "command_normalization": "z-score",
}

# Training numbered list (Section IV.A): jitter → color drop → normalize → resize.
TRAINING_PIPELINE_STAGES = ("augmentation", "normalization", "resize")
# Testing: same normalize→resize order as training, without stochastic aug.
EVAL_PIPELINE_STAGES = ("normalization", "resize")


def assert_plan_preprocessing_config(config: dict[str, Any]) -> None:
    """Raise ValueError when config input/preprocessing deviates from plan §5."""
    inp = config.get("input", {})
    errors: list[str] = []

    jitter = inp.get("color_jitter", {})
    for key, expected in PLAN_PREPROCESSING["color_jitter"].items():
        actual = jitter.get(key)
        if actual is None or abs(float(actual) - float(expected)) > 1e-9:
            errors.append(f"input.color_jitter.{key}: expected {expected}, got {actual}")

    drop_p = inp.get("color_drop_probability")
    if drop_p is None or abs(float(drop_p) - PLAN_PREPROCESSING["color_drop_probability"]) > 1e-9:
        errors.append(
            f"input.color_drop_probability: expected {PLAN_PREPROCESSING['color_drop_probability']}, got {drop_p}"
        )

    norm = inp.get("normalization", {})
    for key in ("mean", "std"):
        expected = PLAN_PREPROCESSING["normalization"][key]
        actual = norm.get(key)
        if actual is None or len(actual) != 3:
            errors.append(f"input.normalization.{key}: expected len-3 list {expected}, got {actual}")
        elif any(abs(float(a) - float(e)) > 1e-9 for a, e in zip(actual, expected)):
            errors.append(f"input.normalization.{key}: expected {expected}, got {actual}")

    kernel = inp.get("gaussian_kernel")
    if kernel != PLAN_PREPROCESSING["gaussian_kernel"]:
        errors.append(f"input.gaussian_kernel: expected {PLAN_PREPROCESSING['gaussian_kernel']}, got {kernel}")

    sigma = inp.get("gaussian_sigma_range")
    if sigma is None or len(sigma) != 2:
        errors.append(f"input.gaussian_sigma_range: expected [0.1, 0.2], got {sigma}")
    elif abs(float(sigma[0]) - 0.1) > 1e-9 or abs(float(sigma[1]) - 0.2) > 1e-9:
        errors.append(f"input.gaussian_sigma_range: expected [0.1, 0.2], got {sigma}")

    resize = inp.get("resize")
    if resize != PLAN_PREPROCESSING["resize"]:
        errors.append(f"input.resize: expected {PLAN_PREPROCESSING['resize']}, got {resize}")

    cmd_norm = inp.get("command_normalization")
    if cmd_norm != PLAN_PREPROCESSING["command_normalization"]:
        errors.append(
            f"input.command_normalization: expected {PLAN_PREPROCESSING['command_normalization']!r}, got {cmd_norm!r}"
        )

    training_order = config.get("preprocessing", {}).get("training", {}).get("order")
    if training_order is not None and list(training_order) != list(TRAINING_PIPELINE_STAGES):
        errors.append(
            f"preprocessing.training.order: expected {list(TRAINING_PIPELINE_STAGES)}, got {training_order}"
        )

    eval_order = config.get("preprocessing", {}).get("evaluation", {}).get("order")
    if eval_order is not None and list(eval_order) != list(EVAL_PIPELINE_STAGES):
        errors.append(
            f"preprocessing.evaluation.order: expected {list(EVAL_PIPELINE_STAGES)}, got {eval_order}"
        )

    if errors:
        raise ValueError("Plan §5 preprocessing config mismatch:\n  - " + "\n  - ".join(errors))

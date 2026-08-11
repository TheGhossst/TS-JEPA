"""
Plan §10 / §26.1: predicted-command training ambiguity.

The paper defines predictor conditioning on predicted commands ũ, but does not
fully specify how the complete command sequence is produced during TS-JEPA
pretraining. This module records the selected implementation candidate, keeps
status OPEN, and prevents silent paper-exact claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# Plan §10 candidate mechanisms (evaluate separately; do not merge without ablation).
COMMAND_SOURCE_CANDIDATES: tuple[str, ...] = (
    "teacher_dp",
    "semantic_actor",
    "sequential_actor_predictor",
    "recovered_from_source",
)

RESOLUTION_STATUS_OPEN = "OPEN"
RESOLUTION_STATUS_DESCRIPTION = "PENDING_PAPER_IMPLEMENTATION_RESOLUTION"

# Only these sources may be selected for training in the current baseline.
IMPLEMENTED_COMMAND_SOURCES: frozenset[str] = frozenset({"teacher_dp"})


@dataclass(frozen=True)
class PredictorCommandResolution:
    """Recorded plan §10 resolution state (not a paper fact)."""

    status: str
    status_description: str
    selected_source: str
    paper_exact: bool
    candidates: tuple[str, ...]
    training_batch_field: str
    training_description: str
    inference_source: str
    inference_description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "status_description": self.status_description,
            "selected_source": self.selected_source,
            "paper_exact": self.paper_exact,
            "candidates": list(self.candidates),
            "training_batch_field": self.training_batch_field,
            "training_description": self.training_description,
            "inference_source": self.inference_source,
            "inference_description": self.inference_description,
        }


def _resolution_block(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("ts_jepa", {}).get("predictor_command_resolution", {}))


def load_predictor_command_resolution(config: dict[str, Any]) -> PredictorCommandResolution:
    pred = config.get("ts_jepa", {}).get("predictor", {})
    block = _resolution_block(config)
    selected = str(pred.get("command_source", block.get("selected_source", "teacher_dp")))

    training_meta = {
        "teacher_dp": (
            "teacher_commands_norm",
            "DP teacher ground-truth u* from trajectory dataset (normalized). "
            "Used as a documented §10 candidate — NOT paper ũ.",
        ),
        "semantic_actor": (
            "semantic_actor_commands_norm",
            "Frozen semantic actor ũ on current/predicted latents (not implemented).",
        ),
        "sequential_actor_predictor": (
            "sequential_actor_predictor_commands_norm",
            "Alternating actor/predictor command rollout (not implemented).",
        ),
        "recovered_from_source": (
            "recovered_source_commands_norm",
            "Mechanism recovered from original implementation/source (not implemented).",
        ),
    }
    field, train_desc = training_meta.get(
        selected,
        ("unknown_commands_norm", f"Unknown command source {selected!r}"),
    )

    return PredictorCommandResolution(
        status=str(block.get("status", RESOLUTION_STATUS_OPEN)),
        status_description=str(block.get("status_description", RESOLUTION_STATUS_DESCRIPTION)),
        selected_source=selected,
        paper_exact=bool(block.get("paper_exact", False)),
        candidates=tuple(block.get("candidates", COMMAND_SOURCE_CANDIDATES)),
        training_batch_field=field,
        training_description=train_desc,
        inference_source=str(block.get("inference_source", "semantic_actor")),
        inference_description=str(
            block.get(
                "inference_description",
                "Packet-lost runtime uses last semantic-actor command (closed loop), "
                "distinct from JEPA-training conditioning.",
            )
        ),
    )


def assert_plan_predictor_command_resolution(config: dict[str, Any]) -> None:
    """
    Validate plan §10 documentation is explicit and internally consistent.

    Fails when:
      - status is not OPEN
      - paper_exact is true (forbidden until paper resolves §10)
      - selected source is undocumented or not in candidates
      - predictor.command_source disagrees with resolution block
      - selected source is not implemented in this codebase
    """
    resolution = load_predictor_command_resolution(config)
    pred_source = str(config.get("ts_jepa", {}).get("predictor", {}).get("command_source", ""))
    block = _resolution_block(config)
    errors: list[str] = []

    if resolution.status != RESOLUTION_STATUS_OPEN:
        errors.append(
            f"predictor_command_resolution.status must remain {RESOLUTION_STATUS_OPEN!r} until "
            f"paper/source resolves §10; got {resolution.status!r}"
        )
    if resolution.status_description != RESOLUTION_STATUS_DESCRIPTION:
        errors.append(
            f"predictor_command_resolution.status_description must be "
            f"{RESOLUTION_STATUS_DESCRIPTION!r}"
        )
    if resolution.paper_exact:
        errors.append(
            "predictor_command_resolution.paper_exact must be false — "
            "§10 mechanism must not be labeled paper-exact while status is OPEN"
        )
    if resolution.selected_source not in resolution.candidates:
        errors.append(
            f"selected_source {resolution.selected_source!r} not listed in candidates "
            f"{list(resolution.candidates)}"
        )
    if pred_source and pred_source != resolution.selected_source:
        errors.append(
            f"predictor.command_source ({pred_source!r}) must match "
            f"predictor_command_resolution.selected_source ({resolution.selected_source!r})"
        )
    block_selected = block.get("selected_source")
    if block_selected is not None and str(block_selected) != resolution.selected_source:
        errors.append(
            f"predictor_command_resolution.selected_source ({block_selected!r}) must match "
            f"predictor.command_source ({resolution.selected_source!r})"
        )
    if resolution.selected_source not in IMPLEMENTED_COMMAND_SOURCES:
        errors.append(
            f"selected_source {resolution.selected_source!r} is not implemented. "
            f"Implemented: {sorted(IMPLEMENTED_COMMAND_SOURCES)}"
        )

    if errors:
        raise ValueError("Plan §10 predictor command resolution invalid:\n  - " + "\n  - ".join(errors))


def select_predictor_conditioning_commands(
    batch: dict[str, torch.Tensor],
    resolution: PredictorCommandResolution,
) -> torch.Tensor:
    """
    Return normalized commands fed to Pφ during JEPA training/eval.

    Naming is intentional: these are conditioning commands for the predictor,
    not necessarily the paper's ũ unless status is resolved and documented.
    """
    if resolution.selected_source == "teacher_dp":
        return batch["teacher_commands_norm"]
    if resolution.selected_source == "semantic_actor":
        raise NotImplementedError(
            "command_source=semantic_actor is a plan §10 candidate but not implemented. "
            "Use teacher_dp for the current baseline or add an ablation implementation."
        )
    if resolution.selected_source == "sequential_actor_predictor":
        raise NotImplementedError(
            "command_source=sequential_actor_predictor is a plan §10 candidate but not implemented."
        )
    if resolution.selected_source == "recovered_from_source":
        raise NotImplementedError(
            "command_source=recovered_from_source requires explicit recovery from the "
            "original implementation/source."
        )
    raise ValueError(f"Unsupported predictor command source: {resolution.selected_source!r}")

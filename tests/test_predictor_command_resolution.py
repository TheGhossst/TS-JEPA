"""Plan §10 predicted-command training ambiguity tests."""

from __future__ import annotations

import copy

import pytest
import torch

from ts_jepa.config import load_config
from ts_jepa.models.predictor_command_resolution import (
    COMMAND_SOURCE_CANDIDATES,
    IMPLEMENTED_COMMAND_SOURCES,
    RESOLUTION_STATUS_DESCRIPTION,
    RESOLUTION_STATUS_OPEN,
    assert_plan_predictor_command_resolution,
    load_predictor_command_resolution,
    select_predictor_conditioning_commands,
)
from ts_jepa.models.ts_jepa import TSJEPA


def test_plan_section10_config_is_explicit_and_open():
    config = load_config()
    assert_plan_predictor_command_resolution(config)
    resolution = load_predictor_command_resolution(config)
    assert resolution.status == RESOLUTION_STATUS_OPEN
    assert resolution.status_description == RESOLUTION_STATUS_DESCRIPTION
    assert resolution.selected_source == "teacher_dp"
    assert resolution.paper_exact is False
    assert tuple(resolution.candidates) == COMMAND_SOURCE_CANDIDATES


def test_teacher_dp_is_only_implemented_training_source():
    assert IMPLEMENTED_COMMAND_SOURCES == frozenset({"teacher_dp"})


def test_select_conditioning_uses_teacher_commands_not_predicted_label():
    config = load_config()
    resolution = load_predictor_command_resolution(config)
    batch = {
        "teacher_commands_norm": torch.tensor([[0.1, 0.2, 0.3]]),
    }
    out = select_predictor_conditioning_commands(batch, resolution)
    assert torch.equal(out, batch["teacher_commands_norm"])
    assert resolution.training_batch_field == "teacher_commands_norm"
    assert "NOT paper" in resolution.training_description or "NOT paper ũ" in resolution.training_description


def test_unimplemented_sources_fail_loudly():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["predictor"]["command_source"] = "semantic_actor"
    config["ts_jepa"]["predictor_command_resolution"]["selected_source"] = "semantic_actor"
    with pytest.raises(ValueError, match="not implemented"):
        assert_plan_predictor_command_resolution(config)

    config = copy.deepcopy(load_config())
    config["ts_jepa"]["predictor"]["command_source"] = "semantic_actor"
    config["ts_jepa"]["predictor_command_resolution"]["selected_source"] = "semantic_actor"
    # bypass assert by constructing resolution manually for select function test
    from ts_jepa.models.predictor_command_resolution import PredictorCommandResolution

    resolution = PredictorCommandResolution(
        status=RESOLUTION_STATUS_OPEN,
        status_description=RESOLUTION_STATUS_DESCRIPTION,
        selected_source="semantic_actor",
        paper_exact=False,
        candidates=COMMAND_SOURCE_CANDIDATES,
        training_batch_field="x",
        training_description="x",
        inference_source="semantic_actor",
        inference_description="x",
    )
    with pytest.raises(NotImplementedError, match="semantic_actor"):
        select_predictor_conditioning_commands({"teacher_commands_norm": torch.zeros(1, 1)}, resolution)


def test_paper_exact_true_is_rejected():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["predictor_command_resolution"]["paper_exact"] = True
    with pytest.raises(ValueError, match="paper_exact"):
        assert_plan_predictor_command_resolution(config)


def test_tsjepa_carries_command_resolution():
    model = TSJEPA(load_config())
    assert model.command_resolution.status == RESOLUTION_STATUS_OPEN
    assert model.command_resolution.selected_source == model.command_source

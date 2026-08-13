"""Checkpoint lineage: eval must follow actor metadata, not seed_0 fallback."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ts_jepa.evaluation.checkpoints import (
    resolve_jepa_checkpoint_from_actor,
    resolve_run_checkpoint,
)


def test_resolve_run_checkpoint_does_not_fall_back_to_seed_0(tmp_path: Path):
    family = tmp_path / "ts_jepa_dp_fixed"
    (family / "seed_0").mkdir(parents=True)
    torch.save({"model": {}}, family / "seed_0" / "best.pt")
    with pytest.raises(FileNotFoundError, match="No ts_jepa_dp_fixed"):
        resolve_run_checkpoint(tmp_path, "ts_jepa_dp_fixed")


def test_resolve_jepa_checkpoint_from_actor_uses_metadata(tmp_path: Path):
    jepa = tmp_path / "jepa_family" / "best.pt"
    jepa.parent.mkdir(parents=True)
    torch.save({"model": {"a": 1}}, jepa)
    actor = tmp_path / "actor" / "best.pt"
    actor.parent.mkdir()
    torch.save({"actor": {}, "jepa_checkpoint": str(jepa)}, actor)
    resolved = resolve_jepa_checkpoint_from_actor(actor, project_dir=tmp_path)
    assert resolved == jepa.resolve()

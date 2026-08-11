"""Evaluation / reporting bugfixes: init_noise wiring and single-seed test_losses."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ts_jepa.config import load_config
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv
from ts_jepa.evaluation.evaluate import (
    _test_losses_from_runs,
    baseline_report,
    evaluate_closed_loop,
)


class _StubController:
    """Minimal stand-in for FrozenRuntimeController in closed-loop tests."""

    def __init__(self) -> None:
        self.device = "cpu"

    def step(self, frame: np.ndarray | None, packet_received: bool) -> float:
        return 0.0


def test_closed_loop_uses_configured_init_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    config = copy.deepcopy(load_config())
    config["simulation"]["init_noise"] = 0.35
    config["simulation"]["trajectory_steps"] = 3
    config["evaluation"]["repetitions"] = 1

    captured: dict[str, Any] = {}
    real_init = InvertedCartPoleEnv.__init__

    def _capturing_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured.clear()
        captured.update(kwargs)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(InvertedCartPoleEnv, "__init__", _capturing_init)
    evaluate_closed_loop(config, _StubController(), steps=2, seed=0)  # type: ignore[arg-type]
    assert "init_noise" in captured
    assert captured["init_noise"] == pytest.approx(0.35)


def test_closed_loop_init_noise_matches_simulation_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = copy.deepcopy(load_config())
    config["simulation"]["init_noise"] = 0.27
    config["simulation"]["trajectory_steps"] = 2

    seen: list[float] = []
    real_init = InvertedCartPoleEnv.__init__

    def _capturing_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "init_noise" in kwargs:
            seen.append(float(kwargs["init_noise"]))
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(InvertedCartPoleEnv, "__init__", _capturing_init)
    evaluate_closed_loop(config, _StubController(), steps=1, seed=1)  # type: ignore[arg-type]
    assert seen == [0.27]


def test_test_losses_fallback_to_seed_metrics(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    jepa_seed = runs / "ts_jepa" / "seed_0"
    actor_seed = runs / "semantic_actor" / "seed_0"
    jepa_seed.mkdir(parents=True)
    actor_seed.mkdir(parents=True)
    (jepa_seed / "metrics.json").write_text(
        json.dumps({"seed": 0, "best_val": 0.01, "best_epoch": 3, "test_loss": 0.0036}),
        encoding="utf-8",
    )
    (actor_seed / "metrics.json").write_text(
        json.dumps({"seed": 0, "best_val": 0.79, "best_epoch": 19, "test_loss": 0.716}),
        encoding="utf-8",
    )

    jepa = _test_losses_from_runs(runs, "ts_jepa", split="jepa_test_untouched")
    actor = _test_losses_from_runs(runs, "semantic_actor", split="actor_test_untouched")

    assert jepa["source"] == "seed_metrics"
    assert jepa["best_seed"] == 0
    assert jepa["best_val_loss"] == pytest.approx(0.01)
    assert jepa["best_test_loss"] == pytest.approx(0.0036)
    assert actor["source"] == "seed_metrics"
    assert actor["best_test_loss"] == pytest.approx(0.716)
    assert actor["split"] == "actor_test_untouched"


def test_test_losses_prefers_repetition_summary(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    family = runs / "ts_jepa"
    seed0 = family / "seed_0"
    seed0.mkdir(parents=True)
    (seed0 / "metrics.json").write_text(
        json.dumps({"seed": 0, "best_val": 0.5, "test_loss": 0.4}),
        encoding="utf-8",
    )
    (family / "repetition_summary.json").write_text(
        json.dumps(
            {
                "best_seed": 1,
                "best_val": 0.02,
                "best_test_loss": 0.015,
                "seed_results": [{"seed": 1, "best_val": 0.02, "test_loss": 0.015}],
            }
        ),
        encoding="utf-8",
    )

    out = _test_losses_from_runs(runs, "ts_jepa", split="jepa_test_untouched")
    assert out["source"] == "repetition_summary"
    assert out["best_seed"] == 1
    assert out["best_test_loss"] == pytest.approx(0.015)
    assert out["best_val_loss"] == pytest.approx(0.02)


def test_baseline_report_single_seed_test_losses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """baseline_report should surface seed_*/metrics.json test_loss without repetition_summary."""
    config = copy.deepcopy(load_config())
    config["paths"]["data_root"] = str(tmp_path / "data")
    config["paths"]["runs_root"] = str(tmp_path / "runs")
    config["evaluation"]["repetitions"] = 1
    config["wireless"]["snr_targets_db"] = [10]
    config["simulation"]["trajectory_steps"] = 2
    config["simulation"]["init_noise"] = 0.35

    runs = Path(config["paths"]["runs_root"])
    for family, test_loss, best_val in (
        ("ts_jepa", 0.0036427, 0.008987),
        ("semantic_actor", 0.715953, 0.790979),
    ):
        seed_dir = runs / family / "seed_0"
        seed_dir.mkdir(parents=True)
        (seed_dir / "metrics.json").write_text(
            json.dumps({"seed": 0, "best_val": best_val, "best_epoch": 1, "test_loss": test_loss}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "ts_jepa.evaluation.evaluate.evaluate_closed_loop",
        lambda *args, **kwargs: {"mean_control_score": 0.0, "forces": [], "scores": []},
    )
    monkeypatch.setattr(
        "ts_jepa.evaluation.evaluate.evaluate_prediction_horizon_nmae",
        lambda *args, **kwargs: {"nmae": 1.0, "nmae_by_horizon": {"1": 1.0}, "kp": 1, "split": "jepa_test_untouched"},
    )
    monkeypatch.setattr(
        "ts_jepa.evaluation.evaluate.evaluate_embedding_tsne",
        lambda *args, **kwargs: {"num_samples": 10, "coords": [[0.0, 0.0]], "cart_positions": [0.0]},
    )

    report = baseline_report(config, _StubController(), data_root=Path(config["paths"]["data_root"]))  # type: ignore[arg-type]
    assert report["test_losses"]["jepa"]["source"] == "seed_metrics"
    assert report["test_losses"]["jepa"]["best_test_loss"] == pytest.approx(0.0036427)
    assert report["test_losses"]["semantic_actor"]["source"] == "seed_metrics"
    assert report["test_losses"]["semantic_actor"]["best_test_loss"] == pytest.approx(0.715953)
    assert "baseline_validation" in report
    assert "wireless" not in report

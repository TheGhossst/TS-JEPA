from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_root(config: dict[str, Any] | None = None) -> Path:
    if config and config.get("paths", {}).get("project_root"):
        return Path(config["paths"]["project_root"]).resolve()
    return repo_root()


def jepa_run_dirname(config: dict[str, Any]) -> str:
    """Checkpoint family under runs_root (default: ts_jepa)."""
    return str(config.get("paths", {}).get("ts_jepa_dirname", "ts_jepa"))


def actor_run_dirname(config: dict[str, Any]) -> str:
    """Checkpoint family under runs_root (default: semantic_actor)."""
    return str(config.get("paths", {}).get("semantic_actor_dirname", "semantic_actor"))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; overlay wins. Non-dict values replace."""
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else repo_root() / "configs" / "ts_jepa_baseline.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    # Optional experiment overlay: `_base: ts_jepa_baseline.yaml` then path overrides only.
    base_name = config.pop("_base", None) or config.pop("extends", None)
    if base_name is not None:
        base_path = Path(base_name)
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base = load_config(base_path)
        config = _deep_merge(base, config)

    if "paths" not in config:
        config["paths"] = {}
    config["paths"].setdefault("project_root", str(repo_root()))
    config["paths"].setdefault("data_root", "data")
    config["paths"].setdefault("runs_root", "runs")
    config["paths"].setdefault("ts_jepa_dirname", "ts_jepa")
    config["paths"].setdefault("semantic_actor_dirname", "semantic_actor")
    return config


def apply_cli_path_overrides(
    config: dict[str, Any],
    *,
    data_root: str | None = None,
    runs_root: str | None = None,
) -> dict[str, Any]:
    """Mutate config paths from CLI without touching model hyperparameters."""
    if data_root is not None:
        config["paths"]["data_root"] = data_root
    if runs_root is not None:
        config["paths"]["runs_root"] = runs_root
    return config

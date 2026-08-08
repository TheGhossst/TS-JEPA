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


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else repo_root() / "configs" / "ts_jepa_baseline.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "paths" not in config:
        config["paths"] = {}
    config["paths"].setdefault("project_root", str(repo_root()))
    config["paths"].setdefault("data_root", "data")
    config["paths"].setdefault("runs_root", "runs")
    return config

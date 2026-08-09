from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ts_jepa.config import load_config
from ts_jepa.data.dataset_sanity import inspect_split, sanity_check_trajectory_root


def _write_traj(path: Path, *, traj_id: int, steps: int, commands: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.zeros((steps, 8, 8, 3), dtype=np.uint8)
    states = np.zeros((steps, 4), dtype=np.float64)
    np.savez_compressed(
        path,
        frames=frames,
        commands=commands.astype(np.float32),
        states=states,
        split=np.asarray("test"),
        seed=np.asarray(10_000 + traj_id),
        trajectory_index=np.asarray(traj_id),
    )


def test_inspect_split_detects_nonzero_commands(tmp_path: Path) -> None:
    split_dir = tmp_path / "traj"
    cmds = np.array([0.0, 4.0, -4.0, 8.0] * 25, dtype=np.float32)  # 100
    _write_traj(split_dir / "00000.npz", traj_id=0, steps=100, commands=cmds)
    out = inspect_split(split_dir, expected_count=1, expected_steps=100)
    assert out["split_pass"]
    assert out["command_stats"]["num_unique_commands"] >= 3
    assert out["command_stats"]["command_std"] > 0


def test_inspect_split_fails_all_zero(tmp_path: Path) -> None:
    split_dir = tmp_path / "traj"
    cmds = np.zeros(100, dtype=np.float32)
    _write_traj(split_dir / "00000.npz", traj_id=0, steps=100, commands=cmds)
    out = inspect_split(split_dir, expected_count=1, expected_steps=100)
    assert not out["pass"]["std_gt_0"]
    assert not out["pass"]["positive_fraction_gt_0"]
    assert not out["pass"]["negative_fraction_gt_0"]
    assert not out["pass"]["unique_commands_ge_3"]
    assert not out["split_pass"]


def test_sanity_check_root_overlap_and_normalizer(tmp_path: Path) -> None:
    config = load_config()
    config = dict(config)
    config["paths"] = dict(config["paths"])
    config["paths"]["data_root"] = str(tmp_path)
    config["simulation"] = dict(config["simulation"])
    config["simulation"]["trajectory_steps"] = 4
    config["ts_jepa"] = dict(config["ts_jepa"])
    config["ts_jepa"]["dataset"] = {"train_trajectories": 2, "test_trajectories": 1}
    config["semantic_actor"] = dict(config["semantic_actor"])
    config["semantic_actor"]["dataset"] = {"train_trajectories": 2, "test_trajectories": 1}

    root = tmp_path
    # JEPA: train ids 0,1 test id 2 — no overlap
    for i, force in enumerate([4.0, -8.0]):
        cmds = np.array([force, 0.0, -force, force], dtype=np.float32)
        _write_traj(root / "trajectories" / "jepa" / "train" / f"{i:05d}.npz", traj_id=i, steps=4, commands=cmds)
    _write_traj(
        root / "trajectories" / "jepa" / "test" / "00002.npz",
        traj_id=2,
        steps=4,
        commands=np.array([4.0, -4.0, 0.0, 8.0], dtype=np.float32),
    )
    for i, force in enumerate([12.0, -12.0]):
        cmds = np.array([force, 0.0, -4.0, 4.0], dtype=np.float32)
        _write_traj(root / "trajectories" / "actor" / "train" / f"{i:05d}.npz", traj_id=i, steps=4, commands=cmds)
    _write_traj(
        root / "trajectories" / "actor" / "test" / "00002.npz",
        traj_id=2,
        steps=4,
        commands=np.array([-20.0, 0.0, 20.0, 4.0], dtype=np.float32),
    )

    report = sanity_check_trajectory_root(config, root, fit_normalizer=True)
    assert report["overall_pass"]
    assert report["command_normalizer"]["nondegenerate"]
    assert (root / "stats" / "command_norm.json").exists()
    payload = json.loads((root / "stats" / "command_norm.json").read_text(encoding="utf-8"))
    assert abs(float(payload["std"])) > 1e-8

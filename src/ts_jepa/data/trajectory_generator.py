from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ts_jepa.config import project_root
from ts_jepa.control.dp_teacher import DPControlTeacher
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv
from ts_jepa.env.factory import build_inverted_cartpole_env

# Plan §4 dataset layout (D_s = JEPA, D_a = semantic actor).
PLAN_DATASET_SPLITS: tuple[tuple[str, str, str, str], ...] = (
    ("D_s", "jepa", "ts_jepa", "train"),
    ("D_s", "jepa", "ts_jepa", "test"),
    ("D_a", "actor", "semantic_actor", "train"),
    ("D_a", "actor", "semantic_actor", "test"),
)

CONTROL_TEACHER_TYPE = "dp_nonlinear"


class UniformRandomControlTeacher:
    """
    Anti-pattern baseline for plan §4.1 verification only.

    DO NOT use for dataset generation — plan forbids independent Uniform(-20, 20) actions.
    """

    def __init__(self, force_min: float = -20.0, force_max: float = 20.0, seed: int = 0) -> None:
        self.force_min = float(force_min)
        self.force_max = float(force_max)
        self._rng = np.random.default_rng(int(seed))

    def act(self, state: np.ndarray) -> float:  # noqa: ARG002
        return float(self._rng.uniform(self.force_min, self.force_max))


def assert_plan_simulation_timing(config: dict[str, Any]) -> None:
    """Plan §4: τ_o = 1 ms sampling period."""
    sim = config["simulation"]
    tau_ms = float(sim["sampling_interval_ms"])
    dt = float(sim["dt"])
    if abs(dt - tau_ms / 1000.0) > 1e-12:
        raise ValueError(
            f"simulation.dt ({dt}) must equal sampling_interval_ms/1000 ({tau_ms / 1000.0}) per plan §4"
        )


def assert_plan_dataset_counts(config: dict[str, Any]) -> None:
    """Plan §4: D_s 200/40, D_a 100/20, 100 steps per trajectory."""
    steps = int(config["simulation"]["trajectory_steps"])
    if steps != 100:
        raise ValueError(f"simulation.trajectory_steps must be 100 per plan §4, got {steps}")
    jepa = config["ts_jepa"]["dataset"]
    actor = config["semantic_actor"]["dataset"]
    expected = {
        ("jepa", "train"): 200,
        ("jepa", "test"): 40,
        ("actor", "train"): 100,
        ("actor", "test"): 20,
    }
    actual = {
        ("jepa", "train"): int(jepa["train_trajectories"]),
        ("jepa", "test"): int(jepa["test_trajectories"]),
        ("actor", "train"): int(actor["train_trajectories"]),
        ("actor", "test"): int(actor["test_trajectories"]),
    }
    if actual != expected:
        raise ValueError(f"Plan §4 dataset counts mismatch: expected {expected}, got {actual}")


def build_env_and_teacher(config: dict[str, Any]) -> tuple[InvertedCartPoleEnv, DPControlTeacher]:
    assert_plan_simulation_timing(config)
    sim = config["simulation"]
    teacher_cfg = config["control_teacher"]
    env = build_inverted_cartpole_env(config)
    teacher = DPControlTeacher(
        env=env,
        grid=teacher_cfg["grid"],
        force_min=teacher_cfg["force_min_N"],
        force_max=teacher_cfg["force_max_N"],
        force_bins=teacher_cfg["force_bins"],
        control_effort_weight=teacher_cfg["control_effort_weight"],
        discount=teacher_cfg["discount"],
        value_iteration_iters=teacher_cfg["value_iteration_iters"],
        desired_state=sim["desired_state"],
        dp_substeps=int(teacher_cfg.get("dp_substeps", 50)),
    )
    return env, teacher


def generate_trajectory(
    env: InvertedCartPoleEnv,
    teacher: DPControlTeacher,
    steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Plan §4.1 loop per step k:
      state_k / frame_k → DP teacher → u*_k → env.step → state_{k+1}
    """
    return env.rollout(teacher, steps=steps, seed=seed)


def _trajectory_metadata(config: dict[str, Any], split: str, trajectory_index: int, seed: int) -> dict[str, Any]:
    sim = config["simulation"]
    return {
        "split": np.asarray(split),
        "seed": np.asarray(seed),
        "trajectory_index": np.asarray(trajectory_index),
        "control_teacher": np.asarray(CONTROL_TEACHER_TYPE),
        "sampling_interval_ms": np.asarray(float(sim["sampling_interval_ms"])),
        "dt": np.asarray(float(sim["dt"])),
        "render_height": np.asarray(int(sim["render_height"])),
        "render_width": np.asarray(int(sim["render_width"])),
    }


def generate_dataset_split(
    config: dict[str, Any],
    split: str,
    count: int,
    start_index: int,
    output_dir: Path,
    env: InvertedCartPoleEnv | None = None,
    teacher: DPControlTeacher | None = None,
) -> None:
    if env is None or teacher is None:
        env, teacher = build_env_and_teacher(config)
    steps = int(config["simulation"]["trajectory_steps"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for offset in range(count):
        trajectory_index = start_index + offset
        seed = 10_000 + trajectory_index
        trajectory = generate_trajectory(env, teacher, steps=steps, seed=seed)
        output_path = output_dir / f"{trajectory_index:05d}.npz"
        meta = _trajectory_metadata(config, split, trajectory_index, seed)
        np.savez_compressed(
            output_path,
            frames=trajectory["frames"],
            commands=trajectory["commands"],
            states=trajectory["states"],
            **meta,
        )


def generate_all_trajectories(
    config: dict[str, Any],
    data_root: Path | None = None,
    *,
    run_sanity_check: bool = True,
) -> Path:
    """
    Generate plan §4 datasets D_s (JEPA) and D_a (semantic actor).

    Raises RuntimeError if post-generation sanity checks fail.
    """
    assert_plan_dataset_counts(config)
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    jepa_cfg = config["ts_jepa"]["dataset"]
    actor_cfg = config["semantic_actor"]["dataset"]
    env, teacher = build_env_and_teacher(config)

    generate_dataset_split(
        config, "jepa_train", jepa_cfg["train_trajectories"], 0, root / "trajectories" / "jepa" / "train", env, teacher
    )
    generate_dataset_split(
        config,
        "jepa_test",
        jepa_cfg["test_trajectories"],
        jepa_cfg["train_trajectories"],
        root / "trajectories" / "jepa" / "test",
        env,
        teacher,
    )
    generate_dataset_split(
        config, "actor_train", actor_cfg["train_trajectories"], 0, root / "trajectories" / "actor" / "train", env, teacher
    )
    generate_dataset_split(
        config,
        "actor_test",
        actor_cfg["test_trajectories"],
        actor_cfg["train_trajectories"],
        root / "trajectories" / "actor" / "test",
        env,
        teacher,
    )

    if run_sanity_check:
        from ts_jepa.data.dataset_sanity import sanity_check_trajectory_root

        report = sanity_check_trajectory_root(config, root, fit_normalizer=True)
        if not report["overall_pass"]:
            raise RuntimeError(f"Plan §4 trajectory sanity check failed: {report['pass']}")
    return root

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ts_jepa.config import project_root
from ts_jepa.control.dp_teacher import DPControlTeacher
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv
from ts_jepa.env.factory import build_inverted_cartpole_env
from ts_jepa.env.renderer import sample_appearance
from ts_jepa.plan.environment import assert_plan_environment_config
from ts_jepa.plan.preprocessing import assert_plan_preprocessing_config

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
    """Plan §4: τ_o = 1 ms sampling period. DP Bellman step equals that τ_o."""
    sim = config["simulation"]
    tau_ms = float(sim["sampling_interval_ms"])
    dt = float(sim["dt"])
    if abs(dt - tau_ms / 1000.0) > 1e-12:
        raise ValueError(
            f"simulation.dt ({dt}) must equal sampling_interval_ms/1000 ({tau_ms / 1000.0}) per plan §4"
        )
    substeps = int(config.get("control_teacher", {}).get("dp_substeps", 1))
    if substeps != 1:
        raise ValueError(
            f"control_teacher.dp_substeps must be 1 (one Bellman transition = one τ_o); got {substeps}"
        )


def dataset_split_index_plan(config: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """
    Disjoint (count, start_index) for D_s then D_a.

    Paper lists D_s and D_a as separate datasets (plan §4). Whether they share
    physical rollouts is NOT SPECIFIED. This repo uses disjoint
    ``trajectory_index`` / seed values so actor test is not a subset of JEPA
    train (see docs/IMPLEMENTATION_CHOICES.md).
    """
    jepa = config["ts_jepa"]["dataset"]
    actor = config["semantic_actor"]["dataset"]
    jepa_tr = int(jepa["train_trajectories"])
    jepa_te = int(jepa["test_trajectories"])
    act_tr = int(actor["train_trajectories"])
    act_te = int(actor["test_trajectories"])
    cursor = 0
    plan: dict[str, tuple[int, int]] = {}
    for name, count in (
        ("jepa_train", jepa_tr),
        ("jepa_test", jepa_te),
        ("actor_train", act_tr),
        ("actor_test", act_te),
    ):
        plan[name] = (count, cursor)
        cursor += count
    return plan


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
    gen = config.get("dataset_generation")
    if gen:
        gen_actual = {
            ("jepa", "train"): int(gen["D_s"]["train_trajectories"]),
            ("jepa", "test"): int(gen["D_s"]["test_trajectories"]),
            ("actor", "train"): int(gen["D_a"]["train_trajectories"]),
            ("actor", "test"): int(gen["D_a"]["test_trajectories"]),
        }
        if gen_actual != actual:
            raise ValueError(
                f"dataset_generation counts must match ts_jepa/semantic_actor dataset blocks: "
                f"dataset blocks {actual}, dataset_generation {gen_actual}"
            )
        if int(gen.get("steps_per_trajectory", steps)) != 100:
            raise ValueError("dataset_generation.steps_per_trajectory must be 100 per plan §4")


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
        dp_substeps=int(teacher_cfg.get("dp_substeps", 1)),
    )
    return env, teacher


def generate_trajectory(
    env: InvertedCartPoleEnv,
    teacher: DPControlTeacher,
    steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Plan §4 loop per stored step k (τ_o = 1 ms, stride=1 in the paper setup):
      state_k / frame_k → DP teacher → u*_k → apply u*_k for observation_stride physics ticks
    """
    appearance = sample_appearance(np.random.default_rng(int(seed)))
    env.renderer.set_appearance(appearance)
    return env.rollout(teacher, steps=steps, seed=seed)


def _trajectory_metadata(config: dict[str, Any], split: str, trajectory_index: int, seed: int) -> dict[str, Any]:
    sim = config["simulation"]
    stride = int(sim.get("observation_stride_steps", 1))
    return {
        "split": np.asarray(split),
        "seed": np.asarray(seed),
        "trajectory_index": np.asarray(trajectory_index),
        "control_teacher": np.asarray(CONTROL_TEACHER_TYPE),
        "sampling_interval_ms": np.asarray(float(sim["sampling_interval_ms"])),
        "dt": np.asarray(float(sim["dt"])),
        "observation_stride_steps": np.asarray(stride),
        "observation_interval_ms": np.asarray(float(sim["sampling_interval_ms"]) * stride),
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
    assert_plan_environment_config(config)
    assert_plan_preprocessing_config(config)
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    env, teacher = build_env_and_teacher(config)

    index_plan = dataset_split_index_plan(config)
    for split_name, family, split in (
        ("jepa_train", "jepa", "train"),
        ("jepa_test", "jepa", "test"),
        ("actor_train", "actor", "train"),
        ("actor_test", "actor", "test"),
    ):
        count, start = index_plan[split_name]
        generate_dataset_split(
            config,
            split_name,
            count,
            start,
            root / "trajectories" / family / split,
            env,
            teacher,
        )

    if run_sanity_check:
        from ts_jepa.data.dataset_sanity import sanity_check_trajectory_root

        report = sanity_check_trajectory_root(config, root, fit_normalizer=True)
        if not report["overall_pass"]:
            raise RuntimeError(f"Plan §4 trajectory sanity check failed: {report['pass']}")
    return root

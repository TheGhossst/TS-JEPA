from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ts_jepa.config import project_root
from ts_jepa.control.dp_teacher import DPControlTeacher
from ts_jepa.env.cartpole_ode import CartPoleParams
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv
from ts_jepa.env.sampling import resolve_observation_cadence_from_config


def build_env_and_teacher(config: dict[str, Any]) -> tuple[InvertedCartPoleEnv, DPControlTeacher]:
    sim = config["simulation"]
    teacher_cfg = config["control_teacher"]
    phys = sim.get("physics", {})
    params = CartPoleParams(
        cart_mass=float(phys.get("cart_mass", 1.0)),
        pole_mass=float(phys.get("pole_mass", 0.1)),
        pole_length=float(phys.get("pole_length", 0.5)),
        gravity=float(phys.get("gravity", 9.81)),
        track_limit=float(phys.get("track_limit", 2.4)),
    )
    env = InvertedCartPoleEnv(
        render_height=sim["render_height"],
        render_width=sim["render_width"],
        dt=sim["dt"],
        force_min=sim["control_min_N"],
        force_max=sim["control_max_N"],
        desired_state=sim["desired_state"],
        params=params,
        process_noise_std=float(sim.get("process_noise_std", 0.0)),
        init_noise=float(sim.get("init_noise", 0.05)),
    )
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
    observation_stride: int = 1,
) -> dict[str, np.ndarray]:
    return env.rollout(
        teacher,
        steps=steps,
        seed=seed,
        observation_stride=observation_stride,
    )


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
    cadence = resolve_observation_cadence_from_config(config, allow_truncated_budget=False)
    num_obs = int(cadence["num_observations"])
    stride = int(cadence["observation_stride"])
    dt = float(config["simulation"]["dt"])
    sampling_interval_ms = int(cadence["sampling_interval_ms"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for offset in range(count):
        trajectory_index = start_index + offset
        seed = 10_000 + trajectory_index
        trajectory = generate_trajectory(
            env,
            teacher,
            steps=num_obs,
            seed=seed,
            observation_stride=stride,
        )
        assert int(trajectory["observation_stride"]) == stride
        assert int(trajectory["physics_steps_total"]) == num_obs * stride
        output_path = output_dir / f"{trajectory_index:05d}.npz"
        np.savez_compressed(
            output_path,
            frames=trajectory["frames"],
            commands=trajectory["commands"],
            states=trajectory["states"],
            observation_indices=trajectory["observation_indices"],
            timestamps_s=trajectory["timestamps_s"],
            observation_stride=trajectory["observation_stride"],
            physics_steps_total=trajectory["physics_steps_total"],
            sampling_interval_ms=np.asarray(sampling_interval_ms, dtype=np.int64),
            dt_s=np.asarray(dt, dtype=np.float64),
            split=np.asarray(split),
            seed=np.asarray(seed),
            trajectory_index=np.asarray(trajectory_index),
        )


def generate_all_trajectories(
    config: dict[str, Any],
    data_root: Path | None = None,
    *,
    pilot_count_per_split: int | None = None,
) -> Path:
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    jepa_cfg = config["ts_jepa"]["dataset"]
    actor_cfg = config["semantic_actor"]["dataset"]
    env, teacher = build_env_and_teacher(config)

    generate_dataset_split(
        config,
        "jepa_train",
        pilot_count_per_split or jepa_cfg["train_trajectories"],
        0,
        root / "trajectories" / "jepa" / "train",
        env,
        teacher,
    )
    generate_dataset_split(
        config,
        "jepa_test",
        pilot_count_per_split or jepa_cfg["test_trajectories"],
        jepa_cfg["train_trajectories"],
        root / "trajectories" / "jepa" / "test",
        env,
        teacher,
    )
    generate_dataset_split(
        config,
        "actor_train",
        pilot_count_per_split or actor_cfg["train_trajectories"],
        0,
        root / "trajectories" / "actor" / "train",
        env,
        teacher,
    )
    generate_dataset_split(
        config,
        "actor_test",
        pilot_count_per_split or actor_cfg["test_trajectories"],
        actor_cfg["train_trajectories"],
        root / "trajectories" / "actor" / "test",
        env,
        teacher,
    )
    return root

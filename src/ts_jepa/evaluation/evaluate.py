from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_jepa.config import project_root
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv
from ts_jepa.evaluation.metrics import (
    communication_bits_embedding,
    communication_bits_rgb,
    control_score,
    nmae,
    summarize_scores,
)
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.wireless.scheduler import ChannelAwareScheduler


def evaluate_closed_loop(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    steps: int | None = None,
    seed: int = 0,
    packet_receive_mask: list[bool] | None = None,
) -> dict[str, Any]:
    env = InvertedCartPoleEnv(
        render_height=config["simulation"]["render_height"],
        render_width=config["simulation"]["render_width"],
        dt=config["simulation"]["dt"],
        force_min=config["simulation"]["control_min_N"],
        force_max=config["simulation"]["control_max_N"],
        desired_state=config["simulation"]["desired_state"],
        process_noise_std=config["simulation"]["process_noise_std"],
    )
    steps = steps or int(config["simulation"]["trajectory_steps"])
    state = env.reset(seed=seed)
    scores = []
    forces = []
    if packet_receive_mask is None:
        packet_receive_mask = [True] * steps

    for t in range(steps):
        frame = env.render(state)
        received = bool(packet_receive_mask[t])
        force = controller.step(frame if received else None, packet_received=received)
        forces.append(force)
        scores.append(
            control_score(
                state,
                desired_x=float(config["simulation"]["desired_state"][0]),
                position_tol=float(config["evaluation"]["control_position_tol"]),
                angle_tol=float(config["evaluation"]["control_angle_tol"]),
            )
        )
        state, _ = env.step(force)

    return {
        "mean_control_score": float(np.mean(scores)),
        "forces": forces,
        "scores": scores,
    }


def evaluate_with_scheduler(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    policy: str = "channel_aware",
    snr_db: float = 10.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Single-device closed loop gated by multi-device scheduler alpha for device 0."""
    scheduler = ChannelAwareScheduler(config, policy=policy)  # type: ignore[arg-type]
    scheduler.set_snr_target(snr_db)
    rng = np.random.default_rng(seed)
    steps = int(config["simulation"]["trajectory_steps"])
    mask = []
    schedule_stats = []
    for _ in range(steps):
        decision = scheduler.schedule(rng)
        mask.append(bool(decision.alphas[0]))
        schedule_stats.append(decision.alphas[0])
    result = evaluate_closed_loop(config, controller, steps=steps, seed=seed, packet_receive_mask=mask)
    result["schedule_receive_rate"] = float(np.mean(schedule_stats))
    result["policy"] = policy
    result["snr_db"] = snr_db
    return result


def baseline_report(config: dict[str, Any], controller: FrozenRuntimeController) -> dict[str, Any]:
    reps = int(config["evaluation"]["repetitions"])
    scores = []
    for r in range(reps):
        out = evaluate_closed_loop(config, controller, seed=100 + r)
        scores.append(out["mean_control_score"])
    bits = {
        "rgb_bits": communication_bits_rgb(
            config["input"]["resize"][0],
            config["input"]["resize"][1],
            channels=3,
        ),
        "embedding_bits": communication_bits_embedding(
            config["ts_jepa"]["encoder"]["embedding_dim"]
        ),
    }
    wireless = {}
    for policy in ("channel_aware", "round_robin", "opportunistic"):
        wireless[policy] = {}
        for snr in config["wireless"]["snr_targets_db"]:
            wireless[policy][snr] = evaluate_with_scheduler(
                config, controller, policy=policy, snr_db=float(snr), seed=7
            )
    return {
        "control": summarize_scores(scores),
        "communication_bits": bits,
        "wireless": {
            p: {str(snr): {"mean_control_score": wireless[p][snr]["mean_control_score"],
                           "schedule_receive_rate": wireless[p][snr]["schedule_receive_rate"]}
                for snr in config["wireless"]["snr_targets_db"]}
            for p in wireless
        },
    }

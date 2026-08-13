"""Closed-loop eval for §16 conventional controllers and §17 Figs. 6–11."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ts_jepa.baselines.controllers import ControlLoopAgent
from ts_jepa.config import actor_run_dirname, jepa_run_dirname, project_root
from ts_jepa.data.trajectory_generator import build_env_and_teacher
from ts_jepa.env.factory import build_inverted_cartpole_env
from ts_jepa.evaluation.evaluate import _apply_held_force, _observation_stride
from ts_jepa.evaluation.metrics import (
    communication_reduction_report,
    control_score,
    nmae,
    physical_force_range_n,
)
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.plan.baseline_validation import PLAN_BASELINE_VALIDATION
from ts_jepa.wireless.scheduler import ChannelAwareScheduler


def evaluate_agent_closed_loop(
    config: dict[str, Any],
    agent: ControlLoopAgent,
    *,
    steps: int | None = None,
    seed: int = 0,
    packet_receive_mask: list[bool] | None = None,
    teacher_forces: bool = True,
) -> dict[str, Any]:
    agent.reset_episode()
    env = build_inverted_cartpole_env(config)
    teacher = None
    if teacher_forces:
        _, teacher = build_env_and_teacher(config)
    steps = int(steps or config["simulation"]["trajectory_steps"])
    stride = _observation_stride(config)
    state = env.reset(seed=seed)
    scores: list[int] = []
    forces: list[float] = []
    teachers: list[float] = []
    if packet_receive_mask is None:
        packet_receive_mask = [True] * steps
    for t in range(steps):
        frame = env.render(state)
        received = bool(packet_receive_mask[t])
        force = agent.step(frame, received, plant_state=state)
        forces.append(float(force))
        if teacher is not None:
            teachers.append(float(teacher.act(state)))
        scores.append(
            control_score(
                state,
                desired_x=float(config["simulation"]["desired_state"][0]),
                position_tol=float(config["evaluation"]["control_position_tol"]),
                angle_tol=float(config["evaluation"]["control_angle_tol"]),
            )
        )
        state = _apply_held_force(env, force, stride)
    payload: dict[str, Any] = {
        "mean_control_score": float(np.mean(scores)) if scores else float("nan"),
        "forces": forces,
        "scores": scores,
        "controlled_steps": int(sum(scores)),
        "num_steps": int(steps),
        "miss_behavior": getattr(agent, "miss_behavior", "unknown"),
        "packet_receive_rate": float(np.mean(packet_receive_mask)),
    }
    if teachers:
        payload["teacher_forces"] = teachers
        payload["nmae"] = nmae(np.asarray(forces), np.asarray(teachers), physical_force_range_n(config))
    return payload


def _prefix_scores(scores: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    acc = 0
    for h, s in enumerate(scores, start=1):
        acc += int(s)
        out[str(h)] = acc / float(h)
    return out


def _prefix_nmae(forces: list[float], teachers: list[float], force_range: float) -> dict[str, float]:
    f = np.asarray(forces, dtype=np.float64)
    t = np.asarray(teachers, dtype=np.float64)
    out: dict[str, float] = {}
    for h in range(1, len(f) + 1):
        out[str(h)] = nmae(f[:h], t[:h], force_range_n=force_range)
    return out


def evaluate_fig6_prediction_modes(
    config: dict[str, Any],
    agents: dict[str, ControlLoopAgent],
    *,
    seed: int = 100,
) -> dict[str, Any]:
    """
    Fig. 6: no-prediction (fresh observation every slot) vs 15-step single initial TX.

    Conventional agents must hold-last or zero; do not attach TS-JEPA Pφ.
    """
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    steps = int(config["simulation"]["trajectory_steps"])
    always = [True] * steps
    single = [True] + [False] * (steps - 1)
    h, w = config["input"]["resize"]
    force_range = physical_force_range_n(config)
    report: dict[str, Any] = {
        "plan_section": "17",
        "figure": "6",
        "kp": kp,
        "modes": ["no_prediction", "single_initial_transmission"],
        "results": {},
    }
    for name, agent in agents.items():
        entry: dict[str, Any] = {}
        for mode, mask in (("no_prediction", always), ("single_initial_transmission", single)):
            out = evaluate_agent_closed_loop(config, agent, seed=seed, packet_receive_mask=mask)
            comm = communication_reduction_report(
                height=int(h),
                width=int(w),
                embedding_dim=int(config["ts_jepa"]["encoder"]["embedding_dim"]),
            )
            is_rgb = name.startswith("dp") or name.startswith("supervised")
            bits_per_tx = comm["rgb_bits"] if is_rgb else comm["embedding_bits"]
            if name.startswith("supervised_kappa"):
                kappa = int(name.split("kappa")[-1].split("_")[0])
                bits_per_tx = comm["rgb_bits"] * kappa
            n_tx = int(sum(mask))
            entry[mode] = {
                "mean_control_score": out["mean_control_score"],
                "nmae": out.get("nmae"),
                "control_score_by_horizon": _prefix_scores(out["scores"][:kp]),
                "nmae_by_horizon": _prefix_nmae(out["forces"][:kp], out.get("teacher_forces", out["forces"])[:kp], force_range)
                if out.get("teacher_forces")
                else {},
                "communication_bits_per_slot_when_tx": bits_per_tx,
                "transmissions": n_tx,
                "total_communication_bits": bits_per_tx * n_tx,
                "miss_behavior": out["miss_behavior"],
            }
        report["results"][name] = entry
    return report


def evaluate_with_scheduler_agent(
    config: dict[str, Any],
    agent: ControlLoopAgent,
    *,
    policy: str,
    snr_db: float,
    seed: int = 0,
    extra_packet_loss: float = 0.0,
) -> dict[str, Any]:
    """
    Wireless closed loop for device 0.

    Plan §16: unscheduled conventional actuators hold the last command.
    Applying the same hold on a scheduled outage / extra drop is IMPLEMENTATION CHOICE
    (plan names unscheduled hold; Fig. 11 packet loss is otherwise unspecified).
    extra_packet_loss rates are IC (not in plan.md).
    """
    agent.reset_episode()
    cfg = copy.deepcopy(config)
    scheduler = ChannelAwareScheduler(cfg, policy=policy)  # type: ignore[arg-type]
    scheduler.set_snr_target(snr_db)
    rng = np.random.default_rng(seed)
    env = build_inverted_cartpole_env(cfg)
    steps = int(cfg["simulation"]["trajectory_steps"])
    stride = _observation_stride(cfg)
    state = env.reset(seed=seed)
    scores: list[int] = []
    forces: list[float] = []
    scheduled: list[int] = []
    delivered: list[int] = []
    p_extra = float(extra_packet_loss)
    for _ in range(steps):
        decision = scheduler.schedule(rng)
        success = bool(decision.successes[0])
        if success and p_extra > 0.0 and rng.random() < p_extra:
            success = False
        scheduled.append(int(decision.alphas[0]))
        delivered.append(int(success))
        frame = env.render(state)
        force = agent.step(frame, success, plant_state=state)
        forces.append(float(force))
        scores.append(
            control_score(
                state,
                desired_x=float(cfg["simulation"]["desired_state"][0]),
                position_tol=float(cfg["evaluation"]["control_position_tol"]),
                angle_tol=float(cfg["evaluation"]["control_angle_tol"]),
            )
        )
        state = _apply_held_force(env, force, stride)
    return {
        "mean_control_score": float(np.mean(scores)) if scores else float("nan"),
        "policy": policy,
        "snr_db": float(snr_db),
        "extra_packet_loss": p_extra,
        "schedule_rate": float(np.mean(scheduled)) if scheduled else 0.0,
        "packet_receive_rate": float(np.mean(delivered)) if delivered else 0.0,
        "miss_behavior": getattr(agent, "miss_behavior", "unknown"),
        "num_devices": int(cfg["wireless"]["num_devices"]),
        "scores": scores,
        "forces": forces,
    }


def max_devices_for_score_band(
    config: dict[str, Any],
    agent: ControlLoopAgent,
    *,
    policy: str,
    snr_db: float,
    device_counts: list[int],
    extra_packet_loss: float = 0.0,
    seed: int = 0,
    j_scheduled: int | None = None,
) -> dict[str, Any]:
    """
    Figs. 10–11: largest I with mean Eq. (28) score in [0.74, 1.0].

    device_counts and J(I) pairing are IMPLEMENTATION CHOICE (paper does not list I).
    """
    lo, hi = PLAN_BASELINE_VALIDATION["acceptable_control_score_band"]
    supported = 0
    per_i: dict[str, Any] = {}
    for i_count in device_counts:
        cfg = copy.deepcopy(config)
        cfg["wireless"]["num_devices"] = int(i_count)
        if j_scheduled is not None:
            cfg["wireless"]["max_devices_scheduled_J"] = int(j_scheduled)
        out = evaluate_with_scheduler_agent(
            cfg, agent, policy=policy, snr_db=snr_db, seed=seed, extra_packet_loss=extra_packet_loss
        )
        score = float(out["mean_control_score"])
        ok = lo <= score <= hi
        per_i[str(i_count)] = {"mean_control_score": score, "in_band": ok, **{k: out[k] for k in ("schedule_rate", "packet_receive_rate")}}
        if ok:
            supported = int(i_count)
        else:
            break
    return {
        "max_devices_in_band": supported,
        "band": [lo, hi],
        "by_device_count": per_i,
        "policy": policy,
        "snr_db": float(snr_db),
        "extra_packet_loss": float(extra_packet_loss),
        "device_counts_implementation_choice": list(device_counts),
    }


def apply_embedding_dim(config: dict[str, Any], dim: int) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg.setdefault("experiments", {})
    cfg["experiments"]["allow_non_baseline_embedding_dim"] = True
    cfg["ts_jepa"]["encoder"]["embedding_dim"] = int(dim)
    cfg["ts_jepa"]["predictor"]["output_dim"] = int(dim)
    cfg["paths"]["ts_jepa_dirname"] = f"{jepa_run_dirname(config)}_emb{dim}"
    cfg["paths"]["semantic_actor_dirname"] = f"{actor_run_dirname(config)}_emb{dim}"
    return cfg


def _require_ic_list(config: dict[str, Any], key: str) -> list[Any]:
    exp = config.get("experiments", {})
    if key not in exp:
        raise ValueError(
            f"experiments.{key} is required as an implementation choice (plan §18); "
            "do not treat a hardcoded fallback as paper-exact"
        )
    return list(exp[key])


def fig7_embedding_dims(config: dict[str, Any]) -> tuple[int, ...]:
    """Fig. 7 grid: plan §17 names the experiment; the numeric axis is IC (not in plan.md)."""
    return tuple(int(x) for x in _require_ic_list(config, "fig7_embedding_dims"))


def fig8_train_sizes(config: dict[str, Any]) -> tuple[int, ...]:
    """Fig. 8 train-set sizes: not in plan.md."""
    return tuple(int(x) for x in _require_ic_list(config, "fig8_train_trajectories"))


def fig10_device_counts(config: dict[str, Any]) -> list[int]:
    """I sweep for Figs. 10–11: not in Table IV / plan.md."""
    return [int(x) for x in _require_ic_list(config, "fig10_device_counts")]


def fig11_packet_losses(config: dict[str, Any]) -> tuple[float, ...]:
    """Fig. 11 packet-loss axis: not in plan.md."""
    return tuple(float(x) for x in _require_ic_list(config, "fig11_packet_loss"))


def fig9_snr_targets(config: dict[str, Any]) -> tuple[int, ...]:
    """Plan §17 / Table wireless: γ_th ∈ {5,10,20} dB."""
    extra = config.get("experiments", {}).get("fig9_snr_db")
    if extra:
        return tuple(int(x) for x in extra)
    return tuple(int(x) for x in config["wireless"]["snr_targets_db"])


def default_jepa_controller(config: dict[str, Any], device: torch.device | None = None) -> FrozenRuntimeController:
    root = project_root(config)
    runs = root / config["paths"]["runs_root"]
    from ts_jepa.evaluation.checkpoints import resolve_jepa_checkpoint_from_actor, resolve_run_checkpoint

    actor_ckpt = resolve_run_checkpoint(runs, actor_run_dirname(config))
    jepa_ckpt = resolve_jepa_checkpoint_from_actor(actor_ckpt, project_dir=root)
    return FrozenRuntimeController.from_checkpoints(config, jepa_ckpt, actor_ckpt, device=device)

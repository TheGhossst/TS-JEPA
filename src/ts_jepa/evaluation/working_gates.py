"""Working-overlay success gates (not paper plan asserts)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, project_root
from ts_jepa.data.datasets import TrajectoryDataset, load_command_normalizer
from ts_jepa.device import select_device
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.evaluation.evaluate import evaluate_actor_nmae, evaluate_closed_loop
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA


def _effective_rank(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float64)
    zc = z - z.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(zc, compute_uv=False)
    energy = singular ** 2
    total = float(energy.sum())
    if total <= 1e-12:
        return 0.0
    p = energy / total
    return float(1.0 / np.sum(p ** 2))


def _predictor_command_sensitivity(
    jepa: TSJEPA,
    z: np.ndarray,
    normalizer: Any,
    device: torch.device,
    *,
    n: int = 32,
) -> dict[str, float]:
    n = min(int(n), int(z.shape[0]))
    z_t = torch.from_numpy(np.asarray(z[:n], dtype=np.float32)).to(device)
    kp = int(jepa.kp)

    def _pred(u_phys: float) -> np.ndarray:
        u_norm = normalizer.normalize(np.full((n, kp), u_phys, dtype=np.float32))
        with torch.no_grad():
            return jepa.predict(z_t, torch.from_numpy(u_norm).to(device)).cpu().numpy()

    zp = _pred(20.0)
    zn = _pred(-20.0)
    l2 = np.linalg.norm(zp - zn, axis=-1).mean()
    denom = np.clip(np.linalg.norm(zp, axis=-1) * np.linalg.norm(zn, axis=-1), 1e-12, None)
    cos = float(np.mean(np.sum(zp * zn, axis=-1) / denom))
    return {
        "mean_l2_plus20_vs_minus20": float(l2),
        "mean_cosine_plus20_vs_minus20": cos,
    }


class _HoldOnMiss:
    miss_behavior = "hold_last_command"

    def __init__(self, inner: FrozenRuntimeController) -> None:
        self.inner = inner
        self.last_force = 0.0

    def reset_episode(self) -> None:
        self.inner.reset_episode()
        self.last_force = 0.0

    def step(self, frame, packet_received: bool, plant_state=None) -> float:
        if packet_received:
            self.last_force = float(self.inner.step(frame, True, plant_state))
            return self.last_force
        if frame is not None:
            self.inner.observe_frame(frame)
        return self.last_force


class _ZeroOnMiss(_HoldOnMiss):
    miss_behavior = "zero_action"

    def step(self, frame, packet_received: bool, plant_state=None) -> float:
        if packet_received:
            self.last_force = float(self.inner.step(frame, True, plant_state))
            return self.last_force
        if frame is not None:
            self.inner.observe_frame(frame)
        return 0.0


def _periodic_receive_mask(steps: int, kp: int) -> list[bool]:
    return [((t % max(int(kp), 1)) == 0) for t in range(int(steps))]


def evaluate_jepa_working_gates(
    config: dict[str, Any],
    *,
    jepa_ckpt: Path,
    data_root: Path,
    device: torch.device,
    max_samples: int = 512,
) -> dict[str, Any]:
    payload = torch.load(jepa_ckpt, map_location=device, weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(payload["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    normalizer = load_command_normalizer(config, data_root=data_root)
    dataset = TrajectoryDataset(
        data_root / "trajectories" / "jepa" / "test",
        config,
        normalizer,
        training=False,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    z_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            z = jepa.encode_context(batch["context"].to(device)).cpu().numpy()
            z_list.append(z)
            if sum(x.shape[0] for x in z_list) >= int(max_samples):
                break
    z_all = np.concatenate(z_list, axis=0)[: int(max_samples)]
    rank = _effective_rank(z_all)
    pred = _predictor_command_sensitivity(jepa, z_all, normalizer, device)
    gates = config.get("evaluation", {}).get("working_gates", {})
    rank_min = float(gates.get("encoder_effective_rank_min", 8.0))
    cos_max = float(gates.get("predictor_cosine_plus20_minus20_max", 0.95))
    l2_min = float(gates.get("predictor_l2_plus20_minus20_min", 0.2))
    rank_ok = rank > rank_min
    pred_ok = (
        pred["mean_cosine_plus20_vs_minus20"] < cos_max
        and pred["mean_l2_plus20_vs_minus20"] > l2_min
    )
    return {
        "num_samples": int(z_all.shape[0]),
        "effective_rank": rank,
        "predictor_command_sensitivity": pred,
        "rank_pass": bool(rank_ok),
        "predictor_pass": bool(pred_ok),
        "pass": bool(rank_ok and pred_ok),
    }


def nmae_label_source_from_actor_payload(payload: dict[str, Any] | None) -> str:
    """DP-teacher NMAE is the BC gate. LQR-DAgger checkpoints must not use it."""
    if not isinstance(payload, dict):
        return "dp_teacher"
    if str(payload.get("method", "")).lower() != "dagger":
        return "dp_teacher"
    expert = payload.get("expert")
    if expert is None:
        dagger = (payload.get("config") or {}).get("semantic_actor", {}).get("dagger") or {}
        expert = dagger.get("expert")
    if str(expert).lower() == "lqr":
        return "lqr"
    return "dp_teacher"


def _load_actor_payload(actor_ckpt: Path | None) -> dict[str, Any] | None:
    if actor_ckpt is None:
        return None
    path = Path(actor_ckpt)
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload if isinstance(payload, dict) else None


def evaluate_actor_working_gates(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    data_root: Path,
    *,
    actor_ckpt: Path | None = None,
) -> dict[str, Any]:
    payload = _load_actor_payload(actor_ckpt)
    labels = nmae_label_source_from_actor_payload(payload)
    nmae = evaluate_actor_nmae(config, controller, data_root=data_root, command_labels=labels)
    require = bool(
        config.get("evaluation", {}).get("working_gates", {}).get("actor_must_beat_mean_command", True)
    )
    actor_ok = bool(nmae.get("beats_mean_command_baseline"))
    out: dict[str, Any] = {
        "actor_nmae": nmae,
        "nmae_command_labels": labels,
        "pass": bool(actor_ok) if require else True,
    }
    if labels == "lqr":
        out["note"] = (
            "Gated NMAE is vs discrete LQR(state) on actor-test frames, not DP-teacher "
            "commands. A DAgger actor trained on LQR is not expected to beat the DP "
            "mean-command baseline; that number is not a working gate for this checkpoint."
        )
    return out


def evaluate_closed_loop_working_gates(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
) -> dict[str, Any]:
    gates = config.get("evaluation", {}).get("working_gates", {})
    require = bool(gates.get("closed_loop_must_beat_hold_and_zero", True))
    seeds = [int(s) for s in gates.get("closed_loop_seeds", [100, 101, 102])]
    steps = int(config["simulation"]["trajectory_steps"])
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    mask = _periodic_receive_mask(steps, kp)

    jepa_scores: list[float] = []
    hold_scores: list[float] = []
    zero_scores: list[float] = []
    for seed in seeds:
        jepa_scores.append(
            float(
                evaluate_closed_loop(
                    config, controller, steps=steps, seed=seed, packet_receive_mask=mask
                )["mean_control_score"]
            )
        )
        hold = _HoldOnMiss(controller)
        hold_scores.append(
            float(
                evaluate_closed_loop(
                    config, hold, steps=steps, seed=seed, packet_receive_mask=mask
                )["mean_control_score"]
            )
        )
        zero = _ZeroOnMiss(controller)
        zero_scores.append(
            float(
                evaluate_closed_loop(
                    config, zero, steps=steps, seed=seed, packet_receive_mask=mask
                )["mean_control_score"]
            )
        )

    jepa_mean = float(np.mean(jepa_scores))
    hold_mean = float(np.mean(hold_scores))
    zero_mean = float(np.mean(zero_scores))
    loop_ok = jepa_mean > hold_mean and jepa_mean > zero_mean
    return {
        "seeds": seeds,
        "receive_every_kp": kp,
        "jepa_mean_control_score": jepa_mean,
        "hold_last_mean_control_score": hold_mean,
        "zero_action_mean_control_score": zero_mean,
        "pass": bool(loop_ok) if require else True,
    }


def run_working_gates(
    config: dict[str, Any],
    *,
    jepa_ckpt: Path | None = None,
    actor_ckpt: Path | None = None,
    device: torch.device | None = None,
    data_root: Path | None = None,
    include_actor: bool = True,
    include_closed_loop: bool = True,
) -> dict[str, Any]:
    device = select_device(device)
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    runs = project_root(config) / config["paths"]["runs_root"]
    jepa_ckpt = resolve_run_checkpoint(
        runs,
        jepa_run_dirname(config),
        explicit=jepa_ckpt,
        seed=0,
    )
    jepa_report = evaluate_jepa_working_gates(
        config, jepa_ckpt=jepa_ckpt, data_root=root, device=device
    )
    report: dict[str, Any] = {
        "jepa_checkpoint": str(jepa_ckpt),
        "jepa": jepa_report,
        "actor": None,
        "closed_loop": None,
        "pass": bool(jepa_report["pass"]),
    }
    if not include_actor:
        return report

    actor_ckpt = resolve_run_checkpoint(
        runs,
        actor_run_dirname(config),
        explicit=actor_ckpt,
        seed=0,
    )
    controller = FrozenRuntimeController.from_checkpoints(
        config, jepa_ckpt, actor_ckpt, device=device
    )
    actor_report = evaluate_actor_working_gates(
        config, controller, root, actor_ckpt=actor_ckpt
    )
    report["actor_checkpoint"] = str(actor_ckpt)
    report["actor"] = actor_report
    report["pass"] = bool(report["pass"] and actor_report["pass"])
    if include_closed_loop:
        loop_report = evaluate_closed_loop_working_gates(config, controller)
        report["closed_loop"] = loop_report
        report["pass"] = bool(report["pass"] and loop_report["pass"])
    return report

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, project_root
from ts_jepa.data.datasets import TrajectoryDataset, load_command_normalizer
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv
from ts_jepa.evaluation.metrics import (
    communication_bits_embedding,
    communication_bits_rgb,
    control_score,
    nmae,
    summarize_scores,
)
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.models.predictor_command_resolution import (
    load_predictor_command_resolution,
    select_predictor_conditioning_commands,
)
from ts_jepa.wireless.scheduler import ChannelAwareScheduler


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


@torch.no_grad()
def evaluate_embedding_tsne(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    data_root: Path | None = None,
    max_samples: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Plan §16.1: t-SNE diagnostic on context-encoder embeddings from JEPA test trajectories.
    """
    from sklearn.manifold import TSNE

    root = data_root or (project_root(config) / config["paths"]["data_root"])
    normalizer = load_command_normalizer(config, data_root=root)
    test_dir = root / "trajectories" / "jepa" / "test"
    dataset = TrajectoryDataset(test_dir, config, normalizer, training=False)
    limit = int(max_samples or config.get("evaluation", {}).get("tsne_max_samples", 500))
    device = controller.device
    jepa = controller.jepa

    embeddings: list[np.ndarray] = []
    cart_positions: list[float] = []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if len(indices) > limit:
        indices = rng.choice(indices, size=limit, replace=False)
    for idx in indices:
        sample = dataset[int(idx)]
        context = sample["context"].unsqueeze(0).to(device)
        z = jepa.encode_context(context).cpu().numpy().reshape(-1)
        embeddings.append(z)
        file_idx, time_index = dataset.index_map[int(idx)]
        with np.load(dataset.files[file_idx]) as data:
            cart_positions.append(float(np.asarray(data["states"])[time_index, 0]))

    if not embeddings:
        return {
            "num_samples": 0,
            "coords": [],
            "cart_positions": [],
            "perplexity": None,
            "seed": seed,
            "split": "jepa_test_untouched",
        }

    matrix = np.stack(embeddings, axis=0)
    perplexity = float(min(30.0, max(5.0, (matrix.shape[0] - 1) / 3.0)))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(matrix)
    return {
        "num_samples": int(matrix.shape[0]),
        "coords": coords.tolist(),
        "cart_positions": cart_positions,
        "perplexity": perplexity,
        "seed": seed,
        "split": "jepa_test_untouched",
    }


def validate_baseline(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Plan §16: mandatory baseline checks before wireless evaluation.
    Records pass/fail per check; wireless is allowed only when all checks pass.
    """
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    nmae_report = report.get("prediction_horizon_nmae") or {}
    control = report.get("control") or {}
    comm = report.get("communication_bits") or {}
    tsne_report = report.get("embedding_tsne") or {}

    horizon_keys = {str(h) for h in range(1, kp + 1)}
    nmae_by_h = nmae_report.get("nmae_by_horizon") or {}
    checks = {
        "embedding_tsne": int(tsne_report.get("num_samples") or 0) > 0,
        "nmae_finite": _is_finite(nmae_report.get("nmae")),
        "horizon_nmae_complete": horizon_keys.issubset(set(nmae_by_h.keys()))
        and all(_is_finite(nmae_by_h[k]) for k in horizon_keys),
        "control_score_finite": _is_finite(control.get("mean")),
        "communication_bits": _is_finite(comm.get("embedding_bits")) and _is_finite(comm.get("rgb_bits")),
        "jepa_test_loss_recorded": report.get("test_losses", {}).get("jepa", {}).get("best_test_loss") is not None,
        "actor_test_loss_recorded": report.get("test_losses", {}).get("semantic_actor", {}).get("best_test_loss")
        is not None,
    }
    passed = all(checks.values())
    resolution = load_predictor_command_resolution(config)
    return {
        "passed": passed,
        "checks": checks,
        "wireless_allowed": passed,
        "predictor_command_resolution": resolution.to_dict(),
    }


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
        init_noise=float(config["simulation"]["init_noise"]),
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


@torch.no_grad()
def evaluate_prediction_horizon_nmae(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """
    NMAE between actor(predicted embeddings) and teacher commands.

    Reports overall NMAE over Kp and per-horizon NMAE for h=1..Kp.
    Uses the untouched JEPA test trajectories. Predictor is conditioned on the
    trajectory/teacher control sequence (same conditioning as JEPA training).
    """
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    normalizer = load_command_normalizer(config, data_root=root)
    test_dir = root / "trajectories" / "jepa" / "test"
    dataset = TrajectoryDataset(test_dir, config, normalizer, training=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    device = controller.device
    jepa = controller.jepa
    actor = controller.actor
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    resolution = jepa.command_resolution

    pred_by_h = {h: [] for h in range(1, kp + 1)}
    tgt_by_h = {h: [] for h in range(1, kp + 1)}
    pred_all = []
    tgt_all = []

    for batch in loader:
        context = batch["context"].to(device)
        conditioning_norm = select_predictor_conditioning_commands(batch, resolution).to(device)
        teacher_phys = batch["teacher_commands"].cpu().numpy()
        z = jepa.encode_context(context)
        z_pred = jepa.predict(z, conditioning_norm)  # [B, Kp, D]
        b = z_pred.shape[0]
        u_norm = actor(z_pred.reshape(b * kp, -1)).reshape(b, kp)
        u_phys = normalizer.denormalize(u_norm.cpu().numpy())
        pred_all.append(u_phys.reshape(-1))
        tgt_all.append(teacher_phys.reshape(-1))
        for h in range(1, kp + 1):
            pred_by_h[h].append(u_phys[:, h - 1].reshape(-1))
            tgt_by_h[h].append(teacher_phys[:, h - 1].reshape(-1))

    if not pred_all:
        return {
            "nmae": float("nan"),
            "nmae_by_horizon": {},
            "kp": kp,
            "split": "jepa_test",
            "num_values": 0,
        }

    nmae_by_horizon = {
        str(h): nmae(np.concatenate(pred_by_h[h]), np.concatenate(tgt_by_h[h])) for h in range(1, kp + 1)
    }
    pred = np.concatenate(pred_all)
    tgt = np.concatenate(tgt_all)
    return {
        "nmae": nmae(pred, tgt),
        "nmae_by_horizon": nmae_by_horizon,
        "kp": kp,
        "split": "jepa_test_untouched",
        "num_values": int(pred.size),
        "predictor_command_resolution": resolution.to_dict(),
    }


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_seed_metrics(family_dir: Path) -> list[dict[str, Any]]:
    """Load runs/<family>/seed_*/metrics.json when present (single-seed / per-seed layout)."""
    if not family_dir.exists():
        return []
    results: list[dict[str, Any]] = []
    for metrics_path in sorted(family_dir.glob("seed_*/metrics.json")):
        payload = _load_json_if_exists(metrics_path)
        if payload is None:
            continue
        seed_name = metrics_path.parent.name  # seed_0
        seed = payload.get("seed")
        if seed is None and seed_name.startswith("seed_"):
            try:
                seed = int(seed_name.split("_", 1)[1])
            except ValueError:
                seed = None
        results.append(
            {
                "seed": seed,
                "best_val": payload.get("best_val"),
                "best_epoch": payload.get("best_epoch"),
                "test_loss": payload.get("test_loss"),
                "checkpoint": str(metrics_path.parent / "best.pt"),
                "metrics_path": str(metrics_path),
            }
        )
    return results


def _test_losses_from_runs(runs_root: Path, family: str, *, split: str) -> dict[str, Any]:
    """
    Prefer 5-seed repetition_summary.json; fall back to seed_*/metrics.json for prelim runs.
    """
    family_dir = runs_root / family
    summary = _load_json_if_exists(family_dir / "repetition_summary.json")
    if summary is not None:
        return {
            "best_seed": summary.get("best_seed"),
            "best_val_loss": summary.get("best_val"),
            "best_test_loss": summary.get("best_test_loss"),
            "seed_results": summary.get("seed_results"),
            "split": split,
            "source": "repetition_summary",
        }

    seed_results = _collect_seed_metrics(family_dir)
    if not seed_results:
        return {
            "best_seed": None,
            "best_val_loss": None,
            "best_test_loss": None,
            "seed_results": None,
            "split": split,
            "source": None,
        }

    # Match repetition protocol: select by best validation loss when available.
    ranked = [r for r in seed_results if r.get("best_val") is not None]
    best = min(ranked, key=lambda r: float(r["best_val"])) if ranked else seed_results[0]
    return {
        "best_seed": best.get("seed"),
        "best_val_loss": best.get("best_val"),
        "best_test_loss": best.get("test_loss"),
        "seed_results": seed_results,
        "split": split,
        "source": "seed_metrics",
    }


def baseline_report(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    data_root: Path | None = None,
    *,
    include_wireless: bool = False,
) -> dict[str, Any]:
    """
    Paper-faithful baseline evaluation (plan §16).

    Wireless scheduling (plan §17) is excluded by default until baseline validation passes.
    Pass include_wireless=True only after validate_baseline()['passed'] is True.
    """
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
    nmae_report = evaluate_prediction_horizon_nmae(config, controller, data_root=data_root)
    tsne_report = evaluate_embedding_tsne(config, controller, data_root=data_root)

    runs_root = project_root(config) / config["paths"]["runs_root"]
    test_losses = {
        "jepa": _test_losses_from_runs(
            runs_root, jepa_run_dirname(config), split="jepa_test_untouched"
        ),
        "semantic_actor": _test_losses_from_runs(
            runs_root, actor_run_dirname(config), split="actor_test_untouched"
        ),
    }

    report: dict[str, Any] = {
        "control": summarize_scores(scores),
        "prediction_horizon_nmae": nmae_report,
        "embedding_tsne": tsne_report,
        "test_losses": test_losses,
        "communication_bits": bits,
        "predictor_command_resolution": load_predictor_command_resolution(config).to_dict(),
    }
    report["baseline_validation"] = validate_baseline(report, config)

    wireless: dict[str, Any] = {}
    if include_wireless and report["baseline_validation"]["wireless_allowed"]:
        for policy in ("channel_aware", "round_robin", "opportunistic"):
            wireless[policy] = {}
            for snr in config["wireless"]["snr_targets_db"]:
                wireless[policy][snr] = evaluate_with_scheduler(
                    config, controller, policy=policy, snr_db=float(snr), seed=7
                )
        report["wireless"] = {
            p: {
                str(snr): {
                    "mean_control_score": wireless[p][snr]["mean_control_score"],
                    "schedule_receive_rate": wireless[p][snr]["schedule_receive_rate"],
                }
                for snr in config["wireless"]["snr_targets_db"]
            }
            for p in wireless
        }
    elif include_wireless:
        report["wireless"] = {
            "skipped": True,
            "reason": "baseline_validation_failed",
            "checks": report["baseline_validation"]["checks"],
        }
    return report


def write_evaluation_artifacts(
    report: dict[str, Any],
    out_dir: Path | str,
) -> dict[str, str]:
    """
    Persist baseline_report.json plus NMAE / t-SNE / wireless plots and side JSON files.
    """
    from ts_jepa.evaluation.plotting import plot_embedding_tsne, plot_nmae_by_horizon, plot_wireless_control_scores

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    report_path = out_dir / "baseline_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    paths["baseline_report"] = str(report_path)

    validation = report.get("baseline_validation") or {}
    validation_json = out_dir / "baseline_validation.json"
    with validation_json.open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)
    paths["baseline_validation"] = str(validation_json)

    nmae_report = report.get("prediction_horizon_nmae") or {}
    nmae_json = out_dir / "nmae_report.json"
    with nmae_json.open("w", encoding="utf-8") as handle:
        json.dump(nmae_report, handle, indent=2)
    paths["nmae_report"] = str(nmae_json)

    if nmae_report.get("nmae_by_horizon"):
        paths["nmae_plot"] = str(plot_nmae_by_horizon(nmae_report, out_dir / "nmae_by_horizon.png"))

    tsne_report = report.get("embedding_tsne") or {}
    tsne_json = out_dir / "embedding_tsne.json"
    with tsne_json.open("w", encoding="utf-8") as handle:
        json.dump(tsne_report, handle, indent=2)
    paths["embedding_tsne"] = str(tsne_json)
    if int(tsne_report.get("num_samples") or 0) > 0:
        paths["embedding_tsne_plot"] = str(
            plot_embedding_tsne(tsne_report, out_dir / "embedding_tsne.png")
        )

    wireless = report.get("wireless") or {}
    if wireless and not wireless.get("skipped"):
        wireless_json = out_dir / "wireless_report.json"
        with wireless_json.open("w", encoding="utf-8") as handle:
            json.dump(wireless, handle, indent=2)
        paths["wireless_report"] = str(wireless_json)
        paths["wireless_plot"] = str(
            plot_wireless_control_scores(wireless, out_dir / "wireless_control_scores.png")
        )

    control_json = out_dir / "closed_loop_report.json"
    with control_json.open("w", encoding="utf-8") as handle:
        json.dump(report.get("control") or {}, handle, indent=2)
    paths["closed_loop_report"] = str(control_json)
    return paths

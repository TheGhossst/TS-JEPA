from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, project_root
from ts_jepa.data.trajectory_generator import build_env_and_teacher
from ts_jepa.data.datasets import TrajectoryDataset, load_command_normalizer
from ts_jepa.env.factory import build_inverted_cartpole_env
from ts_jepa.evaluation.metrics import (
    communication_reduction_report,
    consecutive_frame_mape,
    control_score,
    nmae,
    nmae_report_fields,
    physical_force_range_n,
    summarize_scores,
)
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.losses.jepa_loss import jepa_cosine_similarity
from ts_jepa.models.predictor_command_resolution import (
    load_predictor_command_resolution,
    select_predictor_conditioning_commands,
)
from ts_jepa.plan.baseline_validation import (
    PLAN_BASELINE_VALIDATION,
    STATUS_FAIL,
    STATUS_INCOMPLETE,
    STATUS_PASS,
)
from ts_jepa.wireless.channel import WirelessChannelModel
from ts_jepa.wireless.scheduler import ChannelAwareScheduler


def _observation_stride(config: dict[str, Any]) -> int:
    return max(1, int(config["simulation"].get("observation_stride_steps", 1)))


def control_loop_stride(config: dict[str, Any]) -> int:
    """Physics ticks the actuator is held per closed-loop decision.

    Dataset sampling stays ``simulation.observation_stride_steps`` (paper: 1 ms).
    ``evaluation.control_hold_steps``, if set, is the control period only so a
    100-step Eq. (28) rollout can last ~2 s (working overlay) instead of 100 ms.
    """
    hold = (config.get("evaluation") or {}).get("control_hold_steps")
    if hold is not None:
        return max(1, int(hold))
    return _observation_stride(config)


def closed_loop_eval_seeds(config: dict[str, Any], *, base: int = 100) -> list[int]:
    """Seeds used by control_performance and closed-loop stability."""
    reps = max(1, int(config["evaluation"]["repetitions"]))
    return [int(base) + r for r in range(reps)]


def receive_every_kp_mask(steps: int, kp: int) -> list[bool]:
    """Receive a packet every Kp steps (paper prediction-horizon miss pattern)."""
    period = max(1, int(kp))
    return [((t % period) == 0) for t in range(int(steps))]


def stability_receive_period(config: dict[str, Any]) -> int:
    """Packet-receive period for the §15 lossy stability check.

    Defaults to JEPA ``Kp``. Overlay ``evaluation.stability_receive_every`` when
    receive-every-Kp is past the plant's open-loop cliff (dp_fixed: Kp=15 is
    300 ms and even oracle Observer-LQR scores ~0).
    """
    kp = max(1, int(config["ts_jepa"]["prediction_horizon"]["Kp"]))
    override = (config.get("evaluation") or {}).get("stability_receive_every")
    if override is None:
        return kp
    return max(1, int(override))


def runtime_encoder(controller: Any) -> FrozenRuntimeController:
    """JEPA+actor pair for offline metrics when closed-loop uses Observer-LQR."""
    if isinstance(controller, FrozenRuntimeController):
        return controller
    encoder = getattr(controller, "encoder", None)
    if isinstance(encoder, FrozenRuntimeController):
        return encoder
    raise TypeError(
        f"cannot unwrap FrozenRuntimeController from {type(controller).__name__}"
    )


def _apply_held_force(env, force: float, stride: int):
    state = env.state
    for _ in range(int(stride)):
        state, _ = env.step(force)
    return state


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _section_present(report: dict[str, Any], key: str) -> bool:
    value = report.get(key)
    return isinstance(value, dict) and len(value) > 0


def _finite_coords(coords: Any, expected_n: int) -> bool:
    try:
        arr = np.asarray(coords, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if arr.size == 0 or arr.shape[0] != int(expected_n):
        return False
    return bool(np.isfinite(arr).all())


def _horizon_map_state(payload: Any, kp: int) -> str:
    """Return 'ok', 'missing', or 'invalid' for a 1..Kp metric map."""
    if not isinstance(payload, dict) or not payload:
        return "missing"
    keys = {str(h) for h in range(1, kp + 1)}
    got = {str(k) for k in payload.keys()}
    if not keys.issubset(got):
        return "missing"
    if not all(_is_finite(payload[k]) for k in keys):
        return "invalid"
    return "ok"


def _combine_check_status(*states: str) -> str:
    if any(s == "missing" for s in states):
        return "missing"
    if any(s == "invalid" for s in states):
        return "invalid"
    return "ok"


@torch.no_grad()
def evaluate_consecutive_frame_mape(
    config: dict[str, Any],
    data_root: Path | None = None,
    max_trajectories: int | None = None,
) -> dict[str, Any]:
    """
    Plan §15 Eq. (26): mean consecutive-frame MAPE (%) on JEPA test RGB trajectories.

    Zero-denominator pixels are skipped (IMPLEMENTATION CHOICE).
    """
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    test_dir = root / "trajectories" / "jepa" / "test"
    files = sorted(test_dir.glob("*.npz"))
    if max_trajectories is not None:
        files = files[: int(max_trajectories)]
    pair_values: list[float] = []
    for file_path in files:
        with np.load(file_path) as data:
            frames = np.asarray(data["frames"])
        for t in range(1, int(frames.shape[0])):
            value = consecutive_frame_mape(frames[t], frames[t - 1])
            if np.isfinite(value):
                pair_values.append(float(value))
    mean_mape = float(np.mean(pair_values)) if pair_values else float("nan")
    return {
        "mean_mape_percent": mean_mape,
        "num_pairs": int(len(pair_values)),
        "num_trajectories": int(len(files)),
        "split": "jepa_test_untouched",
        "equation": "26",
        "plan_section": "15",
        "zero_denominator": "skip",
        "units": "percent",
    }


def fig4_sampling_intervals_ms(config: dict[str, Any]) -> list[float]:
    """Fig. 4 sampling-rate axis (IMPLEMENTATION CHOICE; not in plan.md)."""
    exp = config.get("experiments", {})
    if "fig4_sampling_interval_ms" not in exp:
        raise ValueError(
            "experiments.fig4_sampling_interval_ms is required as an implementation choice "
            "(plan §18); Fig. 4 rates are NOT SPECIFIED by the paper"
        )
    values = [float(x) for x in exp["fig4_sampling_interval_ms"]]
    if len(values) < 2:
        raise ValueError("experiments.fig4_sampling_interval_ms must list at least two rates (Fig. 4)")
    return values


def evaluate_fig4_sampling_rate_mape(
    config: dict[str, Any],
    data_root: Path | None = None,
    max_trajectories: int | None = None,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Plan §15 / Fig. 4: consecutive-frame MAPE (Eq. 26) at different sampling rates,
    with and without augmentation.

    Does **not** change simulation τ_o (plan §3). Stored 1 ms frames are subsampled
    at integer strides matching ``experiments.fig4_sampling_interval_ms`` (IC).
    Augmentation is color jitter + color drop only (no ImageNet normalize / resize),
    so MAPE stays on RGB intensities.
    """
    from ts_jepa.preprocessing.pipeline import PreprocessPipeline

    root = data_root or (project_root(config) / config["paths"]["data_root"])
    test_dir = root / "trajectories" / "jepa" / "test"
    files = sorted(test_dir.glob("*.npz"))
    if max_trajectories is not None:
        files = files[: int(max_trajectories)]
    tau_ms = float(config["simulation"]["sampling_interval_ms"])
    intervals = fig4_sampling_intervals_ms(config)
    pipe = PreprocessPipeline(config, training=True)
    rng = np.random.default_rng(seed)

    by_rate: dict[str, dict[str, Any]] = {}
    for interval in intervals:
        stride_f = interval / tau_ms
        stride = int(round(stride_f))
        if stride < 1 or abs(stride_f - stride) > 1e-9:
            raise ValueError(
                f"Fig. 4 sampling interval {interval} ms is not an integer multiple of "
                f"τ_o={tau_ms} ms; choose IC rates that subsample stored frames"
            )
        raw_vals: list[float] = []
        aug_vals: list[float] = []
        for file_path in files:
            with np.load(file_path) as data:
                frames = np.asarray(data["frames"])
            sampled = frames[::stride]
            if sampled.shape[0] < 2:
                continue
            for t in range(1, int(sampled.shape[0])):
                raw = consecutive_frame_mape(sampled[t], sampled[t - 1])
                if np.isfinite(raw):
                    raw_vals.append(float(raw))
                prev = pipe._decode_frame(sampled[t - 1])
                curr = pipe._decode_frame(sampled[t])
                prev_a = pipe._augment(prev, rng) * 255.0
                curr_a = pipe._augment(curr, rng) * 255.0
                aug = consecutive_frame_mape(curr_a.numpy(), prev_a.numpy())
                if np.isfinite(aug):
                    aug_vals.append(float(aug))
        by_rate[str(interval)] = {
            "sampling_interval_ms": float(interval),
            "stride_frames": stride,
            "without_augmentation": {
                "mean_mape_percent": float(np.mean(raw_vals)) if raw_vals else float("nan"),
                "num_pairs": int(len(raw_vals)),
            },
            "with_augmentation": {
                "mean_mape_percent": float(np.mean(aug_vals)) if aug_vals else float("nan"),
                "num_pairs": int(len(aug_vals)),
            },
        }
    return {
        "figure": "4",
        "equation": "26",
        "plan_section": "15",
        "sampling_interval_ms": intervals,
        "sampling_interval_source": "implementation_choice",
        "split": "jepa_test_untouched",
        "zero_denominator": "skip",
        "units": "percent",
        "by_sampling_interval_ms": by_rate,
        "note": (
            "Main simulation remains τ_o=1 ms (plan §3). Fig. 4 rates are IC subsample "
            "strides of stored frames; augmentation is jitter+drop on RGB (not encoder resize)."
        ),
    }


@torch.no_grad()
def evaluate_embedding_tsne(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    data_root: Path | None = None,
    max_samples: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Plan §15: encoder quality t-SNE on context-encoder embeddings from JEPA test trajectories.
    """
    from sklearn.manifold import TSNE

    root = data_root or (project_root(config) / config["paths"]["data_root"])
    normalizer = load_command_normalizer(config, data_root=root)
    test_dir = root / "trajectories" / "jepa" / "test"
    dataset = TrajectoryDataset(test_dir, config, normalizer, training=False)
    limit = int(max_samples or config.get("evaluation", {}).get("tsne_max_samples", 500))
    encoder = runtime_encoder(controller)
    device = encoder.device
    jepa = encoder.jepa

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
            "plan_section": "15",
            "figure": "5",
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
        "plan_section": "15",
        "figure": "5",
    }


def validate_baseline(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Plan §15 paper metrics, plus a closed-loop packet-loss check of plan §14 inference.

    Missing required metrics → INCOMPLETE.
    Present NaN / Inf / structurally invalid metrics → FAIL.
    No paper-performance threshold is invented, but a zero control score is invalid:
    it proves that no evaluated state met Eq. (28). The [0.74, 1.0] band is for
    scalability plots only and is not used as a pass/fail gate.
    """
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    force_range = physical_force_range_n(config)
    reasons: dict[str, str] = {}
    check_states: dict[str, str] = {}

    def record(name: str, state: str, reason: str) -> None:
        check_states[name] = state
        if state != "ok":
            reasons[name] = reason

    # Required report artifacts / sections
    missing_keys = [
        key for key in PLAN_BASELINE_VALIDATION["required_report_keys"] if not _section_present(report, key)
    ]
    artifacts_state = "missing" if missing_keys else "ok"
    record(
        "required_artifacts",
        artifacts_state,
        f"missing report sections: {missing_keys}" if missing_keys else "",
    )

    tsne_report = report.get("embedding_tsne") if isinstance(report.get("embedding_tsne"), dict) else None
    if tsne_report is None:
        record("embedding_quality_tsne", "missing", "embedding_tsne section missing")
    else:
        n_samples = tsne_report.get("num_samples")
        coords = tsne_report.get("coords")
        if n_samples is None or coords is None:
            record("embedding_quality_tsne", "missing", "t-SNE num_samples or coords missing")
        else:
            try:
                n_int = int(n_samples)
            except (TypeError, ValueError):
                n_int = -1
            if n_int <= 0:
                record("embedding_quality_tsne", "invalid", "t-SNE num_samples must be > 0")
            elif not _is_finite(n_samples):
                record("embedding_quality_tsne", "invalid", "t-SNE num_samples is NaN/Inf")
            elif not _finite_coords(coords, n_int):
                record("embedding_quality_tsne", "invalid", "t-SNE coords missing, mismatched, or non-finite")
            else:
                record("embedding_quality_tsne", "ok", "")

    mape_report = report.get("consecutive_frame_mape") if isinstance(report.get("consecutive_frame_mape"), dict) else None
    if mape_report is None:
        record("consecutive_frame_mape", "missing", "consecutive_frame_mape section missing")
    elif "mean_mape_percent" not in mape_report:
        record("consecutive_frame_mape", "missing", "consecutive_frame_mape.mean_mape_percent missing")
    elif not _is_finite(mape_report.get("mean_mape_percent")):
        record("consecutive_frame_mape", "invalid", "consecutive-frame MAPE is NaN/Inf")
    elif int(mape_report.get("num_pairs") or 0) <= 0:
        record("consecutive_frame_mape", "invalid", "consecutive-frame MAPE has no pairs")
    else:
        record("consecutive_frame_mape", "ok", "")

    fig4 = report.get("fig4_mape") if isinstance(report.get("fig4_mape"), dict) else None
    if fig4 is None:
        record("fig4_sampling_rate_mape", "missing", "fig4_mape section missing")
    else:
        by_rate = fig4.get("by_sampling_interval_ms")
        if not isinstance(by_rate, dict) or len(by_rate) < 2:
            record(
                "fig4_sampling_rate_mape",
                "missing",
                "fig4_mape.by_sampling_interval_ms must cover at least two rates",
            )
        else:
            ok = True
            reason = ""
            for key, payload in by_rate.items():
                if not isinstance(payload, dict):
                    ok = False
                    reason = f"Fig. 4 rate {key!r} is not a mapping"
                    break
                for arm in ("without_augmentation", "with_augmentation"):
                    arm_payload = payload.get(arm)
                    if not isinstance(arm_payload, dict) or "mean_mape_percent" not in arm_payload:
                        ok = False
                        reason = f"Fig. 4 rate {key} missing {arm}.mean_mape_percent"
                        break
                    if not _is_finite(arm_payload.get("mean_mape_percent")):
                        ok = False
                        reason = f"Fig. 4 {arm} MAPE at {key} ms is NaN/Inf"
                        break
                    if int(arm_payload.get("num_pairs") or 0) <= 0:
                        ok = False
                        reason = f"Fig. 4 {arm} MAPE at {key} ms has no pairs"
                        break
                if not ok:
                    break
            record("fig4_sampling_rate_mape", "ok" if ok else "invalid", reason)

    actor_nmae = report.get("actor_nmae") if isinstance(report.get("actor_nmae"), dict) else None
    if actor_nmae is None:
        record("actor_prediction_nmae", "missing", "actor_nmae section missing")
    elif "nmae" not in actor_nmae:
        record("actor_prediction_nmae", "missing", "actor_nmae.nmae missing")
    elif not _is_finite(actor_nmae.get("nmae")):
        record("actor_prediction_nmae", "invalid", "actor NMAE is NaN/Inf")
    elif int(actor_nmae.get("num_values") or 0) <= 0:
        record("actor_prediction_nmae", "invalid", "actor NMAE has no values")
    else:
        denom = actor_nmae.get("nmae_denominator")
        if denom is not None and (not _is_finite(denom) or abs(float(denom) - force_range) > 1e-9):
            record(
                "actor_prediction_nmae",
                "invalid",
                f"actor NMAE denominator {denom} != force range {force_range}",
            )
        else:
            record("actor_prediction_nmae", "ok", "")

    control = report.get("control") if isinstance(report.get("control"), dict) else None
    if control is None:
        record("control_performance", "missing", "control section missing")
    elif "mean" not in control:
        record("control_performance", "missing", "control.mean missing")
    elif not _is_finite(control.get("mean")):
        record("control_performance", "invalid", "control.mean is NaN/Inf")
    elif int(control.get("repetitions") or 0) <= 0:
        record("control_performance", "invalid", "control.repetitions must be > 0")
    elif float(control["mean"]) <= 0.0:
        record(
            "control_performance",
            "invalid",
            "control.mean must be > 0; zero means no evaluated state met the control criterion",
        )
    elif float(control["mean"]) > 1.0:
        record("control_performance", "invalid", "control.mean must be <= 1")
    else:
        record("control_performance", "ok", "")

    nmae_report = report.get("prediction_horizon_nmae") if isinstance(report.get("prediction_horizon_nmae"), dict) else None
    if nmae_report is None:
        record("horizon_prediction_1_to_Kp", "missing", "prediction_horizon_nmae section missing")
    else:
        nmae_state = _horizon_map_state(nmae_report.get("nmae_by_horizon"), kp)
        overall_nmae = nmae_report.get("nmae")
        overall_nmae_state = (
            "missing" if overall_nmae is None else ("ok" if _is_finite(overall_nmae) else "invalid")
        )
        combined = _combine_check_status(nmae_state, overall_nmae_state)
        if combined == "missing":
            record(
                "horizon_prediction_1_to_Kp",
                "missing",
                "command NMAE missing for some of 1..Kp (plan §15 Eq. 27)",
            )
        elif combined == "invalid":
            record(
                "horizon_prediction_1_to_Kp",
                "invalid",
                "command NMAE contains NaN/Inf",
            )
        else:
            record("horizon_prediction_1_to_Kp", "ok", "")

    comm = report.get("communication_bits") if isinstance(report.get("communication_bits"), dict) else None
    if comm is None:
        record("communication_reduction", "missing", "communication_bits section missing")
    else:
        needed = ("rgb_bits", "embedding_bits", "reduction_ratio")
        if any(k not in comm for k in needed):
            record("communication_reduction", "missing", "communication bit fields missing")
        elif not all(_is_finite(comm[k]) for k in needed):
            record("communication_reduction", "invalid", "communication bits contain NaN/Inf")
        elif float(comm["rgb_bits"]) <= 0 or float(comm["embedding_bits"]) <= 0:
            record("communication_reduction", "invalid", "communication bit counts must be positive")
        elif float(comm["reduction_ratio"]) <= 1.0:
            record("communication_reduction", "invalid", "reduction_ratio must be > 1 (latent vs RGB)")
        else:
            record("communication_reduction", "ok", "")

    stability = report.get("stability") if isinstance(report.get("stability"), dict) else None
    if stability is None:
        record("closed_loop_stability", "missing", "stability section missing")
    elif "passed" not in stability:
        record("closed_loop_stability", "missing", "stability.passed missing")
    else:
        full = stability.get("full_receive") if isinstance(stability.get("full_receive"), dict) else {}
        lossy = stability.get("intermittent_loss") if isinstance(stability.get("intermittent_loss"), dict) else {}
        score_keys = (full.get("mean_control_score"), lossy.get("mean_control_score"))
        if not full or not lossy or any(v is None for v in score_keys):
            record("closed_loop_stability", "missing", "full/lossy stability control scores missing")
        elif any(not _is_finite(v) for v in score_keys):
            record("closed_loop_stability", "invalid", "stability control scores are NaN/Inf")
        elif any(float(v) <= 0.0 for v in score_keys):
            record(
                "closed_loop_stability",
                "invalid",
                "stability control scores must be > 0; zero means no controlled states",
            )
        elif any(float(v) > 1.0 for v in score_keys):
            record("closed_loop_stability", "invalid", "stability control scores must be <= 1")
        elif stability.get("passed") is not True:
            record("closed_loop_stability", "invalid", "closed-loop stability check did not pass")
        else:
            record("closed_loop_stability", "ok", "")

    def _loss_state(family: str) -> str:
        losses = report.get("test_losses")
        if not isinstance(losses, dict) or family not in losses:
            return "missing"
        value = (losses.get(family) or {}).get("best_test_loss")
        if value is None:
            return "missing"
        return "ok" if _is_finite(value) else "invalid"

    jepa_loss_state = _loss_state("jepa")
    actor_loss_state = _loss_state("semantic_actor")
    record(
        "jepa_test_loss_recorded",
        jepa_loss_state,
        "jepa best_test_loss missing" if jepa_loss_state != "ok" else "",
    )
    record(
        "actor_test_loss_recorded",
        actor_loss_state,
        "semantic_actor best_test_loss missing" if actor_loss_state != "ok" else "",
    )

    checks = {name: state == "ok" for name, state in check_states.items()}
    plan_check_keys = PLAN_BASELINE_VALIDATION["checks"]
    plan_states = [check_states[k] for k in plan_check_keys]
    plan_passed = all(s == "ok" for s in plan_states)
    if any(s == "missing" for s in plan_states):
        plan_status = STATUS_INCOMPLETE
    elif any(s == "invalid" for s in plan_states):
        plan_status = STATUS_FAIL
    else:
        plan_status = STATUS_PASS

    all_states = list(check_states.values())
    if any(s == "missing" for s in all_states):
        status = STATUS_INCOMPLETE
    elif any(s == "invalid" for s in all_states):
        status = STATUS_FAIL
    else:
        status = STATUS_PASS

    resolution = load_predictor_command_resolution(config)
    return {
        "status": status,
        "plan_section_15_status": plan_status,
        "plan_section_16_status": plan_status,
        "passed": status == STATUS_PASS,
        "plan_section_15_passed": plan_passed,
        "plan_section_16_passed": plan_passed,
        "checks": checks,
        "check_states": check_states,
        "reasons": reasons,
        "wireless_allowed": plan_passed and PLAN_BASELINE_VALIDATION["wireless_requires_all_pass"],
        "required_checks": list(plan_check_keys),
        "required_artifacts": list(PLAN_BASELINE_VALIDATION["required_report_keys"]),
        "predictor_command_resolution": resolution.to_dict(),
    }


@torch.no_grad()
def evaluate_actor_nmae(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    data_root: Path | None = None,
    *,
    command_labels: str = "dp_teacher",
) -> dict[str, Any]:
    """
    Actor command NMAE on encoded (received) embeddings vs action labels.

    ``dp_teacher``: untouched actor-test teacher commands (plan §15 BC gate).
    ``lqr``: discrete LQR on the stored plant state at the same timestep.
    LQR labels are for working-overlay DAgger; they are not the paper BC target.
    """
    labels = str(command_labels or "dp_teacher")
    if labels not in {"dp_teacher", "lqr"}:
        raise ValueError(f"command_labels must be 'dp_teacher' or 'lqr', got {labels!r}")
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    normalizer = load_command_normalizer(config, data_root=root)
    test_dir = root / "trajectories" / "actor" / "test"
    dataset = TrajectoryDataset(test_dir, config, normalizer, training=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    encoder = runtime_encoder(controller)
    device = encoder.device
    jepa = encoder.jepa
    actor = encoder.actor
    lqr_gain = None
    if labels == "lqr":
        from ts_jepa.control.lqr import lqr_forces, lqr_gain_from_config

        lqr_gain = lqr_gain_from_config(config)

    preds: list[np.ndarray] = []
    tgts: list[np.ndarray] = []
    for batch in loader:
        context = batch["context"].to(device)
        z = jepa.encode_context(context)
        u_phys = actor(z).cpu().numpy().reshape(-1)
        preds.append(np.asarray(u_phys).reshape(-1))
        if labels == "lqr":
            if "state" not in batch:
                raise RuntimeError("LQR NMAE needs trajectory states; regenerate actor test npz with states.")
            tgts.append(
                lqr_forces(
                    lqr_gain,
                    batch["state"].numpy(),
                    float(config["simulation"]["control_min_N"]),
                    float(config["simulation"]["control_max_N"]),
                )
            )
        else:
            tgts.append(batch["teacher_commands"][:, 0].cpu().numpy().reshape(-1))

    if not preds:
        return {
            **nmae_report_fields(value=float("nan"), force_range_n=physical_force_range_n(config)),
            "num_values": 0,
            "split": "actor_test_untouched",
            "plan_section": "15",
            "command_labels": labels,
        }

    pred = np.concatenate(preds)
    tgt = np.concatenate(tgts)
    force_range = physical_force_range_n(config)
    if labels == "lqr":
        baseline_value = float(np.mean(tgt)) if tgt.size else 0.0
        baseline_source = "mean_lqr_force_on_eval_states"
    else:
        baseline_value = float(normalizer.mean)
        baseline_source = "jepa_train_command_mean_via_normalizer"
    mean_baseline_pred = np.full_like(tgt, baseline_value)
    actor_nmae_value = nmae(pred, tgt, force_range_n=force_range)
    mean_baseline_nmae = nmae(mean_baseline_pred, tgt, force_range_n=force_range)
    return {
        **nmae_report_fields(value=actor_nmae_value, force_range_n=force_range),
        "num_values": int(pred.size),
        "split": "actor_test_untouched",
        "plan_section": "15",
        "command_labels": labels,
        "mean_abs_error": float(np.mean(np.abs(pred - tgt))),
        "mean_command_baseline_nmae": float(mean_baseline_nmae),
        "mean_command_baseline_source": baseline_source,
        "beats_mean_command_baseline": bool(
            np.isfinite(actor_nmae_value)
            and np.isfinite(mean_baseline_nmae)
            and actor_nmae_value < mean_baseline_nmae
        ),
    }


def evaluate_closed_loop_stability(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    *,
    steps: int | None = None,
    seed: int | None = None,
    full_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Plan §14 inference under full receive vs packet loss.

    Control scores use Eq. (28). Both conditions are **means over the same
    seeds as control_performance** (100 + r). A single unlucky init scoring 0
    does not fail the check if the mean is > 0.

    Loss pattern is receive-every-N. N defaults to JEPA Kp. Overlays may set
    ``evaluation.stability_receive_every`` when receive-every-Kp is past the
    plant's open-loop cliff (not specified by the paper).

    ``seed`` is accepted for call-site compatibility and ignored.
    Pass ``full_runs`` to reuse control_performance rollouts.
    """
    del seed  # previously a 2-seed lottery (seed, seed+1); see docstring.
    steps = int(steps or config["simulation"]["trajectory_steps"])
    seeds = closed_loop_eval_seeds(config)
    kp = max(1, int(config["ts_jepa"]["prediction_horizon"]["Kp"]))
    period = stability_receive_period(config)
    mask = receive_every_kp_mask(steps, period)

    if full_runs is None:
        full_runs = [
            evaluate_closed_loop(config, controller, steps=steps, seed=s) for s in seeds
        ]
    if len(full_runs) != len(seeds):
        raise ValueError(
            f"full_runs length {len(full_runs)} != number of closed-loop seeds {len(seeds)}"
        )

    lossy_runs = [
        evaluate_closed_loop(
            config, controller, steps=steps, seed=s, packet_receive_mask=mask
        )
        for s in seeds
    ]

    full_scores = [float(run["mean_control_score"]) for run in full_runs]
    lossy_scores = [float(run["mean_control_score"]) for run in lossy_runs]
    mean_full = float(np.mean(full_scores)) if full_scores else 0.0
    mean_lossy = float(np.mean(lossy_scores)) if lossy_scores else 0.0

    def _forces_finite(runs: list[dict[str, Any]]) -> bool:
        return all(all(np.isfinite(run["forces"])) for run in runs)

    def _scores_finite(runs: list[dict[str, Any]]) -> bool:
        return all(all(np.isfinite(run["scores"])) for run in runs)

    forces_ok = _forces_finite(full_runs) and _forces_finite(lossy_runs)
    scores_ok = _scores_finite(full_runs) and _scores_finite(lossy_runs)
    passed = forces_ok and scores_ok and 0.0 < mean_full <= 1.0 and 0.0 < mean_lossy <= 1.0
    return {
        "passed": passed,
        "plan_section": "14",
        "seeds": seeds,
        "loss_pattern": "receive_every_n",
        "receive_every": period,
        "kp": kp,
        "control_hold_steps": int(control_loop_stride(config)),
        "full_receive": {
            "mean_control_score": mean_full,
            "per_seed": full_scores,
            "finite_forces": _forces_finite(full_runs),
        },
        "intermittent_loss": {
            "mean_control_score": mean_lossy,
            "per_seed": lossy_scores,
            "packet_receive_rate": float(np.mean(mask)) if mask else 0.0,
            "finite_forces": _forces_finite(lossy_runs),
        },
        "predictor_command_resolution": load_predictor_command_resolution(config).to_dict(),
    }


def evaluate_closed_loop(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    steps: int | None = None,
    seed: int = 0,
    packet_receive_mask: list[bool] | None = None,
) -> dict[str, Any]:
    controller.reset_episode()
    env = build_inverted_cartpole_env(config)
    steps = steps or int(config["simulation"]["trajectory_steps"])
    stride = control_loop_stride(config)
    state = env.reset(seed=seed)
    initial_state = state.copy()
    scores = []
    forces = []
    if packet_receive_mask is None:
        packet_receive_mask = [True] * steps

    for t in range(steps):
        frame = env.render(state)
        received = bool(packet_receive_mask[t])
        force = controller.step(frame, packet_received=received, plant_state=state)
        forces.append(force)
        scores.append(
            control_score(
                state,
                desired_x=float(config["simulation"]["desired_state"][0]),
                position_tol=float(config["evaluation"]["control_position_tol"]),
                angle_tol=float(config["evaluation"]["control_angle_tol"]),
            )
        )
        state = _apply_held_force(env, force, stride)

    return {
        "mean_control_score": float(np.mean(scores)),
        "forces": forces,
        "scores": scores,
        "controlled_steps": int(sum(scores)),
        "num_steps": int(steps),
        "initial_state": initial_state.tolist(),
        "final_state": state.tolist(),
        "control_hold_steps": int(stride),
    }


def _finite_or_flag(values: list[float] | np.ndarray) -> bool:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return bool(np.isfinite(arr).all())


def evaluate_closed_loop_full_information_diagnostic(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    *,
    steps: int | None = None,
    seeds: list[int] | None = None,
    teacher: Any | None = None,
) -> dict[str, Any]:
    """
    Full-information closed-loop diagnostic (no packet loss, no predictor rollout).

    Every timestep renders the true RGB frame and calls
    ``FrozenRuntimeController.step_packet_received()``. On the same pre-action
    states, compares semantic-actor force vs the DP teacher force.

    Does not affect baseline validation gates.
    """
    steps = int(steps or config["simulation"]["trajectory_steps"])
    reps = int(config["evaluation"]["repetitions"])
    if seeds is None:
        seeds = [100 + r for r in range(reps)]
    if len(seeds) != reps:
        raise ValueError(f"expected {reps} diagnostic seeds, got {len(seeds)}")

    if teacher is None:
        _, teacher = build_env_and_teacher(config)

    position_tol = float(config["evaluation"]["control_position_tol"])
    angle_tol = float(config["evaluation"]["control_angle_tol"])
    desired_x = float(config["simulation"]["desired_state"][0])

    rollouts: list[dict[str, Any]] = []
    for seed in seeds:
        controller.reset_episode()
        env = build_inverted_cartpole_env(config)
        stride = control_loop_stride(config)
        state = env.reset(seed=seed)
        initial_state = state.copy()
        visited_x = [float(state[0])]
        visited_theta = [float(state[2])]
        forces_actor: list[float] = []
        forces_teacher: list[float] = []
        abs_force_errors: list[float] = []
        scores: list[int] = []
        nan_or_inf = not _finite_or_flag(initial_state)

        for _ in range(steps):
            frame = env.render(state)
            actor_force = float(controller.step_packet_received(frame))
            teacher_force = float(teacher.act(state))
            force_error = abs(actor_force - teacher_force)

            if not _finite_or_flag([actor_force, teacher_force, force_error]):
                nan_or_inf = True
            if not _finite_or_flag(state):
                nan_or_inf = True

            forces_actor.append(actor_force)
            forces_teacher.append(teacher_force)
            abs_force_errors.append(force_error)
            scores.append(
                control_score(
                    state,
                    desired_x=desired_x,
                    position_tol=position_tol,
                    angle_tol=angle_tol,
                )
            )
            if not np.isfinite(actor_force):
                break
            state = _apply_held_force(env, actor_force, stride)
            visited_x.append(float(state[0]))
            visited_theta.append(float(state[2]))
            if not _finite_or_flag(state):
                nan_or_inf = True

        forces_arr = np.asarray(forces_actor, dtype=np.float64)
        errors_arr = np.asarray(abs_force_errors, dtype=np.float64)
        rollouts.append(
            {
                "seed": int(seed),
                "initial_state": initial_state.tolist(),
                "final_state": state.tolist(),
                "x_min": float(np.min(visited_x)),
                "x_max": float(np.max(visited_x)),
                "theta_min": float(np.min(visited_theta)),
                "theta_max": float(np.max(visited_theta)),
                "force_min": float(np.min(forces_arr)),
                "force_max": float(np.max(forces_arr)),
                "force_mean": float(np.mean(forces_arr)),
                "force_mean_abs": float(np.mean(np.abs(forces_arr))),
                "controlled_steps": int(sum(scores)),
                "mean_control_score": float(np.mean(scores)) if scores else float("nan"),
                "num_steps": int(steps),
                "nan_or_inf": bool(nan_or_inf),
                "teacher_comparison": {
                    "mean_abs_force_error": float(np.mean(errors_arr)) if errors_arr.size else float("nan"),
                    "max_abs_force_error": float(np.max(errors_arr)) if errors_arr.size else float("nan"),
                    "forces_actor": forces_actor,
                    "forces_teacher": forces_teacher,
                    "abs_force_errors": abs_force_errors,
                },
            }
        )

    control_scores = [float(r["mean_control_score"]) for r in rollouts]
    controlled_steps = [int(r["controlled_steps"]) for r in rollouts]
    return {
        "mode": "full_information_closed_loop",
        "packet_loss": False,
        "predictor_rollout": False,
        "controller_path": "step_packet_received",
        "num_rollouts": len(rollouts),
        "num_steps": steps,
        "position_tol": position_tol,
        "angle_tol": angle_tol,
        "rollouts": rollouts,
        "summary": {
            "mean_control_score": float(np.mean(control_scores)) if control_scores else float("nan"),
            "controlled_steps_mean": float(np.mean(controlled_steps)) if controlled_steps else float("nan"),
            "controlled_steps_total": int(sum(controlled_steps)),
            "mean_abs_force_error_vs_teacher": float(
                np.mean([r["teacher_comparison"]["mean_abs_force_error"] for r in rollouts])
            ),
            "max_abs_force_error_vs_teacher": float(
                np.max([r["teacher_comparison"]["max_abs_force_error"] for r in rollouts])
            ),
            "any_nan_or_inf": any(bool(r["nan_or_inf"]) for r in rollouts),
        },
    }


def evaluate_with_scheduler(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    policy: str = "channel_aware",
    snr_db: float = 10.0,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Closed loop for device 0: scheduler → InF-SH SNR/outage → packet_received → controller.

    packet_received is True only on a successful transmission (scheduled and γ >= γ_th).
    The controller chooses miss behavior: TS-JEPA predicts (plan §14); §16 conventional
    baselines must hold the last command (not Pφ).
    """
    controller.reset_episode()
    scheduler = ChannelAwareScheduler(config, policy=policy)  # type: ignore[arg-type]
    scheduler.set_snr_target(snr_db)
    rng = np.random.default_rng(seed)
    env = build_inverted_cartpole_env(config)
    steps = int(config["simulation"]["trajectory_steps"])
    stride = control_loop_stride(config)
    state = env.reset(seed=seed)
    scores: list[int] = []
    forces: list[float] = []
    scheduled: list[int] = []
    delivered: list[int] = []
    outaged: list[int] = []

    for _ in range(steps):
        decision = scheduler.schedule(rng)
        packet_received = bool(decision.successes[0])
        scheduled.append(int(decision.alphas[0]))
        delivered.append(int(decision.successes[0]))
        outaged.append(int(decision.outages[0]))
        frame = env.render(state)
        force = controller.step(frame, packet_received=packet_received, plant_state=state)
        forces.append(force)
        scores.append(
            control_score(
                state,
                desired_x=float(config["simulation"]["desired_state"][0]),
                position_tol=float(config["evaluation"]["control_position_tol"]),
                angle_tol=float(config["evaluation"]["control_angle_tol"]),
            )
        )
        state = _apply_held_force(env, force, stride)

    packet_receive_rate = float(np.mean(delivered)) if delivered else 0.0
    return {
        "mean_control_score": float(np.mean(scores)) if scores else float("nan"),
        "forces": forces,
        "scores": scores,
        "policy": policy,
        "snr_db": snr_db,
        "schedule_rate": float(np.mean(scheduled)) if scheduled else 0.0,
        "packet_receive_rate": packet_receive_rate,
        "scheduled_outage_rate": float(np.mean(outaged)) if outaged else 0.0,
        # Actual successful deliveries (not merely scheduled slots).
        "schedule_receive_rate": packet_receive_rate,
        "scheduled_slots": scheduled,
        "delivered_slots": delivered,
    }


@torch.no_grad()
def evaluate_prediction_horizon_nmae(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """
    Per-horizon command NMAE (physical Newtons) and JEPA latent cosine alignment.

    Command NMAE (plan §15 Eq. 27) is actor(z̃_{k+j}) vs teacher u_{k+j} after denorm:
        mean(|u_pred - u_true|) / 40
    Latent cosine is a JEPA diagnostic, not a plan §15 metric.
    Predictor conditioning uses u_k..u_{k+Kp-1} (documented §9 source).
    NMAE targets use u_{k+1}..u_{k+Kp} (aligned with predicted latents).
    """
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    normalizer = load_command_normalizer(config, data_root=root)
    test_dir = root / "trajectories" / "jepa" / "test"
    dataset = TrajectoryDataset(test_dir, config, normalizer, training=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    encoder = runtime_encoder(controller)
    device = encoder.device
    jepa = encoder.jepa
    actor = encoder.actor
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    resolution = jepa.command_resolution
    force_range = physical_force_range_n(config)

    pred_by_h = {h: [] for h in range(1, kp + 1)}
    tgt_by_h = {h: [] for h in range(1, kp + 1)}
    cos_by_h = {h: [] for h in range(1, kp + 1)}
    pred_all = []
    tgt_all = []
    cos_all = []

    for batch in loader:
        context = batch["context"].to(device)
        future = batch["future_frames"].to(device)
        conditioning_norm = select_predictor_conditioning_commands(batch, resolution).to(device)
        # Predictor conditioning is u_k..u_{k+Kp-1}; NMAE targets are u_{k+1}..u_{k+Kp}.
        target_phys = batch["target_commands"].cpu().numpy()
        z = jepa.encode_context(context)
        z_pred = jepa.predict(z, conditioning_norm)  # [B, Kp, D] = z̃_{k+1}..z̃_{k+Kp}
        z_tgt = jepa.encode_targets(future)
        cos = jepa_cosine_similarity(z_pred, z_tgt)  # [B, Kp]
        b = z_pred.shape[0]
        u_phys = actor(z_pred.reshape(b * kp, -1)).reshape(b, kp).cpu().numpy()
        pred_all.append(u_phys.reshape(-1))
        tgt_all.append(target_phys.reshape(-1))
        cos_all.append(cos.detach().cpu().numpy().reshape(-1))
        for h in range(1, kp + 1):
            pred_by_h[h].append(u_phys[:, h - 1].reshape(-1))
            tgt_by_h[h].append(target_phys[:, h - 1].reshape(-1))
            cos_by_h[h].append(cos[:, h - 1].detach().cpu().numpy().reshape(-1))

    empty = {
        **nmae_report_fields(value=float("nan"), force_range_n=force_range),
        "nmae_by_horizon": {},
        "latent_cosine_mean": float("nan"),
        "latent_cosine_by_horizon": {},
        "kp": kp,
        "split": "jepa_test_untouched",
        "num_values": 0,
        "predictor_command_resolution": resolution.to_dict(),
    }
    if not pred_all:
        return empty

    nmae_by_horizon = {
        str(h): nmae(np.concatenate(pred_by_h[h]), np.concatenate(tgt_by_h[h]), force_range_n=force_range)
        for h in range(1, kp + 1)
    }
    latent_cosine_by_horizon = {
        str(h): float(np.mean(np.concatenate(cos_by_h[h]))) for h in range(1, kp + 1)
    }
    pred = np.concatenate(pred_all)
    tgt = np.concatenate(tgt_all)
    return {
        **nmae_report_fields(value=nmae(pred, tgt, force_range_n=force_range), force_range_n=force_range),
        "nmae_by_horizon": nmae_by_horizon,
        "latent_cosine_mean": float(np.mean(np.concatenate(cos_all))),
        "latent_cosine_by_horizon": latent_cosine_by_horizon,
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
    Paper-faithful evaluation (plan §14–§15).

    Wireless scheduling is excluded by default until §15 validation passes.
    Pass include_wireless=True only after validate_baseline()['passed'] / wireless_allowed.
    """
    reported = str(config.get("evaluation", {}).get("reported_result", "best"))
    seeds = closed_loop_eval_seeds(config)
    control_runs = []
    scores = []
    for seed in seeds:
        out = evaluate_closed_loop(config, controller, seed=seed)
        control_runs.append(out)
        scores.append(out["mean_control_score"])
    h, w = config["input"]["resize"]
    bits = communication_reduction_report(
        height=int(h),
        width=int(w),
        embedding_dim=int(config["ts_jepa"]["encoder"]["embedding_dim"]),
        channels=int(config["input"].get("channels_per_rgb_frame", 3)),
    )
    nmae_report = evaluate_prediction_horizon_nmae(config, controller, data_root=data_root)
    actor_nmae = evaluate_actor_nmae(config, controller, data_root=data_root)
    tsne_report = evaluate_embedding_tsne(config, controller, data_root=data_root)
    mape_report = evaluate_consecutive_frame_mape(config, data_root=data_root)
    fig4_report = evaluate_fig4_sampling_rate_mape(config, data_root=data_root)
    stability = evaluate_closed_loop_stability(
        config, controller, full_runs=control_runs
    )

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
        "control": summarize_scores(scores, reported_result=reported),
        "actor_nmae": actor_nmae,
        "prediction_horizon_nmae": nmae_report,
        "embedding_tsne": tsne_report,
        "consecutive_frame_mape": mape_report,
        "fig4_mape": fig4_report,
        "stability": stability,
        "test_losses": test_losses,
        "communication_bits": bits,
        "predictor_command_resolution": load_predictor_command_resolution(config).to_dict(),
        "closed_loop_runtime": {
            "controller_class": type(controller).__name__,
            "miss_behavior": getattr(controller, "miss_behavior", None),
        },
    }
    report["baseline_validation"] = validate_baseline(report, config)

    wireless: dict[str, Any] = {}
    if include_wireless and report["baseline_validation"]["wireless_allowed"]:
        channel = WirelessChannelModel(config)
        outage_capacity = {
            str(snr): channel.estimate_outage_probability(float(snr), seed=0)
            for snr in config["wireless"]["snr_targets_db"]
        }
        for policy in ("channel_aware", "round_robin", "opportunistic"):
            wireless[policy] = {}
            for snr in config["wireless"]["snr_targets_db"]:
                wireless[policy][snr] = evaluate_with_scheduler(
                    config, controller, policy=policy, snr_db=float(snr), seed=7
                )
        report["wireless"] = {
            "channel_model": {
                "scenario": "InF-SH",
                "plan_section": "13",
                "bandwidth_hz": channel.bandwidth_hz,
                "snr_targets_db": list(config["wireless"]["snr_targets_db"]),
                "outage_and_capacity": outage_capacity,
            },
            "policies": {
                p: {
                    str(snr): {
                        "mean_control_score": wireless[p][snr]["mean_control_score"],
                        "schedule_rate": wireless[p][snr]["schedule_rate"],
                        "packet_receive_rate": wireless[p][snr]["packet_receive_rate"],
                        "scheduled_outage_rate": wireless[p][snr]["scheduled_outage_rate"],
                        "schedule_receive_rate": wireless[p][snr]["packet_receive_rate"],
                    }
                    for snr in config["wireless"]["snr_targets_db"]
                }
                for p in wireless
            },
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

    mape_report = report.get("consecutive_frame_mape") or {}
    if mape_report:
        mape_json = out_dir / "consecutive_frame_mape.json"
        with mape_json.open("w", encoding="utf-8") as handle:
            json.dump(mape_report, handle, indent=2)
        paths["consecutive_frame_mape"] = str(mape_json)

    fig4_report = report.get("fig4_mape") or {}
    if fig4_report:
        fig4_json = out_dir / "fig4_mape.json"
        with fig4_json.open("w", encoding="utf-8") as handle:
            json.dump(fig4_report, handle, indent=2)
        paths["fig4_mape"] = str(fig4_json)
        by_rate = fig4_report.get("by_sampling_interval_ms") or {}
        if by_rate:
            from ts_jepa.evaluation.plotting import plot_fig4_mape

            paths["fig4_mape_plot"] = str(plot_fig4_mape(fig4_report, out_dir / "fig4_mape.png"))

    actor_nmae = report.get("actor_nmae") or {}
    if actor_nmae:
        actor_json = out_dir / "actor_nmae_report.json"
        with actor_json.open("w", encoding="utf-8") as handle:
            json.dump(actor_nmae, handle, indent=2)
        paths["actor_nmae_report"] = str(actor_json)

    stability = report.get("stability") or {}
    if stability:
        stab_json = out_dir / "stability_report.json"
        with stab_json.open("w", encoding="utf-8") as handle:
            json.dump(stability, handle, indent=2)
        paths["stability_report"] = str(stab_json)

    control_json = out_dir / "closed_loop_report.json"
    with control_json.open("w", encoding="utf-8") as handle:
        json.dump(report.get("control") or {}, handle, indent=2)
    paths["closed_loop_report"] = str(control_json)
    return paths

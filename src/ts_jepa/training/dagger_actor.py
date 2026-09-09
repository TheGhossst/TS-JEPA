"""DAgger for the frozen-z semantic actor (working overlay, not paper BC).

One-step imitation on teacher trajectories does not keep the pole up in closed
loop. This module keeps Ψθ frozen, rolls out the current Cε, labels visited
states with the queryable DP teacher, aggregates (z, u*), and retrains only Cε.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, project_root
from ts_jepa.control.lqr import lqr_force, lqr_gain_from_config
from ts_jepa.data.datasets import ActorEmbeddingDataset, load_command_normalizer
from ts_jepa.data.trajectory_generator import build_env_and_teacher
from ts_jepa.env.factory import build_inverted_cartpole_env
from ts_jepa.device import select_device
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.evaluation.evaluate import (
    _apply_held_force,
    control_loop_stride,
    evaluate_closed_loop,
)
from ts_jepa.evaluation.raw_state_actor import ActorFeatureDataset, train_feature_actor
from ts_jepa.evaluation.working_gates import (
    _periodic_receive_mask,
    evaluate_actor_working_gates,
    evaluate_closed_loop_working_gates,
)
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.inference.temporal_actor import (
    PairZRuntimeController,
    feature_dim,
    make_pair_actor,
    pair_feature,
)
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.runtime import save_checkpoint, state_dict_to_cpu


DEFAULT_DAGGER = {
    "rounds": 8,
    "rollouts_per_round": 40,
    "beta_start": 0.85,
    "beta_decay": 0.85,
    "epochs_per_round": 12,
    "mix_offline": True,
    "offline_max_samples": 8000,
    "lossy_fraction": 0.25,
    "rollout_seed0": 5000,
    "init_from_bc": True,
    "mix_mode": "convex",
    "checkpoint_selection": "last",
    "round_selection": "last",
    "expert": "lqr",
    "feature": "z",
    "run_dirname": "semantic_actor_working_dagger",
}


def dagger_settings(config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(DEFAULT_DAGGER)
    overlay = config.get("semantic_actor", {}).get("dagger") or {}
    raw.update(overlay)
    return raw


def dagger_beta(round_index: int, beta_start: float, beta_decay: float) -> float:
    """Exponential mixing schedule: P(execute teacher) at 0-based round."""
    if round_index < 0:
        raise ValueError(f"round_index must be >= 0, got {round_index}")
    return float(np.clip(float(beta_start) * (float(beta_decay) ** int(round_index)), 0.0, 1.0))


def mix_execute_action(
    teacher_u: float,
    actor_u: float,
    beta: float,
    rng: np.random.Generator,
    mode: str = "convex",
) -> tuple[float, bool]:
    """Mix teacher and actor forces.

    ``convex``: u = beta * u* + (1-beta) * u_actor. Bernoulli mixing with a
    collapsing actor knocks the pole down in one step and DAgger never sees
    near-upright recoveries.
    """
    beta = float(np.clip(beta, 0.0, 1.0))
    mode = str(mode)
    if mode == "bernoulli":
        use_teacher = bool(rng.random() < beta)
        return (float(teacher_u) if use_teacher else float(actor_u)), use_teacher
    if mode == "convex":
        return float(beta * float(teacher_u) + (1.0 - beta) * float(actor_u)), True
    raise ValueError(f"unknown mix mode {mode!r}")


def embedding_dataset_to_arrays(
    dataset: ActorEmbeddingDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = torch.stack([sample[0] for sample in dataset.samples], dim=0).numpy()
    u = np.asarray([sample[1] for sample in dataset.samples], dtype=np.float32)
    un = np.asarray([sample[2] for sample in dataset.samples], dtype=np.float32)
    return z, u, un


def subsample_offline(
    z: np.ndarray,
    u: np.ndarray,
    un: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(z.shape[0])
    cap = int(max_samples)
    if cap <= 0 or n <= cap:
        return z, u, un
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n, size=cap, replace=False))
    return z[idx], u[idx], un[idx]


def concat_arrays(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonempty = [p for p in parts if p[0].shape[0] > 0]
    if not nonempty:
        raise ValueError("no DAgger samples to concatenate")
    z = np.concatenate([p[0] for p in nonempty], axis=0)
    u = np.concatenate([p[1] for p in nonempty], axis=0)
    un = np.concatenate([p[2] for p in nonempty], axis=0)
    return z, u, un


def _assert_jepa_frozen(jepa: TSJEPA) -> None:
    trainable = [n for n, p in jepa.named_parameters() if p.requires_grad]
    if trainable:
        raise RuntimeError(
            "DAgger must keep TS-JEPA frozen; trainable params remain: "
            f"{trainable[:8]}"
        )


def _load_or_encode_offline(
    *,
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    data_root: Path,
    cache_path: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.is_file():
        with np.load(cache_path) as payload:
            return (
                np.asarray(payload["z"], dtype=np.float32),
                np.asarray(payload["u"], dtype=np.float32),
                np.asarray(payload["u_norm"], dtype=np.float32),
            )
    normalizer = load_command_normalizer(config, data_root=data_root)
    ds = ActorEmbeddingDataset(
        data_root / "trajectories" / "actor" / split,
        config,
        normalizer,
        controller.jepa.context_encoder,
        controller.device,
        training=False,
    )
    z, u, un = embedding_dataset_to_arrays(ds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, z=z, u=u, u_norm=un)
    return z, u, un


@torch.no_grad()
def collect_dagger_round(
    *,
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    teacher: Any,
    env: Any,
    beta: float,
    n_rollouts: int,
    seed0: int,
    lossy_fraction: float,
    mix_mode: str = "convex",
    expert_fn=None,
    feature_kind: str = "z",
    policy_actor=None,
) -> dict[str, Any]:
    """Roll out mixed actor/teacher; label every visited z with teacher.act(state)."""
    steps = int(config["simulation"]["trajectory_steps"])
    stride = control_loop_stride(config)
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    emb_dim = int(config["ts_jepa"]["encoder"]["embedding_dim"])
    lossy_mask = _periodic_receive_mask(steps, kp)
    n_lossy = int(round(float(lossy_fraction) * int(n_rollouts)))
    z_rows: list[np.ndarray] = []
    u_rows: list[float] = []
    scores: list[float] = []
    teacher_frac: list[float] = []

    for i in range(int(n_rollouts)):
        seed = int(seed0) + i
        rng = np.random.default_rng(seed)
        controller.reset_episode()
        state = env.reset(seed=seed)
        mask = lossy_mask if i < n_lossy else [True] * steps
        ep_scores: list[int] = []
        used_teacher = 0
        prev_z: np.ndarray | None = None
        for t in range(steps):
            frame = env.render(state)
            received = bool(mask[t])
            u_star = float(expert_fn(state) if expert_fn is not None else teacher.act(state))
            if received:
                controller.observe_frame(frame)
                context = controller._context_from_buffer()
                z = controller.jepa.encode_context(context)
                controller.latent = z
            else:
                if frame is not None:
                    controller.observe_frame(frame)
                if controller.latent is None:
                    z = None
                else:
                    cmd = torch.tensor(
                        [[controller.last_command_norm]],
                        dtype=torch.float32,
                        device=controller.device,
                    )
                    z = controller.jepa.predictor.forward_step(controller.latent, cmd)
                    controller.latent = z
            if z is None:
                z_np = np.zeros((emb_dim,), dtype=np.float32)
            else:
                z_np = z.detach().cpu().reshape(-1).numpy().astype(np.float32, copy=True)
            feat = pair_feature(z_np, prev_z, feature_kind)  # type: ignore[arg-type]
            prev_z = z_np
            actor_mod = policy_actor if policy_actor is not None else controller.actor
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(controller.device)
            u_actor = float(
                controller.stats.actor_output_to_force_and_norm(actor_mod(feat_t))[0]
            )
            u_exec, took_teacher = mix_execute_action(
                u_star, u_actor, beta, rng, mode=mix_mode
            )
            u_exec, u_norm = controller.stats.actor_output_to_force_and_norm(u_exec)
            controller.last_command_norm = u_norm
            z_rows.append(feat)
            u_rows.append(u_star)
            used_teacher += int(took_teacher)
            ep_scores.append(
                int(
                    abs(float(state[0] - env.desired_state[0]))
                    <= float(config["evaluation"]["control_position_tol"])
                    and abs(float(state[2])) <= float(config["evaluation"]["control_angle_tol"])
                )
            )
            state = _apply_held_force(env, u_exec, stride)
        scores.append(float(np.mean(ep_scores)))
        teacher_frac.append(used_teacher / float(steps))

    z = np.stack(z_rows, axis=0)
    u = np.asarray(u_rows, dtype=np.float32)
    un = np.asarray(
        [(float(v) - controller.stats.mean) / controller.stats.std for v in u_rows],
        dtype=np.float32,
    )
    return {
        "z": z,
        "u": u,
        "u_norm": un,
        "mean_control_score": float(np.mean(scores)) if scores else 0.0,
        "per_rollout_control_score": scores,
        "mean_teacher_execute_frac": float(np.mean(teacher_frac)) if teacher_frac else 0.0,
        "n_samples": int(z.shape[0]),
        "n_lossy_rollouts": n_lossy,
    }


def _eval_round_closed_loop(
    config: dict[str, Any],
    controller: FrozenRuntimeController,
    *,
    include_working_gate: bool = False,
) -> dict[str, Any]:
    seeds = [
        int(s)
        for s in config.get("evaluation", {}).get("working_gates", {}).get(
            "closed_loop_seeds", [100, 101, 102]
        )
    ]
    steps = int(config["simulation"]["trajectory_steps"])
    full = [
        float(evaluate_closed_loop(config, controller, steps=steps, seed=seed)["mean_control_score"])
        for seed in seeds
    ]
    out: dict[str, Any] = {
        "full_information": {
            "seeds": seeds,
            "mean_control_score": float(np.mean(full)),
            "per_seed": full,
        }
    }
    if include_working_gate:
        out["working_gate"] = evaluate_closed_loop_working_gates(config, controller)
    return out


def train_actor_dagger(
    config: dict[str, Any],
    *,
    jepa_checkpoint: Path | None = None,
    actor_checkpoint: Path | None = None,
    device: torch.device | None = None,
    data_root: Path | None = None,
    skip_eval: bool = False,
    settings_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Actor-only DAgger. Does not update JEPA weights or D_s."""
    device = select_device(device)
    settings = dagger_settings(config)
    if settings_overrides:
        settings.update(settings_overrides)
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    project = project_root(config)
    runs = project / config["paths"]["runs_root"]
    jepa_ckpt = resolve_run_checkpoint(
        runs,
        jepa_run_dirname(config),
        explicit=Path(jepa_checkpoint) if jepa_checkpoint else None,
        seed=0,
    )
    actor_ckpt = resolve_run_checkpoint(
        runs,
        actor_run_dirname(config),
        explicit=Path(actor_checkpoint) if actor_checkpoint else None,
    )

    out_dir = runs / str(settings["run_dirname"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("dagger: loading frozen JEPA + BC actor", flush=True)
    controller = FrozenRuntimeController.from_checkpoints(
        config, jepa_ckpt, actor_ckpt, device=device
    )
    for p in controller.jepa.parameters():
        p.requires_grad_(False)
    _assert_jepa_frozen(controller.jepa)

    expert_kind = str(settings.get("expert", "lqr"))
    teacher = None
    expert_fn = None
    if expert_kind == "lqr":
        print("dagger: using true-state LQR expert (DP teacher is not Eq. 28-stable)", flush=True)
        env = build_inverted_cartpole_env(config)
        gain = lqr_gain_from_config(config)
        force_min = float(config["simulation"]["control_min_N"])
        force_max = float(config["simulation"]["control_max_N"])

        def expert_fn(state, _gain=gain, _lo=force_min, _hi=force_max):
            return lqr_force(_gain, state, _lo, _hi)

        settings["mix_offline"] = False
    else:
        print("dagger: solving DP teacher (one-time)", flush=True)
        env, teacher = build_env_and_teacher(config)
    normalizer = load_command_normalizer(config, data_root=root)

    offline_z = offline_u = offline_un = None
    if bool(settings["mix_offline"]):
        print("dagger: loading/encoding offline D_a embeddings", flush=True)
        offline_z, offline_u, offline_un = _load_or_encode_offline(
            config=config,
            controller=controller,
            data_root=root,
            cache_path=out_dir / "offline_train_embeddings.npz",
            split="train",
        )
        offline_z, offline_u, offline_un = subsample_offline(
            offline_z,
            offline_u,
            offline_un,
            int(settings["offline_max_samples"]),
            seed=int(settings["rollout_seed0"]),
        )
        test_z, test_u, test_un = _load_or_encode_offline(
            config=config,
            controller=controller,
            data_root=root,
            cache_path=out_dir / "offline_test_embeddings.npz",
            split="test",
        )
    feature_kind = str(settings.get("feature", "z"))
    z_dim = int(config["ts_jepa"]["encoder"]["embedding_dim"])
    if feature_kind != "z":
        settings["init_from_bc"] = False
        if str(settings.get("run_dirname")) == "semantic_actor_working_dagger":
            settings["run_dirname"] = f"semantic_actor_working_dagger_{feature_kind}"
        out_dir = runs / str(settings["run_dirname"])
        out_dir.mkdir(parents=True, exist_ok=True)
    policy_actor = (
        controller.actor
        if feature_kind == "z"
        else make_pair_actor(config, feature_kind).to(device)  # type: ignore[arg-type]
    )
    policy_actor.eval()

    else_dim = feature_dim(z_dim, feature_kind)  # type: ignore[arg-type]
    if not bool(settings["mix_offline"]):
        test_z = np.zeros((1, else_dim), dtype=np.float32)
        test_u = np.zeros((1,), dtype=np.float32)
        test_un = np.zeros((1,), dtype=np.float32)

    test_ds = ActorFeatureDataset(test_z, test_u, test_un)
    agg_z: list[np.ndarray] = []
    agg_u: list[np.ndarray] = []
    agg_un: list[np.ndarray] = []
    round_logs: list[dict[str, Any]] = []
    actor_state = None if feature_kind != "z" or not bool(settings["init_from_bc"]) else state_dict_to_cpu(
        controller.actor.state_dict()
    )
    if not bool(settings["init_from_bc"]):
        actor_state = None
    round_sel = str(settings.get("round_selection", "last"))
    best_loop_score = float("-inf")
    best_round_state = None
    best_round_i: int | None = None

    def _loop_controller():
        if feature_kind == "z":
            return controller
        return PairZRuntimeController(
            controller, policy_actor, feature=feature_kind, full_information=True  # type: ignore[arg-type]
        )

    for round_i in range(int(settings["rounds"])):
        beta = dagger_beta(round_i, float(settings["beta_start"]), float(settings["beta_decay"]))
        seed0 = int(settings["rollout_seed0"]) + round_i * int(settings["rollouts_per_round"])
        print(
            f"dagger: collect round={round_i} beta={beta:.3f} mix={settings.get('mix_mode', 'convex')} "
            f"feature={feature_kind}",
            flush=True,
        )
        collected = collect_dagger_round(
            config=config,
            controller=controller,
            teacher=teacher,
            env=env,
            beta=beta,
            n_rollouts=int(settings["rollouts_per_round"]),
            seed0=seed0,
            lossy_fraction=float(settings["lossy_fraction"]),
            mix_mode=str(settings.get("mix_mode", "convex")),
            expert_fn=expert_fn,
            feature_kind=feature_kind,
            policy_actor=policy_actor,
        )
        agg_z.append(collected["z"])
        agg_u.append(collected["u"])
        agg_un.append(collected["u_norm"])
        parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        if offline_z is not None:
            parts.append((offline_z, offline_u, offline_un))
        parts.append((np.concatenate(agg_z, axis=0), np.concatenate(agg_u, axis=0), np.concatenate(agg_un, axis=0)))
        z, u, un = concat_arrays(parts)
        train_ds = ActorFeatureDataset(z, u, un)
        trained = train_feature_actor(
            config,
            train_ds,
            test_ds,
            device=device,
            max_epochs=int(settings["epochs_per_round"]),
            run_dir=out_dir / f"round_{round_i:02d}",
            seed=round_i,
            init_state_dict=actor_state,
            epoch_desc=f"dagger round {round_i} epoch",
            checkpoint_selection=str(settings.get("checkpoint_selection", "last")),
        )
        actor_state = state_dict_to_cpu(trained["actor"].state_dict())
        policy_actor.load_state_dict(actor_state)
        policy_actor.eval()
        for p in policy_actor.parameters():
            p.requires_grad_(False)
        if feature_kind == "z":
            controller.actor.load_state_dict(actor_state)
            controller.actor.eval()

        loop = None if skip_eval else _eval_round_closed_loop(config, _loop_controller())
        log = {
            "round": round_i,
            "beta": beta,
            "feature": feature_kind,
            "collect_mean_control_score": collected["mean_control_score"],
            "collect_teacher_execute_frac": collected["mean_teacher_execute_frac"],
            "n_new_samples": collected["n_samples"],
            "n_train_samples": int(z.shape[0]),
            "best_val": trained["best_val"],
            "test_loss": trained["test_loss"],
            "closed_loop": loop,
        }
        round_logs.append(log)
        print(json.dumps({k: v for k, v in log.items() if k != "closed_loop"}, default=str), flush=True)
        if loop is not None:
            print(
                f"dagger round={round_i} full_info="
                f"{loop['full_information']['mean_control_score']:.4f}",
                flush=True,
            )
            if round_sel == "best_closed_loop":
                score = float(loop["full_information"]["mean_control_score"])
                if score > best_loop_score:
                    best_loop_score = score
                    best_round_state = actor_state
                    best_round_i = round_i

    if round_sel == "best_closed_loop" and best_round_state is not None:
        actor_state = best_round_state
        policy_actor.load_state_dict(actor_state)
        policy_actor.eval()
        if feature_kind == "z":
            controller.actor.load_state_dict(actor_state)
            controller.actor.eval()
        print(
            f"dagger: selected round={best_round_i} closed_loop={best_loop_score:.4f}",
            flush=True,
        )

    save_checkpoint(
        out_dir / "best.pt",
        {
            "actor": actor_state,
            "jepa_checkpoint": str(jepa_ckpt.resolve()),
            "normalizer": normalizer.to_dict(),
            "config": config,
            "method": "dagger",
            "expert": expert_kind,
            "feature": feature_kind,
            "embedding_dim": feature_dim(z_dim, feature_kind),  # type: ignore[arg-type]
            "rounds": round_logs,
            "init_actor_checkpoint": str(Path(actor_ckpt).resolve()),
            "round_selection": round_sel,
            "selected_round": best_round_i,
        },
    )
    report = {
        "checkpoint": str(out_dir / "best.pt"),
        "jepa_checkpoint": str(jepa_ckpt),
        "init_actor_checkpoint": str(actor_ckpt),
        "settings": settings,
        "rounds": round_logs,
        "jepa_updated": False,
        "round_selection": round_sel,
        "selected_round": best_round_i,
    }
    if not skip_eval:
        loop_ctrl = _loop_controller()
        if feature_kind == "z":
            report["actor_nmae"] = evaluate_actor_working_gates(
                config, controller, root, actor_ckpt=out_dir / "best.pt"
            )
        else:
            report["actor_nmae"] = {
                "skipped": True,
                "note": "single-z actor_test NMAE does not apply to [z_k, z_{k-1}] input",
            }
        report["closed_loop"] = _eval_round_closed_loop(
            config, loop_ctrl, include_working_gate=True
        )
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report

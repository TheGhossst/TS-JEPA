"""Stabilizing actor: LQR on a linear z→state probe (frozen JEPA).

The DP teacher used for behavior cloning does not itself stay in the Eq. 28
band (cart walks off the track). Imitating it — including DAgger — cannot
produce a closed-loop policy that beats hold/zero. This module keeps Ψθ
frozen, reads kinematics out of z, and applies a discrete LQR law designed
for the control period.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, project_root
from ts_jepa.control.lqr import (
    discrete_linearization,
    discrete_lqr_gain,
    default_lqr_weights,
    lqr_force,
)
from ts_jepa.device import select_device
from ts_jepa.env.factory import build_inverted_cartpole_env
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.evaluation.command_linear_probe import (
    _standardize_apply,
    _standardize_fit,
    fit_linear_ols,
    predict_linear_ols,
)
from ts_jepa.evaluation.evaluate import (
    _apply_held_force,
    _observation_stride,
    evaluate_closed_loop,
)
from ts_jepa.evaluation.metrics import control_score
from ts_jepa.evaluation.working_gates import evaluate_closed_loop_working_gates
from ts_jepa.inference.infer import FrozenRuntimeController


def fit_z_state_probe(
    z: np.ndarray,
    states: np.ndarray,
    *,
    ridge: float = 1e-4,
) -> dict[str, np.ndarray]:
    z = np.asarray(z, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64).reshape(z.shape[0], -1)[:, :4]
    mean, std = _standardize_fit(z)
    zs = _standardize_apply(z, mean, std)
    coefs = np.stack(
        [fit_linear_ols(zs, states[:, i], ridge=ridge) for i in range(4)],
        axis=0,
    )
    return {"kind": "linear", "z_mean": np.asarray(mean), "z_std": np.asarray(std), "coefs": coefs}


def predict_state_from_z(z: np.ndarray, probe: dict[str, Any]) -> np.ndarray:
    if str(probe.get("kind", "linear")) == "mlp":
        from ts_jepa.evaluation.mlp_state_decoder import predict_mlp_state

        return predict_mlp_state(z, probe)
    z = np.asarray(z, dtype=np.float64)
    single = z.ndim == 1
    z2 = z.reshape(1, -1) if single else z
    zs = _standardize_apply(z2, probe["z_mean"], probe["z_std"])
    pred = np.stack(
        [predict_linear_ols(zs, probe["coefs"][i]) for i in range(4)],
        axis=1,
    )
    return pred.reshape(4) if single else pred


class ProbeLQRRuntimeController:
    """RGB → frozen Ψθ → linear state probe → discrete LQR."""

    miss_behavior = "lqr_on_linear_prediction"

    def __init__(
        self,
        encoder: FrozenRuntimeController,
        probe: dict[str, np.ndarray],
        gain: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        *,
        full_information: bool = False,
    ) -> None:
        self.encoder = encoder
        self.probe = probe
        self.gain = np.asarray(gain, dtype=np.float64).reshape(1, 4)
        self.a = np.asarray(a, dtype=np.float64).reshape(4, 4)
        self.b = np.asarray(b, dtype=np.float64).reshape(4)
        self.full_information = bool(full_information)
        self.force_min = encoder.stats.force_min
        self.force_max = encoder.stats.force_max
        self.kappa = max(1, int(encoder.config["input"]["kappa"]))
        self.s_hat: np.ndarray | None = None
        self.last_u = 0.0

    def reset_episode(self) -> None:
        self.encoder.reset_episode()
        self.s_hat = None
        self.last_u = 0.0

    def observe_frame(self, frame: np.ndarray) -> None:
        self.encoder.observe_frame(frame)

    def _encode_z(self, frame: np.ndarray) -> np.ndarray:
        self.encoder.observe_frame(frame)
        context = self.encoder._context_from_buffer()
        z = self.encoder.jepa.encode_context(context)
        return z.detach().cpu().reshape(-1).numpy().astype(np.float64)

    @torch.no_grad()
    def step(
        self,
        frame,
        packet_received: bool,
        plant_state: np.ndarray | None = None,
    ) -> float:
        del plant_state
        if frame is None:
            raise ValueError("ProbeLQRRuntimeController requires an RGB frame")
        use_encode = self.full_information or packet_received or self.s_hat is None
        if use_encode:
            z = self._encode_z(frame)
            self.s_hat = predict_state_from_z(z, self.probe)
        else:
            self.encoder.observe_frame(frame)
            assert self.s_hat is not None
            self.s_hat = self.a @ self.s_hat + self.b * float(self.last_u)
        u = lqr_force(self.gain, self.s_hat, self.force_min, self.force_max)
        if len(self.encoder.frame_buffer) < self.kappa:
            # Duplicate-frame kappa padding makes velocity unobservable.
            theta = float(self.s_hat[2])
            u = float(np.clip(-float(self.gain[0, 2]) * theta, self.force_min, self.force_max))
        self.last_u = u
        return u


class ObserverLQRRuntimeController:
    """RGB → frozen Ψθ → z → decoded (x, θ) → α-β observer → discrete LQR.

    Memoryless z→velocity decodes are MSE-attenuated (R²≈0.45 for θ̇) and
    attenuated velocity feedback destroys the loop even though extra *white*
    force noise up to ±5 N does not (see runs/eval/lqr_noise_sensitivity.json
    and runs/eval/velocity_observability_control.json). Positions decode well
    (MAE ~0.008 m / 0.004 rad), so this controller ignores decoded velocities
    entirely: it predicts the state with the discrete linearization and the
    actually-applied force, then corrects positions with α-β gains. Velocity
    information is accumulated across timesteps instead of read from one z.

    Packet lost → predict-only step (no measurement correction), mirroring the
    plan §14 miss path without using Pφ.
    """

    miss_behavior = "observer_predict_no_correction"

    def __init__(
        self,
        encoder: FrozenRuntimeController,
        probe: dict[str, Any],
        gain: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        *,
        alpha: float = 0.5,
        beta: float = 0.3,
        full_information: bool = False,
    ) -> None:
        self.encoder = encoder
        self.probe = probe
        self.gain = np.asarray(gain, dtype=np.float64).reshape(1, 4)
        self.a = np.asarray(a, dtype=np.float64).reshape(4, 4)
        self.b = np.asarray(b, dtype=np.float64).reshape(4)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.full_information = bool(full_information)
        self.force_min = encoder.stats.force_min
        self.force_max = encoder.stats.force_max
        stride = max(1, int(encoder.config["simulation"].get("observation_stride_steps", 1)))
        self.dt_obs = float(stride) * float(encoder.config["simulation"]["dt"])
        self.est: np.ndarray | None = None
        self.last_u = 0.0

    def reset_episode(self) -> None:
        self.encoder.reset_episode()
        self.est = None
        self.last_u = 0.0

    def observe_frame(self, frame: np.ndarray) -> None:
        self.encoder.observe_frame(frame)

    def _measure(self, frame: np.ndarray) -> tuple[float, float]:
        self.encoder.observe_frame(frame)
        context = self.encoder._context_from_buffer()
        z = self.encoder.jepa.encode_context(context)
        s_hat = predict_state_from_z(z.detach().cpu().reshape(-1).numpy().astype(np.float64), self.probe)
        return float(s_hat[0]), float(s_hat[2])

    @torch.no_grad()
    def step(
        self,
        frame,
        packet_received: bool,
        plant_state: np.ndarray | None = None,
    ) -> float:
        del plant_state
        if frame is None:
            raise ValueError("ObserverLQRRuntimeController requires an RGB frame")
        use_measurement = self.full_information or packet_received
        if use_measurement:
            x_m, th_m = self._measure(frame)
        else:
            self.encoder.observe_frame(frame)
            x_m = th_m = None  # type: ignore[assignment]
        if self.est is None:
            if x_m is None:
                return 0.0  # no embedding received yet (IC, matches FrozenRuntimeController)
            self.est = np.array([x_m, 0.0, th_m, 0.0], dtype=np.float64)
        else:
            self.est = self.a @ self.est + self.b * float(self.last_u)
            if x_m is not None:
                rx = float(x_m) - float(self.est[0])
                rth = float(th_m) - float(self.est[2])
                self.est[0] += self.alpha * rx
                self.est[1] += self.beta * rx / self.dt_obs
                self.est[2] += self.alpha * rth
                self.est[3] += self.beta * rth / self.dt_obs
        u = lqr_force(self.gain, self.est, self.force_min, self.force_max)
        self.last_u = u
        return u


class TrueStateLQRController:
    """Oracle LQR using plant_state (sanity check that the gain is stabilizing)."""

    miss_behavior = "lqr_true_state"

    def __init__(self, config: dict[str, Any], gain: np.ndarray) -> None:
        self.gain = np.asarray(gain, dtype=np.float64).reshape(1, 4)
        self.force_min = float(config["simulation"]["control_min_N"])
        self.force_max = float(config["simulation"]["control_max_N"])
        self.full_information = True

    def reset_episode(self) -> None:
        return None

    def step(self, frame, packet_received: bool, plant_state: np.ndarray | None = None) -> float:
        del frame, packet_received
        if plant_state is None:
            raise ValueError("TrueStateLQRController requires plant_state")
        return lqr_force(self.gain, plant_state, self.force_min, self.force_max)


def evaluate_true_state_lqr(
    config: dict[str, Any],
    gain: np.ndarray,
    *,
    seeds: list[int],
    steps: int | None = None,
) -> dict[str, Any]:
    env = build_inverted_cartpole_env(config)
    stride = _observation_stride(config)
    steps = int(steps or config["simulation"]["trajectory_steps"])
    pos_tol = float(config["evaluation"]["control_position_tol"])
    ang_tol = float(config["evaluation"]["control_angle_tol"])
    desired_x = float(config["simulation"]["desired_state"][0])
    per_seed: list[float] = []
    finals: list[list[float]] = []
    for seed in seeds:
        state = env.reset(seed=int(seed))
        scores = []
        for _ in range(steps):
            scores.append(
                control_score(state, desired_x=desired_x, position_tol=pos_tol, angle_tol=ang_tol)
            )
            force = lqr_force(
                gain,
                state,
                float(config["simulation"]["control_min_N"]),
                float(config["simulation"]["control_max_N"]),
            )
            state = _apply_held_force(env, force, stride)
        per_seed.append(float(np.mean(scores)))
        finals.append(np.asarray(state, dtype=np.float64).reshape(-1).tolist())
    return {
        "seeds": [int(s) for s in seeds],
        "mean_control_score": float(np.mean(per_seed)),
        "per_seed": per_seed,
        "final_states": finals,
    }


@torch.no_grad()
def collect_balanced_probe_pairs(
    config: dict[str, Any],
    encoder: FrozenRuntimeController,
    gain: np.ndarray,
    *,
    n_rollouts: int = 50,
    seed0: int = 9000,
    action_noise_std: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode RGB on true-state-LQR trajectories near upright (not drifting D_a)."""
    env = build_inverted_cartpole_env(config)
    steps = int(config["simulation"]["trajectory_steps"])
    stride = _observation_stride(config)
    z_rows: list[np.ndarray] = []
    s_rows: list[np.ndarray] = []
    force_min = float(config["simulation"]["control_min_N"])
    force_max = float(config["simulation"]["control_max_N"])
    for i in range(int(n_rollouts)):
        seed = int(seed0) + i
        rng = np.random.default_rng(seed)
        encoder.reset_episode()
        state = env.reset(seed=seed)
        for _ in range(steps):
            frame = env.render(state)
            encoder.observe_frame(frame)
            context = encoder._context_from_buffer()
            z = encoder.jepa.encode_context(context)
            z_rows.append(z.detach().cpu().reshape(-1).numpy().astype(np.float32))
            s_rows.append(np.asarray(state, dtype=np.float32).reshape(4)[:4].copy())
            u = lqr_force(gain, state, force_min, force_max)
            u = float(np.clip(u + rng.normal(0.0, action_noise_std), force_min, force_max))
            state = _apply_held_force(env, u, stride)
    return np.stack(z_rows, axis=0), np.stack(s_rows, axis=0)


def load_state_decoder_checkpoint(path: Path) -> dict[str, Any]:
    """Load a probe-LQR decoder checkpoint (decoder.pt or probe_lqr.npz)."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as payload:
            probe = {
                "kind": "linear",
                "z_mean": np.asarray(payload["probe_z_mean"]),
                "z_std": np.asarray(payload["probe_z_std"]),
                "coefs": np.asarray(payload["probe_coefs"]),
            }
            return {
                "probe": probe,
                "gain": np.asarray(payload["lqr_gain"]),
                "a": np.asarray(payload["A"]),
                "b": np.asarray(payload["B"]).reshape(4),
                "jepa_checkpoint": str(payload["jepa_checkpoint"]),
            }
    payload = torch.load(path, map_location="cpu", weights_only=False)
    from ts_jepa.evaluation.mlp_state_decoder import StateDecoderMLP

    z_mean = np.asarray(payload["probe_z_mean"])
    model = StateDecoderMLP(
        z_dim=int(z_mean.reshape(-1).shape[0]),
        hidden=tuple(int(h) for h in payload["hidden"]),
    )
    model.load_state_dict(payload["mlp_state_dict"])
    model.eval()
    probe = {
        "kind": "mlp",
        "module": model,
        "z_mean": z_mean,
        "z_std": np.asarray(payload["probe_z_std"]),
        "y_mean": np.asarray(payload["y_mean"]),
        "y_std": np.asarray(payload["y_std"]),
    }
    return {
        "probe": probe,
        "gain": np.asarray(payload["lqr_gain"]),
        "a": np.asarray(payload["A"]),
        "b": np.asarray(payload["B"]).reshape(4),
        "jepa_checkpoint": str(payload["jepa_checkpoint"]),
    }


def evaluate_observer_lqr(
    config: dict[str, Any],
    *,
    decoder_checkpoint: Path,
    jepa_checkpoint: Path | None = None,
    actor_checkpoint: Path | None = None,
    device: torch.device | None = None,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> dict[str, Any]:
    """Closed-loop evaluation of ObserverLQRRuntimeController from a saved decoder."""
    device = select_device(device)
    project = project_root(config)
    runs = project / config["paths"]["runs_root"]
    loaded = load_state_decoder_checkpoint(Path(decoder_checkpoint))
    jepa_ckpt = resolve_run_checkpoint(
        runs,
        jepa_run_dirname(config),
        explicit=Path(jepa_checkpoint) if jepa_checkpoint else Path(loaded["jepa_checkpoint"]),
        seed=0,
    )
    actor_ckpt = resolve_run_checkpoint(
        runs,
        actor_run_dirname(config),
        explicit=Path(actor_checkpoint) if actor_checkpoint else None,
    )
    encoder = FrozenRuntimeController.from_checkpoints(config, jepa_ckpt, actor_ckpt, device=device)
    for p in encoder.jepa.parameters():
        p.requires_grad_(False)

    gain, a, b = loaded["gain"], loaded["a"], loaded["b"]
    seeds = [
        int(s)
        for s in config.get("evaluation", {}).get("working_gates", {}).get(
            "closed_loop_seeds", [100, 101, 102]
        )
    ]
    true_lqr = evaluate_true_state_lqr(config, gain, seeds=seeds)

    def _make(full_information: bool) -> ObserverLQRRuntimeController:
        return ObserverLQRRuntimeController(
            encoder,
            loaded["probe"],
            gain,
            a,
            b,
            alpha=alpha,
            beta=beta,
            full_information=full_information,
        )

    full_scores = [
        float(evaluate_closed_loop(config, _make(True), seed=seed)["mean_control_score"])
        for seed in seeds
    ]
    memoryless = ProbeLQRRuntimeController(encoder, loaded["probe"], gain, a, b, full_information=True)
    memoryless_scores = [
        float(evaluate_closed_loop(config, memoryless, seed=seed)["mean_control_score"])
        for seed in seeds
    ]
    working = evaluate_closed_loop_working_gates(config, _make(False))

    out_dir = runs / "semantic_actor_working_observer_lqr"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "observer_lqr",
        "jepa_updated": False,
        "jepa_checkpoint": str(jepa_ckpt),
        "decoder_checkpoint": str(Path(decoder_checkpoint).resolve()),
        "decoder_kind": str(loaded["probe"].get("kind")),
        "observer": {"alpha": float(alpha), "beta": float(beta)},
        "lqr_gain": np.asarray(gain).reshape(-1).tolist(),
        "true_state_lqr": true_lqr,
        "observer_lqr_full_information": {
            "seeds": seeds,
            "mean_control_score": float(np.mean(full_scores)),
            "per_seed": full_scores,
        },
        "memoryless_probe_lqr_full_information": {
            "seeds": seeds,
            "mean_control_score": float(np.mean(memoryless_scores)),
            "per_seed": memoryless_scores,
        },
        "working_gate": working,
        "note": (
            "Velocities are NOT read from z (memoryless decode is MSE-attenuated, which "
            "destroys the loop); they are re-estimated by an alpha-beta observer over the "
            "decoded (x, theta). See runs/eval/velocity_observability_control.json."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_probe_lqr_actor(
    config: dict[str, Any],
    *,
    jepa_checkpoint: Path | None = None,
    actor_checkpoint: Path | None = None,
    device: torch.device | None = None,
    data_root: Path | None = None,
    decoder: str = "linear",
) -> dict[str, Any]:
    """Fit z→state probe + LQR. Does not update JEPA. Writes a readout checkpoint."""
    device = select_device(device)
    decoder = str(decoder)
    if decoder not in {"linear", "mlp"}:
        raise ValueError(f"decoder must be linear or mlp, got {decoder!r}")
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
    out_dir = runs / (
        "semantic_actor_working_probe_lqr_mlp" if decoder == "mlp" else "semantic_actor_working_probe_lqr"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = FrozenRuntimeController.from_checkpoints(config, jepa_ckpt, actor_ckpt, device=device)
    for p in encoder.jepa.parameters():
        p.requires_grad_(False)

    env = build_inverted_cartpole_env(config)
    stride = _observation_stride(config)
    a, b = discrete_linearization(env.ode, stride)
    q, r = default_lqr_weights()
    gain = discrete_lqr_gain(a, b, q, r)

    print("probe-lqr: collecting z,state on true-state LQR rollouts", flush=True)
    z_train, states = collect_balanced_probe_pairs(config, encoder, gain)
    if decoder == "mlp":
        from ts_jepa.evaluation.mlp_state_decoder import train_mlp_state_decoder

        print("probe-lqr: training MLP z->state decoder", flush=True)
        probe = train_mlp_state_decoder(z_train, states, device=device)
    else:
        probe = fit_z_state_probe(z_train, states)
    s_hat = predict_state_from_z(z_train, probe)
    probe_mae = np.mean(np.abs(s_hat - states[:, :4]), axis=0)
    print(
        f"probe-lqr decoder={decoder}: MAE x={probe_mae[0]:.4f} xd={probe_mae[1]:.4f} "
        f"th={probe_mae[2]:.4f} thd={probe_mae[3]:.4f}",
        flush=True,
    )

    seeds = [
        int(s)
        for s in config.get("evaluation", {}).get("working_gates", {}).get(
            "closed_loop_seeds", [100, 101, 102]
        )
    ]
    true_lqr = evaluate_true_state_lqr(config, gain, seeds=seeds)
    controller = ProbeLQRRuntimeController(encoder, probe, gain, a, b[:, 0], full_information=False)
    full_scores = [
        float(evaluate_closed_loop(config, controller, seed=seed)["mean_control_score"])
        for seed in seeds
    ]
    controller = ProbeLQRRuntimeController(encoder, probe, gain, a, b[:, 0], full_information=False)
    working = evaluate_closed_loop_working_gates(config, controller)

    payload = {
        "decoder": decoder,
        "probe_z_mean": probe["z_mean"],
        "probe_z_std": probe["z_std"],
        "lqr_gain": gain,
        "A": a,
        "B": b,
        "jepa_checkpoint": str(jepa_ckpt.resolve()),
    }
    if decoder == "linear":
        payload["probe_coefs"] = probe["coefs"]
        np.savez_compressed(out_dir / "probe_lqr.npz", **payload)
    else:
        torch.save(
            {
                **payload,
                "mlp_state_dict": probe["module"].state_dict(),
                "y_mean": probe["y_mean"],
                "y_std": probe["y_std"],
                "hidden": probe["hidden"],
            },
            out_dir / "decoder.pt",
        )
    report = {
        "method": "probe_lqr",
        "decoder": decoder,
        "jepa_updated": False,
        "jepa_checkpoint": str(jepa_ckpt),
        "checkpoint": str(out_dir / ("decoder.pt" if decoder == "mlp" else "probe_lqr.npz")),
        "probe_mae_state": {
            "x": float(probe_mae[0]),
            "x_dot": float(probe_mae[1]),
            "theta": float(probe_mae[2]),
            "theta_dot": float(probe_mae[3]),
        },
        "n_probe_samples": int(z_train.shape[0]),
        "lqr_gain": gain.reshape(-1).tolist(),
        "true_state_lqr": true_lqr,
        "probe_lqr_full_information": {
            "seeds": seeds,
            "mean_control_score": float(np.mean(full_scores)),
            "per_seed": full_scores,
        },
        "working_gate": working,
        "note": (
            "DP teacher rollouts on D_a already have Eq. 28 score 0 (cart walks off). "
            "This actor is LQR on a frozen-z state probe, not behavior cloning of the teacher."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

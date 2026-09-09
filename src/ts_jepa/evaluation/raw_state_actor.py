"""Raw-state SemanticActor diagnostic (no JEPA updates).

Trains the plan §12 MLP (same hidden dims / AdamW / physical MSE) on
cart-pole state or κ-state history instead of frozen z, then compares
held-out NMAE and closed-loop scores to the existing z-actor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, project_root
from ts_jepa.data.datasets import load_command_normalizer
from ts_jepa.device import select_device
from ts_jepa.evaluation.checkpoints import resolve_jepa_checkpoint_from_actor, resolve_run_checkpoint
from ts_jepa.evaluation.command_linear_probe import load_commands_aligned, load_states_aligned
from ts_jepa.evaluation.evaluate import evaluate_actor_nmae, evaluate_closed_loop
from ts_jepa.evaluation.metrics import nmae, nmae_report_fields, physical_force_range_n
from ts_jepa.evaluation.state_probe import kappa_for_state_history, load_state_history_aligned
from ts_jepa.evaluation.working_gates import _periodic_receive_mask
from ts_jepa.inference.infer import FrozenRuntimeController, RuntimeCommandStats
from ts_jepa.models.actor import SemanticActor
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.runtime import make_dataloader, state_dict_to_cpu
from ts_jepa.training.actor_helpers import (
    actor_loss_kwargs,
    actor_regression_loss,
    evaluate_actor_objective,
    evaluate_mse,
    mean_command_baseline_mse,
    split_train_val_actor,
)

FeatureKind = Literal["current", "history"]


class ActorFeatureDataset(Dataset):
    """Same batch keys as ActorEmbeddingDataset, but features are raw state."""

    def __init__(
        self,
        features: np.ndarray,
        commands_phys: np.ndarray,
        commands_norm: np.ndarray,
    ) -> None:
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(commands_phys, dtype=np.float32).reshape(-1)
        yn = np.asarray(commands_norm, dtype=np.float32).reshape(-1)
        if x.shape[0] != y.shape[0] or y.shape[0] != yn.shape[0]:
            raise ValueError(
                f"feature/command length mismatch: x={x.shape[0]} y={y.shape[0]} yn={yn.shape[0]}"
            )
        self.features = torch.from_numpy(x)
        self.commands_phys = y
        self.commands_norm = yn

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "embedding": self.features[index],
            "command": torch.tensor([self.commands_phys[index]], dtype=torch.float32),
            "command_norm": torch.tensor([self.commands_norm[index]], dtype=torch.float32),
        }


class RawStateRuntimeController:
    """Closed-loop actor that reads plant_state instead of RGB→Ψθ."""

    miss_behavior = "hold_last_feature"

    def __init__(
        self,
        config: dict[str, Any],
        actor: nn.Module,
        normalizer: CommandNormalizer,
        *,
        feature: FeatureKind = "current",
        full_information: bool = True,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = select_device(device)
        self.actor = actor.to(self.device).eval()
        for p in self.actor.parameters():
            p.requires_grad_(False)
        self.stats = RuntimeCommandStats(
            mean=normalizer.mean,
            std=normalizer.std,
            force_min=float(config["simulation"]["control_min_N"]),
            force_max=float(config["simulation"]["control_max_N"]),
        )
        self.feature = feature
        self.full_information = bool(full_information)
        self.kappa = kappa_for_state_history(config) if feature == "history" else 1
        self.state_buffer: list[np.ndarray] = []
        self.last_feature: torch.Tensor | None = None

    def reset_episode(self) -> None:
        self.state_buffer.clear()
        self.last_feature = None

    def _feature_tensor(self) -> torch.Tensor:
        if not self.state_buffer:
            raise RuntimeError("RawStateRuntimeController has no plant_state yet")
        if self.feature == "current":
            vec = np.asarray(self.state_buffer[-1], dtype=np.float32).reshape(-1)[:4]
        else:
            k = self.kappa
            parts: list[np.ndarray] = []
            for i in range(k):
                idx = len(self.state_buffer) - k + i
                parts.append(self.state_buffer[0] if idx < 0 else self.state_buffer[idx])
            vec = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1)[:4] for p in parts], axis=0)
        return torch.from_numpy(vec).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def step(
        self,
        frame,  # noqa: ARG002 — RGB is unused; evaluate_closed_loop still renders it
        packet_received: bool,
        plant_state: np.ndarray | None = None,
    ) -> float:
        if plant_state is None:
            raise ValueError("RawStateRuntimeController requires plant_state every step")
        self.state_buffer.append(np.asarray(plant_state, dtype=np.float64).reshape(-1).copy())
        if len(self.state_buffer) > max(self.kappa + 5, 8):
            self.state_buffer = self.state_buffer[-(self.kappa + 5) :]
        if self.full_information or packet_received or self.last_feature is None:
            feat = self._feature_tensor()
            self.last_feature = feat
        else:
            feat = self.last_feature
        u_actor = self.actor(feat)
        force, _u_norm = self.stats.actor_output_to_force_and_norm(u_actor)
        return force


def _actor_hparams(config: dict[str, Any]) -> dict[str, Any]:
    opt = config["semantic_actor"]["optimizer"]
    early = config["semantic_actor"]["early_stopping"]
    arch = config["semantic_actor"]["architecture"]
    return {
        "hidden_dims": tuple(arch["hidden_dims"]),
        "dropout": float(arch.get("dropout", 0.2)),
        "lr": float(opt["learning_rate"]),
        "weight_decay": float(opt.get("weight_decay", 0.01)),
        "batch_size": int(opt["batch_size"]),
        "epochs": int(opt["epochs"]),
        "patience": int(early["patience"]),
        "val_fraction": float(early.get("val_fraction", 0.2)),
        "loss": "MSE",
        "loss_domain": "physical",
    }


def train_feature_actor(
    config: dict[str, Any],
    train_ds: ActorFeatureDataset,
    test_ds: ActorFeatureDataset,
    *,
    device: torch.device,
    max_epochs: int | None = None,
    run_dir: Path | None = None,
    seed: int = 0,
    init_state_dict: dict[str, Any] | None = None,
    epoch_desc: str = "raw-state actor epoch",
    checkpoint_selection: str = "best_val",
) -> dict[str, Any]:
    """Same optimizer / physical MSE / val split as train_semantic_actor, no JEPA."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    hp = _actor_hparams(config)
    epochs = int(max_epochs if max_epochs is not None else hp["epochs"])
    feat_dim = int(train_ds.features.shape[1])
    arch = config["semantic_actor"]["architecture"]
    jepa_dim = int(config["ts_jepa"]["encoder"]["embedding_dim"])
    actor = SemanticActor(
        embedding_dim=feat_dim,
        hidden_dims=hp["hidden_dims"],
        dropout=hp["dropout"],
        layernorm=bool(arch.get("layernorm", False)) and feat_dim == jepa_dim,
        output_init_scale=float(arch.get("output_init_scale", 1.0)),
    ).to(device)
    if init_state_dict is not None:
        actor.load_state_dict(init_state_dict)
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
    )
    criterion = nn.MSELoss()
    loss_kwargs = actor_loss_kwargs(config)
    val_fraction = hp["val_fraction"]
    split_mode = str(config["semantic_actor"]["early_stopping"].get("split", "contiguous"))
    train_split, val_split = split_train_val_actor(
        train_ds, val_fraction, seed=seed, mode=split_mode
    )
    batch_size = hp["batch_size"]
    train_loader = make_dataloader(
        train_split,
        batch_size=min(batch_size, max(1, len(train_split))),
        shuffle=True,
        device=device,
        config=config,
        num_workers=0,
    )
    val_loader = make_dataloader(
        val_split,
        batch_size=min(batch_size, max(1, len(val_split))),
        shuffle=False,
        device=device,
        config=config,
        num_workers=0,
    )
    test_loader = make_dataloader(
        test_ds,
        batch_size=min(batch_size, max(1, len(test_ds))),
        shuffle=False,
        device=device,
        config=config,
        num_workers=0,
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    patience = hp["patience"]
    for epoch in range(1, epochs + 1):
        actor.train()
        train_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"{epoch_desc} {epoch}", leave=False, mininterval=1.0):
            emb = batch["embedding"].to(device)
            target = batch["command"].to(device)
            pred = actor(emb)
            loss = actor_regression_loss(pred, target, **loss_kwargs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().item())
            n_batches += 1
        train_loss /= max(1, n_batches)
        val_loss = evaluate_actor_objective(
            actor, val_loader, device, loss_kwargs=loss_kwargs
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"{epoch_desc}={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = state_dict_to_cpu(actor.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience and str(checkpoint_selection) == "best_val":
                print(f"{epoch_desc} early stop epoch={epoch} stale={stale}")
                break
    if str(checkpoint_selection) == "last":
        best_state = state_dict_to_cpu(actor.state_dict())
        best_epoch = epochs
        best_val = float(history[-1]["val_loss"]) if history else best_val
    elif best_state is not None:
        actor.load_state_dict(best_state)
    test_loss = evaluate_mse(actor, test_loader, device, criterion)
    train_targets = np.asarray(train_ds.commands_phys, dtype=np.float64)
    test_targets = np.asarray(test_ds.commands_phys, dtype=np.float64)
    train_mean = float(train_targets.mean()) if train_targets.size else float("nan")
    mean_baseline_test = mean_command_baseline_mse(test_targets, train_mean)
    result = {
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "test_loss": float(test_loss),
        "history": history,
        "feature_dim": feat_dim,
        "hidden_dims": list(hp["hidden_dims"]),
        "loss_domain": "physical",
        "mean_command_baseline_mse_test": mean_baseline_test,
        "beats_mean_command_baseline": bool(np.isfinite(test_loss) and test_loss < mean_baseline_test),
        "seed": int(seed),
    }
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": state_dict_to_cpu(actor.state_dict()),
                "feature_dim": feat_dim,
                "config": config,
                "val_loss": best_val,
                "test_loss": test_loss,
                "history": history,
                "loss_domain": "physical",
            },
            run_dir / "best.pt",
        )
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["checkpoint"] = str(run_dir / "best.pt")
    result["actor"] = actor
    return result


def evaluate_feature_actor_nmae(
    config: dict[str, Any],
    actor: SemanticActor,
    features: np.ndarray,
    commands_phys: np.ndarray,
    normalizer: CommandNormalizer,
    *,
    device: torch.device,
) -> dict[str, Any]:
    actor.eval()
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 256):
            chunk = x[start : start + 256].to(device)
            preds.append(actor(chunk).detach().cpu().numpy().reshape(-1))
    pred = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.float32)
    tgt = np.asarray(commands_phys, dtype=np.float64).reshape(-1)
    force_range = physical_force_range_n(config)
    actor_nmae = nmae(pred, tgt, force_range_n=force_range)
    mean_baseline_pred = np.full_like(tgt, float(normalizer.mean))
    mean_baseline_nmae = nmae(mean_baseline_pred, tgt, force_range_n=force_range)
    return {
        **nmae_report_fields(value=actor_nmae, force_range_n=force_range),
        "num_values": int(pred.size),
        "split": "actor_test_untouched",
        "mean_abs_error": float(np.mean(np.abs(pred - tgt))) if pred.size else float("nan"),
        "mean_command_baseline_nmae": float(mean_baseline_nmae),
        "mean_command_baseline_source": "jepa_train_command_mean_via_normalizer",
        "beats_mean_command_baseline": bool(
            np.isfinite(actor_nmae) and np.isfinite(mean_baseline_nmae) and actor_nmae < mean_baseline_nmae
        ),
        "pred_mean_N": float(pred.mean()) if pred.size else float("nan"),
        "pred_std_N": float(pred.std()) if pred.size else float("nan"),
        "target_std_N": float(tgt.std()) if tgt.size else float("nan"),
    }


def _closed_loop_block(
    config: dict[str, Any],
    controller: Any,
    *,
    seeds: list[int],
    steps: int,
    packet_receive_mask: list[bool] | None,
) -> dict[str, Any]:
    scores: list[float] = []
    for seed in seeds:
        out = evaluate_closed_loop(
            config,
            controller,
            steps=steps,
            seed=int(seed),
            packet_receive_mask=packet_receive_mask,
        )
        scores.append(float(out["mean_control_score"]))
    return {
        "seeds": [int(s) for s in seeds],
        "mean_control_score": float(np.mean(scores)) if scores else float("nan"),
        "per_seed": scores,
    }


def audit_action_scaling(
    config: dict[str, Any],
    *,
    actor_outputs: np.ndarray,
    commands_phys: np.ndarray,
    normalizer: CommandNormalizer,
    checkpoint_normalizer: dict[str, Any] | None = None,
    train_loss_domain: str = "physical",
) -> dict[str, Any]:
    """Compare actor outputs to physical commands vs z-scored commands."""
    u = np.asarray(actor_outputs, dtype=np.float64).reshape(-1)
    y = np.asarray(commands_phys, dtype=np.float64).reshape(-1)
    y_norm = np.asarray(normalizer.normalize(y.astype(np.float32)), dtype=np.float64).reshape(-1)
    roundtrip = np.asarray(normalizer.denormalize(y_norm.astype(np.float32)), dtype=np.float64)
    stats = RuntimeCommandStats(
        mean=normalizer.mean,
        std=normalizer.std,
        force_min=float(config["simulation"]["control_min_N"]),
        force_max=float(config["simulation"]["control_max_N"]),
    )
    clipped = np.array([stats.actor_output_to_force_and_norm(v)[0] for v in u[: min(len(u), 4096)]])
    mse_phys = float(np.mean((u - y) ** 2)) if u.size and y.size else float("nan")
    mse_if_denorm = float(np.mean((roundtrip[: u.size] - y[: u.size]) ** 2)) if u.size else float("nan")
    # If the head emitted z-scores, denormalizing would match y better than raw u.
    mse_denorm_u = float(np.mean((u * normalizer.std + normalizer.mean - y) ** 2)) if u.size else float("nan")
    out_std = float(u.std()) if u.size else float("nan")
    cmd_std = float(y.std()) if y.size else float("nan")
    suspected_zscore_head = bool(
        np.isfinite(out_std) and np.isfinite(cmd_std) and out_std < 2.5 and cmd_std > 5.0
        and mse_denorm_u + 1e-6 < mse_phys
    )
    ckpt_mean = None if checkpoint_normalizer is None else float(checkpoint_normalizer.get("mean", float("nan")))
    ckpt_std = None if checkpoint_normalizer is None else float(checkpoint_normalizer.get("std", float("nan")))
    normalizer_match = True
    if ckpt_mean is not None and np.isfinite(ckpt_mean):
        normalizer_match = abs(ckpt_mean - float(normalizer.mean)) < 1e-4 and abs(
            float(ckpt_std or 0.0) - float(normalizer.std)
        ) < 1e-4
    inference_treats_output_as = "physical_newtons_then_clip_then_zscore_for_predictor"
    train_target = "physical_newtons"
    consistent = (str(train_loss_domain) == "physical") and (not suspected_zscore_head)
    return {
        "train_loss_domain": train_loss_domain,
        "train_target": train_target,
        "inference_treats_output_as": inference_treats_output_as,
        "actor_output_mean": float(u.mean()) if u.size else float("nan"),
        "actor_output_std": out_std,
        "actor_output_min": float(u.min()) if u.size else float("nan"),
        "actor_output_max": float(u.max()) if u.size else float("nan"),
        "command_phys_mean": float(y.mean()) if y.size else float("nan"),
        "command_phys_std": cmd_std,
        "command_norm_std": float(y_norm.std()) if y_norm.size else float("nan"),
        "mse_actor_vs_physical_command": mse_phys,
        "mse_if_denormalize_actor_output": mse_denorm_u,
        "command_roundtrip_mae": float(np.mean(np.abs(roundtrip - y))) if y.size else float("nan"),
        "clip_changes_fraction": float(np.mean(np.abs(clipped - u[: clipped.size]) > 1e-6)) if u.size else float("nan"),
        "disk_normalizer_matches_checkpoint": bool(normalizer_match),
        "suspected_zscore_head": suspected_zscore_head,
        "train_infer_domain_consistent": bool(consistent),
        "note": (
            "train_semantic_actor fits MSE(actor(z), u_phys). "
            "FrozenRuntimeController.actor_output_to_force_and_norm clips Newtons and "
            "only z-scores the clipped force for Pφ on packet loss. "
            "A z-score head would have std ~1 and denormalizing it would beat raw-Newton MSE."
        ),
    }


def _load_split_features(
    config: dict[str, Any],
    split_dir: Path,
    kind: FeatureKind,
    normalizer: CommandNormalizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    commands = load_commands_aligned(split_dir)
    if kind == "history":
        features = load_state_history_aligned(split_dir, kappa_for_state_history(config))
    else:
        features = load_states_aligned(split_dir)[:, :4]
    commands_norm = normalizer.normalize(commands.astype(np.float32))
    return features, commands.astype(np.float32), commands_norm


def _actor_outputs_on_first_trajectory(
    controller: FrozenRuntimeController,
    test_dir: Path,
    *,
    max_steps: int = 128,
) -> tuple[list[float], list[float]]:
    """Encode one test rollout the same way closed-loop inference does."""
    files = sorted(Path(test_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no actor-test npz files in {test_dir}")
    with np.load(files[0]) as data:
        frames = np.asarray(data["frames"])
        commands = np.asarray(data["commands"], dtype=np.float32).reshape(-1)
    outs: list[float] = []
    tgts: list[float] = []
    controller.reset_episode()
    n = min(int(max_steps), int(commands.shape[0]), int(frames.shape[0]))
    with torch.no_grad():
        for t in range(n):
            controller.observe_frame(frames[t])
            context = controller._context_from_buffer()
            z = controller.jepa.encode_context(context)
            raw = float(controller.actor(z).reshape(-1)[0].item())
            outs.append(raw)
            tgts.append(float(commands[t]))
    return outs, tgts


def run_raw_state_actor_diagnostic(
    config: dict[str, Any],
    *,
    device: torch.device,
    data_root: Path | None = None,
    jepa_ckpt: Path | None = None,
    actor_ckpt: Path | None = None,
    max_epochs: int | None = None,
    kinds: tuple[FeatureKind, ...] = ("current", "history"),
    skip_closed_loop: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    runs = project_root(config) / config["paths"]["runs_root"]
    normalizer = load_command_normalizer(config, data_root=root)
    train_dir = root / "trajectories" / "actor" / "train"
    test_dir = root / "trajectories" / "actor" / "test"
    hp = _actor_hparams(config)
    gates = config.get("evaluation", {}).get("working_gates", {})
    loop_seeds = [int(s) for s in gates.get("closed_loop_seeds", [100, 101, 102])]
    steps = int(config["simulation"]["trajectory_steps"])
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    mask = _periodic_receive_mask(steps, kp)

    try:
        z_actor_ckpt = resolve_run_checkpoint(
            runs,
            actor_run_dirname(config),
            explicit=actor_ckpt,
            seed=None,
        )
    except FileNotFoundError:
        z_actor_ckpt = resolve_run_checkpoint(
            runs,
            actor_run_dirname(config),
            explicit=actor_ckpt,
            seed=0,
        )
    if jepa_ckpt is None:
        try:
            jepa_ckpt = resolve_jepa_checkpoint_from_actor(z_actor_ckpt, project_dir=project_root(config))
        except FileNotFoundError:
            jepa_ckpt = resolve_run_checkpoint(runs, jepa_run_dirname(config), seed=0)

    z_controller = FrozenRuntimeController.from_checkpoints(config, jepa_ckpt, z_actor_ckpt, device=device)
    z_nmae = evaluate_actor_nmae(config, z_controller, data_root=root)
    z_loop_full = None
    z_loop_gated = None
    if not skip_closed_loop:
        z_loop_full = _closed_loop_block(
            config, z_controller, seeds=loop_seeds, steps=steps, packet_receive_mask=[True] * steps
        )
        z_loop_gated = _closed_loop_block(
            config, z_controller, seeds=loop_seeds, steps=steps, packet_receive_mask=mask
        )

    z_payload = torch.load(z_actor_ckpt, map_location="cpu", weights_only=False)
    print("action-scaling audit: one actor-test trajectory through the closed-loop encode path ...")
    z_outs, z_tgts = _actor_outputs_on_first_trajectory(
        z_controller, test_dir, max_steps=128
    )
    z_scaling = audit_action_scaling(
        config,
        actor_outputs=np.asarray(z_outs),
        commands_phys=np.asarray(z_tgts),
        normalizer=normalizer,
        checkpoint_normalizer=z_payload.get("normalizer"),
        train_loss_domain=str(config["semantic_actor"].get("loss_domain", "physical")),
    )

    raw_reports: dict[str, Any] = {}
    for kind in kinds:
        print(f"training raw-state SemanticActor on feature={kind} ...")
        x_tr, y_tr, yn_tr = _load_split_features(config, train_dir, kind, normalizer)
        x_te, y_te, yn_te = _load_split_features(config, test_dir, kind, normalizer)
        train_ds = ActorFeatureDataset(x_tr, y_tr, yn_tr)
        test_ds = ActorFeatureDataset(x_te, y_te, yn_te)
        run_dir = runs / "eval" / f"raw_state_actor_{kind}"
        trained = train_feature_actor(
            config,
            train_ds,
            test_ds,
            device=device,
            max_epochs=max_epochs,
            run_dir=run_dir,
            seed=seed,
        )
        actor = trained.pop("actor")
        raw_nmae = evaluate_feature_actor_nmae(
            config, actor, x_te, y_te, normalizer, device=device
        )
        with torch.no_grad():
            pred_te = []
            xt = torch.from_numpy(x_te)
            for start in range(0, xt.shape[0], 256):
                pred_te.append(actor(xt[start : start + 256].to(device)).detach().cpu().numpy().reshape(-1))
        pred_te_np = np.concatenate(pred_te) if pred_te else np.zeros((0,), dtype=np.float32)
        raw_scaling = audit_action_scaling(
            config,
            actor_outputs=pred_te_np,
            commands_phys=y_te,
            normalizer=normalizer,
            train_loss_domain="physical",
        )
        loops: dict[str, Any] = {}
        if not skip_closed_loop:
            for mode, full_info in (("full_information", True), ("working_gate_hold_last", False)):
                ctrl = RawStateRuntimeController(
                    config,
                    actor,
                    normalizer,
                    feature=kind,
                    full_information=full_info,
                    device=device,
                )
                pkt = [True] * steps if full_info else mask
                loops[mode] = _closed_loop_block(
                    config, ctrl, seeds=loop_seeds, steps=steps, packet_receive_mask=pkt
                )
        raw_reports[kind] = {
            "feature_kind": kind,
            "feature_dim": int(x_tr.shape[1]),
            "training": {
                "best_epoch": trained["best_epoch"],
                "best_val": trained["best_val"],
                "test_loss": trained["test_loss"],
                "beats_mean_command_baseline_mse": trained["beats_mean_command_baseline"],
                "mean_command_baseline_mse_test": trained["mean_command_baseline_mse_test"],
            },
            "history_tail": trained["history"][-5:],
            "actor_nmae": raw_nmae,
            "action_scaling": raw_scaling,
            "closed_loop": loops,
            "checkpoint": trained.get("checkpoint"),
        }

    interpretation = interpret_raw_vs_jepa(
        z_nmae=z_nmae,
        z_loop_full=z_loop_full,
        raw_reports=raw_reports,
        z_scaling=z_scaling,
    )
    return {
        "jepa_checkpoint": str(Path(jepa_ckpt).resolve()),
        "z_actor_checkpoint": str(Path(z_actor_ckpt).resolve()),
        "data_root": str(root),
        "jepa_modified": False,
        "shared_training": hp,
        "z_actor": {
            "actor_nmae": z_nmae,
            "closed_loop_full_information": z_loop_full,
            "closed_loop_working_gate": z_loop_gated,
            "action_scaling": z_scaling,
        },
        "raw_state_actor": raw_reports,
        "interpretation": interpretation,
    }


def interpret_raw_vs_jepa(
    *,
    z_nmae: dict[str, Any],
    z_loop_full: dict[str, Any] | None,
    raw_reports: dict[str, Any],
    z_scaling: dict[str, Any],
) -> dict[str, Any]:
    raw_current = raw_reports.get("current") or next(iter(raw_reports.values()), None)
    if raw_current is None:
        return {"code": "missing_raw_actor", "honest_conclusion": "No raw-state actor was trained."}
    raw_nmae = raw_current["actor_nmae"]
    raw_full = (raw_current.get("closed_loop") or {}).get("full_information")
    z_loop = float(z_loop_full["mean_control_score"]) if z_loop_full else float("nan")
    raw_loop = float(raw_full["mean_control_score"]) if raw_full else float("nan")
    raw_beats = bool(raw_nmae.get("beats_mean_command_baseline"))
    z_beats = bool(z_nmae.get("beats_mean_command_baseline"))
    scaling_bug = bool(z_scaling.get("suspected_zscore_head")) or not bool(
        z_scaling.get("train_infer_domain_consistent", True)
    )

    if scaling_bug:
        code = "action_scaling_bug"
        conclusion = (
            "Actor outputs look like z-scores (or train/infer domains disagree). "
            "Fix command scaling before blaming the representation."
        )
    elif np.isfinite(raw_loop) and raw_loop > 0.05:
        if (not np.isfinite(z_loop)) or z_loop <= 0.05:
            code = "z_is_bottleneck"
            conclusion = (
                "The same MLP controls the plant from raw state but not from frozen z. "
                "The representation is missing usable control state; do not retrain the actor recipe first."
            )
        else:
            code = "both_control"
            conclusion = "Both raw-state and z actors produce nonzero closed-loop scores."
    elif np.isfinite(raw_loop) and raw_loop <= 0.05:
        if raw_beats and abs(float(raw_nmae.get("nmae", 1.0)) - float(z_nmae.get("nmae", 1.0))) < 0.02:
            code = "actor_cannot_learn_control"
            conclusion = (
                "Even an oracle-state copy of SemanticActor stays near the mean command and "
                "scores 0 in closed loop. The actor pipeline (capacity, loss, teacher pairing, "
                "or evaluation) cannot express a stabilizing policy — not just z."
            )
        elif not raw_beats:
            code = "actor_cannot_learn_command"
            conclusion = (
                "Raw state does not beat the mean-command baseline with this MLP/training. "
                "The imitation target or actor optimization is the bottleneck."
            )
        else:
            code = "actor_imitates_but_cannot_stabilize"
            conclusion = (
                "Raw-state actor beats mean-command NMAE but still scores ~0 closed-loop. "
                "Offline imitation of the DP teacher is too weak to stabilize, even with true state."
            )
    else:
        code = "inconclusive"
        conclusion = "Closed-loop numbers are missing; use NMAE comparison only."
        if raw_beats and not z_beats:
            code = "z_is_bottleneck_offline"
            conclusion = "Raw state beats the mean-command baseline; frozen z does not. Representation bottleneck (offline)."
        elif z_beats and raw_beats:
            conclusion = (
                "Both beat the mean-command baseline offline. "
                "Closed-loop was not decisive; rerun without --skip-closed-loop."
            )
    return {
        "code": code,
        "honest_conclusion": conclusion,
        "z_nmae": float(z_nmae.get("nmae", float("nan"))),
        "raw_nmae": float(raw_nmae.get("nmae", float("nan"))),
        "z_closed_loop_full": z_loop,
        "raw_closed_loop_full": raw_loop,
        "z_beats_mean_command": z_beats,
        "raw_beats_mean_command": raw_beats,
    }

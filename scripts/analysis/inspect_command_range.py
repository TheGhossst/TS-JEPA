"""Diagnostic: raw vs normalized teacher-command range vs the ±20 gate probe."""

from __future__ import annotations

import argparse

import numpy as np

from ts_jepa.config import load_config, project_root
from ts_jepa.data.datasets import load_command_normalizer
from ts_jepa.data.temporal import command_indices, max_valid_time_index


PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def _summarize(name: str, x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    out: dict[str, float] = {
        "n": float(x.size),
        "min": float(x.min()) if x.size else float("nan"),
        "max": float(x.max()) if x.size else float("nan"),
        "mean": float(x.mean()) if x.size else float("nan"),
        "std": float(x.std()) if x.size else float("nan"),
    }
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(x, p)) if x.size else float("nan")
    print(f"\n{name}  n={int(out['n'])}")
    print(
        f"  min={out['min']:.6g}  max={out['max']:.6g}  "
        f"mean={out['mean']:.6g}  std={out['std']:.6g}"
    )
    pct = "  ".join(f"p{p}={out[f'p{p}']:.6g}" for p in PERCENTILES)
    print(f"  {pct}")
    return out


def _percentile_of(value: float, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(100.0 * np.mean(x <= value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ts_jepa_working.yaml")
    parser.add_argument("--max-windows", type=int, default=20000)
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root(config)
    data_root = root / config["paths"]["data_root"]
    normalizer = load_command_normalizer(config, data_root=data_root)
    print(f"config={args.config}")
    print(f"data_root={data_root}")
    print(f"normalizer mean={normalizer.mean:.8g}  std={normalizer.std:.8g}")

    raw_traj = []
    train_dir = data_root / "trajectories" / "jepa" / "train"
    for path in sorted(train_dir.glob("*.npz")):
        with np.load(path) as data:
            raw_traj.append(np.asarray(data["commands"], dtype=np.float32))
    raw_all = np.concatenate(raw_traj, axis=0)
    _summarize("RAW teacher commands (all JEPA train npz steps)", raw_all)

    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    windows = []
    for commands in raw_traj:
        max_start = max_valid_time_index(int(commands.shape[0]), kp)
        for k in range(max(0, max_start + 1)):
            idx = command_indices(k, kp)
            windows.append(commands[idx[0] : idx[0] + kp])
    windows_a = np.stack(windows, axis=0)
    n = min(int(args.max_windows), windows_a.shape[0])
    rng = np.random.default_rng(0)
    pick = rng.choice(windows_a.shape[0], size=n, replace=False)
    raw_win_a = windows_a[pick].reshape(-1)
    norm_win_a = normalizer.normalize(raw_win_a)
    _summarize(
        f"RAW teacher_commands (predictor windows u_k..u_k+Kp-1, {n} of {windows_a.shape[0]})",
        raw_win_a,
    )
    _summarize("NORMALIZED teacher_commands_norm (same windows)", norm_win_a)

    probe = 20.0
    plus_n = float(normalizer.normalize(np.array([probe], dtype=np.float32))[0])
    minus_n = float(normalizer.normalize(np.array([-probe], dtype=np.float32))[0])
    print(f"\nGate probe +/-{probe:g} N in normalized space")
    print(
        f"  +{probe:g} N -> {plus_n:.6g}   percentile of train teacher_commands_norm: "
        f"{_percentile_of(plus_n, norm_win_a):.3f}"
    )
    print(
        f"  -{probe:g} N -> {minus_n:.6g}   percentile of train teacher_commands_norm: "
        f"{_percentile_of(minus_n, norm_win_a):.3f}"
    )
    print(
        f"  |u_norm| >= |(+{probe:g} N)| fraction: "
        f"{float(np.mean(np.abs(norm_win_a) >= abs(plus_n))):.4f}"
    )
    lo = float(config["simulation"]["control_min_N"])
    hi = float(config["simulation"]["control_max_N"])
    lo_n = float(normalizer.normalize(np.array([lo], dtype=np.float32))[0])
    hi_n = float(normalizer.normalize(np.array([hi], dtype=np.float32))[0])
    print(f"\nConfig actuator range [{lo:g}, {hi:g}] N → normalized [{lo_n:.6g}, {hi_n:.6g}]")
    print(
        f"Empirical teacher_commands_norm range [{float(norm_win_a.min()):.6g}, {float(norm_win_a.max()):.6g}]"
    )


if __name__ == "__main__":
    main()

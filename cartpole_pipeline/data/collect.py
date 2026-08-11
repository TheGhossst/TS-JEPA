"""Collect CartPole trajectories with an LQR controller."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from cartpole_pipeline.controllers.lqr import LQRController
from cartpole_pipeline.envs.continuous_cartpole import ContinuousCartPoleEnv

FRAME_HEIGHT = 64
FRAME_WIDTH = 128
EPISODE_LENGTH = 100
DEFAULT_TRAIN_EPISODES = 200
DEFAULT_TEST_EPISODES = 40


def resize_frame(frame: np.ndarray, height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH) -> np.ndarray:
    """Resize an RGB frame to (height, width)."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def collect_trajectories(
    num_episodes: int,
    episode_length: int = EPISODE_LENGTH,
    seed: int = 0,
    render_mode: str = "rgb_array",
) -> dict[str, np.ndarray]:
    """
    Roll out LQR-controlled episodes and return stacked trajectory arrays.

    If an episode terminates early (pole falls), collection stops for that
    episode and remaining timesteps are padded by repeating the last frame
    and state with a zero action, preserving falling dynamics while keeping
    fixed tensor shapes.

    Returns:
        Dictionary with keys ``frames``, ``states``, and ``actions`` shaped
        (num_episodes, episode_length, ...).
    """
    env = ContinuousCartPoleEnv(render_mode=render_mode)
    controller = LQRController(force_limit=env.force_limit)

    frames = np.zeros((num_episodes, episode_length, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    states = np.zeros((num_episodes, episode_length, 4), dtype=np.float32)
    actions = np.zeros((num_episodes, episode_length, 1), dtype=np.float32)

    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        controller.reset()
        last_step = -1

        for step in range(episode_length):
            rgb = env.render()
            frames[ep, step] = resize_frame(rgb)

            states[ep, step] = state
            action = controller(state)
            actions[ep, step, 0] = action
            last_step = step

            state, _, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))
            if terminated or truncated:
                break

        # Pad remaining timesteps after early termination.
        if last_step >= 0 and last_step < episode_length - 1:
            pad_start = last_step + 1
            frames[ep, pad_start:] = frames[ep, last_step]
            states[ep, pad_start:] = states[ep, last_step]
            actions[ep, pad_start:] = 0.0

    env.close()

    # Empirical visual-diversity check: pixel MAPE between frame 0 and frame 50.
    if episode_length > 50:
        f0 = frames[:, 0].astype(np.float64)
        f50 = frames[:, 50].astype(np.float64)
        # +1 avoids division by zero on black background pixels (uint8).
        mape = float(np.mean(np.abs(f50 - f0) / (np.abs(f0) + 1.0)) * 100.0)
        print(f"Average pixel MAPE (frame 0 vs frame 50): {mape:.4f}%")

    return {"frames": frames, "states": states, "actions": actions}


def save_trajectories(data: dict[str, np.ndarray], output_path: Path) -> None:
    """Save trajectory arrays to a compressed .npz file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        frames=data["frames"],
        states=data["states"],
        actions=data["actions"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CartPole LQR trajectories.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--test-episodes", type=int, default=DEFAULT_TEST_EPISODES)
    parser.add_argument("--episode-length", type=int, default=EPISODE_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Collecting {args.train_episodes} training trajectories...")
    train_data = collect_trajectories(
        num_episodes=args.train_episodes,
        episode_length=args.episode_length,
        seed=args.seed,
    )
    train_path = args.output_dir / "train_data.npz"
    save_trajectories(train_data, train_path)
    print(f"Saved training data to {train_path}")

    print(f"Collecting {args.test_episodes} testing trajectories...")
    test_data = collect_trajectories(
        num_episodes=args.test_episodes,
        episode_length=args.episode_length,
        seed=args.seed + 10_000,
    )
    test_path = args.output_dir / "test_data.npz"
    save_trajectories(test_data, test_path)
    print(f"Saved testing data to {test_path}")


if __name__ == "__main__":
    main()

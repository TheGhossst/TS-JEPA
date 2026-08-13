"""Checkpoint path helpers for single-seed vs 5-seed layouts (IMPLEMENTATION CHOICE)."""

from __future__ import annotations

from pathlib import Path


def resolve_run_checkpoint(
    runs_root: Path,
    family: str,
    *,
    explicit: Path | None = None,
    seed: int | None = None,
) -> Path:
    """
    Resolve a JEPA / actor checkpoint.

    Preference:
      1) explicit path
      2) runs/<family>/seed_<seed>/best.pt when seed is set
      3) runs/<family>/best.pt (5-seed selection / promoted)

    Does not fall back to seed_0/best.pt. Pair eval with
    ``resolve_jepa_checkpoint_from_actor`` so the actor's stored encoder is used.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    family_dir = Path(runs_root) / family
    candidates: list[Path] = []
    if seed is not None:
        candidates.append(family_dir / f"seed_{int(seed)}" / "best.pt")
    candidates.append(family_dir / "best.pt")

    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No {family} checkpoint found. Tried: {tried}")


def resolve_jepa_checkpoint_from_actor(
    actor_checkpoint: Path | str,
    *,
    project_dir: Path | str | None = None,
) -> Path:
    """
    Load the JEPA encoder path recorded in an actor checkpoint.

    Actor training writes ``jepa_checkpoint`` into best.pt / last.pt. Evaluation
    and audit must use that file, not a default seed_0 encoder.
    """
    import torch

    actor_path = Path(actor_checkpoint)
    if not actor_path.exists():
        raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")
    payload = torch.load(actor_path, map_location="cpu", weights_only=False)
    stored = payload.get("jepa_checkpoint")
    if not stored:
        raise FileNotFoundError(
            f"Actor checkpoint {actor_path} has no jepa_checkpoint metadata. "
            "Retrain the actor so the JEPA path is recorded."
        )
    stored_path = Path(stored)
    candidates = [stored_path]
    if project_dir is not None:
        root = Path(project_dir)
        if not stored_path.is_absolute():
            candidates.append(root / stored_path)
    for path in candidates:
        if path.exists():
            return path.resolve()
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Actor checkpoint {actor_path} records jepa_checkpoint={stored!r} "
        f"but the file was not found. Tried: {tried}"
    )

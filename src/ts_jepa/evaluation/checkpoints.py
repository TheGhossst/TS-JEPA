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
      4) runs/<family>/seed_0/best.pt (common prelim layout)
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
    candidates.append(family_dir / "seed_0" / "best.pt")

    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No {family} checkpoint found. Tried: {tried}")

#!/usr/bin/env python
"""Backward-compatible entry point. Prefer ``scripts/pipeline/run_full_baseline.py``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "pipeline" / "run_full_baseline.py"

if __name__ == "__main__":
    sys.argv[0] = str(_SCRIPT)
    runpy.run_path(str(_SCRIPT), run_name="__main__")

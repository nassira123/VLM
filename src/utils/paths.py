"""Path utilities for run management."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path


def make_run_id() -> str:
    """Create a unique run id using timestamp + short hash."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    digest = hashlib.sha256(stamp.encode("utf-8")).hexdigest()[:6]
    return f"{stamp}-{digest}"


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return its Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

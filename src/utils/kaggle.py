"""Kaggle download helper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def _kaggle_config_path() -> Optional[Path]:
    env_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if env_dir:
        candidate = Path(env_dir) / "kaggle.json"
        if candidate.exists():
            return candidate
    default_path = Path.home() / ".kaggle" / "kaggle.json"
    if default_path.exists():
        return default_path
    return None


def kaggle_download(dataset: str, out_dir: str | Path, unzip: bool = True) -> None:
    """Download a Kaggle dataset via CLI."""
    cfg = _kaggle_config_path()
    if not cfg:
        raise FileNotFoundError(
            "Missing kaggle.json. Place it in ~/.kaggle/kaggle.json or set KAGGLE_CONFIG_DIR."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir)]
    if unzip:
        command.append("--unzip")
    print("[kaggle]", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed: {result.stderr.strip()}\nCommand: {' '.join(command)}"
        )

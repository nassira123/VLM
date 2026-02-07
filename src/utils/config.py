"""Configuration helpers for YAML-driven experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fill minimal defaults for required keys."""
    if "project_name" not in cfg:
        raise ValueError("Missing required key: project_name")
    if "paths" not in cfg:
        raise ValueError("Missing required key: paths")
    cfg.setdefault("seed", 42)
    cfg.setdefault("run_id", "")
    return cfg


def save_resolved_config(cfg: Dict[str, Any], out_path: str | Path) -> None:
    """Persist a resolved config to disk."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

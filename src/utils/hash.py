"""Stable hashing for cache keys."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import yaml


def stable_hash_dict(d: Dict[str, Any]) -> str:
    """Hash a dict with stable ordering."""
    dumped = yaml.safe_dump(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(dumped).hexdigest()[:12]


def stable_hash_file(path: str | Path, block_size: int = 1 << 20) -> str:
    """Hash file contents for cache keys."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:12]

import hashlib
from pathlib import Path
from typing import Any

import yaml


def hash_dict(payload: dict) -> str:
    dumped = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(dumped).hexdigest()[:12]


def hash_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:12]

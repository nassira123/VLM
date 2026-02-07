import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def resolve_config(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(config)
    return resolved


def env_override(config: Dict[str, Any], prefix: str = "VLM_") -> Dict[str, Any]:
    updated = copy.deepcopy(config)
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].lower().split("__")
        cursor = updated
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = value
    return updated

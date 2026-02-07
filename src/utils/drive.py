"""Google Drive helpers for Colab environments."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple


def mount_drive_if_needed() -> bool:
    """Mount Google Drive in Colab if available."""
    if "COLAB_RELEASE_TAG" in os.environ:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
        return True
    return False


def init_run_folders(project_name: str, run_id: str) -> Tuple[Path, Optional[Path]]:
    """Initialize local and Drive run directories."""
    if "COLAB_RELEASE_TAG" in os.environ:
        run_dir_local = Path("/content") / project_name / "runs" / run_id
    else:
        run_dir_local = Path.cwd() / "runs" / run_id
    run_dir_local.mkdir(parents=True, exist_ok=True)
    for sub in ["configs", "logs", "plots", "tables", "checkpoints", "cache/features"]:
        (run_dir_local / sub).mkdir(parents=True, exist_ok=True)

    drive_root = Path("/content/drive/MyDrive") / project_name
    run_dir_drive = None
    if drive_root.exists():
        run_dir_drive = drive_root / "runs" / run_id
        run_dir_drive.mkdir(parents=True, exist_ok=True)
        for sub in ["configs", "logs", "plots", "tables", "checkpoints", "cache/features"]:
            (run_dir_drive / sub).mkdir(parents=True, exist_ok=True)
    return run_dir_local, run_dir_drive


def mirror_to_drive(local_path: Path, drive_root: Optional[Path]) -> None:
    """Mirror a local artifact to Drive if mounted."""
    if not drive_root:
        print("[drive] Drive not mounted; skipping mirror for", local_path)
        return
    local_path = Path(local_path)
    rel = local_path.relative_to(local_path.parents[1])
    dest = drive_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, dest)

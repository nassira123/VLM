from pathlib import Path
from typing import Optional


def default_drive_root(project_name: str) -> Path:
    return Path("/content/drive/MyDrive") / project_name


def ensure_run_dirs(run_dir: Path, drive_root: Optional[Path] = None):
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["configs", "logs", "plots", "tables", "checkpoints", "cache/features"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    mirror_dir = None
    if drive_root:
        mirror_dir = drive_root / "runs" / run_dir.name
        mirror_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["configs", "logs", "plots", "tables", "checkpoints", "cache/features"]:
            (mirror_dir / sub).mkdir(parents=True, exist_ok=True)
    return mirror_dir


def mirror_file(local_path: Path, run_dir: Path, drive_root: Optional[Path]):
    if not drive_root:
        return
    rel = local_path.relative_to(run_dir)
    dest = drive_root / "runs" / run_dir.name / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(local_path.read_bytes())

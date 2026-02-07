"""CheXpert dataset processing and manifest creation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

CHEXPERT_5_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def _parse_patient_id(path: str) -> str:
    match = re.search(r"patient(\d+)", path)
    return match.group(1) if match else "unknown"


def _parse_study_id(path: str) -> Optional[str]:
    match = re.search(r"study(\d+)", path)
    return match.group(1) if match else None


def _parse_view(path: str) -> str:
    lowered = path.lower()
    if "frontal" in lowered:
        return "frontal"
    if "lateral" in lowered:
        return "lateral"
    return "unknown"


def _map_uncertain(value: float, policy: str, soft_weight: float) -> Tuple[float, float]:
    if pd.isna(value) or value == "":
        return 0.0, 0.0
    if value == -1:
        if policy == "u_ignore":
            return 0.0, 0.0
        if policy == "u_ones":
            return 1.0, 1.0
        if policy == "soft_u":
            return 0.5, soft_weight
    return float(value), 1.0


def build_manifest(
    chexpert_root: Path,
    run_dir: Path,
    view_filter: str = "frontal",
    soft_weight: float = 0.3,
) -> Tuple[pd.DataFrame, Path]:
    """Build a CheXpert manifest and return (df, manifest_path)."""
    rows: List[Dict[str, object]] = []
    for split_name, csv_name in [("train", "train.csv"), ("valid", "valid.csv")]:
        csv_path = chexpert_root / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}. Is CheXpert extracted?")
        df = pd.read_csv(csv_path)
        for _, record in df.iterrows():
            rel_path = record["Path"]
            view = _parse_view(rel_path)
            if view_filter and view != view_filter:
                continue
            full_path = chexpert_root / rel_path
            row: Dict[str, object] = {
                "image_path": str(full_path),
                "patient_id": _parse_patient_id(rel_path),
                "study_id": _parse_study_id(rel_path),
                "split": split_name,
                "view": view,
            }
            for label in CHEXPERT_5_LABELS:
                raw_value = record.get(label, "")
                row[f"label_{label}_raw"] = raw_value
                for policy in ["u_ignore", "u_ones", "soft_u"]:
                    mapped, weight = _map_uncertain(raw_value, policy, soft_weight)
                    row[f"label_{label}_{policy}"] = mapped
                    row[f"weight_{label}_{policy}"] = weight
            rows.append(row)

    manifest = pd.DataFrame(rows)
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tables_dir / "manifest_chexpert.csv"
    manifest.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return manifest, manifest_path


def qc_manifest(manifest: pd.DataFrame, run_dir: Path) -> Path:
    """Run QC checks and write a report to run_dir/tables."""
    missing = []
    corrupted = []
    sizes = []
    for path in manifest["image_path"].tolist()[:5000]:
        if not Path(path).exists():
            missing.append(path)
            continue
        try:
            with Image.open(path) as img:
                sizes.append(img.size)
        except Exception:
            corrupted.append(path)
    dup_paths = int(manifest["image_path"].duplicated().sum())
    leakage = set(manifest[manifest["split"] == "train"]["patient_id"]).intersection(
        set(manifest[manifest["split"] == "valid"]["patient_id"])
    )
    report = {
        "missing": len(missing),
        "corrupted": len(corrupted),
        "duplicate_paths": dup_paths,
        "patient_leakage": len(leakage),
        "sample_image_sizes": sizes[:50],
    }
    report_path = run_dir / "tables" / "qc_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path

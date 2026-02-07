import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

CHEXPERT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def parse_patient_study(path: str) -> Tuple[str, Optional[str]]:
    patient_match = re.search(r"patient(\d+)", path)
    study_match = re.search(r"study(\d+)", path)
    patient_id = patient_match.group(1) if patient_match else "unknown"
    study_id = study_match.group(1) if study_match else None
    return patient_id, study_id


def parse_view(path: str) -> str:
    lowered = path.lower()
    if "frontal" in lowered:
        return "frontal"
    if "lateral" in lowered:
        return "lateral"
    return "unknown"


@dataclass
class ChexpertRow:
    image_path: str
    patient_id: str
    study_id: Optional[str]
    split: str
    view: str
    labels: Dict[str, float]


class ChexpertDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        split: str,
        transform=None,
        label_policy: str = "u_ignore",
    ):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.label_policy = label_policy

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = [row[f"label_{label}_{self.label_policy}"] for label in CHEXPERT_LABELS]
        return image, labels


def map_uncertain(value: float, policy: str, soft_weight: float = 0.3) -> Tuple[float, float]:
    if pd.isna(value):
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
    output_csv: Path,
    view_filter: str = "frontal",
    soft_weight: float = 0.3,
) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for split_name, csv_name in [("train", "train.csv"), ("valid", "valid.csv")]:
        csv_path = chexpert_root / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}. Is CheXpert extracted?")
        df = pd.read_csv(csv_path)
        for _, record in df.iterrows():
            path = record["Path"]
            full_path = chexpert_root / path
            view = parse_view(path)
            if view_filter and view != view_filter:
                continue
            patient_id, study_id = parse_patient_study(path)
            row = {
                "image_path": str(full_path),
                "patient_id": patient_id,
                "study_id": study_id,
                "split": split_name,
                "view": view,
            }
            for label in CHEXPERT_LABELS:
                row[f"label_{label}_raw"] = record.get(label, "")
                for policy in ["u_ignore", "u_ones", "soft_u"]:
                    value, weight = map_uncertain(record.get(label, ""), policy, soft_weight)
                    row[f"label_{label}_{policy}"] = value
                    row[f"weight_{label}_{policy}"] = weight
            rows.append(row)
    manifest = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return manifest

"""Common dataset utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Base dataset backed by a manifest CSV."""

    def __init__(self, manifest: pd.DataFrame, label_columns: List[str], transform=None):
        self.manifest = manifest.reset_index(drop=True)
        self.label_columns = label_columns
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = row[self.label_columns].to_numpy(dtype=float)
        meta = {"image_path": row["image_path"], "patient_id": row.get("patient_id", "")}
        return image, labels, meta


def split_external_dataset(df: pd.DataFrame, seed: int, val_ratio: float = 0.2) -> pd.DataFrame:
    """Split external datasets into val/test for reporting only."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * val_ratio)
    df.loc[: split_idx - 1, "split"] = "val"
    df.loc[split_idx:, "split"] = "test"
    return df

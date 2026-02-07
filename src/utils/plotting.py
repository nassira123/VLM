"""Plotting utilities for EDA and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_bar(counts: Dict[str, int], title: str, output_path: Path) -> None:
    labels = list(counts.keys())
    values = list(counts.values())
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_hist(values: Sequence[int], title: str, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    for label, (x, y) in curves.items():
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=7)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_image_grid(image_paths: List[str], title: str, output_path: Path, max_images: int = 9) -> None:
    sample_paths = image_paths[:max_images]
    n = len(sample_paths)
    if n == 0:
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for idx, path in enumerate(sample_paths, start=1):
        plt.subplot(rows, cols, idx)
        img = Image.open(path).convert("RGB")
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

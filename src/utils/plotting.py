from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_label_distribution(counts: Dict[str, int], output_path: Path, title: str):
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


def plot_curve(curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], title: str, xlabel: str, ylabel: str, output_path: Path):
    plt.figure(figsize=(6, 5))
    for label, (x, y, _) in curves.items():
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=7)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

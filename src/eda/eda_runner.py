"""Lightweight EDA runner for CheXpert manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.plotting import plot_bar, plot_hist, plot_image_grid


def run_eda(manifest_csv: Path, run_dir: Path, labels: list[str]) -> Dict[str, Path]:
    df = pd.read_csv(manifest_csv)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    split_counts = df["split"].value_counts().to_dict()
    plot_bar(split_counts, "Split counts", plots_dir / "split_counts.png")

    label_counts = {label: int((df[f"label_{label}_raw"] == 1).sum()) for label in labels}
    plot_bar(label_counts, "Positive label counts", plots_dir / "label_pos_counts.png")

    uncertain_counts = {label: int((df[f"label_{label}_raw"] == -1).sum()) for label in labels}
    plot_bar(uncertain_counts, "Uncertain label counts", plots_dir / "uncertain_counts.png")

    sizes = []
    for path in df["image_path"].tolist()[:500]:
        try:
            from PIL import Image

            with Image.open(path) as img:
                sizes.append(img.size[0] * img.size[1])
        except Exception:
            continue
    if sizes:
        plot_hist(sizes, "Image size histogram (area)", plots_dir / "image_size_hist.png")

    plot_image_grid(df["image_path"].tolist(), "Sample images", plots_dir / "sample_grid.png")

    report_path = run_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# EDA Report",
                f"- Total images: {len(df)}",
                f"- Splits: {split_counts}",
                f"- Positive counts: {label_counts}",
                f"- Uncertain counts: {uncertain_counts}",
            ]
        )
    )

    return {
        "split_counts": plots_dir / "split_counts.png",
        "label_pos_counts": plots_dir / "label_pos_counts.png",
        "uncertain_counts": plots_dir / "uncertain_counts.png",
        "image_size_hist": plots_dir / "image_size_hist.png",
        "sample_grid": plots_dir / "sample_grid.png",
        "report": report_path,
    }

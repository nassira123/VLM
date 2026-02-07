import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, resolve_config, save_config
from src.utils.drive import ensure_run_dirs, mirror_file
from src.utils.logger import setup_logger
from src.utils.plotting import plot_label_distribution
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = resolve_config(load_config(args.config))
    set_seed(config["seed"])
    logger = setup_logger()

    manifest_csv = Path(config["paths"]["manifest_csv"])
    df = pd.read_csv(manifest_csv)

    run_id = config.get("run_id") or "eda"
    run_dir = Path(config["paths"]["runs_dir"]) / run_id
    drive_root = Path(config["paths"].get("drive_root", "")) if config["paths"].get("drive_root") else None
    ensure_run_dirs(run_dir, drive_root)
    plots_dir = run_dir / "plots"

    split_counts = df["split"].value_counts().to_dict()
    plot_label_distribution(split_counts, plots_dir / "split_counts.png", "Split counts")

    label_counts = {}
    for label in config["labels"]:
        label_counts[label] = int((df[f"label_{label}_raw"] == 1).sum())
    plot_label_distribution(label_counts, plots_dir / "label_pos_counts.png", "Positive label counts")

    logger.info(f"EDA plots saved to {plots_dir}")
    config_path = run_dir / "configs" / "resolved_config.yaml"
    save_config(config, config_path)
    mirror_file(config_path, run_dir, drive_root)
    mirror_file(plots_dir / "split_counts.png", run_dir, drive_root)
    mirror_file(plots_dir / "label_pos_counts.png", run_dir, drive_root)


if __name__ == "__main__":
    main()

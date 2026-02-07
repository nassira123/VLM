"""Prepare datasets: Kaggle download, CheXpert manifest, QC."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets.chexpert import build_manifest, qc_manifest
from src.utils.config import load_config, resolve_config, save_resolved_config
from src.utils.drive import init_run_folders, mirror_to_drive, mount_drive_if_needed
from src.utils.kaggle import kaggle_download
from src.utils.logger import get_logger
from src.utils.paths import make_run_id
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = resolve_config(load_config(args.config))
    set_seed(config["seed"])

    mount_drive_if_needed()
    run_id = config.get("run_id") or make_run_id()
    run_dir, drive_dir = init_run_folders(config["project_name"], run_id)
    log_path = run_dir / "logs" / "train.log"
    logger = get_logger(log_path=log_path)

    data_dir = Path(config["paths"]["data_dir"])
    chexpert_id = config["kaggle"]["datasets"]["chexpert"]
    if config["kaggle"].get("download", True):
        logger.info("Downloading datasets via Kaggle API.")
        kaggle_download(chexpert_id, data_dir / "chexpert")
        kaggle_download(config["kaggle"]["datasets"]["pneumonia"], data_dir / "pneumonia")
        kaggle_download(config["kaggle"]["datasets"]["covid"], data_dir / "covid")

    manifest_df, manifest_path = build_manifest(
        chexpert_root=data_dir / "chexpert",
        run_dir=run_dir,
        view_filter=config["chexpert"]["view"],
        soft_weight=config["chexpert"]["soft_u_weight"],
    )
    logger.info("CheXpert manifest written to %s", manifest_path)

    qc_path = qc_manifest(manifest_df, run_dir)
    logger.info("QC report written to %s", qc_path)

    resolved_path = run_dir / "configs" / "resolved_config.yaml"
    config["run_id"] = run_id
    save_resolved_config(config, resolved_path)

    for artifact in [manifest_path, qc_path, resolved_path, log_path]:
        mirror_to_drive(artifact, drive_dir)

    latest_link = run_dir.parent / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir, target_is_directory=True)


if __name__ == "__main__":
    main()

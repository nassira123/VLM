"""Run lightweight EDA plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eda.eda_runner import run_eda
from src.utils.config import load_config, resolve_config, save_resolved_config
from src.utils.drive import init_run_folders, mirror_to_drive, mount_drive_if_needed
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
    logger = get_logger(log_path=run_dir / "logs" / "train.log")

    manifest_csv = Path(config["paths"]["manifest_csv"])
    outputs = run_eda(manifest_csv, run_dir, config["labels"])

    resolved_path = run_dir / "configs" / "resolved_config.yaml"
    config["run_id"] = run_id
    save_resolved_config(config, resolved_path)

    for path in outputs.values():
        mirror_to_drive(path, drive_dir)
    mirror_to_drive(resolved_path, drive_dir)
    logger.info("EDA complete. Artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.datasets.chexpert import CHEXPERT_LABELS, build_manifest
from src.utils.config import load_config, resolve_config, save_config
from src.utils.drive import mirror_file
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def kaggle_download(dataset: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(dest),
        "--unzip",
    ]
    subprocess.run(command, check=True)


def qc_manifest(manifest: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    corrupted = []
    sizes = []
    for path in tqdm(manifest["image_path"], desc="QC images"):
        if not Path(path).exists():
            missing.append(path)
            continue
        try:
            with Image.open(path) as img:
                sizes.append(img.size)
        except Exception:
            corrupted.append(path)
    dup_paths = manifest["image_path"].duplicated().sum()
    leakage = set(manifest[manifest["split"] == "train"]["patient_id"]).intersection(
        set(manifest[manifest["split"] == "valid"]["patient_id"])
    )
    qc_report = {
        "missing": len(missing),
        "corrupted": len(corrupted),
        "duplicate_paths": int(dup_paths),
        "patient_leakage": len(leakage),
        "image_size_sample": sizes[:50],
    }
    with open(output_dir / "qc_report.json", "w", encoding="utf-8") as handle:
        json.dump(qc_report, handle, indent=2)
    if missing:
        (output_dir / "missing_paths.txt").write_text("\n".join(missing))
    if corrupted:
        (output_dir / "corrupted_paths.txt").write_text("\n".join(corrupted))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = resolve_config(load_config(args.config))
    set_seed(config["seed"])
    logger = setup_logger()

    data_dir = Path(config["paths"]["data_dir"])
    drive_root = Path(config["paths"].get("drive_root", "")) if config["paths"].get("drive_root") else None
    if config["kaggle"]["download"]:
        logger.info("Downloading datasets via Kaggle API...\n")
        kaggle_download(config["kaggle"]["datasets"]["chexpert"], data_dir / "chexpert")
        kaggle_download(config["kaggle"]["datasets"]["pneumonia"], data_dir / "pneumonia")
        kaggle_download(config["kaggle"]["datasets"]["covid"], data_dir / "covid")

    chexpert_root = data_dir / "chexpert"
    manifest_csv = data_dir / "chexpert_manifest.csv"
    manifest = build_manifest(
        chexpert_root=chexpert_root,
        output_csv=manifest_csv,
        view_filter=config["chexpert"]["view"],
        soft_weight=config["chexpert"]["soft_u_weight"],
    )
    logger.info(f"CheXpert manifest saved to {manifest_csv}\n")

    qc_dir = data_dir / "qc"
    qc_manifest(manifest, qc_dir)
    logger.info("QC completed.\n")

    summary = manifest["split"].value_counts().to_dict()
    label_summary = {}
    for label in CHEXPERT_LABELS:
        label_summary[label] = int((manifest[f"label_{label}_raw"] == 1).sum())
    summary_path = data_dir / "chexpert_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"split_counts": summary, "label_pos_counts": label_summary}, handle, indent=2)

    resolved_path = data_dir / "resolved_data_config.yaml"
    save_config(config, resolved_path)

    if drive_root:
        mirror_file(manifest_csv, data_dir, drive_root)
        mirror_file(summary_path, data_dir, drive_root)
        mirror_file(resolved_path, data_dir, drive_root)
        for report in ["qc_report.json", "missing_paths.txt", "corrupted_paths.txt"]:
            path = qc_dir / report
            if path.exists():
                mirror_file(path, data_dir, drive_root)


if __name__ == "__main__":
    main()

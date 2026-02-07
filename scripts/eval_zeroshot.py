"""Zero-shot evaluation entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import json
import pandas as pd
import torch

from src.datasets.chexpert import CHEXPERT_5_LABELS
from src.eval.zeroshot import eval_zeroshot
from src.models.openclip import load_biomedclip_hf, load_clip_openai_vitb32
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]["name"]
    if model_name == "biomedclip":
        model, preprocess, tokenizer = load_biomedclip_hf(device)
        model_id = "biomedclip"
    else:
        model, preprocess, tokenizer = load_clip_openai_vitb32(device)
        model_id = "vit-b-32"

    manifest_csv = Path(config["paths"]["manifest_csv"])
    df = pd.read_csv(manifest_csv)
    split = config["model"].get("split", "valid")
    df = df[df["split"] == split].reset_index(drop=True)
    df.attrs["manifest_csv"] = str(manifest_csv)
    df.attrs["split"] = split

    label_policy = config["model"]["label_policy"]
    label_columns = [f"label_{label}_{label_policy}" for label in CHEXPERT_5_LABELS]

    metrics = eval_zeroshot(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        manifest_df=df,
        label_columns=label_columns,
        labels=CHEXPERT_5_LABELS,
        templates=config["prompts"],
        batch_size=config["model"]["batch_size"],
        num_workers=config["model"]["num_workers"],
        device=device,
        thresholds=config["model"]["thresholds"],
        cache_dir=run_dir / "cache" / "features" / model_id / split,
        use_cache=config["model"].get("cache_features", False),
        model_id=model_id,
        image_size=config["model"]["image_size"],
        run_dir=run_dir,
    )

    metrics_path = run_dir / "tables" / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2))

    auc_table = pd.DataFrame([metrics["auc"]])
    auc_table.to_csv(run_dir / "tables" / "auc.csv", index=False)
    thresh_rows = []
    for threshold, values in metrics["thresholds"].items():
        row = {"threshold": threshold}
        row.update(values)
        thresh_rows.append(row)
    pd.DataFrame(thresh_rows).to_csv(run_dir / "tables" / "threshold_metrics.csv", index=False)

    resolved_path = run_dir / "configs" / "resolved_config.yaml"
    config["run_id"] = run_id
    save_resolved_config(config, resolved_path)

    for artifact in [
        metrics_path,
        summary_path,
        resolved_path,
        run_dir / "tables" / "auc.csv",
        run_dir / "tables" / "threshold_metrics.csv",
        run_dir / "plots" / "roc.png",
        run_dir / "plots" / "pr.png",
    ]:
        if artifact.exists():
            mirror_to_drive(artifact, drive_dir)
    logger.info("Zero-shot evaluation complete. Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()

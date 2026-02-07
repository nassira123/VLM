import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.chexpert import CHEXPERT_LABELS, ChexpertDataset
from src.models.openclip import load_openclip, load_openclip_hf
from src.utils.config import load_config, resolve_config, save_config
from src.utils.hash import hash_dict, hash_file
from src.utils.drive import ensure_run_dirs, mirror_file
from src.utils.logger import setup_logger
from src.utils.metrics import compute_auc_per_label, compute_curves, compute_map, threshold_metrics
from src.utils.plotting import plot_curve
from src.utils.seed import set_seed

PROMPT_TEMPLATES = {
    "chexpert": {
        "positive": "chest x-ray showing {label}",
        "negative": "chest x-ray with no {label}",
    },
    "pneumonia": {
        "positive": "chest x-ray showing pneumonia",
        "negative": "chest x-ray with no pneumonia",
    },
    "covid": {
        "positive": "chest x-ray consistent with COVID-19 pneumonia",
        "negative": "chest x-ray not consistent with COVID-19",
    },
}


def build_text_features(model, tokenizer, labels: List[str], device: torch.device):
    prompts = []
    for label in labels:
        prompts.append(PROMPT_TEMPLATES["chexpert"]["positive"].format(label=label))
        prompts.append(PROMPT_TEMPLATES["chexpert"]["negative"].format(label=label))
    tokens = tokenizer(prompts)
    tokens = tokens.to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def compute_logits(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    scores = image_features @ text_features.T
    scores = scores.reshape(scores.shape[0], -1, 2)
    logits = scores[:, :, 0] - scores[:, :, 1]
    return logits


def cache_key(config: Dict, manifest_csv: Path, split: str) -> str:
    payload = {
        "model": config["model"],
        "split": split,
        "manifest_hash": hash_file(manifest_csv),
    }
    return hash_dict(payload)


def load_or_compute_features(
    model,
    dataloader,
    device: torch.device,
    cache_dir: Path,
    cache_id: str,
    use_cache: bool,
):
    meta_path = cache_dir / "cache_meta.json"
    if use_cache and cache_dir.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("cache_id") == cache_id:
            parts = sorted(cache_dir.glob("part_*.pt"))
            if parts:
                features = [torch.load(part, map_location=device) for part in parts]
                return torch.cat(features, dim=0)
    features = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for idx, (images, _) in enumerate(tqdm(dataloader, desc="Encoding images")):
        images = images.to(device)
        with torch.no_grad():
            feat = model.encode_image(images)
        features.append(feat.cpu())
        torch.save(feat.cpu(), cache_dir / f"part_{idx:03d}.pt")
    meta_path.write_text(json.dumps({"cache_id": cache_id}, indent=2))
    return torch.cat(features, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = resolve_config(load_config(args.config))
    set_seed(config["seed"])
    logger = setup_logger()

    run_id = config.get("run_id") or "zeroshot_clip"
    run_dir = Path(config["paths"]["runs_dir"]) / run_id
    drive_root = Path(config["paths"].get("drive_root", "")) if config["paths"].get("drive_root") else None
    ensure_run_dirs(run_dir, drive_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]
    hf_id = config["model"].get("hf_id")
    if hf_id:
        model, preprocess, tokenizer = load_openclip_hf(hf_id, device)
    else:
        model, preprocess, tokenizer = load_openclip(model_name, pretrained, device)

    dataset = ChexpertDataset(
        manifest_csv=config["paths"]["manifest_csv"],
        split="valid",
        transform=preprocess,
        label_policy=config["model"]["label_policy"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["model"]["batch_size"],
        num_workers=config["model"]["num_workers"],
        shuffle=False,
    )

    cache_id = cache_key(config, Path(config["paths"]["manifest_csv"]), "valid")
    cache_dir = run_dir / "cache" / "features" / model_name.replace("/", "-") / "valid"

    image_features = load_or_compute_features(
        model,
        dataloader,
        device,
        cache_dir,
        cache_id,
        config["model"]["cache_features"],
    )

    text_features = build_text_features(model, tokenizer, CHEXPERT_LABELS, device)
    logits = compute_logits(image_features.to(device), text_features)
    y_score = torch.sigmoid(logits).cpu().numpy()

    y_true = dataset.df[[f"label_{label}_{config['model']['label_policy']}" for label in CHEXPERT_LABELS]].to_numpy()
    aucs = compute_auc_per_label(y_true, y_score, CHEXPERT_LABELS)
    mean_ap = compute_map(y_true, y_score)
    thresh = threshold_metrics(y_true, y_score, config["model"]["thresholds"])

    roc_curves, pr_curves = compute_curves(y_true, y_score, CHEXPERT_LABELS)
    roc_path = run_dir / "plots" / "roc.png"
    pr_path = run_dir / "plots" / "pr.png"
    if roc_curves:
        plot_curve(roc_curves, "ROC curves", "FPR", "TPR", roc_path)
        mirror_file(roc_path, run_dir, drive_root)
    if pr_curves:
        plot_curve(pr_curves, "PR curves", "Recall", "Precision", pr_path)
        mirror_file(pr_path, run_dir, drive_root)

    metrics = {"auc": aucs, "map": mean_ap, "thresholds": thresh}
    metrics_path = run_dir / "tables" / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    mirror_file(metrics_path, run_dir, drive_root)

    logger.info(f"Zero-shot evaluation complete. Metrics saved to {run_dir / 'tables' / 'metrics.json'}\n")
    config_path = run_dir / "configs" / "resolved_config.yaml"
    save_config(config, config_path)
    mirror_file(config_path, run_dir, drive_root)


if __name__ == "__main__":
    main()

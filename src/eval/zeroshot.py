"""Zero-shot evaluator for CLIP-style models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.common import ImageDataset
from src.utils.hash import stable_hash_dict, stable_hash_file
from src.utils.metrics import compute_all_metrics
from src.utils.plotting import plot_curves


def build_prompts(labels: List[str], templates: Dict[str, str]) -> List[str]:
    prompts = []
    for label in labels:
        prompts.append(templates["positive"].format(label=label))
        prompts.append(templates["negative"].format(label=label))
    return prompts


def cache_key(model_id: str, manifest_csv: Path, split: str, image_size: int) -> str:
    payload = {
        "model_id": model_id,
        "split": split,
        "image_size": image_size,
        "manifest_hash": stable_hash_file(manifest_csv),
    }
    return stable_hash_dict(payload)


def eval_zeroshot(
    model: torch.nn.Module,
    preprocess,
    tokenizer,
    manifest_df,
    label_columns: List[str],
    labels: List[str],
    templates: Dict[str, str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    thresholds: List[float],
    cache_dir: Path,
    use_cache: bool,
    model_id: str,
    image_size: int,
    run_dir: Path,
) -> Dict[str, object]:
    dataset = ImageDataset(manifest_df, label_columns=label_columns, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    prompts = build_prompts(labels, templates)
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    cache_id = cache_key(model_id, Path(manifest_df.attrs["manifest_csv"]), manifest_df.attrs["split"], image_size)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "cache_meta.json"

    y_true_list = []
    y_score_list = []

    if use_cache and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("cache_id") == cache_id:
            feature_files = sorted(cache_dir.glob("part_*.pt"))
            if feature_files:
                for idx, feature_path in enumerate(feature_files):
                    feats = torch.load(feature_path, map_location=device)
                    logits = feats @ text_features.T
                    logits = logits.reshape(logits.shape[0], -1, 2)
                    logits = logits[:, :, 0] - logits[:, :, 1]
                    y_score_list.append(torch.sigmoid(logits).cpu().numpy())
                y_true_list = [manifest_df[label_columns].to_numpy(dtype=float)]
    if not y_score_list:
        feature_files = []
        for idx, (images, labels_batch, _) in enumerate(tqdm(dataloader, desc="Zero-shot eval")):
            images = images.to(device)
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            logits = logits.reshape(logits.shape[0], -1, 2)
            logits = logits[:, :, 0] - logits[:, :, 1]
            y_score_list.append(torch.sigmoid(logits).cpu().numpy())
            y_true_list.append(labels_batch.numpy())
            if use_cache:
                torch.save(image_features.cpu(), cache_dir / f"part_{idx:03d}.pt")
                feature_files.append(cache_dir / f"part_{idx:03d}.pt")
        if use_cache:
            meta_path.write_text(json.dumps({"cache_id": cache_id}, indent=2))

    y_true = np.concatenate(y_true_list, axis=0)
    y_score = np.concatenate(y_score_list, axis=0)

    metrics = compute_all_metrics(y_true, y_score, labels, thresholds)

    roc_curves = {}
    pr_curves = {}
    for idx, label in enumerate(labels):
        label_true = y_true[:, idx]
        label_score = y_score[:, idx]
        if len(np.unique(label_true)) < 2:
            continue
        from sklearn.metrics import roc_curve, precision_recall_curve

        fpr, tpr, _ = roc_curve(label_true, label_score)
        precision, recall, _ = precision_recall_curve(label_true, label_score)
        roc_curves[label] = (fpr, tpr)
        pr_curves[label] = (recall, precision)

    if roc_curves:
        plot_curves(roc_curves, "ROC curves", "FPR", "TPR", run_dir / "plots" / "roc.png")
    if pr_curves:
        plot_curves(pr_curves, "PR curves", "Recall", "Precision", run_dir / "plots" / "pr.png")

    return metrics

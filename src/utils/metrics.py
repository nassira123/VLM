"""Metrics for multi-label evaluation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def auc_per_label(y_true: np.ndarray, y_score: np.ndarray, labels: List[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for idx, label in enumerate(labels):
        result[label] = _safe_auc(y_true[:, idx], y_score[:, idx])
    valid = [v for v in result.values() if not np.isnan(v)]
    result["macro"] = float(np.mean(valid)) if valid else float("nan")
    return result


def map_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    scores = []
    for idx in range(y_true.shape[1]):
        if len(np.unique(y_true[:, idx])) < 2:
            continue
        scores.append(average_precision_score(y_true[:, idx], y_score[:, idx]))
    return float(np.mean(scores)) if scores else float("nan")


def threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: List[float],
) -> Dict[str, Dict[str, float]]:
    results = {}
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = (y_pred * y_true).sum(axis=0)
        fp = (y_pred * (1 - y_true)).sum(axis=0)
        fn = ((1 - y_pred) * y_true).sum(axis=0)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)
        results[str(threshold)] = {
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "micro_precision": float(tp.sum() / max((tp + fp).sum(), 1)),
            "micro_recall": float(tp.sum() / max((tp + fn).sum(), 1)),
            "micro_f1": float(2 * tp.sum() / max((2 * tp.sum() + fp.sum() + fn.sum()), 1)),
        }
    return results


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    per_label = np.mean((y_score - y_true) ** 2, axis=0)
    return float(np.mean(per_label))


def ece_score(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece_per_label = []
    for label_idx in range(y_true.shape[1]):
        label_true = y_true[:, label_idx]
        label_score = y_score[:, label_idx]
        ece = 0.0
        for i in range(n_bins):
            mask = (label_score >= bins[i]) & (label_score < bins[i + 1])
            if not np.any(mask):
                continue
            acc = np.mean(label_true[mask])
            conf = np.mean(label_score[mask])
            ece += np.abs(acc - conf) * np.mean(mask)
        ece_per_label.append(ece)
    return float(np.mean(ece_per_label))


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: List[str],
    thresholds: List[float],
) -> Dict[str, object]:
    return {
        "auc": auc_per_label(y_true, y_score, labels),
        "map": map_score(y_true, y_score),
        "thresholds": threshold_metrics(y_true, y_score, thresholds),
        "ece": ece_score(y_true, y_score),
        "brier": brier_score(y_true, y_score),
    }

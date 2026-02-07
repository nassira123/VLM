from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def compute_auc_per_label(y_true: np.ndarray, y_score: np.ndarray, labels: List[str]) -> Dict[str, float]:
    result = {}
    for idx, label in enumerate(labels):
        result[label] = safe_auc(y_true[:, idx], y_score[:, idx])
    valid = [v for v in result.values() if not np.isnan(v)]
    result["macro"] = float(np.mean(valid)) if valid else float("nan")
    return result


def compute_map(y_true: np.ndarray, y_score: np.ndarray) -> float:
    scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        scores.append(average_precision_score(y_true[:, i], y_score[:, i]))
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
            "micro_f1": float(
                2 * tp.sum() / max((2 * tp.sum() + fp.sum() + fn.sum()), 1)
            ),
        }
    return results


def compute_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: List[str],
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    roc = {}
    pr = {}
    for idx, label in enumerate(labels):
        if len(np.unique(y_true[:, idx])) < 2:
            continue
        fpr, tpr, roc_thresh = roc_curve(y_true[:, idx], y_score[:, idx])
        precision, recall, pr_thresh = precision_recall_curve(y_true[:, idx], y_score[:, idx])
        roc[label] = (fpr, tpr, roc_thresh)
        pr[label] = (precision, recall, pr_thresh)
    return roc, pr

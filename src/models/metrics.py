from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def get_binary_probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1 or logits.shape[1] == 1:
        return torch.sigmoid(logits.reshape(-1))
    return torch.softmax(logits, dim=1)[:, 1]


def compute_binary_classification_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> Dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)
    y_pred = (y_score >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = 0.0

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_binary_threshold(
    y_true,
    y_score,
    strategy: str = "max_f1",
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> Dict[str, float | int | str]:
    """
    Select a decision threshold on validation scores for heavily imbalanced fraud detection.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    candidates = []
    for idx, threshold in enumerate(thresholds):
        p = precision[idx]
        r = recall[idx]
        if min_precision is not None and p < min_precision:
            continue
        if min_recall is not None and r < min_recall:
            continue
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        candidates.append((float(threshold), float(p), float(r), float(f1)))

    if not candidates:
        selected_threshold = 0.5
        selected_metrics = compute_binary_classification_metrics(y_true, y_score, threshold=selected_threshold)
        selected_metrics["threshold_strategy"] = "fallback_0.5"
        return selected_metrics

    if strategy == "max_f1":
        selected_threshold, _, _, _ = max(candidates, key=lambda item: (item[3], item[2], item[1]))
    elif strategy == "max_recall":
        selected_threshold, _, _, _ = max(candidates, key=lambda item: (item[2], item[1], item[3]))
    elif strategy == "max_precision":
        selected_threshold, _, _, _ = max(candidates, key=lambda item: (item[1], item[2], item[3]))
    else:
        raise ValueError(f"Unknown threshold selection strategy: {strategy}")

    selected_metrics = compute_binary_classification_metrics(y_true, y_score, threshold=selected_threshold)
    selected_metrics["threshold_strategy"] = strategy
    return selected_metrics


def update_epoch_metric_buffer(
    buffers: Dict[str, Tuple[list, list]],
    stage: str,
    y: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    y_score = get_binary_probabilities_from_logits(logits)
    y_true_buffer, y_score_buffer = buffers.setdefault(stage, ([], []))
    y_true_buffer.append(y.detach().cpu().numpy())
    y_score_buffer.append(y_score.detach().cpu().numpy())


def compute_buffered_epoch_metrics(
    buffers: Dict[str, Tuple[list, list]],
    stage: str,
) -> Dict[str, float | int]:
    y_true_buffer, y_score_buffer = buffers.get(stage, ([], []))
    if not y_true_buffer:
        return {}
    y_true = np.concatenate(y_true_buffer)
    y_score = np.concatenate(y_score_buffer)
    buffers[stage] = ([], [])
    return compute_binary_classification_metrics(y_true, y_score)

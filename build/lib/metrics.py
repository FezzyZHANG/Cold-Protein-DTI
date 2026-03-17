"""Classification and ranking metrics for DTI experiments."""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def binary_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    """Compute conservative binary metrics without requiring scikit-learn."""
    y_true = labels.astype(np.int64)
    y_score = scores.astype(np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    accuracy = _safe_divide(tp + tn, len(y_true))

    f1: float | None = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))

    mcc_den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = None if mcc_den == 0 else float(((tp * tn) - (fp * fn)) / mcc_den)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    auroc: float | None = None
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(y_score, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
        auroc = float((np.sum(ranks[pos_mask]) - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))

    auprc: float | None = None
    if n_pos > 0:
        order = np.argsort(-y_score, kind="mergesort")
        y_sorted = y_true[order]
        tp_cum = np.cumsum(y_sorted == 1)
        fp_cum = np.cumsum(y_sorted == 0)
        precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        recall_curve = tp_cum / n_pos
        ap = 0.0
        prev_recall = 0.0
        for idx, label in enumerate(y_sorted):
            if label != 1:
                continue
            ap += float(recall_curve[idx] - prev_recall) * float(precision_curve[idx])
            prev_recall = float(recall_curve[idx])
        auprc = float(ap)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "auroc": auroc,
        "auprc": auprc,
        "n_examples": float(len(y_true)),
        "n_positive": float(n_pos),
        "n_negative": float(n_neg),
    }


def _ranking_metrics_for_group(
    labels: np.ndarray,
    scores: np.ndarray,
    ks: list[int],
    ef_fractions: list[float],
) -> dict[str, float | None]:
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = labels[order]
    n_total = len(y_sorted)
    n_pos = int(np.sum(y_sorted))

    metrics: dict[str, float | None] = {}
    for k in ks:
        top_k = min(k, n_total)
        if top_k <= 0:
            metrics[f"recall@{k}"] = None
            metrics[f"hit@{k}"] = None
            continue
        top_labels = y_sorted[:top_k]
        top_pos = int(np.sum(top_labels))
        metrics[f"recall@{k}"] = _safe_divide(top_pos, n_pos)
        metrics[f"hit@{k}"] = 1.0 if top_pos > 0 else 0.0

    for frac in ef_fractions:
        top_k = max(1, ceil(n_total * frac))
        top_labels = y_sorted[:top_k]
        top_pos = int(np.sum(top_labels))
        baseline = (n_pos / n_total) * top_k if n_total > 0 else 0.0
        metrics[f"ef@{int(frac * 100)}pct"] = _safe_divide(top_pos, baseline)

    return metrics


def _average_metric_dicts(metric_dicts: list[dict[str, float | None]]) -> dict[str, float | None]:
    if not metric_dicts:
        return {}

    keys = sorted({key for item in metric_dicts for key in item})
    averaged: dict[str, float | None] = {}
    for key in keys:
        values = [item[key] for item in metric_dicts if key in item and item[key] is not None]
        averaged[key] = None if not values else float(np.mean(values))
    return averaged


def ranking_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    group_ids: list[str],
    ks: list[int],
    ef_fractions: list[float],
) -> dict[str, Any]:
    """Compute pooled and per-target macro ranking metrics."""
    pooled = _ranking_metrics_for_group(labels=labels, scores=scores, ks=ks, ef_fractions=ef_fractions)

    per_target: list[dict[str, float | None]] = []
    unique_groups = sorted(set(group_ids))
    for group_id in unique_groups:
        mask = np.asarray([current == group_id for current in group_ids], dtype=bool)
        group_labels = labels[mask]
        if len(group_labels) == 0 or np.sum(group_labels) == 0:
            continue
        group_scores = scores[mask]
        per_target.append(_ranking_metrics_for_group(group_labels, group_scores, ks, ef_fractions))

    return {
        "pooled": pooled,
        "macro_by_target": _average_metric_dicts(per_target),
        "n_targets_with_positive": len(per_target),
    }


def build_metrics_payload(
    labels: np.ndarray,
    scores: np.ndarray,
    group_ids: list[str],
    threshold: float,
    ks: list[int],
    ef_fractions: list[float],
    loss: float | None = None,
) -> dict[str, Any]:
    classification = binary_classification_metrics(labels=labels, scores=scores, threshold=threshold)
    ranking = ranking_metrics(labels=labels, scores=scores, group_ids=group_ids, ks=ks, ef_fractions=ef_fractions)

    payload: dict[str, Any] = {
        "classification": classification,
        "ranking": ranking,
    }
    if loss is not None:
        payload["loss"] = float(loss)
    return payload


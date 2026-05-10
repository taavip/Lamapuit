"""Pixel-level segmentation metrics and threshold sweep.

Ported from scripts/train_deeplabv3plus_manual_masks.py lines 693-827.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def compute_metrics(
    probs: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute pixel-level Dice, IoU, precision, recall on valid pixels only."""
    v = valid > 0.5
    if not np.any(v):
        return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    pred = (probs >= threshold) & v
    gt = (target > 0.5) & v

    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2.0 * precision * recall) / (precision + recall + EPS)
    acc = (tp + tn) / (tp + fp + fn + tn + EPS)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }


def accumulate_pixel_metrics(
    prob_list: list[np.ndarray],
    target_list: list[np.ndarray],
    valid_list: list[np.ndarray],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Global pixel-level metrics pooled across multiple arrays."""
    tp = fp = fn = tn = 0.0
    for probs, target, valid in zip(prob_list, target_list, valid_list):
        v = valid > 0.5
        if not np.any(v):
            continue
        pred = (probs >= threshold) & v
        gt = (target > 0.5) & v
        tp += float(np.logical_and(pred, gt).sum())
        fp += float(np.logical_and(pred, ~gt).sum())
        fn += float(np.logical_and(~pred, gt).sum())
        tn += float(np.logical_and(~pred, ~gt).sum())

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2.0 * precision * recall) / (precision + recall + EPS)
    acc = (tp + tn) / (tp + fp + fn + tn + EPS)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def threshold_sweep(
    prob_list: list[np.ndarray],
    target_list: list[np.ndarray],
    valid_list: list[np.ndarray],
    thresholds: list[float] | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Sweep thresholds and return (best_by_f1_row, all_rows)."""
    if thresholds is None:
        thresholds = [round(float(x), 2) for x in np.arange(0.10, 0.91, 0.05)]

    rows: list[dict[str, float]] = []
    best: dict[str, float] | None = None

    for thr in thresholds:
        row = accumulate_pixel_metrics(prob_list, target_list, valid_list, threshold=thr)
        row["threshold"] = float(thr)
        rows.append(row)
        if best is None or row["f1"] > best["f1"]:
            best = row

    if best is None:
        raise RuntimeError("Threshold sweep found no valid metrics")
    return best, rows

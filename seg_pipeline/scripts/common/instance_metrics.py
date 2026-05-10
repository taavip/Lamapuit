"""Instance segmentation metrics: AP@IoU, mAP@50:95, count error, size bias.

All metrics are computed against ground-truth polygon instances at configurable
IoU thresholds. Predictions are lists of (score, mask/polygon) pairs; GTs are
lists of polygon instances.

Design follows COCO evaluation conventions:
    - AP is computed per-image then averaged across the test set
    - IoU between two instances is intersection-over-union of pixel masks or polygon areas
    - Matching uses greedy assignment (highest-score prediction matched first)
    - Unmatched predictions are FP; unmatched GTs are FN
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# Polygon IoU
# ---------------------------------------------------------------------------


def polygon_iou(poly_a, poly_b) -> float:
    """Intersection-over-union of two Shapely polygons."""
    from shapely.validation import make_valid
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0
    try:
        a = make_valid(poly_a)
        b = make_valid(poly_b)
        intersection = a.intersection(b).area
        union = a.union(b).area
    except Exception:
        return 0.0
    return intersection / union if union > 0 else 0.0


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union of two binary masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Matching at a single IoU threshold
# ---------------------------------------------------------------------------


def match_predictions_to_gt(
    pred_scores: Sequence[float],
    pred_polys: list,
    gt_polys: list,
    iou_threshold: float = 0.5,
) -> tuple[list[bool], list[bool]]:
    """Greedy matching: highest-score predictions matched to GT first.

    Returns:
        pred_matched: bool list — True if prediction matched a GT
        gt_matched:   bool list — True if GT was matched by a prediction
    """
    n_pred = len(pred_scores)
    n_gt = len(gt_polys)
    pred_matched = [False] * n_pred
    gt_matched = [False] * n_gt

    if n_pred == 0 or n_gt == 0:
        return pred_matched, gt_matched

    # Sort predictions by descending score
    sorted_idx = sorted(range(n_pred), key=lambda i: -pred_scores[i])

    for pred_i in sorted_idx:
        best_iou = iou_threshold - 1e-6
        best_gt = -1
        for gt_j in range(n_gt):
            if gt_matched[gt_j]:
                continue
            iou = polygon_iou(pred_polys[pred_i], gt_polys[gt_j])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_j
        if best_gt >= 0:
            pred_matched[pred_i] = True
            gt_matched[best_gt] = True

    return pred_matched, gt_matched


# ---------------------------------------------------------------------------
# Precision-recall curve + AP
# ---------------------------------------------------------------------------


def compute_ap(
    all_scores: list[float],
    all_tp: list[bool],
    n_gt_total: int,
) -> float:
    """Compute Average Precision from sorted detections.

    Uses COCO 101-point interpolation.
    """
    if n_gt_total == 0:
        return 0.0

    order = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp, dtype=float)[order]
    fp = 1 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-8)
    recall = cum_tp / n_gt_total

    # 101-point interpolation
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        p_at_r = precision[recall >= thr].max() if (recall >= thr).any() else 0.0
        ap += p_at_r / 101
    return float(ap)


# ---------------------------------------------------------------------------
# Full instance metric computation
# ---------------------------------------------------------------------------


class InstanceMetrics:
    """Accumulate per-image predictions and compute final metrics.

    Usage:
        metrics = InstanceMetrics()
        for image in test_images:
            metrics.update(pred_scores, pred_polys, gt_polys)
        results = metrics.compute()
    """

    IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)  # COCO: 0.50:0.95:0.05 (10 thresholds)

    def __init__(self) -> None:
        # Accumulated per threshold: list of (score, is_tp)
        self._detections: dict[float, list[tuple[float, bool]]] = {
            thr: [] for thr in self.IOU_THRESHOLDS
        }
        self._n_gt = 0
        self._pred_areas: list[float] = []
        self._gt_areas: list[float] = []
        self._n_pred_per_image: list[int] = []
        self._n_gt_per_image: list[int] = []

    def update(
        self,
        pred_scores: list[float],
        pred_polys: list,
        gt_polys: list,
    ) -> None:
        """Add predictions for one image (test-stripe patch or full image)."""
        self._n_gt += len(gt_polys)
        self._n_pred_per_image.append(len(pred_polys))
        self._n_gt_per_image.append(len(gt_polys))
        self._gt_areas.extend(p.area for p in gt_polys)
        self._pred_areas.extend(p.area for p in pred_polys)

        for thr in self.IOU_THRESHOLDS:
            pred_matched, _ = match_predictions_to_gt(pred_scores, pred_polys, gt_polys, thr)
            for i, score in enumerate(pred_scores):
                self._detections[thr].append((score, pred_matched[i]))

    def compute(self) -> dict:
        """Compute AP@50, AP@75, mAP@50:95, precision@50, recall@50, count error, size Δ."""
        aps = {}
        for thr in self.IOU_THRESHOLDS:
            dets = self._detections[thr]
            if not dets:
                aps[thr] = 0.0
                continue
            scores = [d[0] for d in dets]
            is_tp = [d[1] for d in dets]
            aps[thr] = compute_ap(scores, is_tp, self._n_gt)

        ap50 = aps[0.50]
        ap75 = aps[0.75]
        map_50_95 = float(np.mean(list(aps.values())))

        # Precision and recall at IoU=0.50
        dets_50 = self._detections[0.50]
        if dets_50:
            scores = np.array([d[0] for d in dets_50])
            tp_arr = np.array([d[1] for d in dets_50], dtype=float)
            order = np.argsort(-scores)
            tp_sorted = tp_arr[order]
            fp_sorted = 1 - tp_sorted
            tp_total = tp_sorted.sum()
            fp_total = fp_sorted.sum()
            precision_50 = float(tp_total / (tp_total + fp_total + 1e-8))
            recall_50 = float(tp_total / (self._n_gt + 1e-8))
        else:
            precision_50 = recall_50 = 0.0

        # Count error: (N_pred_total - N_gt_total) / N_gt_total
        n_pred_total = sum(self._n_pred_per_image)
        n_gt_total = self._n_gt
        count_error = float(n_pred_total - n_gt_total) / max(1, n_gt_total)

        # Size Δ: mean predicted area - mean GT area (in pixels)
        mean_pred_area = float(np.mean(self._pred_areas)) if self._pred_areas else 0.0
        mean_gt_area = float(np.mean(self._gt_areas)) if self._gt_areas else 0.0
        size_delta = mean_pred_area - mean_gt_area

        return {
            "ap50": ap50,
            "ap75": ap75,
            "map_50_95": map_50_95,
            "precision_50": precision_50,
            "recall_50": recall_50,
            "n_gt": n_gt_total,
            "n_pred": n_pred_total,
            "count_error": count_error,
            "mean_pred_area_px": mean_pred_area,
            "mean_gt_area_px": mean_gt_area,
            "size_delta_px": size_delta,
            "ap_per_threshold": {f"{thr:.2f}": float(ap) for thr, ap in aps.items()},
        }


# ---------------------------------------------------------------------------
# Helper: convert YOLO result masks to Shapely polygons
# ---------------------------------------------------------------------------


def yolo_result_to_polygons(result, conf_threshold: float = 0.25) -> tuple[list[float], list]:
    """Extract scored Shapely polygons from an ultralytics YOLO result object.

    Returns (scores, polygons) in the same order.
    """
    from shapely.geometry import Polygon as SPolygon

    scores, polys = [], []
    if result.masks is None:
        return scores, polys

    from shapely.validation import make_valid
    for i, conf in enumerate(result.boxes.conf.cpu().numpy()):
        if conf < conf_threshold:
            continue
        xy = result.masks.xy[i]  # (N, 2) array of pixel coords
        if len(xy) < 3:
            continue
        poly = make_valid(SPolygon(xy))
        if poly.is_empty or poly.area < 1:
            continue
        scores.append(float(conf))
        polys.append(poly)
    return scores, polys

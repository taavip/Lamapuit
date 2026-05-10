"""Extended metrics for ablation study: Boundary IoU, clDice, AP@IoU."""

import numpy as np
from scipy.ndimage import binary_dilation, label
from skimage.morphology import skeletonize


def boundary_iou(
    pred_bin: np.ndarray, gt_bin: np.ndarray, dilation_px: int = 3
) -> float:
    """Intersection over Union computed only on boundary regions.

    Measures shape boundary alignment — important for thin, elongated structures.
    Only pixels within `dilation_px` of either predicted or GT boundary are scored.

    Args:
        pred_bin: Binary prediction (H, W), dtype bool or uint8/float32.
        gt_bin: Binary ground truth (H, W), dtype bool or uint8/float32.
        dilation_px: Dilation radius in pixels for boundary detection.

    Returns:
        Boundary IoU in [0, 1].
    """
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)

    # Extract boundary rings via dilation XOR
    pred_dilated = binary_dilation(pred_bin, iterations=dilation_px)
    pred_boundary = pred_dilated ^ binary_dilation(~pred_bin, iterations=dilation_px)

    gt_dilated = binary_dilation(gt_bin, iterations=dilation_px)
    gt_boundary = gt_dilated ^ binary_dilation(~gt_bin, iterations=dilation_px)

    # Region of interest: union of both boundaries
    boundary_region = pred_boundary | gt_boundary

    # IoU on boundary region only
    tp = float((pred_bin & gt_bin & boundary_region).sum())
    fp = float((pred_bin & ~gt_bin & boundary_region).sum())
    fn = float((~pred_bin & gt_bin & boundary_region).sum())

    denominator = tp + fp + fn
    return tp / denominator if denominator > 0 else 0.0


def cldice_metric(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """Centerline Dice: Dice coefficient of skeletons.

    Measures connectivity and topology of thin structures (e.g., log centerlines).
    Complements pixel-level Dice by emphasizing skeleton preservation.

    Args:
        pred_bin: Binary prediction (H, W), dtype bool or uint8/float32.
        gt_bin: Binary ground truth (H, W), dtype bool or uint8/float32.

    Returns:
        clDice in [0, 1]. Returns 0 if either skeleton is empty.
    """
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)

    # Skip if either region is empty
    if not pred_bin.any() or not gt_bin.any():
        return 0.0

    # Extract skeletons
    skel_pred = skeletonize(pred_bin).astype(bool)
    skel_gt = skeletonize(gt_bin).astype(bool)

    # Skip if skeletonization failed (e.g., single-pixel regions)
    if not skel_pred.any() or not skel_gt.any():
        return 0.0

    # Skeleton-level recall and precision
    tprec = (skel_pred & gt_bin).sum() / skel_pred.sum()  # Pred skeleton coverage in GT
    tsens = (skel_gt & pred_bin).sum() / skel_gt.sum()    # GT skeleton coverage in Pred

    # Harmonic mean
    denominator = tprec + tsens
    return (2 * tprec * tsens) / denominator if denominator > 0 else 0.0


def ap_at_iou(
    pred_bin: np.ndarray, gt_bin: np.ndarray, iou_thresholds: tuple = (0.25, 0.50)
) -> dict[str, float]:
    """Average Precision at multiple IoU thresholds (component-level detection).

    Measures per-instance detection quality: how many predicted components overlap
    with GT components at varying IoU thresholds.

    Args:
        pred_bin: Binary prediction (H, W), dtype bool or uint8/float32.
        gt_bin: Binary ground truth (H, W), dtype bool or uint8/float32.
        iou_thresholds: Tuple of IoU thresholds, e.g., (0.25, 0.50).

    Returns:
        Dict with keys like "ap_iou_25", "ap_iou_50" (as percentages, e.g., 85.5).
    """
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)

    # Label connected components
    pred_labeled, n_pred = label(pred_bin)
    gt_labeled, n_gt = label(gt_bin)

    if n_pred == 0 or n_gt == 0:
        # No predictions or no ground truth → AP = 0
        return {f"ap_iou_{int(t * 100)}": 0.0 for t in iou_thresholds}

    # Build bipartite matching: each predicted component to best GT component
    results = {}
    for threshold in iou_thresholds:
        # For each predicted component, find best matching GT component
        matches = 0
        for pred_id in range(1, n_pred + 1):
            pred_mask = pred_labeled == pred_id
            pred_area = pred_mask.sum()

            # Find best IoU with any GT component
            best_iou = 0.0
            for gt_id in range(1, n_gt + 1):
                gt_mask = gt_labeled == gt_id
                intersection = (pred_mask & gt_mask).sum()
                union = (pred_mask | gt_mask).sum()
                iou = intersection / union if union > 0 else 0.0
                best_iou = max(best_iou, iou)

            # Count as correct if IoU exceeds threshold
            if best_iou >= threshold:
                matches += 1

        # AP = recall at this threshold (assuming all predictions are positive)
        ap = (matches / n_pred) * 100 if n_pred > 0 else 0.0
        results[f"ap_iou_{int(threshold * 100)}"] = ap

    return results

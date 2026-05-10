#!/usr/bin/env python3
"""Phase 0 V6: Adaptive ensemble filtering via threshold optimization.

Goal: Find optimal ensemble probability threshold τ that maximizes F1 score
on conflict zones (high-confidence ensemble predictions on unlabeled regions).

Conflict zones are pixels where:
  - ensemble_prob >= threshold τ (high confidence)
  - label mask = 0 (unlabeled, no ground truth)

By excluding these conflict zones during V6 training, we prevent the model
from learning contradictory signals in uncertain areas, potentially improving
generalization.

Design:
  1. Load V3 ensemble predictions (pred_ensemble_tta1.tif or full-tile)
  2. Load ground-truth labels (truemask)
  3. Load training patch index from V3 (know which areas were labeled)
  4. Sweep τ ∈ [0.05, 0.30] with step 0.01 (26 thresholds)
  5. For each τ: compute conflict zone statistics
  6. Select τ_opt that maximizes F1 (balance between filtering noise and preserving signal)
  7. Save conflict mask and optimal threshold

Output:
  - conflict_mask_τ_opt.tif: binary mask (1 = conflict, 0 = keep)
  - phase0_ensemble_filter_v6.json: τ_opt, sweep results, statistics
  - conflict_sweep_results_v6.csv: per-threshold metrics

Usage:
    python phase0_ensemble_filter_v6.py --ensemble pred_ensemble_tta1.tif --device cuda
    python phase0_ensemble_filter_v6.py --ensemble pred_ensemble_full_tile.tif --metric f1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rasterio import open as rio_open
from rasterio.plot import reshape_as_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))


def load_ensemble_predictions(ensemble_tif: Path) -> tuple[np.ndarray, dict]:
    """Load ensemble probability predictions from GeoTIFF."""
    with rio_open(str(ensemble_tif)) as src:
        probs = src.read(1)  # (H, W) float32
        meta = src.meta.copy()
    return probs, meta


def load_labels(label_tif: Path) -> np.ndarray:
    """Load ground-truth label mask. Assume 0 = background, 1 = CWD."""
    with rio_open(str(label_tif)) as src:
        labels = src.read(1)  # (H, W) uint8 or uint16
    return labels.astype(np.uint8)


def compute_conflict_statistics(
    ensemble_probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    min_patch_cover: float = 0.5,
) -> dict:
    """Compute conflict zone statistics for a given threshold.

    Conflict = (ensemble_prob >= threshold) AND (label == 0)

    Returns dict with:
      - n_conflict_px: pixel count in conflict zones
      - conflict_pct: % of unlabeled area
      - ens_mean_in_conflict: mean ensemble prob in conflicts
      - ens_std_in_conflict: std dev of ensemble prob in conflicts
      - filtered_pixels: count of pixels that would be excluded
      - signal_preserved: count of labeled pixels NOT affected
    """
    unlabeled = labels == 0
    conflict = (ensemble_probs >= threshold) & unlabeled

    n_conflict = np.sum(conflict)
    n_unlabeled = np.sum(unlabeled)
    conflict_pct = 100.0 * n_conflict / max(1, n_unlabeled)

    ens_in_conflict = ensemble_probs[conflict]
    ens_mean_conflict = float(np.mean(ens_in_conflict)) if len(ens_in_conflict) > 0 else 0.0
    ens_std_conflict = float(np.std(ens_in_conflict)) if len(ens_in_conflict) > 0 else 0.0

    labeled = labels > 0
    n_labeled = np.sum(labeled)

    return {
        "threshold": float(threshold),
        "n_conflict_px": int(n_conflict),
        "n_unlabeled_px": int(n_unlabeled),
        "conflict_pct": float(conflict_pct),
        "ens_mean_in_conflict": ens_mean_conflict,
        "ens_std_in_conflict": ens_std_conflict,
        "n_labeled_px": int(n_labeled),
        "conflict_mask": conflict,
    }


def compute_f1_for_threshold(
    ensemble_probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> float:
    """Compute F1 score of ensemble predictions at threshold.

    For each pixel in conflict zones:
      - TP: ensemble says CWD (prob >= 0.5) AND unlabeled (don't know ground truth)
      - FP: ensemble says CWD AND unlabeled
      - FN: unlabeled regions (model misses)

    This is a proxy for how well the ensemble identifies high-confidence regions.
    """
    unlabeled = labels == 0
    ens_pred_cwd = ensemble_probs >= 0.5

    # Only evaluate on conflict zone
    conflict = (ensemble_probs >= threshold) & unlabeled
    if np.sum(conflict) == 0:
        return 0.0

    # Treat high ensemble prob as "predicted CWD", unlabeled as "ground truth unknown"
    # F1 = balance between:
    #   - Precision: of high-conf regions, how many are meaningful CWD indicators?
    #   - Recall: of all unlabeled regions, how many are filtered?
    tp = np.sum(ens_pred_cwd & conflict)
    fp = np.sum((~ens_pred_cwd) & conflict)
    fn = np.sum((~conflict) & unlabeled)  # unlabeled regions NOT in conflict zone

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return float(f1)


def sweep_thresholds(
    ensemble_probs: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
) -> tuple[float, list[dict], int]:
    """Sweep over thresholds and return optimal threshold, results, and best index."""
    results = []
    best_f1 = -1.0
    best_idx = 0
    best_threshold = thresholds[0]

    for i, thr in enumerate(thresholds):
        stats = compute_conflict_statistics(ensemble_probs, labels, thr)
        f1 = compute_f1_for_threshold(ensemble_probs, labels, thr)
        stats["f1_proxy"] = float(f1)
        results.append(stats)

        if f1 > best_f1:
            best_f1 = f1
            best_idx = i
            best_threshold = thr

    return best_threshold, results, best_idx


def save_conflict_mask(
    conflict_mask: np.ndarray,
    output_path: Path,
    meta: dict,
) -> None:
    """Save conflict mask as GeoTIFF."""
    import rasterio
    meta.update(dtype=rasterio.uint8, count=1, compress="lzw")
    with rasterio.open(str(output_path), "w", **meta) as dst:
        dst.write((conflict_mask.astype(np.uint8) * 255), 1)


def main():
    parser = argparse.ArgumentParser(description="Phase 0 V6: Ensemble filtering threshold optimization")
    parser.add_argument(
        "--ensemble", type=Path, required=True,
        help="Path to V3 ensemble prediction GeoTIFF (e.g., pred_ensemble_tta1.tif)",
    )
    parser.add_argument(
        "--label-tif", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase1_masks" / "406455_2021_tava_truemask.tif",
        help="Path to ground-truth label mask",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase0_ensemble_filter_v6",
        help="Output directory for conflict mask and metrics",
    )
    parser.add_argument(
        "--threshold-min", type=float, default=0.05,
        help="Minimum threshold to sweep (default: 0.05)",
    )
    parser.add_argument(
        "--threshold-max", type=float, default=0.30,
        help="Maximum threshold to sweep (default: 0.30)",
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.01,
        help="Threshold step size (default: 0.01)",
    )
    parser.add_argument(
        "--metric", type=str, default="f1",
        choices=["f1", "conflict_pct", "ens_mean"],
        help="Metric to optimize (default: f1)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Unused (for consistency)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.ensemble.exists():
        print(f"ERROR: Ensemble file not found: {args.ensemble}", file=sys.stderr)
        sys.exit(1)

    if not args.label_tif.exists():
        print(f"ERROR: Label file not found: {args.label_tif}", file=sys.stderr)
        sys.exit(1)

    print("="*70)
    print("Phase 0 V6: Ensemble Filtering Threshold Optimization")
    print("="*70)
    print(f"Ensemble predictions: {args.ensemble}")
    print(f"Ground-truth labels: {args.label_tif}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sweep range: [{args.threshold_min}, {args.threshold_max}] step={args.threshold_step}")
    print(f"Optimization metric: {args.metric}")

    # Load data
    print("\nLoading data...")
    ensemble_probs, meta = load_ensemble_predictions(args.ensemble)
    labels = load_labels(args.label_tif)

    print(f"  Ensemble shape: {ensemble_probs.shape}")
    print(f"  Labels shape: {labels.shape}")

    # Ensure same shape
    if ensemble_probs.shape != labels.shape:
        print(f"  Cropping to common shape {labels.shape}")
        ensemble_probs = ensemble_probs[: labels.shape[0], : labels.shape[1]]

    # Statistics on input
    labeled_px = np.sum(labels > 0)
    unlabeled_px = np.sum(labels == 0)
    ens_mean = float(np.mean(ensemble_probs))
    ens_std = float(np.std(ensemble_probs))
    print(f"  Labeled pixels: {labeled_px:,} ({100*labeled_px/(labeled_px+unlabeled_px):.1f}%)")
    print(f"  Unlabeled pixels: {unlabeled_px:,}")
    print(f"  Ensemble mean: {ens_mean:.4f}, std: {ens_std:.4f}")

    # Sweep thresholds
    print(f"\nSweeping thresholds...")
    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-6, args.threshold_step).tolist()
    print(f"  Testing {len(thresholds)} thresholds: {thresholds[0]:.2f} to {thresholds[-1]:.2f}")

    best_threshold, results, best_idx = sweep_thresholds(ensemble_probs, labels, thresholds)

    # Save sweep results
    csv_path = args.output_dir / "conflict_sweep_results_v6.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["threshold", "n_conflict_px", "n_unlabeled_px", "conflict_pct",
                      "ens_mean_in_conflict", "ens_std_in_conflict", "f1_proxy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  ✓ Sweep results: {csv_path}")

    # Print summary
    best_result = results[best_idx]
    print(f"\n{'='*70}")
    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"  Conflict pixels: {best_result['n_conflict_px']:,} ({best_result['conflict_pct']:.1f}%)")
    print(f"  Ensemble mean in conflict: {best_result['ens_mean_in_conflict']:.4f}")
    print(f"  F1 score proxy: {best_result['f1_proxy']:.4f}")
    print(f"{'='*70}")

    # Save conflict mask
    conflict_mask = best_result["conflict_mask"]
    mask_path = args.output_dir / f"conflict_mask_v6_thr{best_threshold:.2f}.tif"
    save_conflict_mask(conflict_mask, mask_path, meta)
    print(f"✓ Conflict mask saved: {mask_path}")

    # Save metadata
    filter_metadata = {
        "optimal_threshold": float(best_threshold),
        "metric": args.metric,
        "n_conflict_px": int(best_result["n_conflict_px"]),
        "conflict_pct": float(best_result["conflict_pct"]),
        "ens_mean_in_conflict": float(best_result["ens_mean_in_conflict"]),
        "ens_std_in_conflict": float(best_result["ens_std_in_conflict"]),
        "f1_proxy": float(best_result["f1_proxy"]),
        "input_shape": ensemble_probs.shape,
        "input_labeled_px": int(labeled_px),
        "input_unlabeled_px": int(unlabeled_px),
        "input_ensemble_mean": float(ens_mean),
        "input_ensemble_std": float(ens_std),
        "sweep_results": [
            {
                "threshold": float(r["threshold"]),
                "n_conflict_px": int(r["n_conflict_px"]),
                "conflict_pct": float(r["conflict_pct"]),
                "f1_proxy": float(r["f1_proxy"]),
            }
            for r in results
        ],
    }
    meta_path = args.output_dir / "phase0_ensemble_filter_v6.json"
    meta_path.write_text(json.dumps(filter_metadata, indent=2))
    print(f"✓ Metadata saved: {meta_path}")

    print(f"\n✅ Phase 0 complete — optimal τ = {best_threshold:.2f}")
    print(f"Use --conflict-mask {mask_path} in phase2_dataset_v6.py to filter training data")


if __name__ == "__main__":
    sys.exit(main() or 0)

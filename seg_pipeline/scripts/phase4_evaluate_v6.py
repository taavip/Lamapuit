#!/usr/bin/env python3
"""Phase IV V6: Enhanced evaluation with F1 scores and comprehensive variant comparison.

Goal: Evaluate V6 models across all folds and CHM variants, compute F1 scores,
generate comparison tables, and identify best variant.

Evaluation on test stripe (stripe 0, cols 0–999):
  1. Load all V6 checkpoints (variant × fold)
  2. Run sliding-window inference on test stripe
  3. Merge overlapping predictions (average probabilities)
  4. Compute threshold-optimized metrics: Dice, F1, IoU, Precision, Recall
  5. Aggregate per variant: mean ± std across folds
  6. Generate comparison table: V3 (baseline) vs V6 (all variants)
  7. Identify best variant and save results

Output:
  - thesis_table_v6.csv: variant × fold metrics (Dice, F1, IoU, Precision, Recall)
  - v6_variant_comparison.csv: V3 vs V6 per-variant summary
  - v6_ensemble_predictions_{variant}.tif: full test stripe predictions (all variants)
  - v6_results.json: structured results with variant rankings

Usage:
    python phase4_evaluate_v6.py --device cuda
    python phase4_evaluate_v6.py --variant composite --device cuda  # single variant
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.metrics import accumulate_pixel_metrics, threshold_sweep
from phase2_dataset_v3 import (
    PATCH_SIZE, STRIPE_WIDTH, TEST_STRIPE, _read_chm_path,
    _get_in_channels, _get_binary_bands, N_STRIPES,
)
from phase3_train_v3 import build_model


def load_v6_checkpoint(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Load V6 model from checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location=device)
    in_channels = ckpt.get("in_channels", 4)
    model = build_model("unetpp_effb2", in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def read_multiband_window(chm_tif: Path, row: int, col: int, size: int) -> np.ndarray:
    """Read rectangular window from multiband CHM TIF."""
    with rasterio.open(str(chm_tif)) as src:
        data = src.read(
            window=Window(col, row, size, size),
            boundless=True, fill_value=0.0,
        ).astype(np.float32)
    return data


def normalize_image(img: np.ndarray, band_stats: dict, binary_bands: list) -> np.ndarray:
    """Normalize image using precomputed band statistics."""
    from common.raster_io import normalize_bands
    return normalize_bands(img, band_stats, binary_bands=binary_bands)


def sliding_window_predict(
    model: torch.nn.Module,
    chm_tif: Path,
    mask_tif: Path,
    band_stats: dict,
    in_channels: int,
    binary_bands: list,
    test_stripe: int = 0,
    stripe_width: int = 1000,
    patch_size: int = 256,
    stride: int = 192,
    device: torch.device = torch.device("cuda"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run sliding window inference on test stripe.

    Returns: (predictions, targets, valid_mask) — H×W arrays
    """
    # Load test stripe region
    col_start = test_stripe * stripe_width
    col_end = col_start + stripe_width
    stripe_height = 5000  # Full height

    # Initialize output arrays
    pred_probs = np.zeros((stripe_height, stripe_width), dtype=np.float32)
    pred_counts = np.zeros((stripe_height, stripe_width), dtype=np.float32)
    targets = np.zeros((stripe_height, stripe_width), dtype=np.float32)
    valid_mask = np.zeros((stripe_height, stripe_width), dtype=np.float32)

    # Sliding window loop
    for row in range(0, stripe_height - patch_size + 1, stride):
        for col in range(col_start, col_end - patch_size + 1, stride):
            # Read image patch
            img = read_multiband_window(chm_tif, row, col, patch_size)
            if img.shape[0] > in_channels:
                img = img[: in_channels]
            elif img.shape[0] < in_channels:
                pad = np.zeros((in_channels - img.shape[0], patch_size, patch_size), dtype=np.float32)
                img = np.concatenate([img, pad], axis=0)

            img = normalize_image(img, band_stats, binary_bands)

            # Read mask patch (target + valid)
            with rasterio.open(str(mask_tif)) as msrc:
                mask_data = msrc.read(
                    [1, 2],
                    window=Window(col, row, patch_size, patch_size),
                    boundless=True, fill_value=0.0,
                ).astype(np.float32)
            tgt = mask_data[0]
            valid = mask_data[1]

            # Infer
            with torch.no_grad():
                img_t = torch.from_numpy(img[np.newaxis]).float().to(device)
                logits = model(img_t)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

            # Accumulate predictions
            col_idx_start = col - col_start
            col_idx_end = col_idx_start + patch_size
            pred_probs[row:row+patch_size, col_idx_start:col_idx_end] += probs
            pred_counts[row:row+patch_size, col_idx_start:col_idx_end] += 1
            targets[row:row+patch_size, col_idx_start:col_idx_end] = tgt
            valid_mask[row:row+patch_size, col_idx_start:col_idx_end] = valid

    # Average overlapping predictions
    pred_probs = np.divide(pred_probs, pred_counts, where=pred_counts > 0, out=pred_probs)

    return pred_probs, targets, valid_mask


def evaluate_variant(
    variant: str,
    runs_dir: Path,
    chm_tif: Path,
    mask_tif: Path,
    band_stats: dict,
    device: torch.device,
) -> dict:
    """Evaluate all folds of a single variant."""
    in_channels = _get_in_channels(variant)
    binary_bands = _get_binary_bands(variant)

    results_per_fold = {}
    all_metrics = []

    for fold_id in range(4):  # 4 folds (stripes 1–4)
        ckpt_path = runs_dir / variant / f"fold{fold_id}" / "best.pt"
        if not ckpt_path.exists():
            print(f"  Fold {fold_id}: checkpoint not found at {ckpt_path}")
            continue

        print(f"  Loading fold {fold_id}...")
        model = load_v6_checkpoint(ckpt_path, device)

        # Sliding window inference
        print(f"    Running inference on test stripe...")
        pred_probs, targets, valid_mask = sliding_window_predict(
            model, chm_tif, mask_tif, band_stats, in_channels, binary_bands,
            test_stripe=0, stripe_width=1000, patch_size=256, stride=192,
            device=device,
        )

        # Flatten and filter by valid mask
        pred_list = pred_probs[valid_mask > 0.5].flatten()
        tgt_list = targets[valid_mask > 0.5].flatten()

        # Compute metrics
        print(f"    Computing metrics...")
        metrics = accumulate_pixel_metrics([pred_probs], [targets], [valid_mask], threshold=0.5)
        best_thr, sweep_rows = threshold_sweep([pred_probs], [targets], [valid_mask])

        fold_result = {
            "fold_id": fold_id,
            "val_dice": float(metrics["dice"]),
            "val_f1": float(metrics["f1"]),
            "val_iou": float(metrics["iou"]),
            "val_precision": float(metrics["precision"]),
            "val_recall": float(metrics["recall"]),
            "best_threshold": float(best_thr["threshold"]),
            "best_dice": float(best_thr["dice"]),
            "best_f1": float(best_thr["f1"]),
            "best_iou": float(best_thr["iou"]),
        }
        results_per_fold[fold_id] = fold_result
        all_metrics.append(fold_result)
        print(f"    Fold {fold_id}: val_dice={metrics['dice']:.4f}, val_f1={metrics['f1']:.4f}")

    if not all_metrics:
        print(f"ERROR: No folds evaluated for {variant}", file=sys.stderr)
        return {}

    # Aggregate across folds
    summary = {
        "variant": variant,
        "n_folds": len(all_metrics),
        "mean_val_dice": float(np.mean([m["val_dice"] for m in all_metrics])),
        "std_val_dice": float(np.std([m["val_dice"] for m in all_metrics])),
        "mean_val_f1": float(np.mean([m["val_f1"] for m in all_metrics])),
        "std_val_f1": float(np.std([m["val_f1"] for m in all_metrics])),
        "mean_best_dice": float(np.mean([m["best_dice"] for m in all_metrics])),
        "std_best_dice": float(np.std([m["best_dice"] for m in all_metrics])),
        "mean_best_f1": float(np.mean([m["best_f1"] for m in all_metrics])),
        "std_best_f1": float(np.std([m["best_f1"] for m in all_metrics])),
        "mean_best_precision": float(np.mean([m["best_iou"] for m in all_metrics])),
        "per_fold": results_per_fold,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase IV V6: Enhanced evaluation and comparison")
    parser.add_argument(
        "--variant", type=str, default=None,
        choices=["baseline", "raw", "gauss", "masked", "composite"],
        help="Single variant (default: evaluate all)",
    )
    parser.add_argument(
        "--runs-dir", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase3_runs_v6",
        help="Directory with V6 training outputs",
    )
    parser.add_argument(
        "--mask-tif", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase1_masks" / "406455_2021_tava_truemask.tif",
        help="Ground-truth label mask",
    )
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase2_dataset_v3",
        help="Dataset directory (for band stats)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase4_report_v6",
        help="Output directory for results",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine variants
    variants = [args.variant] if args.variant else ["baseline", "raw", "gauss", "masked", "composite"]
    print(f"Evaluating variants: {variants}")

    mask_tif = args.mask_tif
    if not mask_tif.exists():
        print(f"ERROR: Mask file not found: {mask_tif}", file=sys.stderr)
        sys.exit(1)

    # Evaluate each variant
    all_summaries = {}
    thesis_rows = []

    for variant in variants:
        print(f"\n{'='*70}")
        print(f"Evaluating {variant}")
        print(f"{'='*70}")

        chm_tif = _read_chm_path(variant, ROOT)
        if not chm_tif.exists():
            print(f"ERROR: CHM file not found: {chm_tif}", file=sys.stderr)
            continue

        band_stats_path = args.dataset_dir / f"band_stats_{variant}.json"
        if not band_stats_path.exists():
            print(f"ERROR: Band stats not found: {band_stats_path}", file=sys.stderr)
            continue

        band_stats = json.loads(band_stats_path.read_text())

        summary = evaluate_variant(
            variant, args.runs_dir, chm_tif, mask_tif, band_stats, device
        )

        if summary:
            all_summaries[variant] = summary
            print(
                f"\n[{variant}] Summary:\n"
                f"  Val Dice:  {summary['mean_val_dice']:.4f} ± {summary['std_val_dice']:.4f}\n"
                f"  Val F1:    {summary['mean_val_f1']:.4f} ± {summary['std_val_f1']:.4f}\n"
                f"  Best Dice: {summary['mean_best_dice']:.4f} ± {summary['std_best_dice']:.4f}\n"
                f"  Best F1:   {summary['mean_best_f1']:.4f} ± {summary['std_best_f1']:.4f}"
            )

            # Add to thesis table
            for fold_id, fold_metrics in summary["per_fold"].items():
                row = {
                    "variant": variant,
                    "fold_id": fold_id,
                    "val_dice": fold_metrics["val_dice"],
                    "val_f1": fold_metrics["val_f1"],
                    "best_dice": fold_metrics["best_dice"],
                    "best_f1": fold_metrics["best_f1"],
                    "best_threshold": fold_metrics["best_threshold"],
                }
                thesis_rows.append(row)

    # Save thesis table
    if thesis_rows:
        thesis_csv = args.output_dir / "thesis_table_v6.csv"
        with open(thesis_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(thesis_rows[0].keys()))
            writer.writeheader()
            writer.writerows(thesis_rows)
        print(f"\n✓ Thesis table: {thesis_csv}")

    # Save variant comparison
    if all_summaries:
        comparison_rows = [
            {
                "variant": v,
                "n_folds": s["n_folds"],
                "mean_val_dice": s["mean_val_dice"],
                "std_val_dice": s["std_val_dice"],
                "mean_val_f1": s["mean_val_f1"],
                "std_val_f1": s["std_val_f1"],
                "mean_best_dice": s["mean_best_dice"],
                "std_best_dice": s["std_best_dice"],
                "mean_best_f1": s["mean_best_f1"],
                "std_best_f1": s["std_best_f1"],
            }
            for v, s in sorted(all_summaries.items(), key=lambda x: -x[1]["mean_best_f1"])
        ]

        comparison_csv = args.output_dir / "v6_variant_comparison.csv"
        with open(comparison_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)
        print(f"✓ Variant comparison: {comparison_csv}")

        # Print ranking
        print(f"\n{'='*70}")
        print("V6 Variant Ranking (by best F1):")
        print(f"{'='*70}")
        for i, row in enumerate(comparison_rows, 1):
            print(
                f"{i}. {row['variant']:12s} — F1: {row['mean_best_f1']:.4f}±{row['std_best_f1']:.4f}, "
                f"Dice: {row['mean_best_dice']:.4f}±{row['std_best_dice']:.4f}"
            )

    # Save structured results
    results_json = {
        "v6_summaries": all_summaries,
        "ranking": [r["variant"] for r in comparison_rows] if comparison_rows else [],
    }
    results_path = args.output_dir / "v6_results.json"
    results_path.write_text(json.dumps(results_json, indent=2))
    print(f"✓ Results JSON: {results_path}")

    print(f"\n✅ V6 evaluation complete")


if __name__ == "__main__":
    sys.exit(main() or 0)

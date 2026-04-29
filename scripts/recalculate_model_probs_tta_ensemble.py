#!/usr/bin/env python3
"""
Recalculate model probabilities using the exact ensemble TTA (Test Time Augmentation) logic.

This script reproduces the EXACT probabilities from train_ensemble.py by:
1. Loading all 4 models (cnn_seed42, cnn_seed43, cnn_seed44, effnet_b2)
2. For each label, performing 8x TTA (4 rotations × 2 flips per rotation)
3. Soft-voting probabilities across all 4 models
4. Normalizing with correct CHM formula: clip(raw, 0, 20) / 20

This matches the original ensemble probability generation exactly.

Usage:
  python scripts/recalculate_model_probs_tta_ensemble.py \\
    --labels data/chm_variants/labels_canonical_with_splits.csv \\
    --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop \\
    --output data/chm_variants/labels_canonical_with_splits_tta_ensemble.csv
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window

# Import model builders from label_tiles
sys.path.insert(0, str(Path(__file__).parent))
from label_tiles import _get_build_fn, _instantiate_model_from_build_fn


def load_chm_window(chm_dir: Path, raster_name: str, row_off: int, col_off: int) -> np.ndarray | None:
    """Load 128×128 CHM window from GeoTIFF."""
    chm_path = chm_dir / raster_name
    if not chm_path.exists():
        return None

    try:
        with rasterio.open(chm_path) as src:
            window = Window(col_off, row_off, 128, 128)
            data = src.read(1, window=window).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            # Replace NaN with 0 to avoid numerical issues
            data = np.nan_to_num(data, nan=0.0)
            return data
    except Exception:
        return None


def normalize_chm(tile: np.ndarray) -> np.ndarray:
    """Normalize CHM tile: clip to [0-20m] and scale to [0,1] (matches train_ensemble.py)."""
    return np.clip(tile, 0.0, 20.0) / 20.0


def load_model(checkpoint_path: Path, device) -> nn.Module | None:
    """Load a single model checkpoint."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        build_fn_name = ckpt.get("build_fn_name", "_build_deep_cnn_attn")
        build_fn = _get_build_fn(build_fn_name)
        if build_fn is None:
            return None
        model = _instantiate_model_from_build_fn(build_fn).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
    except Exception as e:
        print(f"ERROR loading {checkpoint_path}: {e}")
        return None


def predict_proba_cdw_tta(model: nn.Module, device, chm_tile: np.ndarray) -> float | None:
    """
    Compute P(CDW) using 8x TTA (4 rotations × 2 flips per rotation).

    Exactly matches the TTA logic from train_ensemble.py lines 285-298:
    - For each of 4 rotations (0°, 90°, 180°, 270°)
    - For each rotation, compute both original and horizontally flipped
    - Average all 8 predictions
    """
    if chm_tile is None or np.all(~np.isfinite(chm_tile)):
        return None

    try:
        # Normalize CHM
        chm_norm = normalize_chm(chm_tile)
        x = torch.tensor(chm_norm[np.newaxis, np.newaxis], dtype=torch.float32).to(device)

        with torch.no_grad():
            views = []
            # 4 rotations
            for k in range(4):
                v = torch.rot90(x, k, [-2, -1])
                # Original rotation
                views.append(torch.softmax(model(v), dim=1)[0, 1].item())
                # Horizontal flip
                views.append(torch.softmax(model(torch.flip(v, [-1])), dim=1)[0, 1].item())

            # Average 8 views
            prob_cwd = float(np.mean(views))
            return prob_cwd
    except Exception:
        return None


def recalculate_probabilities_tta_ensemble(df, baseline_chm_dir, device):
    """Recalculate model_prob using exact TTA + soft-voting ensemble."""
    df = df.copy()

    # Load all 4 models
    models_to_load = [
        ("cnn_seed42", Path("output/tile_labels/cnn_seed42.pt")),
        ("cnn_seed43", Path("output/tile_labels/cnn_seed43.pt")),
        ("cnn_seed44", Path("output/tile_labels/cnn_seed44.pt")),
        ("effnet_b2", Path("output/tile_labels/effnet_b2.pt")),
    ]

    print("Loading 4-model ensemble with TTA...")
    models = {}
    for name, path in models_to_load:
        print(f"  Loading {name}...", end=" ", flush=True)
        model = load_model(path, device)
        if model is None:
            print("FAILED")
            return df
        models[name] = model
        print("OK")

    print(f"✓ All 4 models loaded successfully\n")

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    baseline_chm_dir = Path(baseline_chm_dir)

    # Track changes
    processed = 0
    failed = 0
    skipped = 0
    prob_changed = 0
    prob_changes = []

    print("Processing labels with 8x TTA + soft-voting (this will take a while)...")
    total = len(df)

    for idx, row in df.iterrows():
        if (idx + 1) % max(1, total // 20) == 0:  # Progress every 5%
            pct = 100.0 * (idx + 1) / total
            print(f"  [{idx + 1}/{total}] {pct:.1f}% | processed={processed}, changed={prob_changed}, failed={failed}")

        raster_name = row["raster"]
        row_off = int(row["row_off"])
        col_off = int(row["col_off"])

        try:
            # Load CHM window
            chm_tile = load_chm_window(baseline_chm_dir, raster_name, row_off, col_off)
            if chm_tile is None:
                skipped += 1
                continue

            # Run TTA inference on all 4 models and soft-vote
            ensemble_probs = []
            for model in models.values():
                prob = predict_proba_cdw_tta(model, device, chm_tile)
                if prob is None:
                    failed += 1
                    break
                ensemble_probs.append(prob)

            if len(ensemble_probs) != len(models):
                failed += 1
                continue

            # Soft-vote: average probabilities across all 4 models
            prob_cwd = float(np.mean(ensemble_probs))

            # Track old vs new
            old_prob = float(row["model_prob"]) if pd.notna(row["model_prob"]) else None
            prob_diff = None
            if old_prob is not None:
                prob_diff = abs(prob_cwd - old_prob)
                prob_changes.append(prob_diff)
                if prob_diff > 0.01:  # Changed by >1%
                    prob_changed += 1

            # Update row
            df.at[idx, "model_prob"] = prob_cwd
            df.at[idx, "model_name"] = "Ensemble(4-TTA)"
            df.at[idx, "timestamp"] = timestamp

            processed += 1

        except Exception as e:
            failed += 1
            if idx < 5:
                print(f"    Error row {idx}: {e}")

    print(f"\nProcessing complete:")
    print(f"  Total processed: {processed:,}")
    print(f"  Failed: {failed:,}")
    print(f"  Skipped: {skipped:,}")
    print(f"  Probabilities changed (>1%): {prob_changed:,}")

    if prob_changes:
        prob_changes_arr = np.array(prob_changes)
        print(f"\nProbability change statistics:")
        print(f"  Mean change: {prob_changes_arr.mean():.4f}")
        print(f"  Std dev: {prob_changes_arr.std():.4f}")
        print(f"  Min: {prob_changes_arr.min():.4f}")
        print(f"  Max: {prob_changes_arr.max():.4f}")
        print(f"  Median: {np.median(prob_changes_arr):.4f}")

    return df


def print_stats(df):
    """Print statistics of recalculated probabilities."""
    print("\n" + "=" * 75)
    print("RECALCULATED MODEL PROBABILITY STATISTICS (TTA ENSEMBLE)")
    print("=" * 75)
    print()

    print("Overall statistics:")
    print(f"  Mean prob: {df['model_prob'].mean():.4f}")
    print(f"  Std dev:   {df['model_prob'].std():.4f}")
    print(f"  Min:       {df['model_prob'].min():.4f}")
    print(f"  Max:       {df['model_prob'].max():.4f}")
    print(f"  Median:    {df['model_prob'].median():.4f}")
    print()

    # By class
    print("By class label:")
    for label in ["cdw", "no_cdw"]:
        subset = df[df["label"] == label]
        if len(subset) > 0:
            print(f"\n  {label.upper()}:")
            print(f"    Count:     {len(subset):,}")
            print(f"    Mean prob: {subset['model_prob'].mean():.4f}")
            print(f"    Std dev:   {subset['model_prob'].std():.4f}")

    # By split
    print()
    print("By split:")
    for split in ["test", "val", "train", "none"]:
        subset = df[df["split"] == split]
        if len(subset) > 0:
            print(f"\n  {split.upper()}:")
            print(f"    Count:     {len(subset):,}")
            print(f"    Mean prob: {subset['model_prob'].mean():.4f}")
            print(f"    Std dev:   {subset['model_prob'].std():.4f}")

    print()
    print("=" * 75 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Recalculate model probabilities with exact TTA ensemble from train_ensemble.py"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/chm_variants/labels_canonical_with_splits.csv"),
        help="Path to labels CSV",
    )
    parser.add_argument(
        "--baseline-chm-dir",
        type=Path,
        default=Path("data/lamapuit/chm_max_hag_13_drop"),
        help="Directory containing baseline CHM 20cm tif files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: overwrite input)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing")
    parser.add_argument(
        "--sample",
        type=int,
        help="Process only a sample of N labels (for testing)",
    )
    args = parser.parse_args()

    print(f"Loading labels from {args.labels}...")
    df = pd.read_csv(args.labels)
    print(f"Loaded {len(df)} labels")

    if args.sample:
        df = df.sample(n=args.sample, random_state=42)
        print(f"Sampled {len(df)} labels for testing")

    print()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Recalculate probabilities with TTA
    df = recalculate_probabilities_tta_ensemble(df, args.baseline_chm_dir, device)

    # Print stats
    print_stats(df)

    if not args.dry_run:
        output_path = args.output or args.labels
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

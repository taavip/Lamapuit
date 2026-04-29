#!/usr/bin/env python3
"""
Recalculate model probabilities using the 4-model ensemble from label_tiles.CNNPredictor.

This script properly uses the ensemble of 4 models (cnn_seed42/43/44 + effnet_b2)
with soft-voting to compute ensemble probabilities, matching the original
pipeline that generated the existing model_prob values.

Usage:
  python scripts/recalculate_model_probs_ensemble.py \
    --labels data/chm_variants/labels_canonical_with_splits.csv \
    --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop \
    --output data/chm_variants/labels_canonical_with_splits_recalculated.csv
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

# Import CNNPredictor from label_tiles
sys.path.insert(0, str(Path(__file__).parent))
from label_tiles import CNNPredictor


def load_chm_window(chm_dir: Path, raster_name: str, row_off: int, col_off: int) -> np.ndarray | None:
    """Load 128×128 CHM window from GeoTIFF."""
    chm_path = chm_dir / raster_name
    if not chm_path.exists():
        return None

    try:
        with rasterio.open(chm_path) as src:
            window = Window(col_off, row_off, 128, 128)
            data = src.read(1, window=window)
            data = data.astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            # Replace NaN with 0 to avoid softmax issues in neural network
            data = np.nan_to_num(data, nan=0.0)
            return data
    except Exception:
        return None


def recalculate_probabilities_ensemble(df, baseline_chm_dir):
    """Recalculate model_prob using CNNPredictor ensemble."""
    df = df.copy()

    print("Loading ensemble model with 4-model soft-voting...")
    pred = CNNPredictor()

    # Load ensemble from ensemble_meta.json
    ensemble_meta_path = Path("output/tile_labels/ensemble_meta.json")
    if ensemble_meta_path.exists():
        print(f"Loading ensemble metadata from {ensemble_meta_path}...")
        ok = pred.load_ensemble_meta(ensemble_meta_path)
        if ok:
            print(f"  ✓ Ensemble loaded successfully")
        else:
            print(f"  ✗ Failed to load ensemble, falling back to single model")
            ok = pred.load_from_disk(Path("output/tile_labels/ensemble_model.pt"))
            if not ok:
                print("ERROR: Could not load any model")
                return df
    else:
        print(f"No ensemble metadata found, loading single model...")
        ok = pred.load_from_disk(Path("output/tile_labels/ensemble_model.pt"))
        if not ok:
            print("ERROR: Could not load model")
            return df

    print("Model loaded and ready for inference\n")

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    baseline_chm_dir = Path(baseline_chm_dir)

    # Track changes
    processed = 0
    failed = 0
    skipped = 0
    prob_changed = 0
    prob_changes = []

    print("Processing labels (this may take a while)...")
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

            # Run inference via ensemble predictor
            prob_cwd = pred.predict_proba_cdw(chm_tile)
            if prob_cwd is None:
                failed += 1
                continue

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
            df.at[idx, "model_name"] = "Ensemble(ConvNeXt+EffNet)"
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
    print("RECALCULATED MODEL PROBABILITY STATISTICS")
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

    parser = argparse.ArgumentParser(description="Recalculate model probabilities with ensemble CNN")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/chm_variants/labels_canonical_with_splits.csv"),
        help="Path to labels CSV with splits",
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

    # Recalculate probabilities
    df = recalculate_probabilities_ensemble(df, args.baseline_chm_dir)

    # Print stats
    print_stats(df)

    if not args.dry_run:
        output_path = args.output or args.labels
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

        # Compare with original if available
        if output_path != args.labels:
            orig_df = pd.read_csv(args.labels)
            merged = pd.merge(
                orig_df[['raster', 'row_off', 'col_off', 'model_prob']].rename(columns={'model_prob': 'original'}),
                df[['raster', 'row_off', 'col_off', 'model_prob']].rename(columns={'model_prob': 'recalculated'}),
                on=['raster', 'row_off', 'col_off']
            )
            merged['diff'] = np.abs(merged['original'] - merged['recalculated'])

            print("\n" + "="*75)
            print("COMPARISON WITH ORIGINAL PROBABILITIES")
            print("="*75)
            print(f"\nMatched pairs: {len(merged)}")
            valid_pairs = merged[merged['original'].notna() & merged['recalculated'].notna()]
            print(f"Valid pairs (both non-NaN): {len(valid_pairs)}")
            if len(valid_pairs) > 0:
                print(f"  Mean difference: {valid_pairs['diff'].mean():.6f}")
                print(f"  Median difference: {valid_pairs['diff'].median():.6f}")
                print(f"  Max difference: {valid_pairs['diff'].max():.6f}")
                print(f"  Std dev: {valid_pairs['diff'].std():.6f}")


if __name__ == "__main__":
    main()

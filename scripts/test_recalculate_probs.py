#!/usr/bin/env python3
"""Test recalculate_model_probs on a small sample to verify probability accuracy."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Test on 100 random labels from the original dataset
def main():
    csv_path = Path("data/chm_variants/labels_canonical_with_splits.csv")

    print("Loading original labels...")
    df = pd.read_csv(csv_path)

    # Sample 100 random labels for testing
    sample_df = df.sample(n=100, random_state=42).copy()
    sample_path = Path("/tmp/test_sample_100.csv")
    sample_df.to_csv(sample_path, index=False)

    print(f"Saved test sample of 100 labels to {sample_path}")
    print(f"\nOriginal probabilities (first 10):")
    print(sample_df[['raster', 'row_off', 'col_off', 'label', 'model_prob']].head(10))

    print("\n" + "="*80)
    print("Running recalculation on test sample...")
    print("="*80 + "\n")

    # Run recalculation on test sample
    import subprocess
    result = subprocess.run([
        sys.executable,
        "scripts/recalculate_model_probs.py",
        "--labels", str(sample_path),
        "--baseline-chm-dir", "data/lamapuit/chm_max_hag_13_drop",
        "--model-path", "output/tile_labels/ensemble_model.pt",
        "--output", "/tmp/test_sample_recalculated.csv"
    ])

    if result.returncode != 0:
        print("ERROR: Recalculation failed!")
        return 1

    # Compare results
    print("\n" + "="*80)
    print("Comparing original vs recalculated probabilities...")
    print("="*80 + "\n")

    recalc_df = pd.read_csv("/tmp/test_sample_recalculated.csv")

    # Compare by merging
    comparison = pd.merge(
        sample_df[['raster', 'row_off', 'col_off', 'model_prob']].rename(columns={'model_prob': 'original_prob'}),
        recalc_df[['raster', 'row_off', 'col_off', 'model_prob']].rename(columns={'model_prob': 'recalc_prob'}),
        on=['raster', 'row_off', 'col_off']
    )

    comparison['abs_diff'] = np.abs(comparison['original_prob'] - comparison['recalc_prob'])
    comparison['pct_diff'] = 100 * comparison['abs_diff'] / (np.abs(comparison['original_prob']) + 1e-8)

    print("Top 20 samples with largest absolute difference:")
    print(comparison.nlargest(20, 'abs_diff')[['original_prob', 'recalc_prob', 'abs_diff', 'pct_diff']])

    print(f"\nComparison statistics:")
    print(f"  Mean absolute difference: {comparison['abs_diff'].mean():.6f}")
    print(f"  Std dev: {comparison['abs_diff'].std():.6f}")
    print(f"  Min difference: {comparison['abs_diff'].min():.6f}")
    print(f"  Max difference: {comparison['abs_diff'].max():.6f}")
    print(f"  Median difference: {comparison['abs_diff'].median():.6f}")

    # Check if close enough (< 0.01 absolute difference)
    close_threshold = 0.01
    n_close = (comparison['abs_diff'] < close_threshold).sum()
    pct_close = 100 * n_close / len(comparison)

    print(f"\nLabels with difference < {close_threshold}: {n_close}/{len(comparison)} ({pct_close:.1f}%)")

    # Determine if test passes
    if pct_close > 95:  # If 95%+ are very close
        print("\n✅ TEST PASSED: Recalculated probabilities match original!")
        return 0
    else:
        print("\n❌ TEST FAILED: Recalculated probabilities don't match original")
        print("\nShowing all samples with diff > 0.01:")
        print(comparison[comparison['abs_diff'] >= close_threshold])
        return 1

if __name__ == "__main__":
    sys.exit(main())

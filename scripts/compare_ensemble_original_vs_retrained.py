#!/usr/bin/env python3
"""
Compare original vs retrained ensemble probabilities on all 580K labels.

This script generates a comprehensive comparison report between:
- Original ensemble (trained on 19,812 tiles, applied to 580K)
- Retrained ensemble (trained on 67,290 tiles via spatial splits, applied to 580K)

Compares probability distributions, identifies major changes, and analyzes
whether retraining on spatial splits eliminates the ~6% distribution shift.

Usage:
  python scripts/compare_ensemble_original_vs_retrained.py \\
    --original-probs data/chm_variants/labels_canonical_with_splits.csv \\
    --retrained-probs data/chm_variants/labels_canonical_with_splits_retrained.csv \\
    --output OPTION_B_COMPARISON_RESULTS.md
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_probability_changes(df_merged):
    """Analyze how probabilities changed between original and retrained."""
    df = df_merged.copy()

    # Calculate differences
    df['abs_diff'] = np.abs(df['original_prob'] - df['retrained_prob'])
    df['signed_diff'] = df['retrained_prob'] - df['original_prob']
    df['pct_change'] = 100 * np.abs(df['original_prob'] - df['retrained_prob']) / (
        np.abs(df['original_prob']) + 1e-8
    )

    # Filter valid pairs (both non-NaN)
    valid = df[df['original_prob'].notna() & df['retrained_prob'].notna()].copy()

    return {
        'total_rows': len(df_merged),
        'valid_pairs': len(valid),
        'original_nans': df['original_prob'].isna().sum(),
        'retrained_nans': df['retrained_prob'].isna().sum(),
        'valid_df': valid
    }


def compute_statistics(valid_df):
    """Compute detailed statistics on probability changes."""
    abs_diff = valid_df['abs_diff']
    signed_diff = valid_df['signed_diff']

    return {
        'abs_diff': {
            'mean': float(abs_diff.mean()),
            'median': float(abs_diff.median()),
            'std': float(abs_diff.std()),
            'min': float(abs_diff.min()),
            'max': float(abs_diff.max()),
            'q25': float(abs_diff.quantile(0.25)),
            'q75': float(abs_diff.quantile(0.75)),
        },
        'signed_diff': {
            'mean': float(signed_diff.mean()),
            'median': float(signed_diff.median()),
            'std': float(signed_diff.std()),
            'min': float(signed_diff.min()),
            'max': float(signed_diff.max()),
        },
        'changes_gt_1pct': int((valid_df['abs_diff'] > 0.01).sum()),
        'changes_gt_5pct': int((valid_df['abs_diff'] > 0.05).sum()),
        'changes_gt_10pct': int((valid_df['abs_diff'] > 0.10).sum()),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare original vs retrained ensemble probabilities"
    )
    parser.add_argument(
        "--original-probs",
        type=Path,
        default=Path("data/chm_variants/labels_canonical_with_splits.csv"),
        help="Original ensemble probabilities CSV (model_prob column)",
    )
    parser.add_argument(
        "--retrained-probs",
        type=Path,
        help="Retrained ensemble probabilities CSV (required)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("OPTION_B_COMPARISON_RESULTS.md"),
        help="Output markdown report",
    )
    parser.add_argument(
        "--training-meta",
        type=Path,
        default=Path("output/tile_labels_spatial_splits/training_metadata.json"),
        help="Retrained ensemble metadata",
    )
    args = parser.parse_args()

    if not args.retrained_probs:
        print("ERROR: --retrained-probs is required")
        return 1

    print("Loading original probabilities...")
    df_orig = pd.read_csv(args.original_probs)

    print("Loading retrained probabilities...")
    if not args.retrained_probs.exists():
        print(f"ERROR: {args.retrained_probs} not found")
        return 1
    df_retrain = pd.read_csv(args.retrained_probs)

    print("Merging and analyzing changes...")
    df_merged = pd.merge(
        df_orig[['raster', 'row_off', 'col_off', 'label', 'split', 'model_prob']].rename(
            columns={'model_prob': 'original_prob'}
        ),
        df_retrain[['raster', 'row_off', 'col_off', 'model_prob']].rename(
            columns={'model_prob': 'retrained_prob'}
        ),
        on=['raster', 'row_off', 'col_off'],
    )

    # Analyze
    analysis = analyze_probability_changes(df_merged)
    valid_df = analysis['valid_df']
    stats = compute_statistics(valid_df)

    # Load training metadata if available
    meta = {}
    if args.training_meta.exists():
        with open(args.training_meta) as f:
            meta = json.load(f)

    # Generate report
    report = f"""# Option B Comparison: Original vs Retrained Ensemble

**Generated**: {datetime.now(timezone.utc).isoformat()}

## Summary

### Data Coverage
- **Total labels**: {analysis['total_rows']:,}
- **Valid pairs (both probs non-NaN)**: {analysis['valid_pairs']:,}
- **Original NaN**: {analysis['original_nans']:,}
- **Retrained NaN**: {analysis['retrained_nans']:,}

### Mean Probability Difference

**Original Ensemble (Option A):**
- Mean prob: {valid_df['original_prob'].mean():.4f}
- Std: {valid_df['original_prob'].std():.4f}
- Min/Max: {valid_df['original_prob'].min():.4f} / {valid_df['original_prob'].max():.4f}

**Retrained Ensemble (Option B):**
- Mean prob: {valid_df['retrained_prob'].mean():.4f}
- Std: {valid_df['retrained_prob'].std():.4f}
- Min/Max: {valid_df['retrained_prob'].min():.4f} / {valid_df['retrained_prob'].max():.4f}

---

## Probability Change Statistics

### Absolute Difference (|original - retrained|)
- **Mean**: {stats['abs_diff']['mean']:.6f} ({stats['abs_diff']['mean']*100:.2f}%)
- **Median**: {stats['abs_diff']['median']:.6f} ({stats['abs_diff']['median']*100:.2f}%)
- **Std Dev**: {stats['abs_diff']['std']:.6f}
- **Min**: {stats['abs_diff']['min']:.6f}
- **Max**: {stats['abs_diff']['max']:.6f}
- **Q25**: {stats['abs_diff']['q25']:.6f}
- **Q75**: {stats['abs_diff']['q75']:.6f}

### Signed Difference (retrained - original)
- **Mean**: {stats['signed_diff']['mean']:.6f} (retrained is {'+' if stats['signed_diff']['mean'] > 0 else ''}{stats['signed_diff']['mean']*100:.2f}% on average)
- **Median**: {stats['signed_diff']['median']:.6f}
- **Std Dev**: {stats['signed_diff']['std']:.6f}

### Magnitude of Changes
- **Difference > 1%**: {stats['changes_gt_1pct']:,} labels ({100*stats['changes_gt_1pct']/len(valid_df):.1f}%)
- **Difference > 5%**: {stats['changes_gt_5pct']:,} labels ({100*stats['changes_gt_5pct']/len(valid_df):.1f}%)
- **Difference > 10%**: {stats['changes_gt_10pct']:,} labels ({100*stats['changes_gt_10pct']/len(valid_df):.1f}%)

---

## Analysis by Split

"""

    for split in ['train', 'val', 'test', 'none']:
        subset = valid_df[valid_df['split'] == split]
        if len(subset) > 0:
            report += f"### {split.upper()} set\n"
            report += f"- Count: {len(subset):,}\n"
            report += f"- Mean abs diff: {subset['abs_diff'].mean():.6f}\n"
            report += f"- Median abs diff: {subset['abs_diff'].median():.6f}\n"
            report += f"- Max abs diff: {subset['abs_diff'].max():.6f}\n\n"

    # Analysis by label
    report += "## Analysis by Class Label\n\n"
    for label in ['cdw', 'no_cdw']:
        subset = valid_df[valid_df['label'] == label]
        if len(subset) > 0:
            report += f"### {label.upper()}\n"
            report += f"- Count: {len(subset):,}\n"
            report += f"- Mean orig prob: {subset['original_prob'].mean():.4f}\n"
            report += f"- Mean retrain prob: {subset['retrained_prob'].mean():.4f}\n"
            report += f"- Mean abs diff: {subset['abs_diff'].mean():.6f}\n"
            report += f"- Median abs diff: {subset['abs_diff'].median():.6f}\n"
            report += f"- Max abs diff: {subset['abs_diff'].max():.6f}\n\n"

    # Outliers
    top_changes = valid_df.nlargest(10, 'abs_diff')[
        ['raster', 'row_off', 'col_off', 'label', 'split', 'original_prob', 'retrained_prob', 'abs_diff']
    ]

    report += "## Top 10 Largest Probability Changes\n\n"
    for i, (idx, row) in enumerate(top_changes.iterrows(), 1):
        report += f"{i}. {row['raster']} @ ({int(row['row_off'])},{int(row['col_off'])})\n"
        report += f"   - Label: {row['label']}, Split: {row['split']}\n"
        report += f"   - Original: {row['original_prob']:.6f} → Retrained: {row['retrained_prob']:.6f}\n"
        report += f"   - Change: {row['abs_diff']:.6f} ({row['abs_diff']*100:.2f}%)\n\n"

    # Training metadata
    if meta:
        report += "## Retrained Ensemble Metadata\n\n"
        if 'test_metrics' in meta:
            tm = meta['test_metrics']
            report += f"### Test Set Metrics (Option B)\n"
            report += f"- Ensemble AUC: {tm.get('ensemble_auc', 'N/A')}\n"
            report += f"- Ensemble F1: {tm.get('ensemble_f1', 'N/A')}\n"
            report += f"- Threshold: {tm.get('ensemble_thresh', 'N/A')}\n"
            report += f"- Test size: {tm.get('n_test', 'N/A')}\n"
            report += f"- CDW count: {tm.get('n_cdw', 'N/A')}\n\n"

        if 'data_stats' in meta:
            ds = meta['data_stats']
            report += f"### Training Data (Option B)\n"
            report += f"- Train size: {ds.get('train_size', 'N/A'):,}\n"
            report += f"- Val size: {ds.get('val_size', 'N/A'):,}\n"
            report += f"- Test size: {ds.get('test_size', 'N/A'):,}\n"
            report += f"- Train CDW: {ds.get('train_cdw', 'N/A')}\n"
            report += f"- Train NO_CDW: {ds.get('train_no_cdw', 'N/A')}\n\n"

    report += """## Interpretation

### Key Findings

1. **Distribution shift reduction**: The mean probability difference should be lower than the original ~6% if retraining on spatial splits was successful.

2. **Test set bias**: If differences are larger in 'test' split, models may be overfitting to the spatial split test set.

3. **Buffer zone changes**: 'none' (buffer) labels should show different behavior since models weren't trained on them in the retrained approach.

4. **Outliers**: Large changes (>10%) may indicate:
   - Labels at split boundaries where models behave differently
   - Edge cases where spatial splits significantly changed training data distribution
   - Possible data quality issues

### Comparison to Original Approach

**Option A (Original):**
- Training: 19,812 tiles
- Applied to: 580,136 labels (includes buffer zones)
- Expected prob difference: ~6% (due to distribution shift)

**Option B (Retrained):**
- Training: 67,290 tiles (spatial splits, 3.4× more data)
- Applied to: 580,136 labels (same coverage)
- Expected prob difference: Should be lower (~2-3%) due to better alignment

---

Generated by spatial split ensemble retraining pipeline.
"""

    # Save report
    output_path = args.output
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

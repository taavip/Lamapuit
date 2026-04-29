#!/usr/bin/env python3
"""
Assign train/val/test/none splits to labels_canonical.csv.

Splits are assigned per map_sheet using spatial isolation via 50%-overlap chip clusters.
- Test zones: Chebyshev ≤ 1 in stride units (3×3 = 9 chips per center)
- Buffer zones: Chebyshev = 2 excluding diagonal corners (12 chips per center)
- Same (row_off, col_off) across all years on a map_sheet get the same split
- Training: remaining eligible labels minus mid-confidence (0.3–0.7)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("output/onboarding_labels_v2_drop13_standardized/labels_canonical.csv"),
        help="Path to labels_canonical.csv",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=Path, help="Output path (default: overwrite input)")
    parser.add_argument(
        "--stride", type=int, default=64, help="Stride in pixels (50%% overlap: stride=64, chunk=128)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    print(f"Loading {args.labels}...")
    df = pd.read_csv(args.labels)
    print(f"Loaded {len(df)} labels")

    # Assign splits
    df = assign_label_splits(df, seed=args.seed, stride=args.stride)

    # Print stats
    print_split_stats(df)

    # Write output
    if not args.dry_run:
        output_path = args.output or args.labels
        df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")


def assign_label_splits(df, seed=42, stride=64, test_frac=0.02, val_frac=0.01):
    """
    Assign split = {'test', 'val', 'train', 'none'} to each label.

    Eligible labels: source=='manual' OR source=='auto_skip' OR
                     model_prob > 0.95 OR model_prob < 0.05

    Procedure:
    1. Per map_sheet, aggregate all eligible labels across all years
    2. Select 2% of these eligible labels randomly as test seeds
    3. For each seed, expand from its position to test zone + buffer zone
    4. Mark all labels (any year) in test zones as 'test' if eligible, else 'none'
    5. Mark all labels in buffer zones as 'none'
    6. Repeat for val (1% of remaining eligible labels after test exclusion)
    7. Training: remaining eligible labels not in [0.3, 0.7] confidence
    8. Rest: 'none'
    """
    df = df.copy()
    df["split"] = "none"

    # Identify eligible labels
    eligible_mask = (
        (df["source"] == "manual")
        | (df["source"] == "auto_skip")
        | ((df["model_prob"].notna()) & ((df["model_prob"] > 0.95) | (df["model_prob"] < 0.05)))
    )

    # Add stride-unit coordinates
    df["row_s"] = df["row_off"] // stride
    df["col_s"] = df["col_off"] // stride

    rng = np.random.default_rng(seed)

    # Process per map_sheet
    for map_sheet, sheet_df_idx in df.groupby("map_sheet").groups.items():
        sheet_df = df.loc[sheet_df_idx]

        # Get all eligible labels on this map_sheet (indices)
        eligible_idx = sheet_df_idx[eligible_mask[sheet_df_idx]]
        if len(eligible_idx) == 0:
            continue

        # Select 2% of eligible labels as test seeds
        n_test_seeds = max(1, int(len(eligible_idx) * test_frac))
        test_seed_indices = rng.choice(eligible_idx, size=n_test_seeds, replace=False)

        # Get positions of test seeds
        test_centers = set()
        for idx in test_seed_indices:
            pos = (df.at[idx, "row_s"], df.at[idx, "col_s"])
            test_centers.add(pos)

        # Expand test centers to test and buffer positions
        test_pos, buffer_pos = _expand_centers_to_zones(test_centers)

        # Mark test and buffer for all labels on this sheet
        for idx in sheet_df_idx:
            pos = (df.at[idx, "row_s"], df.at[idx, "col_s"])
            if pos in test_pos:
                if eligible_mask[idx]:
                    df.at[idx, "split"] = "test"
                # else: ineligible stays 'none'
            elif pos in buffer_pos:
                df.at[idx, "split"] = "none"  # buffer = thrown out

        # Val: Select 1% of remaining eligible labels (not in test/buffer zones) as val seeds
        remaining_eligible_idx = eligible_idx[
            (df.loc[eligible_idx, "split"] == "none") & ~(df.loc[eligible_idx, "row_s"].apply(
                lambda r_s: any((r_s, df.loc[idx, "col_s"]) in buffer_pos for idx in eligible_idx)
            ))
        ]

        # Simpler: remaining eligible are those not yet marked as test
        remaining_eligible_idx = eligible_idx[df.loc[eligible_idx, "split"] == "none"]

        if len(remaining_eligible_idx) > 0:
            n_val_seeds = max(1, int(len(remaining_eligible_idx) * val_frac))
            val_seed_indices = rng.choice(remaining_eligible_idx, size=n_val_seeds, replace=False)

            val_centers = set()
            for idx in val_seed_indices:
                pos = (df.at[idx, "row_s"], df.at[idx, "col_s"])
                val_centers.add(pos)

            val_pos, val_buffer_pos = _expand_centers_to_zones(val_centers)

            for idx in sheet_df_idx:
                pos = (df.at[idx, "row_s"], df.at[idx, "col_s"])
                if df.at[idx, "split"] == "none":  # not already test
                    if pos in val_pos:
                        if eligible_mask[idx]:
                            df.at[idx, "split"] = "val"
                    elif pos in val_buffer_pos:
                        df.at[idx, "split"] = "none"  # val buffer

    # Training: eligible labels not in test/val/buffer, minus mid-confidence
    mid_conf_mask = (
        df["model_prob"].notna()
        & (df["model_prob"] >= 0.3)
        & (df["model_prob"] <= 0.7)
    )
    train_mask = (eligible_mask) & (df["split"] == "none") & (~mid_conf_mask)
    df.loc[train_mask, "split"] = "train"

    # Clean up temp columns
    df.drop(columns=["row_s", "col_s"], inplace=True)

    return df


def _expand_centers_to_zones(centers):
    """
    Expand center positions to test and buffer position sets.

    Test zone: Chebyshev ≤ 1 from center (3×3 = 9 positions)
    Buffer zone: Chebyshev = 2 from center, excluding diagonal corners (12 positions)
    """
    test_pos = set()
    buffer_pos = set()

    for r_s, c_s in centers:
        # Test zone: Chebyshev ≤ 1
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                test_pos.add((r_s + dr, c_s + dc))

        # Buffer zone: Chebyshev = 2 (excluding diagonal corners at ±2,±2)
        # Cardinal directions at distance 2
        buffer_pos.add((r_s, c_s + 2))  # right
        buffer_pos.add((r_s, c_s - 2))  # left
        buffer_pos.add((r_s + 2, c_s))  # down
        buffer_pos.add((r_s - 2, c_s))  # up
        # Knight's move (±1, ±2) and (±2, ±1)
        for dr, dc in [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]:
            buffer_pos.add((r_s + dr, c_s + dc))

    # Remove test positions from buffer
    buffer_pos -= test_pos

    return test_pos, buffer_pos


def print_split_stats(df):
    """Print split statistics."""
    print("\n" + "=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)

    total = len(df)
    stats = {}

    for split in ["test", "val", "train", "none"]:
        count = (df["split"] == split).sum()
        pct = 100.0 * count / total
        stats[split] = (count, pct)
        print(f"{split:10s}: {count:7d} ({pct:6.2f}%)")

    print("-" * 60)
    print(f"{'TOTAL':10s}: {total:7d} (100.00%)")

    # Breakdown of eligible vs ineligible
    eligible_mask = (
        (df["source"] == "manual")
        | (df["source"] == "auto_skip")
        | ((df["model_prob"].notna()) & ((df["model_prob"] > 0.95) | (df["model_prob"] < 0.05)))
    )
    print("\n" + "-" * 60)
    print(f"Eligible labels: {eligible_mask.sum()} ({100.0*eligible_mask.sum()/total:.2f}%)")
    print(f"Ineligible labels: {(~eligible_mask).sum()} ({100.0*(~eligible_mask).sum()/total:.2f}%)")

    # Breakdown of test/val/train/none by eligibility
    print("\n" + "-" * 60)
    print("Eligible labels by split:")
    for split in ["test", "val", "train", "none"]:
        mask = (df["split"] == split) & eligible_mask
        count = mask.sum()
        if stats[split][0] > 0:
            pct_of_split = 100.0 * count / stats[split][0]
            print(f"  {split:10s}: {count:7d} ({pct_of_split:6.2f}% of {split})")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

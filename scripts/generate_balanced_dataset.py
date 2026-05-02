#!/usr/bin/env python
"""
Generate balanced dataset with additional no-CDW examples for better model training.

This script:
1. Adds more empty/background tiles (no CDW)
2. Balances positive (CDW) and negative (no CDW) examples
3. Helps prevent model bias towards always predicting CDW

Usage:
    python scripts/generate_balanced_dataset.py --dataset data/dataset_enhanced_robust --balance_ratio 0.6
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import random
import shutil


def generate_negative_samples(
    chm_path: str,
    output_dir: Path,
    split: str,
    n_samples: int,
    tile_size: int = 640,
    max_nodata_frac: float = 0.8,
):
    """
    Generate negative (no CDW) samples from CHM areas without labels.

    Args:
        chm_path: Path to CHM raster
        output_dir: Output directory
        split: 'train' or 'val'
        n_samples: Number of negative samples to generate
        tile_size: Size of tiles
        max_nodata_frac: Maximum allowed nodata fraction
    """
    img_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split

    with rasterio.open(chm_path) as src:
        width, height = src.width, src.height
        nodata = src.nodata or -9999.0

        generated = 0
        attempts = 0
        max_attempts = n_samples * 10

        while generated < n_samples and attempts < max_attempts:
            attempts += 1

            # Random position
            col = random.randint(0, width - tile_size)
            row = random.randint(0, height - tile_size)

            # Read tile
            window = Window(col, row, tile_size, tile_size)
            tile_data = src.read(1, window=window)

            # Check nodata
            nodata_mask = np.isnan(tile_data) | (tile_data == nodata) | (tile_data < 0)
            if nodata_mask.sum() / nodata_mask.size > max_nodata_frac:
                continue

            # Normalize
            valid = tile_data[~nodata_mask]
            if len(valid) == 0 or valid.max() <= valid.min():
                continue

            tile_norm = np.clip((tile_data - valid.min()) / (valid.max() - valid.min()), 0, 1)
            tile_img = (tile_norm * 255).astype(np.uint8)
            tile_img[nodata_mask] = 0

            # Save image
            idx = len(list(img_dir.glob("*.png")))
            img_path = img_dir / f"negative_{idx:05d}.png"
            cv2.imwrite(str(img_path), tile_img)

            # Create empty label file
            label_path = label_dir / f"negative_{idx:05d}.txt"
            label_path.touch()

            generated += 1

    return generated


def balance_dataset(dataset_dir: str, balance_ratio: float = 0.6):
    """
    Balance dataset by adding negative samples.

    Args:
        dataset_dir: Path to dataset directory
        balance_ratio: Target ratio of CDW samples (0.6 = 60% CDW, 40% no-CDW)
    """
    dataset_dir = Path(dataset_dir)

    print(f"Balancing dataset at {dataset_dir}")
    print(f"Target CDW ratio: {balance_ratio:.1%}")

    # Find available CHM files
    chm_files = list(Path("data/chm_max_hag").glob("*.tif"))
    if not chm_files:
        print("Error: No CHM files found")
        return

    # Random CHM for negative sampling
    chm_path = str(random.choice(chm_files))
    print(f"Using CHM for negative sampling: {Path(chm_path).name}")

    for split in ["train", "val"]:
        img_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            continue

        # Count existing samples
        img_files = list(img_dir.glob("*.png"))
        cdw_count = 0
        empty_count = 0

        for img_file in img_files:
            label_file = label_dir / (img_file.stem + ".txt")
            if label_file.exists() and label_file.stat().st_size > 0:
                cdw_count += 1
            else:
                empty_count += 1

        print(f"\n{split.upper()} split:")
        print(f"  Current: {cdw_count} CDW, {empty_count} no-CDW")

        # Calculate needed negative samples
        target_total = int(cdw_count / balance_ratio)
        needed_negatives = max(0, target_total - (cdw_count + empty_count))

        if needed_negatives > 0:
            print(f"  Generating {needed_negatives} additional no-CDW samples...")
            generated = generate_negative_samples(chm_path, dataset_dir, split, needed_negatives)
            print(f"  Generated: {generated} no-CDW samples")

            final_total = cdw_count + empty_count + generated
            final_ratio = cdw_count / final_total if final_total > 0 else 0
            print(f"  Final: {cdw_count} CDW, {empty_count + generated} no-CDW")
            print(f"  CDW ratio: {final_ratio:.1%}")
        else:
            print(f"  Already balanced (CDW ratio: {cdw_count/(cdw_count+empty_count):.1%})")


def main():
    parser = argparse.ArgumentParser(description="Balance dataset with no-CDW samples")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument(
        "--balance_ratio",
        type=float,
        default=0.6,
        help="Target ratio of CDW samples (0.6 = 60%% CDW, 40%% no-CDW)",
    )

    args = parser.parse_args()
    balance_dataset(args.dataset, args.balance_ratio)


if __name__ == "__main__":
    main()

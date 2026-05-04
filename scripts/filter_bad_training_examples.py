#!/usr/bin/env python
"""
Filter out bad or uninformative training examples from a YOLO dataset to improve model performance.

- Removes empty images (no CDW objects)
- Removes images with excessive nodata
- Optionally removes outliers based on mask area or other heuristics

Usage:
    python scripts/filter_bad_training_examples.py --dataset data/dataset_enhanced_robust --min_mask_pixels 30 --max_nodata_frac 0.95
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import shutil
from tqdm import tqdm


def is_bad_example(img_path, label_path, min_mask_pixels=30, max_nodata_frac=0.95):
    # Check if label file is empty (no CDW objects)
    if not label_path.exists() or label_path.stat().st_size == 0:
        return True
    # Check mask area and nodata
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return True
    # Nodata: assume 0 is nodata (as in your pipeline)
    nodata_frac = (img == 0).sum() / img.size
    if nodata_frac > max_nodata_frac:
        return True
    # Check mask area
    mask_pixels = 0
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 2:
                mask_pixels += len(parts[1:]) // 2
    if mask_pixels < min_mask_pixels:
        return True
    return False


def filter_dataset(dataset_dir, min_mask_pixels=30, max_nodata_frac=0.95, split="train"):
    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    filtered_img_dir = dataset_dir / "images" / f"{split}_filtered"
    filtered_label_dir = dataset_dir / "labels" / f"{split}_filtered"
    filtered_img_dir.mkdir(parents=True, exist_ok=True)
    filtered_label_dir.mkdir(parents=True, exist_ok=True)

    kept, removed = 0, 0
    for img_path in tqdm(list(img_dir.glob("*.png"))):
        label_path = label_dir / (img_path.stem + ".txt")
        if is_bad_example(img_path, label_path, min_mask_pixels, max_nodata_frac):
            removed += 1
            continue
        shutil.copy2(img_path, filtered_img_dir / img_path.name)
        shutil.copy2(label_path, filtered_label_dir / label_path.name)
        kept += 1
    print(f"Filtered {split}: Kept {kept}, Removed {removed}")


def main():
    parser = argparse.ArgumentParser(description="Filter bad training examples from YOLO dataset")
    parser.add_argument("--dataset", required=True, help="Path to YOLO dataset root")
    parser.add_argument(
        "--min_mask_pixels", type=int, default=30, help="Minimum mask pixels to keep example"
    )
    parser.add_argument(
        "--max_nodata_frac", type=float, default=0.95, help="Maximum allowed nodata fraction"
    )
    args = parser.parse_args()

    for split in ["train", "val"]:
        filter_dataset(args.dataset, args.min_mask_pixels, args.max_nodata_frac, split)
    print("Filtering complete. Use *_filtered folders for training.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create a proper test dataset split from existing data.

This script:
1. Loads the existing enhanced_robust dataset
2. Creates a proper 70/15/15 train/val/test split
3. Ensures class balance across splits
4. Creates dataset.yaml files for each configuration
"""

import shutil
from pathlib import Path
import random
import yaml
import cv2
import numpy as np


def create_test_split(
    source_dataset: str = "data/dataset_enhanced_robust",
    output_dataset: str = "data/dataset_final",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Create train/val/test split from existing dataset.

    Args:
        source_dataset: Path to source dataset
        output_dataset: Path to output dataset with test split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    src = Path(source_dataset)
    dst = Path(output_dataset)

    # Create output structure
    for split in ["train", "val", "test"]:
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Collect all images
    all_images = []
    for split in ["train", "val"]:
        split_dir = src / "images" / split
        if split_dir.exists():
            all_images.extend(list(split_dir.glob("*.png")))

    print(f"Found {len(all_images)} total images")

    # Separate images with and without CDW
    images_with_cdw = []
    images_without_cdw = []

    for img_path in all_images:
        label_path = src / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path) as f:
                content = f.read().strip()
            if content:
                images_with_cdw.append(img_path)
            else:
                images_without_cdw.append(img_path)
        else:
            images_without_cdw.append(img_path)

    print(f"  With CDW: {len(images_with_cdw)}")
    print(f"  Without CDW: {len(images_without_cdw)}")

    # Shuffle
    random.shuffle(images_with_cdw)
    random.shuffle(images_without_cdw)

    # Calculate split indices for each class
    def split_list(lst, ratios):
        n = len(lst)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        return (lst[:n_train], lst[n_train : n_train + n_val], lst[n_train + n_val :])

    ratios = (train_ratio, val_ratio, test_ratio)
    train_cdw, val_cdw, test_cdw = split_list(images_with_cdw, ratios)
    train_empty, val_empty, test_empty = split_list(images_without_cdw, ratios)

    # Combine splits
    splits = {
        "train": train_cdw + train_empty,
        "val": val_cdw + val_empty,
        "test": test_cdw + test_empty,
    }

    # Copy files to new structure
    for split_name, images in splits.items():
        print(f"\n{split_name.capitalize()} split: {len(images)} images")

        for img_path in images:
            # Copy image
            dst_img = dst / "images" / split_name / img_path.name
            shutil.copy(img_path, dst_img)

            # Copy label
            src_label = src / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
            dst_label = dst / "labels" / split_name / f"{img_path.stem}.txt"

            if src_label.exists():
                shutil.copy(src_label, dst_label)
            else:
                # Create empty label file
                dst_label.touch()

    # Create dataset.yaml files
    _create_yaml_configs(dst)

    # Print statistics
    _print_statistics(dst)

    print(f"\n✓ Dataset created at: {dst}")


def _create_yaml_configs(dataset_path: Path):
    """Create dataset.yaml configurations."""

    # Full dataset (train + val + test)
    yaml_full = {
        "path": str(dataset_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "cdw"},
    }

    with open(dataset_path / "dataset.yaml", "w") as f:
        yaml.dump(yaml_full, f, default_flow_style=False)

    # Train+Val only (for training)
    yaml_trainval = {
        "path": str(dataset_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "cdw"},
    }

    with open(dataset_path / "dataset_trainval.yaml", "w") as f:
        yaml.dump(yaml_trainval, f, default_flow_style=False)

    print("\nCreated dataset configurations:")
    print("  dataset.yaml - Full dataset with test split")
    print("  dataset_trainval.yaml - Training configuration (train+val only)")


def _print_statistics(dataset_path: Path):
    """Print detailed dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        img_dir = dataset_path / "images" / split
        label_dir = dataset_path / "labels" / split

        if not img_dir.exists():
            continue

        images = list(img_dir.glob("*.png"))

        with_cdw = 0
        total_instances = 0

        for img_path in images:
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path) as f:
                    lines = f.readlines()
                if lines:
                    with_cdw += 1
                    total_instances += len(lines)

        empty = len(images) - with_cdw
        balance = with_cdw / len(images) * 100 if images else 0

        print(f"\n{split.upper()}:")
        print(f"  Total images: {len(images)}")
        print(f"  With CDW: {with_cdw} ({balance:.1f}%)")
        print(f"  Empty: {empty} ({100-balance:.1f}%)")
        print(f"  Total instances: {total_instances}")
        if with_cdw > 0:
            print(f"  Instances per positive: {total_instances/with_cdw:.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create test dataset split")
    parser.add_argument(
        "--source", default="data/dataset_enhanced_robust", help="Source dataset directory"
    )
    parser.add_argument("--output", default="data/dataset_final", help="Output dataset directory")
    parser.add_argument(
        "--train-ratio", type=float, default=0.70, help="Training set ratio (default: 0.70)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 0.01:
        print(f"Error: Ratios must sum to 1.0, got {total}")
        exit(1)

    create_test_split(
        source_dataset=args.source,
        output_dataset=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

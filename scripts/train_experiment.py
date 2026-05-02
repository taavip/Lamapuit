#!/usr/bin/env python
"""
Flexible training script for experiments with different augmentation presets.
"""

import argparse
import time
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Augmentation presets
AUGMENTATION_PRESETS = {
    "none": {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    },
    "light": {
        "hsv_h": 0.01,
        "hsv_s": 0.3,
        "hsv_v": 0.2,
        "degrees": 10.0,
        "translate": 0.05,
        "scale": 0.3,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.3,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    },
    "moderate": {
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 15.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.05,
        "copy_paste": 0.0,
        "erasing": 0.0,
    },
    "heavy": {
        "hsv_h": 0.03,
        "hsv_s": 0.8,
        "hsv_v": 0.5,
        "degrees": 30.0,
        "translate": 0.2,
        "scale": 0.7,
        "shear": 5.0,
        "perspective": 0.0001,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.3,
        "erasing": 0.2,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Experiment training with configurable settings")
    parser.add_argument(
        "--data",
        default="data/dataset_enhanced_robust/dataset_filtered.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--model",
        default="yolo11s-seg.pt",
        help="Model: yolo11n-seg.pt, yolo11s-seg.pt, yolo11m-seg.pt",
    )
    parser.add_argument("--name", default="experiment", help="Run name")
    parser.add_argument("--epochs", type=int, default=150, help="Max epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument(
        "--augmentation",
        default="light",
        choices=["none", "light", "moderate", "heavy"],
        help="Augmentation preset",
    )

    args = parser.parse_args()

    # Auto-detect device
    if torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")
    else:
        device = "cpu"
        print("⚠ No GPU, using CPU")

    # Get augmentation settings
    aug_config = AUGMENTATION_PRESETS[args.augmentation]

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {args.name}")
    print(f"{'='*70}")
    print(f"Dataset:       {args.data}")
    print(f"Model:         {args.model}")
    print(f"Batch size:    {args.batch}")
    print(f"Epochs:        {args.epochs} (patience={args.patience})")
    print(f"Augmentation:  {args.augmentation}")
    print(f"Device:        {device}")
    print(f"{'='*70}\n")

    start_time = time.time()

    from ultralytics import YOLO
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    yolo = YOLO(args.model)

    # Training with experiment settings
    results = yolo.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        name=args.name,
        project="runs/cdw_detect",
        imgsz=640,
        # Use preset augmentation
        **aug_config,
        # Fixed settings
        patience=args.patience,
        save_period=10,
        cache="disk",
        workers=8,
        cos_lr=True,
        close_mosaic=10,
        dropout=0.1,
        weight_decay=0.0005,
        exist_ok=True,
        verbose=True,
        amp=True if device != "cpu" else False,
        cls=0.25,
        box=8.0,
        dfl=1.5,
        val=True,
        plots=True,
    )

    elapsed = time.time() - start_time
    best_model = Path("runs/cdw_detect") / args.name / "weights" / "best.pt"

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE: {args.name}")
    print(f"{'='*70}")
    print(f"Duration:    {elapsed/60:.1f} minutes")
    print(f"Best model:  {best_model}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

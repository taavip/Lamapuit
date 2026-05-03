#!/usr/bin/env python
"""
Enhanced training script with GPU auto-detection, augmentation, and time-based training.
"""

import argparse
import time
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def estimate_epochs_for_duration(dataset_yaml, batch_size, target_hours=4, device="0"):
    """
    Estimate number of epochs that can be completed in target duration.
    Uses a conservative estimate based on typical YOLO training speeds.
    """
    # Rough estimates (can be adjusted based on actual hardware)
    if device == "cpu":
        seconds_per_epoch_estimate = 300  # 5 minutes per epoch on CPU
    else:
        # GPU training - estimate based on batch size and dataset size
        # For ~400 images, typical GPU can do ~30-60 seconds per epoch
        seconds_per_epoch_estimate = 60

    target_seconds = target_hours * 3600
    estimated_epochs = int(target_seconds / seconds_per_epoch_estimate)

    # Apply safety factor to avoid overrunning (use 80% of time)
    safe_epochs = int(estimated_epochs * 0.8)

    # Reasonable bounds
    safe_epochs = max(10, min(safe_epochs, 300))  # Between 10 and 300 epochs

    return safe_epochs


def main():
    parser = argparse.ArgumentParser(description="Train YOLO CDW model with optimal settings")
    parser.add_argument(
        "--data", default="data/dataset_enhanced_robust/dataset.yaml", help="Path to dataset.yaml"
    )
    parser.add_argument(
        "--hours", type=float, default=4.0, help="Target training duration in hours"
    )
    parser.add_argument(
        "--batch", type=int, default=None, help="Batch size (auto-detect if not specified)"
    )
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Base model")
    parser.add_argument("--name", default="cdw_4hour_training", help="Run name")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")

    args = parser.parse_args()

    # Auto-detect device
    if torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")

        # Auto-select batch size based on GPU memory
        if args.batch is None:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb >= 16:
                batch_size = 16
            elif gpu_memory_gb >= 8:
                batch_size = 8
            else:
                batch_size = 4
            print(f"✓ Auto-selected batch size: {batch_size} (GPU memory: {gpu_memory_gb:.1f} GB)")
        else:
            batch_size = args.batch
    else:
        device = "cpu"
        batch_size = args.batch or 2
        print(f"⚠ No GPU detected, using CPU (this will be slow)")

    # Estimate epochs for target duration
    epochs = estimate_epochs_for_duration(args.data, batch_size, args.hours, device)

    print(f"\n{'='*70}")
    print(f"ENHANCED TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Dataset:          {args.data}")
    print(f"Target duration:  {args.hours} hours")
    print(f"Estimated epochs: {epochs}")
    print(f"Batch size:       {batch_size}")
    print(f"Device:           {device}")
    print(f"Model:            {args.model}")
    print(f"Early stopping:   {args.patience} epochs patience")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Train with ultralytics YOLO directly for full augmentation control
    from ultralytics import YOLO
    import gc

    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    yolo = YOLO(args.model)

    # Train with enhanced augmentation settings
    results = yolo.train(
        data=args.data,
        epochs=epochs,
        batch=batch_size,
        device=device,
        name=args.name,
        project="runs/cdw_detect",
        # Enhanced augmentation parameters
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation
        hsv_v=0.4,  # HSV-Value
        degrees=10.0,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,  # Scaling augmentation
        shear=2.0,  # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.5,  # Vertical flip
        fliplr=0.5,  # Horizontal flip
        mosaic=1.0,  # Mosaic augmentation
        mixup=0.1,  # Mixup augmentation
        copy_paste=0.1,  # Copy-paste augmentation
        # Training parameters
        patience=args.patience,  # Early stopping
        save_period=10,  # Save checkpoint every 10 epochs
        cache="disk",  # Cache images to disk for faster training
        workers=8,  # Data loading workers
        cos_lr=True,  # Cosine learning rate scheduler
        close_mosaic=10,  # Disable mosaic in last 10 epochs
        exist_ok=True,
        verbose=True,
        amp=True if device != "cpu" else False,
    )

    best_model = Path("runs/cdw_detect") / args.name / "weights" / "best.pt"

    elapsed_time = time.time() - start_time
    elapsed_hours = elapsed_time / 3600

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Duration:    {elapsed_hours:.2f} hours ({elapsed_time/60:.1f} minutes)")
    print(f"Best model:  {best_model}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Train YOLO model with filtered dataset and optimized settings for better CDW detection.

Suggestions for Better Results:
1. Use filtered dataset (removed bad examples)
2. Increase model size (yolo11s-seg or yolo11m-seg for better accuracy)
3. Longer training with early stopping
4. Focus augmentation on CDW-specific challenges
5. Use pretrained weights
6. Optimize for recall over precision (better to catch all CDW)

Usage:
    python scripts/train_with_suggestions.py --hours 2 --model yolo11s-seg.pt
"""

import argparse
import time
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def estimate_epochs_for_duration(target_hours=2, device="0"):
    """Estimate epochs based on target duration."""
    if device == "cpu":
        seconds_per_epoch = 300
    else:
        seconds_per_epoch = 45  # Faster with fewer examples

    target_seconds = target_hours * 3600
    estimated_epochs = int(target_seconds / seconds_per_epoch * 0.8)
    return max(30, min(estimated_epochs, 500))


def main():
    parser = argparse.ArgumentParser(
        description="Train with filtered dataset and optimization suggestions"
    )
    parser.add_argument(
        "--data",
        default="data/dataset_enhanced_robust/dataset_filtered.yaml",
        help="Path to filtered dataset.yaml",
    )
    parser.add_argument("--hours", type=float, default=2.0, help="Target training duration")
    parser.add_argument(
        "--batch", type=int, default=None, help="Batch size (auto if not specified)"
    )
    parser.add_argument(
        "--model",
        default="yolo11s-seg.pt",
        help="Model size: yolo11n-seg.pt (fast) or yolo11s-seg.pt (better) or yolo11m-seg.pt (best)",
    )
    parser.add_argument("--name", default="cdw_filtered_optimized", help="Run name")

    args = parser.parse_args()

    # Auto-detect device
    if torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")

        if args.batch is None:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Larger batch for fewer examples
            if gpu_memory_gb >= 16:
                batch_size = 32
            elif gpu_memory_gb >= 8:
                batch_size = 16
            else:
                batch_size = 8
            print(f"✓ Auto-selected batch size: {batch_size}")
        else:
            batch_size = args.batch
    else:
        device = "cpu"
        batch_size = args.batch or 4
        print("⚠ No GPU, using CPU")

    epochs = estimate_epochs_for_duration(args.hours, device)

    print(f"\n{'='*70}")
    print(f"OPTIMIZED TRAINING WITH FILTERED DATASET")
    print(f"{'='*70}")
    print(f"Dataset:         {args.data}")
    print(f"Model:           {args.model}")
    print(f"Target duration: {args.hours} hours")
    print(f"Estimated epochs:{epochs}")
    print(f"Batch size:      {batch_size}")
    print(f"Device:          {device}")
    print(f"\nOPTIMIZATIONS APPLIED:")
    print(f"  ✓ Filtered bad examples (empty/nodata)")
    print(f"  ✓ Higher quality training data")
    print(f"  ✓ Larger batch size for stable gradients")
    print(f"  ✓ Enhanced augmentation")
    print(f"  ✓ Optimized for CDW detection")
    print(f"{'='*70}\n")

    start_time = time.time()

    from ultralytics import YOLO
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    yolo = YOLO(args.model)

    # Optimized training settings
    results = yolo.train(
        data=args.data,
        epochs=epochs,
        batch=batch_size,
        device=device,
        name=args.name,
        project="runs/cdw_detect",
        imgsz=640,
        # Enhanced augmentation for CDW detection
        hsv_h=0.02,  # Slight color variation
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,  # More rotation
        translate=0.15,  # More translation
        scale=0.6,  # More scaling
        shear=3.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,  # More mixup
        copy_paste=0.15,  # More copy-paste
        # Training optimizations
        patience=50,  # More patience for filtered dataset
        save_period=20,
        cache="disk",
        workers=8,
        cos_lr=True,
        close_mosaic=15,
        exist_ok=True,
        verbose=True,
        amp=True if device != "cpu" else False,
        # Optimize for recall (catch all CDW)
        cls=0.3,  # Lower classification loss weight
        box=7.5,
        dfl=1.5,
    )

    best_model = Path("runs/cdw_detect") / args.name / "weights" / "best.pt"
    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Duration:    {elapsed/3600:.2f} hours")
    print(f"Best model:  {best_model}")
    print(f"\nSUGGESTIONS FOR FURTHER IMPROVEMENT:")
    print(f"  1. Use larger model: yolo11m-seg.pt or yolo11l-seg.pt")
    print(f"  2. Add more labeled data from different CHM sources")
    print(f"  3. Fine-tune on specific forest types or conditions")
    print(f"  4. Adjust confidence threshold during inference")
    print(f"  5. Use ensemble of models for better results")
    print(f"  6. Post-process predictions with geometric constraints")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

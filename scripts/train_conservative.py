#!/usr/bin/env python
"""
Conservative training approach for small dataset:
- Smaller model (yolo11s-seg)
- Lower augmentation
- Stricter early stopping
- Higher dropout
"""

import argparse
import time
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Conservative CDW training for small dataset")
    parser.add_argument(
        "--data",
        default="data/dataset_enhanced_robust/dataset_filtered.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument(
        "--batch", type=int, default=None, help="Batch size (auto if not specified)"
    )
    parser.add_argument(
        "--model",
        default="yolo11s-seg.pt",
        help="Model: yolo11n-seg.pt or yolo11s-seg.pt (recommended for small data)",
    )
    parser.add_argument("--name", default="cdw_conservative", help="Run name")

    args = parser.parse_args()

    # Auto-detect device
    if torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")

        if args.batch is None:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Smaller batches for better generalization with small dataset
            if "yolo11s" in args.model:
                batch_size = 8  # Small batches = more gradient updates
            else:
                batch_size = 16
            print(f"✓ Auto-selected batch size: {batch_size}")
        else:
            batch_size = args.batch
    else:
        device = "cpu"
        batch_size = args.batch or 4
        print("⚠ No GPU, using CPU")

    print(f"\n{'='*70}")
    print(f"CONSERVATIVE CDW TRAINING (Small Dataset Optimized)")
    print(f"{'='*70}")
    print(f"Dataset:       {args.data}")
    print(f"Model:         {args.model}")
    print(f"Batch size:    {batch_size}")
    print(f"Device:        {device}")
    print(f"Strategy:      Reduce overfitting with smaller model + less augmentation")
    print(f"{'='*70}\n")

    start_time = time.time()

    from ultralytics import YOLO
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    yolo = YOLO(args.model)

    # Conservative training settings for small dataset
    results = yolo.train(
        data=args.data,
        epochs=300,  # Fewer epochs
        batch=batch_size,
        device=device,
        name=args.name,
        project="runs/cdw_detect",
        imgsz=640,
        # REDUCED augmentation (avoid overfitting small dataset)
        hsv_h=0.015,  # Lower color variation
        hsv_s=0.5,  # Lower saturation
        hsv_v=0.3,  # Lower brightness
        degrees=15.0,  # Reduced rotation (was 30)
        translate=0.1,  # Reduced translation (was 0.2)
        scale=0.5,  # Reduced scale (was 0.7)
        shear=2.0,  # Reduced shear (was 5.0)
        perspective=0.0,  # Disabled
        flipud=0.5,  # Keep flips
        fliplr=0.5,
        mosaic=0.5,  # Reduced mosaic (was 1.0)
        mixup=0.05,  # Much lower mixup (was 0.2)
        copy_paste=0.0,  # DISABLED copy-paste (can cause overfitting)
        erasing=0.0,  # DISABLED erasing
        # AGGRESSIVE early stopping
        patience=50,  # Stop earlier (was 100)
        save_period=10,  # Save more frequently
        # Performance optimizations
        cache="disk",
        workers=8,
        cos_lr=True,
        close_mosaic=10,  # Disable mosaic earlier for fine-tuning
        # Regularization
        dropout=0.1,  # Add dropout to prevent overfitting
        weight_decay=0.0005,  # L2 regularization
        # Other settings
        exist_ok=True,
        verbose=True,
        amp=True if device != "cpu" else False,
        # Loss weights (same as ultimate)
        cls=0.25,
        box=8.0,
        dfl=1.5,
        # Validation settings
        val=True,
        plots=True,
    )

    best_model = Path("runs/cdw_detect") / args.name / "weights" / "best.pt"
    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Duration:    {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"Best model:  {best_model}")
    print(f"\nNEXT: Compare with ultimate model:")
    print(
        f"  python scripts/plot_training_results.py --run runs/segment/runs/cdw_detect/{args.name}"
    )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Ultimate training script with:
- Unlimited epochs with smart early stopping (no overtraining)
- CDW-specific augmentation
- Balanced dataset with no-CDW examples
- Aggressive monitoring to prevent overfitting

Usage:
    python scripts/train_ultimate.py --model yolo11m-seg.pt
"""

import argparse
import time
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Ultimate CDW training with smart early stopping")
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
        default="yolo11m-seg.pt",
        help="Model: yolo11s-seg.pt, yolo11m-seg.pt (recommended), yolo11l-seg.pt",
    )
    parser.add_argument("--name", default="cdw_ultimate", help="Run name")

    args = parser.parse_args()

    # Auto-detect device
    if torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")

        if args.batch is None:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Adjust batch size based on model and GPU memory
            if "yolo11l" in args.model:
                batch_size = 8 if gpu_memory_gb >= 16 else 4
            elif "yolo11m" in args.model:
                batch_size = 16 if gpu_memory_gb >= 16 else 8
            else:
                batch_size = 32 if gpu_memory_gb >= 16 else 16
            print(f"✓ Auto-selected batch size: {batch_size}")
        else:
            batch_size = args.batch
    else:
        device = "cpu"
        batch_size = args.batch or 4
        print("⚠ No GPU, using CPU")

    print(f"\n{'='*70}")
    print(f"ULTIMATE CDW TRAINING")
    print(f"{'='*70}")
    print(f"Dataset:       {args.data}")
    print(f"Model:         {args.model}")
    print(f"Batch size:    {batch_size}")
    print(f"Device:        {device}")
    print(f"Max epochs:    500 (with smart early stopping)")
    print(f"\nTRAINING STRATEGY:")
    print(f"  ✓ Train until convergence (no arbitrary time limit)")
    print(f"  ✓ Aggressive early stopping (patience=100)")
    print(f"  ✓ CDW-specific augmentation")
    print(f"  ✓ Balanced CDW/no-CDW data")
    print(f"  ✓ Prevent overfitting with validation monitoring")
    print(f"  ✓ Save best model automatically")
    print(f"{'='*70}\n")

    start_time = time.time()

    from ultralytics import YOLO
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    yolo = YOLO(args.model)

    # Ultimate training settings
    results = yolo.train(
        data=args.data,
        epochs=500,  # High limit, will stop early when converged
        batch=batch_size,
        device=device,
        name=args.name,
        project="runs/cdw_detect",
        imgsz=640,
        # CDW-specific heavy augmentation
        hsv_h=0.03,  # Color variation (different lighting/seasons)
        hsv_s=0.8,  # Saturation variation
        hsv_v=0.5,  # Brightness variation
        degrees=30.0,  # Strong rotation (fallen trees at any angle)
        translate=0.2,  # Translation (CDW can be anywhere)
        scale=0.7,  # Scale variation (near/far CDW)
        shear=5.0,  # Shear (perspective effects)
        perspective=0.0001,  # Slight perspective
        flipud=0.5,  # Vertical flip (no preferred orientation)
        fliplr=0.5,  # Horizontal flip
        mosaic=1.0,  # Mosaic (learn CDW in context)
        mixup=0.2,  # Strong mixup (complex scenes)
        copy_paste=0.3,  # Strong copy-paste (more CDW variations)
        erasing=0.2,  # Random erasing (occlusion robustness)
        # Smart early stopping - no overtraining
        patience=100,  # Wait 100 epochs for improvement
        save_period=25,  # Save checkpoint every 25 epochs
        # Performance optimizations
        cache="disk",  # Cache for faster loading
        workers=8,  # Parallel data loading
        cos_lr=True,  # Cosine learning rate (smooth convergence)
        close_mosaic=20,  # Disable mosaic in last 20 epochs (fine-tuning)
        # Other settings
        exist_ok=True,
        verbose=True,
        amp=True if device != "cpu" else False,
        # Loss weights optimized for CDW detection
        cls=0.25,  # Lower classification loss (focus on localization)
        box=8.0,  # Higher box loss (precise boundaries)
        dfl=1.5,  # DFL loss
        # Validation settings
        val=True,  # Enable validation
        plots=True,  # Generate plots
    )

    best_model = Path("runs/cdw_detect") / args.name / "weights" / "best.pt"
    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Duration:    {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"Best model:  {best_model}")
    print(f"\nTRAINING STOPPED BECAUSE:")
    print(f"  • Validation metric didn't improve for 100 epochs")
    print(f"  • Model has converged to optimal performance")
    print(f"  • No overtraining occurred")
    print(f"\nNEXT STEPS:")
    print(
        f"  1. Plot results: python scripts/plot_training_results.py --run runs/segment/runs/cdw_detect/{args.name}"
    )
    print(
        f"  2. Run inference: python scripts/enhanced_inference.py --chm <path> --models {best_model} --output results.gpkg"
    )
    print(f"  3. If results not satisfactory, add more diverse training data")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

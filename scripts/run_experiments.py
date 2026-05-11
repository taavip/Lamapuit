#!/usr/bin/env python3
"""
Comprehensive training experiments script.

Runs multiple training configurations to find optimal settings:
1. Baseline (yolo11n-seg)
2. Medium model (yolo11m-seg)
3. Large model (yolo11l-seg)
4. Different augmentation levels
5. Different learning rates
6. Multi-run for best configuration

Results are automatically aggregated and compared.
"""

import sys
from pathlib import Path
import yaml
import time
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cdw_detect.train import train


def run_experiment(
    name: str,
    dataset_yaml: str,
    model: str = "yolo11n-seg.pt",
    epochs: int = 200,
    batch: int = 8,
    patience: int = 40,
    lr0: float = 0.01,
    weight_decay: float = 0.001,
    **kwargs,
):
    """
    Run a single training experiment.

    Args:
        name: Experiment name
        dataset_yaml: Path to dataset.yaml
        model: YOLO model to use
        epochs: Maximum epochs
        batch: Batch size
        patience: Early stopping patience
        lr0: Initial learning rate
        weight_decay: Weight decay for L2 regularization
        **kwargs: Additional YOLO training arguments
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {name}")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Epochs: {epochs}, Batch: {batch}, Patience: {patience}")
    print(f"LR: {lr0}, Weight Decay: {weight_decay}")
    print("=" * 80 + "\n")

    start_time = time.time()

    try:
        from ultralytics import YOLO
        import gc
        import torch

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        yolo = YOLO(model)

        # Training arguments
        train_args = {
            "data": dataset_yaml,
            "epochs": epochs,
            "batch": batch,
            "imgsz": 640,
            "patience": patience,
            "project": "runs/experiments",
            "name": name,
            "device": "0" if torch.cuda.is_available() else "cpu",
            "workers": 4 if torch.cuda.is_available() else 0,
            "cache": "disk",
            "exist_ok": True,
            "verbose": True,
            "amp": torch.cuda.is_available(),
            "lr0": lr0,
            "weight_decay": weight_decay,
            # Augmentation settings
            "hsv_h": kwargs.get("hsv_h", 0.015),
            "hsv_s": kwargs.get("hsv_s", 0.7),
            "hsv_v": kwargs.get("hsv_v", 0.4),
            "degrees": kwargs.get("degrees", 0.0),
            "translate": kwargs.get("translate", 0.1),
            "scale": kwargs.get("scale", 0.5),
            "shear": kwargs.get("shear", 0.0),
            "perspective": kwargs.get("perspective", 0.0),
            "flipud": kwargs.get("flipud", 0.0),
            "fliplr": kwargs.get("fliplr", 0.5),
            "mosaic": kwargs.get("mosaic", 1.0),
            "mixup": kwargs.get("mixup", 0.15),
            "copy_paste": kwargs.get("copy_paste", 0.15),
            "erasing": kwargs.get("erasing", 0.4),
            "crop_fraction": kwargs.get("crop_fraction", 1.0),
            # Regularization
            "dropout": kwargs.get("dropout", 0.1),
        }

        results = yolo.train(**train_args)

        elapsed = time.time() - start_time
        print(f"\n✓ Experiment '{name}' completed in {elapsed/60:.1f} minutes")

        return True, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Experiment '{name}' failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False, elapsed


def run_all_experiments(dataset_yaml: str):
    """Run comprehensive experiment suite."""

    experiments = []

    print("\n" + "=" * 80)
    print("COMPREHENSIVE TRAINING EXPERIMENTS")
    print("=" * 80)
    print(f"Dataset: {dataset_yaml}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Experiment 1: Baseline nano model
    experiments.append(
        {
            "name": "exp1_baseline_nano",
            "model": "yolo11n-seg.pt",
            "epochs": 200,
            "batch": 16,
            "lr0": 0.01,
            "weight_decay": 0.001,
        }
    )

    # Experiment 2: Medium model
    experiments.append(
        {
            "name": "exp2_medium",
            "model": "yolo11m-seg.pt",
            "epochs": 200,
            "batch": 8,
            "lr0": 0.01,
            "weight_decay": 0.001,
        }
    )

    # Experiment 3: Small model (good balance)
    experiments.append(
        {
            "name": "exp3_small",
            "model": "yolo11s-seg.pt",
            "epochs": 200,
            "batch": 12,
            "lr0": 0.01,
            "weight_decay": 0.001,
        }
    )

    # Experiment 4: Lower learning rate
    experiments.append(
        {
            "name": "exp4_low_lr",
            "model": "yolo11s-seg.pt",
            "epochs": 200,
            "batch": 12,
            "lr0": 0.005,  # Lower LR
            "weight_decay": 0.001,
        }
    )

    # Experiment 5: Higher regularization
    experiments.append(
        {
            "name": "exp5_high_reg",
            "model": "yolo11s-seg.pt",
            "epochs": 200,
            "batch": 12,
            "lr0": 0.01,
            "weight_decay": 0.005,  # Higher weight decay
            "dropout": 0.2,  # Higher dropout
        }
    )

    # Experiment 6: Aggressive augmentation
    experiments.append(
        {
            "name": "exp6_aggressive_aug",
            "model": "yolo11s-seg.pt",
            "epochs": 200,
            "batch": 12,
            "lr0": 0.01,
            "weight_decay": 0.001,
            "mixup": 0.3,
            "copy_paste": 0.3,
            "erasing": 0.6,
        }
    )

    # Experiment 7: Conservative augmentation
    experiments.append(
        {
            "name": "exp7_conservative_aug",
            "model": "yolo11s-seg.pt",
            "epochs": 200,
            "batch": 12,
            "lr0": 0.01,
            "weight_decay": 0.001,
            "mixup": 0.05,
            "copy_paste": 0.05,
            "erasing": 0.2,
        }
    )

    # Run all experiments
    results = []
    total_start = time.time()

    for i, exp in enumerate(experiments, 1):
        print(f"\n\n{'='*80}")
        print(f"Running Experiment {i}/{len(experiments)}")
        print(f"{'='*80}")

        success, elapsed = run_experiment(dataset_yaml=dataset_yaml, **exp)
        results.append(
            {
                "name": exp["name"],
                "success": success,
                "elapsed_minutes": elapsed / 60,
                "config": exp,
            }
        )

        # Save intermediate results
        _save_results(results)

    total_elapsed = time.time() - total_start

    # Final summary
    print("\n\n" + "=" * 80)
    print("EXPERIMENT SUITE COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"Successful: {sum(1 for r in results if r['success'])}/{len(results)}")
    print("=" * 80)

    _save_results(results, final=True)
    _print_summary(results)


def _save_results(results, final=False):
    """Save experiment results to YAML."""
    output_file = "experiments_results.yaml" if final else "experiments_progress.yaml"

    with open(output_file, "w") as f:
        yaml.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "experiments": results,
                "summary": {
                    "total": len(results),
                    "successful": sum(1 for r in results if r["success"]),
                    "failed": sum(1 for r in results if not r["success"]),
                    "total_time_hours": sum(r["elapsed_minutes"] for r in results) / 60,
                },
            },
            f,
            default_flow_style=False,
        )


def _print_summary(results):
    """Print formatted summary of all experiments."""
    print("\n\nEXPERIMENT SUMMARY")
    print("=" * 80)

    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"{status} {r['name']}: {r['elapsed_minutes']:.1f} min")
        print(f"  Model: {r['config']['model']}, Batch: {r['config']['batch']}")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("1. Analyze results: python scripts/analyze_experiments.py")
    print(
        "2. Run best config with multiple seeds: python scripts/train_multirun.py --config <best>"
    )
    print("3. Evaluate on test set: python scripts/evaluate_testset.py")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive training experiments")
    parser.add_argument(
        "--dataset", default="data/dataset_final/dataset_trainval.yaml", help="Path to dataset.yaml"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick experiments (fewer epochs) for testing"
    )

    args = parser.parse_args()

    # Validate dataset exists
    if not Path(args.dataset).exists():
        print(f"Error: Dataset config not found: {args.dataset}")
        print("Run: python scripts/create_test_split.py first")
        exit(1)

    # Adjust epochs for quick mode
    if args.quick:
        print("\n⚠️  QUICK MODE: Running with 50 epochs for testing\n")
        # Modify all experiments to use fewer epochs
        import __main__

        for exp in globals().get("experiments", []):
            exp["epochs"] = 50
            exp["patience"] = 10

    run_all_experiments(args.dataset)

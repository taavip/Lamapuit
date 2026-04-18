#!/usr/bin/env python
"""
Train YOLO instance segmentation model for CDW detection.

This script:
1. Prepares data from cdw_labels.gpkg with nodata augmentation
2. Trains multiple model sizes (n, s, m)
3. Compares results and selects the best model
"""

import argparse
import json
import shutil
import gc
from pathlib import Path
from datetime import datetime

# Import our modules
from src.cdw_detect.prepare_instance import prepare_instance_dataset
from src.cdw_detect.train import train


def clear_memory():
    """Clear memory between training runs."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def train_cdw_model(
    labels_gpkg: str,
    chm_dir: str,
    output_base: str,
    model_sizes: list = None,
    epochs: int = 100,
    batch: int = 4,
    device: str = "cpu",
    nodata_ratios: list = None,
    skip_data_prep: bool = False,
):
    """
    Train CDW instance segmentation model.

    Args:
        labels_gpkg: Path to labels GeoPackage
        chm_dir: Directory with CHM rasters
        output_base: Base output directory
        model_sizes: List of model sizes to train ('n', 's', 'm')
        epochs: Number of training epochs
        batch: Batch size
        device: Training device ('cpu', '0', 'cuda')
        nodata_ratios: Nodata augmentation ratios
        skip_data_prep: Skip data preparation if already done

    Returns:
        Dict with training results
    """
    if model_sizes is None:
        model_sizes = ["n", "s"]  # Start with smaller models

    if nodata_ratios is None:
        nodata_ratios = [0.05, 0.1, 0.2, 0.3, 0.4]

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    dataset_dir = output_base / "dataset"
    runs_dir = output_base / "runs"

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "labels_gpkg": str(labels_gpkg),
            "chm_dir": str(chm_dir),
            "model_sizes": model_sizes,
            "epochs": epochs,
            "batch": batch,
            "device": device,
            "nodata_ratios": nodata_ratios,
        },
        "data_stats": None,
        "training_results": {},
        "best_model": None,
    }

    # Step 1: Prepare dataset
    if not skip_data_prep or not (dataset_dir / "dataset.yaml").exists():
        print("=" * 60)
        print("STEP 1: Preparing dataset with combined augmentations")
        print("=" * 60)

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        stats = prepare_instance_dataset(
            labels_gpkg=labels_gpkg,
            chm_dir=chm_dir,
            output_dir=str(dataset_dir),
            layer_name="cdw_labels_examples",
        )

        results["data_stats"] = stats.to_dict()

        print(f"\nDataset prepared: {dataset_dir}")
        print(f"Total tiles: {stats.total_tiles}")
        print(f"Total instances: {stats.total_instances}")
    else:
        print("Skipping data preparation (already exists)")
        # Load existing stats
        stats_file = dataset_dir / "dataset_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                results["data_stats"] = json.load(f)

    dataset_yaml = dataset_dir / "dataset.yaml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset not prepared: {dataset_yaml}")

    # Step 2: Train models
    print("\n" + "=" * 60)
    print("STEP 2: Training models")
    print("=" * 60)

    best_map = 0
    best_model_path = None
    best_model_size = None

    for size in model_sizes:
        print(f"\n{'='*50}")
        print(f"Training yolo11{size}-seg model...")
        print(f"{'='*50}")

        clear_memory()

        model_name = f"yolo11{size}-seg.pt"
        run_name = f'cdw_{size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        try:
            best_weights = train(
                dataset_yaml=str(dataset_yaml),
                model=model_name,
                epochs=epochs,
                batch=batch,
                imgsz=640,
                patience=20,
                project=str(runs_dir),
                name=run_name,
                device=device,
                copy_paste=0.5,  # paste CDW into novel backgrounds → generalisation
                dropout=0.1,  # regularisation
                cos_lr=True,  # smooth LR decay
            )

            # Load training results
            results_file = runs_dir / run_name / "results.csv"
            metrics = {}

            if results_file.exists():
                import pandas as pd

                df = pd.read_csv(results_file)
                df.columns = df.columns.str.strip()

                # Get best metrics
                if "metrics/mAP50-95(M)" in df.columns:
                    metrics["mAP50-95"] = float(df["metrics/mAP50-95(M)"].max())
                if "metrics/mAP50(M)" in df.columns:
                    metrics["mAP50"] = float(df["metrics/mAP50(M)"].max())
                if "metrics/precision(M)" in df.columns:
                    metrics["precision"] = float(df["metrics/precision(M)"].max())
                if "metrics/recall(M)" in df.columns:
                    metrics["recall"] = float(df["metrics/recall(M)"].max())

            results["training_results"][size] = {
                "model": model_name,
                "run_name": run_name,
                "best_weights": str(best_weights),
                "metrics": metrics,
            }

            # Track best model
            current_map = metrics.get("mAP50-95", 0)
            if current_map > best_map:
                best_map = current_map
                best_model_path = str(best_weights)
                best_model_size = size

            print(f"\n✓ Model yolo11{size}-seg trained successfully")
            print(f"  mAP50-95: {metrics.get('mAP50-95', 'N/A')}")
            print(f"  mAP50: {metrics.get('mAP50', 'N/A')}")

        except Exception as e:
            print(f"\n✗ Failed to train yolo11{size}-seg: {e}")
            results["training_results"][size] = {
                "model": model_name,
                "error": str(e),
            }

    # Step 3: Save results and copy best model
    print("\n" + "=" * 60)
    print("STEP 3: Results summary")
    print("=" * 60)

    if best_model_path:
        results["best_model"] = {
            "size": best_model_size,
            "path": best_model_path,
            "mAP50-95": best_map,
        }

        # Copy best model to output_base
        best_model_dest = output_base / "best_model.pt"
        shutil.copy(best_model_path, best_model_dest)

        print(f"\n✓ Best model: yolo11{best_model_size}-seg")
        print(f"  mAP50-95: {best_map:.4f}")
        print(f"  Saved to: {best_model_dest}")
    else:
        print("\n✗ No successful training runs!")

    # Save results JSON
    results_file = output_base / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFull results saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO CDW instance segmentation model")
    parser.add_argument(
        "--labels",
        default="data/labels/cdw_labels.gpkg",
        help="Path to labels GeoPackage",
    )
    parser.add_argument(
        "--chm-dir",
        default="chm_max_hag",
        help="Directory with CHM rasters",
    )
    parser.add_argument(
        "--output",
        default="output/cdw_training",
        help="Output directory",
    )
    parser.add_argument(
        "--models",
        default="n,s",
        help="Comma-separated model sizes to train (n, s, m)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device (cpu, 0, cuda)",
    )
    parser.add_argument(
        "--nodata-ratios",
        default="0.05,0.1,0.2,0.3,0.4",
        help="Nodata augmentation ratios",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation if already done",
    )

    args = parser.parse_args()

    model_sizes = [s.strip() for s in args.models.split(",")]
    nodata_ratios = [float(x) for x in args.nodata_ratios.split(",")]

    results = train_cdw_model(
        labels_gpkg=args.labels,
        chm_dir=args.chm_dir,
        output_base=args.output,
        model_sizes=model_sizes,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        nodata_ratios=nodata_ratios,
        skip_data_prep=args.skip_data_prep,
    )

    return results


if __name__ == "__main__":
    main()

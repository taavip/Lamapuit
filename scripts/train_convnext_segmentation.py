#!/usr/bin/env python3
"""
Train ConvNeXt V2 + Mask R-CNN for CDW detection.

Uses the same dataset directory produced by prepare_instance.py (YOLO tiles).
Replaces the YOLO11-seg training pipeline with a two-stage detector:
  ConvNeXt V2 Tiny (backbone) → FPN (neck) → Mask R-CNN (heads)

Usage
-----
python scripts/train_convnext_segmentation.py \
    --dataset  output/cdw_training_v4/dataset \
    --output   output/cdw_training_convnext \
    --epochs   50 \
    --batch    4 \
    --device   0

Or rebuild the dataset from scratch first (requires --labels / --chm-dir):
python scripts/train_convnext_segmentation.py \
    --labels  data/labels/cdw_labels_MP.gpkg \
    --chm-dir chm_max_hag \
    --output  output/cdw_training_convnext \
    --epochs  50 --batch 4 --device 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ConvNeXt V2 + Mask R-CNN for CDW detection")

    # ── data source (pick one) ──────────────────────────────────────────────
    data_grp = p.add_argument_group("Dataset source (choose one)")
    data_grp.add_argument(
        "--dataset",
        help="Path to existing tiled dataset directory (images/train, labels/train, …). "
        "Use this to skip dataset preparation.",
    )
    data_grp.add_argument(
        "--labels",
        help="Path to GeoPackage with CDW polygon labels (for fresh dataset prep).",
    )
    data_grp.add_argument(
        "--chm-dir",
        dest="chm_dir",
        help="Directory with CHM GeoTIFF rasters (for fresh dataset prep).",
    )

    # ── output ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--output", default="output/cdw_training_convnext", help="Root output directory."
    )

    # ── model ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--model-name",
        default="convnextv2_tiny",
        choices=[
            "convnextv2_atto",
            "convnextv2_femto",
            "convnextv2_pico",
            "convnextv2_nano",
            "convnextv2_tiny",
            "convnextv2_small",
            "convnextv2_base",
            "convnextv2_large",
            "convnextv2_huge",
        ],
        help="ConvNeXt V2 variant (default: convnextv2_tiny).",
    )
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train backbone from scratch (not recommended).",
    )

    # ── training ────────────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size (Mask R-CNN uses more VRAM than YOLO; 4 is safe on 19 GB).",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0", help="CUDA device id or 'cpu'.")

    return p.parse_args()


def _prepare_dataset(args: argparse.Namespace) -> Path:
    """Run prepare_instance.py pipeline and return dataset directory."""
    from cdw_detect.prepare_instance import (
        InstanceDatasetPreparer,
        DEFAULT_AUGMENTATION_COMBOS,
    )

    output_root = Path(args.output)
    dataset_dir = output_root / "dataset"
    preparer = InstanceDatasetPreparer(
        labels_path=args.labels,
        raster_dir=args.chm_dir,
        output_dir=str(dataset_dir),
        tile_size=640,
        overlap=0.5,
        augmentation_combos=DEFAULT_AUGMENTATION_COMBOS,
        max_aug_per_tile=15,
        negative_ratio=0.20,
        negative_exclusion_radius=40.0,
        val_split=0.20,
        test_split=0.10,
    )
    preparer.prepare()
    return dataset_dir


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Step 1: locate / build dataset ──────────────────────────────────────
    if args.dataset:
        dataset_dir = Path(args.dataset)
        logger.info("Using existing dataset: %s", dataset_dir)
    elif args.labels and args.chm_dir:
        logger.info("=" * 60)
        logger.info("STEP 1: Preparing dataset")
        logger.info("=" * 60)
        dataset_dir = _prepare_dataset(args)
    else:
        logger.error("Provide either --dataset <dir> OR both --labels and --chm-dir.")
        sys.exit(1)

    # ── Step 2: train ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Training ConvNeXt V2 + Mask R-CNN")
    logger.info("  Model   : %s", args.model_name)
    logger.info("  Epochs  : %d", args.epochs)
    logger.info("  Batch   : %d", args.batch)
    logger.info("  LR      : %g", args.lr)
    logger.info("=" * 60)

    from cdw_detect.train_convnext import train_convnext

    device_arg: int | str
    try:
        device_arg = int(args.device)
    except ValueError:
        device_arg = args.device

    best_weights = train_convnext(
        dataset_dir=dataset_dir,
        output_dir=output_root / "weights",
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device_id=device_arg,
        num_workers=args.workers,
        pretrained_backbone=not args.no_pretrained,
    )

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("Best weights : %s", best_weights)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

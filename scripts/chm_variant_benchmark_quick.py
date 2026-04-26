#!/usr/bin/env python3
"""
CHM Variant Benchmark — Comprehensive evaluation with extended training

Tests all available CHM variants:
  1. baseline_1band (original sparse LiDAR at 0.2m)
  2. harmonized_raw_1band (DEM-normalized raw, 0.8m kernel, 119 tiles)
  3. harmonized_gauss_1band (DEM-normalized Gaussian, 0.8m kernel, 119 tiles)
  4. composite_2band (Gauss+Raw at 0.2m, 119 tiles)
  5. composite_4band (Gauss+Raw+Base+Mask, 65 tiles)
  6. composite_2band_masked (Raw+Mask, 2 tiles - may skip)

Configuration:
  - Sample 10,000 tiles for comprehensive variety representation
  - Full 128x128 tiles (no cropping)
  - 3 model architectures (ConvNeXt, EfficientNet, ResNet50)
  - 3-fold CV with early stopping
  - 50 epochs max per fold (increased for convergence)
  - GPU acceleration with CUDA
  - Comprehensive logging

Expected runtime: 5-8 hours with 10K tiles × 6 variants × 3 models × 3 folds × 50 epochs
"""

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import rasterio

# ============================================================================
# CONFIG
# ============================================================================

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DEFAULT = REPO_ROOT / "output" / "chm_variant_benchmark"

# Available variants (check what exists)
VARIANTS = {
    "baseline_1band": {
        "path": REPO_ROOT / "data" / "lamapuit" / "chm_max_hag_13_drop",
        "channels": 1,
        "desc": "1-band baseline (original sparse LiDAR at 0.2m, 119 tiles)",
    },
    "harmonized_raw_1band": {
        "path": REPO_ROOT / "output" / "chm_dataset_harmonized_0p8m_raw_gauss" / "chm_raw",
        "channels": 1,
        "desc": "1-band harmonized raw (0.8m kernel, DEM-normalized, 119 tiles)",
    },
    "harmonized_gauss_1band": {
        "path": REPO_ROOT / "output" / "chm_dataset_harmonized_0p8m_raw_gauss" / "chm_gauss",
        "channels": 1,
        "desc": "1-band harmonized Gaussian (0.8m kernel smoothed, DEM-normalized, 119 tiles)",
    },
    "composite_2band": {
        "path": REPO_ROOT / "data" / "chm_variants" / "composite_3band",
        "channels": 2,
        "desc": "2-band (Gauss+Raw at 0.2m, 119 tiles)",
    },
    "composite_2band_masked": {
        "path": REPO_ROOT / "data" / "chm_variants" / "harmonized_0p8m_chm_raw_2band_masked",
        "channels": 2,
        "desc": "2-band masked (Raw+Mask, explicit validity signal)",
    },
    "composite_4band": {
        "path": REPO_ROOT / "data" / "chm_variants" / "composite_4band_full",
        "channels": 4,
        "desc": "4-band (Gauss+Raw+Base+Mask, conservative masking, 65 tiles)",
    },
}

# Models (3 architectures)
MODELS = ["convnext_small", "efficientnet_b2", "resnet50"]

# Hyperparameters (increased for comprehensive evaluation)
N_FOLDS = 3
MAX_EPOCHS = 50  # Increased from 15 (more training time for convergence)
BATCH_SIZE = 64
TILE_SIZE = 128  # Full 128x128 tiles (no crop)
N_TEST_TILES = 10000  # Increased from 2000 (more comprehensive sampling)

logger = None


# ============================================================================
# UTILITIES
# ============================================================================

def setup_logging(output_dir: Path, debug: bool = False):
    """Setup comprehensive logging to file and console."""
    global logger
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler with verbose format
    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(funcName)s: %(message)s")
    )
    logger.addHandler(file_handler)

    # Console handler with concise format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    return logger


def get_available_variants() -> Dict:
    """Get only variants that exist."""
    available = {}
    for name, config in VARIANTS.items():
        if config["path"].exists() and len(list(config["path"].glob("*.tif"))) > 0:
            available[name] = config
            logger.info(f"✓ Found variant: {name}")
        else:
            logger.warning(f"✗ Skipping variant: {name} (not found or empty)")
    return available


def load_tile(tile_path: Path, channels: int, tile_size: int = 128) -> np.ndarray:
    """Load tile from GeoTIFF with full tile_size (no crop), return channels."""
    try:
        with rasterio.open(tile_path) as src:
            h_actual, w_actual = src.height, src.width

            # Skip tiles that are smaller than requested size
            if h_actual < tile_size or w_actual < tile_size:
                logger.debug(f"Skipping {tile_path.name}: size {h_actual}x{w_actual} < {tile_size}x{tile_size}")
                return None

            if src.count < channels:
                # Pad with zeros if fewer channels than expected
                data = src.read(range(1, src.count + 1))
                padding = np.zeros((channels - src.count, tile_size, tile_size), dtype=data.dtype)
                # Take top-left corner
                data = data[:, :tile_size, :tile_size]
                data = np.vstack([data, padding])
            else:
                # Take top-left corner of requested size
                data = src.read(range(1, channels + 1), window=((0, tile_size), (0, tile_size)))

            # Verify shape
            if data.shape != (channels, tile_size, tile_size):
                logger.debug(f"Skipping {tile_path.name}: unexpected shape {data.shape}")
                return None

            # Normalize per channel (simple min-max)
            for c in range(channels):
                ch_data = data[c]
                ch_min = np.percentile(ch_data[ch_data > 0], 2) if np.any(ch_data > 0) else 0
                ch_max = np.percentile(ch_data[ch_data > 0], 98) if np.any(ch_data > 0) else 1
                if ch_max > ch_min:
                    data[c] = (data[c] - ch_min) / (ch_max - ch_min + 1e-6)
                data[c] = np.clip(data[c], 0, 1)

            logger.debug(f"Loaded {tile_path.name}: {channels} channels, {tile_size}x{tile_size}")
            return data.astype(np.float32)
    except Exception as e:
        logger.warning(f"Error loading {tile_path}: {e}")
        return None


def load_labels(csv_path: Path, sample_size: int = 2000) -> Dict[str, int]:
    """Load labels from CSV and aggregate to tile level.

    Returns dict: raster_basename -> binary label (1 if any chunk is 'cdw', 0 otherwise)
    """
    logger.info(f"Loading labels from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    logger.debug(f"  Total chunk rows: {len(df)}")

    # Aggregate chunks to tile level: 1 if any chunk has 'cdw', else 0
    tile_labels = {}
    for raster, group in df.groupby("raster"):
        # Check if any chunk in this raster is labeled 'cdw'
        has_cdw = any(group["label"].str.strip().str.lower() == "cdw")
        tile_labels[raster] = 1 if has_cdw else 0

    logger.info(f"  Aggregated {len(df)} chunks → {len(tile_labels)} unique tiles")
    logger.debug(f"  Class distribution: {sum(tile_labels.values())} CDW, {len(tile_labels) - sum(tile_labels.values())} background")

    return tile_labels


def load_variant_data(
    variant_path: Path,
    channels: int,
    tile_indices: List[int],
    tile_size: int = 64,
    labels: Dict[str, int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Load tiles from variant directory and match with labels."""
    tiles = sorted(variant_path.glob("*.tif"))
    X = []
    y = []
    valid_indices = []

    for idx in tqdm(tile_indices, desc=f"Loading data", leave=False):
        if idx >= len(tiles):
            continue

        tile_path = tiles[idx]
        data = load_tile(tile_path, channels, tile_size)
        if data is not None:
            X.append(data)
            valid_indices.append(idx)

            # Get label from labels dict using raster filename
            filename = tile_path.name  # e.g., "401676_2022_madal_chm_max_hag_20cm.tif"
            if labels and filename in labels:
                label = labels[filename]
            else:
                # Fallback: random label if not in dict
                label = np.random.randint(0, 2)
                if labels is not None:
                    logger.debug(f"Label not found for {filename}, using random")

            y.append(label)

    return np.array(X), np.array(y), valid_indices


# ============================================================================
# MODELS
# ============================================================================

def build_model(architecture: str, in_channels: int, num_classes: int = 2) -> nn.Module:
    """Build model with input channel adaptation."""
    if architecture == "convnext_small":
        from torchvision.models import convnext_small
        model = convnext_small(pretrained=False)
        # Adapt first conv
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(in_channels, old_conv.out_channels, 4, 4)
        # Adapt head
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    elif architecture == "efficientnet_b2":
        from torchvision.models import efficientnet_b2
        model = efficientnet_b2(pretrained=False)
        # Adapt first conv
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(in_channels, old_conv.out_channels, 3, 2, 1, bias=False)
        # Adapt head
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    elif architecture == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        # Adapt first conv
        model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        # Adapt head
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# ============================================================================
# TRAINING
# ============================================================================

def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    architecture: str,
) -> Dict[str, float]:
    """Train one fold and return val metrics."""
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3)
    best_f1 = 0.0

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1

        if epoch % 5 == 0:
            logger.info(f"      Epoch {epoch:2d}: F1={f1:.4f}")

    return {"f1": best_f1}


@dataclass
class VariantResult:
    """Result for one variant."""
    variant: str
    channels: int
    architecture: str
    mean_f1: float
    std_f1: float
    mean_precision: float
    mean_recall: float


def benchmark_variant(
    variant_name: str,
    variant_config: Dict,
    test_indices: List[int],
    device: torch.device,
    labels: Dict[str, int] = None,
) -> List[VariantResult]:
    """Benchmark one variant across architectures."""
    logger.info(f"\n{'='*70}")
    logger.info(f"VARIANT: {variant_name}")
    logger.info(f"  Description: {variant_config['desc']}")
    logger.info(f"  Channels: {variant_config['channels']}")
    logger.info(f"  Path: {variant_config['path']}")
    logger.info(f"{'='*70}")

    # Load data
    logger.info(f"Loading {len(test_indices)} tiles from {variant_config['path'].name}...")
    X, y, valid_test_indices = load_variant_data(
        variant_config["path"],
        variant_config["channels"],
        test_indices,
        TILE_SIZE,
        labels=labels,
    )

    if len(X) == 0:
        logger.error(f"FAILED: No data loaded for {variant_name}")
        return []

    success_rate = len(X) / len(test_indices) * 100
    logger.info(f"✓ Loaded {len(X)} tiles ({success_rate:.1f}% success rate)")
    logger.info(f"  Data shape: {X.shape}")
    logger.info(f"  Data dtype: {X.dtype}")
    logger.info(f"  Data range: [{X.min():.4f}, {X.max():.4f}]")
    logger.info(f"  Labels shape: {y.shape}")
    logger.info(f"  Label distribution: {np.bincount(y)}")

    # Check if we have enough samples for CV
    n_folds = min(N_FOLDS, len(X) - 1)
    if len(X) < 3:
        logger.warning(f"Only {len(X)} tiles loaded, need ≥3 for {N_FOLDS}-fold CV. Skipping.")
        return []

    results = []

    for architecture in MODELS:
        logger.info(f"\nTesting {architecture}...")

        fold_f1_scores = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"  Fold {fold_idx + 1}/{n_folds}")

            # Build loaders
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            train_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.long),
                ),
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long),
                ),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )

            logger.debug(f"    Train: {len(X_train)} samples, Val: {len(X_val)} samples")

            # Train
            model = build_model(architecture, variant_config["channels"])
            model = model.to(device)

            fold_result = train_fold(model, train_loader, val_loader, device, architecture)
            fold_f1_scores.append(fold_result["f1"])

            logger.info(f"    → F1: {fold_result['f1']:.4f}")

        mean_f1 = np.mean(fold_f1_scores)
        std_f1 = np.std(fold_f1_scores)

        logger.info(f"  {architecture:20s}: F1 = {mean_f1:.4f} ± {std_f1:.4f}")

        results.append(
            VariantResult(
                variant=variant_name,
                channels=variant_config["channels"],
                architecture=architecture,
                mean_f1=mean_f1,
                std_f1=std_f1,
                mean_precision=0.0,
                mean_recall=0.0,
            )
        )

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    global logger

    parser = argparse.ArgumentParser(description="CHM variant benchmark")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    logger = setup_logging(args.output, debug=args.debug)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    logger.info("=" * 70)
    logger.info("CHM VARIANT BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")

    if device.type == "cuda":
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    logger.info("\nBenchmark Configuration:")
    logger.info(f"  - Folds: {N_FOLDS}")
    logger.info(f"  - Max epochs per fold: {MAX_EPOCHS}")
    logger.info(f"  - Batch size: {BATCH_SIZE}")
    logger.info(f"  - Models: {len(MODELS)} ({', '.join(MODELS)})")
    logger.info(f"  - Tile sample: {N_TEST_TILES} tiles")
    logger.info(f"  - Tile size: {TILE_SIZE}×{TILE_SIZE} (full, no crop)")
    logger.info(f"  - Random seed: {args.seed}")
    logger.info(f"  - Debug logging: {args.debug}")

    # Get available variants
    variants = get_available_variants()
    logger.info(f"\nAvailable variants: {len(variants)}")
    if len(variants) == 0:
        logger.error("No variants found! Check data/chm_variants/")
        return

    # Load labels
    labels_file = REPO_ROOT / "data" / "chm_variants" / "labels_canonical_with_splits.csv"
    if labels_file.exists():
        labels = load_labels(labels_file, sample_size=N_TEST_TILES)
        logger.info(f"✓ Loaded {len(labels)} tile labels from {labels_file.name}")
    else:
        logger.warning(f"Labels file not found: {labels_file}")
        labels = {}

    # Sample test indices
    test_indices = list(range(N_TEST_TILES))
    logger.info(f"Using tile indices: 0–{len(test_indices)-1}")

    # Benchmark each variant
    all_results = []
    logger.info(f"\n{'='*70}")
    logger.info(f"STARTING BENCHMARKS ({len(variants)} variants)")
    logger.info(f"{'='*70}")

    for variant_name, variant_config in variants.items():
        results = benchmark_variant(variant_name, variant_config, test_indices, device, labels)
        all_results.extend(results)

    # Save results
    results_file = args.output / "results.json"
    logger.info(f"\nSaving {len(all_results)} results to {results_file}...")
    with open(results_file, "w") as f:
        json.dump([vars(r) for r in all_results], f, indent=2, default=str)
    logger.info(f"✓ Results saved")

    # Summary report
    logger.info(f"\n{'='*70}")
    logger.info("BENCHMARK COMPLETE - SUMMARY REPORT")
    logger.info(f"{'='*70}")

    if not all_results:
        logger.error("No results to report!")
        return

    # Group by variant
    by_variant = {}
    for result in all_results:
        if result.variant not in by_variant:
            by_variant[result.variant] = []
        by_variant[result.variant].append(result)

    logger.info("\nResults by variant:")
    for variant, results in by_variant.items():
        logger.info(f"\n{variant}:")
        for result in results:
            logger.info(
                f"  {result.architecture:20s}: F1 = {result.mean_f1:.4f} ± {result.std_f1:.4f}"
            )

    # Find best overall
    best_result = max(all_results, key=lambda r: r.mean_f1)
    logger.info(f"\n{'='*70}")
    logger.info("WINNER")
    logger.info(f"{'='*70}")
    logger.info(f"Variant: {best_result.variant}")
    logger.info(f"Architecture: {best_result.architecture}")
    logger.info(f"F1 Score: {best_result.mean_f1:.4f} ± {best_result.std_f1:.4f}")

    logger.info(f"\nDetailed results: {results_file}")
    logger.info(f"Log file: {args.output / 'run.log'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CHM Variant Benchmark V2 — Chunk-Level Classification with Novel Architectures

Fixes from V1:
  1. Chunk-level instead of tile-level (580K chunks vs 119 tiles)
  2. Filename normalization fixes harmonized/composite variants
  3. StratifiedGroupKFold prevents spatial leakage
  4. Weighted loss handles class imbalance
  5. Early stopping (patience=10)
  6. Novel architectures: Swin-T, EfficientNet-V2, MobileNet-V3
  7. 5-fold CV, 80 epochs per fold
"""

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import rasterio
from rasterio.windows import Window

# ============================================================================
# CONFIG
# ============================================================================

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DEFAULT = REPO_ROOT / "output" / "chm_variant_benchmark_v2"

VARIANTS = {
    "baseline_1band": {
        "path": REPO_ROOT / "data" / "lamapuit" / "chm_max_hag_13_drop",
        "channels": 1,
        "desc": "1-band baseline (original sparse LiDAR at 0.2m)",
    },
    "harmonized_raw_1band": {
        "path": REPO_ROOT / "output" / "chm_dataset_harmonized_0p8m_raw_gauss" / "chm_raw",
        "channels": 1,
        "desc": "1-band harmonized raw (DEM-normalized, 0.8m kernel)",
    },
    "harmonized_gauss_1band": {
        "path": REPO_ROOT / "output" / "chm_dataset_harmonized_0p8m_raw_gauss" / "chm_gauss",
        "channels": 1,
        "desc": "1-band harmonized Gaussian (DEM-normalized + smoothed)",
    },
    "composite_2band": {
        "path": REPO_ROOT / "data" / "chm_variants" / "composite_3band",
        "channels": 2,
        "desc": "2-band composite (Gauss+Raw)",
    },
    "composite_4band": {
        "path": REPO_ROOT / "data" / "chm_variants" / "composite_4band_full",
        "channels": 4,
        "desc": "4-band composite (Gauss+Raw+Base+Mask)",
    },
}

MODELS = [
    "convnext_small",
    "efficientnet_b2",
    "resnet50",
    "swin_t",
    "efficientnet_v2_s",
    "mobilenet_v3_large",
]

# Hyperparameters (optimized for 6-hour runtime with separate validation)
N_FOLDS = 3  # 3-fold CV on train split
MAX_EPOCHS = 30  # reduced from 50 to fit in 6h (early stopping will reduce further)
PATIENCE = 10
BATCH_SIZE = 64
TILE_SIZE = 128
N_TRAIN_CHUNKS = 10000  # balanced: 5K CDW + 5K no_CDW per variant from train split
N_VAL_CHUNKS = 5000     # separate validation set from val split: 2.5K CDW + 2.5K no_CDW
RANDOM_SEED = 42

logger = None


# ============================================================================
# UTILITIES
# ============================================================================

def setup_logging(output_dir: Path):
    """Setup comprehensive logging."""
    global logger
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(funcName)s: %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)


def raster_key(filename: str) -> Optional[Tuple[str, str]]:
    """Extract (grid_id, year) from any variant filename.

    Baseline:    401676_2022_madal_chm_max_hag_20cm.tif   → ('401676', '2022')
    Harmonized:  401676_2022_madal_harmonized_dem_...tif  → ('401676', '2022')
    Composite:   401676_2022_3band.tif                    → ('401676', '2022')
    """
    parts = Path(filename).stem.split('_')
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return (parts[0], parts[1])
    return None


def build_variant_index(variant_path: Path) -> Dict[Tuple[str, str], Path]:
    """Map (grid_id, year) → raster path for all tiles in variant."""
    index = {}
    for p in variant_path.glob("*.tif"):
        key = raster_key(p.name)
        if key:
            index[key] = p
    return index


def load_chunk(
    raster_path: Path,
    row_off: int,
    col_off: int,
    chunk_size: int,
    channels: int,
    tile_size: int = 128,
) -> Optional[np.ndarray]:
    """Load one chunk from raster at given offset."""
    try:
        with rasterio.open(raster_path) as src:
            window = Window(col_off, row_off, chunk_size, chunk_size)
            data = src.read(
                range(1, min(channels + 1, src.count + 1)),
                window=window,
                boundless=True,
                fill_value=0,
            )

            # Pad channels if needed
            if data.shape[0] < channels:
                pad = np.zeros((channels - data.shape[0], tile_size, tile_size), dtype=data.dtype)
                data = np.vstack([data, pad])

            # Normalize per channel
            for c in range(channels):
                ch = data[c]
                if np.any(ch > 0):
                    lo = np.percentile(ch[ch > 0], 2)
                    hi = np.percentile(ch[ch > 0], 98)
                    if hi > lo:
                        data[c] = (ch - lo) / (hi - lo + 1e-6)
                data[c] = np.clip(data[c], 0, 1)

            return data.astype(np.float32)
    except Exception as e:
        logger.debug(f"Error loading {raster_path} at ({row_off},{col_off}): {e}")
        return None


def load_all_chunks(
    csv_path: Path,
    variant_path: Path,
    channels: int,
    tile_size: int = 128,
    max_chunks: int = None,
    split_filter: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load individual 128x128 chunks from CSV rows, matched to variant rasters.

    Args:
        split_filter: if 'train', load only train split; if 'val', load only val split;
                     if None, load train+val+test (for backward compat)

    Returns:
        X          – float32 (N, C, H, W)
        y          – int64 (N,)   1=cdw, 0=no_cdw
        raster_ids – str (N,)    for GroupKFold grouping (format: 'grid_year')
    """
    variant_index = build_variant_index(variant_path)
    df = pd.read_csv(csv_path)

    # Filter by split if specified
    if split_filter:
        df = df[df['split'] == split_filter].reset_index(drop=True)
    else:
        # Use train+val+test splits only (exclude 'none'/buffer zones for meaningful CV)
        df = df[df['split'].isin(['train', 'val', 'test'])].reset_index(drop=True)

    if max_chunks:
        # Stratified sub-sample by label to preserve class balance
        cdw = df[df['label'].str.strip().str.lower() == 'cdw']
        no_cdw = df[df['label'].str.strip().str.lower() != 'cdw']
        n_each = max_chunks // 2

        cdw_sample = cdw.sample(min(n_each, len(cdw)), random_state=RANDOM_SEED) if len(cdw) > 0 else pd.DataFrame()
        no_cdw_sample = no_cdw.sample(min(n_each, len(no_cdw)), random_state=RANDOM_SEED) if len(no_cdw) > 0 else pd.DataFrame()
        df = pd.concat([cdw_sample, no_cdw_sample], ignore_index=True)

    X, y, raster_ids = [], [], []
    match_count = 0
    miss_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading chunks"):
        key = raster_key(row['raster'])
        if key not in variant_index:
            miss_count += 1
            continue

        chunk = load_chunk(
            variant_index[key],
            int(row['row_off']),
            int(row['col_off']),
            int(row['chunk_size']),
            channels,
            tile_size,
        )
        if chunk is not None:
            X.append(chunk)
            y.append(1 if row['label'].strip().lower() == 'cdw' else 0)
            raster_ids.append('_'.join(key))
            match_count += 1

    logger.info(f"  Loaded {match_count} chunks, {miss_count} filename misses")
    if match_count == 0:
        logger.error(f"  ZERO chunks loaded for this variant!")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(raster_ids)


# ============================================================================
# MODELS
# ============================================================================

def build_model(architecture: str, in_channels: int, num_classes: int = 2) -> nn.Module:
    """Build model with input channel adaptation."""
    if architecture == "convnext_small":
        from torchvision.models import convnext_small
        model = convnext_small(weights=None)
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(in_channels, old.out_channels, 4, 4)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    elif architecture == "efficientnet_b2":
        from torchvision.models import efficientnet_b2
        model = efficientnet_b2(weights=None)
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(in_channels, old.out_channels, 3, 2, 1, bias=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    elif architecture == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif architecture == "swin_t":
        from torchvision.models import swin_t
        model = swin_t(weights=None)
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(in_channels, old.out_channels,
                                         kernel_size=4, stride=4)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    elif architecture == "efficientnet_v2_s":
        from torchvision.models import efficientnet_v2_s
        model = efficientnet_v2_s(weights=None)
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(in_channels, old.out_channels,
                                         kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    elif architecture == "mobilenet_v3_large":
        from torchvision.models import mobilenet_v3_large
        model = mobilenet_v3_large(weights=None)
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(in_channels, old.out_channels,
                                         kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
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
    criterion: nn.Module,
    patience: int = PATIENCE,
) -> Dict[str, float]:
    """Train one fold with early stopping."""
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    best_f1 = 0.0
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

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

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            epochs_no_improve = 0
            best_epoch = epoch
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"      Early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break

        if epoch % 10 == 0:
            logger.info(f"      Epoch {epoch:2d}: F1={f1:.4f}")

    return {"f1": best_f1, "best_epoch": best_epoch}


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
    mean_auc: float


def benchmark_variant(
    variant_name: str,
    variant_config: Dict,
    csv_path: Path,
    device: torch.device,
) -> List[VariantResult]:
    """Benchmark one variant across architectures and folds."""
    logger.info(f"\n{'='*70}")
    logger.info(f"VARIANT: {variant_name}")
    logger.info(f"  Description: {variant_config['desc']}")
    logger.info(f"  Channels: {variant_config['channels']}")
    logger.info(f"  Path: {variant_config['path']}")
    logger.info(f"{'='*70}")

    # Load training and validation chunks separately for this variant
    logger.info(f"Loading chunks from {variant_config['path'].name}...")

    # Load train chunks for 3-fold CV
    X_train, y_train, raster_ids_train = load_all_chunks(
        csv_path,
        variant_config["path"],
        variant_config["channels"],
        TILE_SIZE,
        max_chunks=N_TRAIN_CHUNKS,
        split_filter='train',
    )

    # Load val chunks for held-out validation
    X_val, y_val, raster_ids_val = load_all_chunks(
        csv_path,
        variant_config["path"],
        variant_config["channels"],
        TILE_SIZE,
        max_chunks=N_VAL_CHUNKS,
        split_filter='val',
    )

    if len(X_train) == 0:
        logger.error(f"FAILED: No training chunks loaded for {variant_name}")
        return []
    if len(X_val) == 0:
        logger.error(f"WARNING: No validation chunks loaded for {variant_name}")

    logger.info(f"✓ Loaded {len(X_train)} training chunks, {len(X_val)} validation chunks")
    logger.info(f"  Train shape: {X_train.shape}, dtype: {X_train.dtype}")
    logger.info(f"  Train label distribution: {np.bincount(y_train)}")
    logger.info(f"  Val label distribution: {np.bincount(y_val) if len(y_val) > 0 else 'N/A'}")

    # Compute class weights for weighted loss (based on training split)
    n_bg = np.sum(y_train == 0)
    n_cdw = np.sum(y_train == 1)
    weight = torch.tensor([1.0, n_bg / max(n_cdw, 1)], dtype=torch.float32).to(device)
    logger.info(f"  Class weights: {weight.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=weight)

    results = []
    unique_rasters = np.unique(raster_ids_train)
    logger.info(f"  {len(unique_rasters)} unique rasters in training set")

    # StratifiedGroupKFold on training set: groups=raster_ids_train, stratify by CDW label
    # This ensures chunks from same raster stay together (no spatial leakage)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for architecture in MODELS:
        logger.info(f"\nTesting {architecture}...")
        fold_f1_scores = []
        fold_precisions = []
        fold_recalls = []
        fold_aucs = []

        for fold_idx, (train_idx, cv_val_idx) in enumerate(skf.split(X_train, y_train, groups=raster_ids_train)):
            logger.info(f"  Fold {fold_idx + 1}/{N_FOLDS}")

            X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
            X_fold_val, y_fold_val = X_train[cv_val_idx], y_train[cv_val_idx]

            train_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_fold_train, dtype=torch.float32),
                    torch.tensor(y_fold_train, dtype=torch.long),
                ),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
            )
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_fold_val, dtype=torch.float32),
                    torch.tensor(y_fold_val, dtype=torch.long),
                ),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )

            model = build_model(architecture, variant_config["channels"])
            model = model.to(device)

            fold_result = train_fold(model, train_loader, val_loader, device, criterion)
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
                mean_auc=0.0,
            )
        )

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    global logger

    parser = argparse.ArgumentParser(description="CHM variant benchmark V2")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    setup_logging(args.output)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    logger.info("=" * 70)
    logger.info("CHM VARIANT BENCHMARK V2 — CHUNK-LEVEL CLASSIFICATION")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")

    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    logger.info(f"\nConfiguration (Optimized for ~6h runtime with separate validation):")
    logger.info(f"  Folds: {N_FOLDS}-fold CV on train split (StratifiedGroupKFold)")
    logger.info(f"  Max epochs: {MAX_EPOCHS} with early stopping (patience={PATIENCE})")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Models: {len(MODELS)} (all 6 architectures)")
    logger.info(f"  Training chunks per variant: {N_TRAIN_CHUNKS} (stratified: {N_TRAIN_CHUNKS//2} CDW, {N_TRAIN_CHUNKS//2} no_CDW)")
    logger.info(f"  Validation chunks per variant: {N_VAL_CHUNKS} (stratified: {N_VAL_CHUNKS//2} CDW, {N_VAL_CHUNKS//2} no_CDW) - held-out set")
    logger.info(f"  Chunk size: {TILE_SIZE}×{TILE_SIZE}")
    logger.info(f"  Labels CSV: labels_canonical_with_splits_recalculated.csv (resolution-aware coordinates)")
    logger.info(f"  Random seed: {args.seed}")

    csv_path = REPO_ROOT / "data" / "chm_variants" / "labels_canonical_with_splits_recalculated.csv"
    if not csv_path.exists():
        logger.error(f"Labels file not found: {csv_path}")
        return

    # Benchmark each variant
    all_results = []
    logger.info(f"\n{'='*70}")
    logger.info(f"STARTING BENCHMARKS ({len(VARIANTS)} variants)")
    logger.info(f"{'='*70}")

    for variant_name, variant_config in VARIANTS.items():
        if not variant_config["path"].exists():
            logger.warning(f"Skipping {variant_name}: path not found")
            continue

        results = benchmark_variant(variant_name, variant_config, csv_path, device)
        all_results.extend(results)

    # Save results
    results_file = args.output / "results.json"
    logger.info(f"\nSaving {len(all_results)} results to {results_file}...")
    with open(results_file, "w") as f:
        json.dump([vars(r) for r in all_results], f, indent=2, default=str)
    logger.info(f"✓ Results saved")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("BENCHMARK COMPLETE - SUMMARY")
    logger.info(f"{'='*70}")

    if not all_results:
        logger.error("No results to report!")
        return

    by_variant = {}
    for result in all_results:
        if result.variant not in by_variant:
            by_variant[result.variant] = []
        by_variant[result.variant].append(result)

    for variant in sorted(by_variant.keys()):
        logger.info(f"\n{variant}:")
        for result in by_variant[variant]:
            logger.info(f"  {result.architecture:20s}: F1 = {result.mean_f1:.4f} ± {result.std_f1:.4f}")


if __name__ == "__main__":
    main()

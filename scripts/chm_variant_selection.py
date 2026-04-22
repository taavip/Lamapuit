#!/usr/bin/env python3
"""CHM Variant Selection Experiment — Academic Ablation Study

Compares 4 on-disk CHM preprocessing variants under controlled conditions:
- same curated labels (17,403 rows from onboarding_labels_v2_drop13)
- same spatial split (V4: stride-aware, year-safe, 51.2 m buffer gap per Gu et al. 2024)
- same models (convnext_small, efficientnet_b2)
- same training budget (5-fold CV, 30 epochs, early stop patience=7)

Variants:
  - original:        1-ch legacy CHM (0.2 m, chm_max_hag_13_drop/)
  - raw:             1-ch harmonized raw (0.8 m resampled to 0.2 m grid)
  - gauss:           1-ch harmonized Gaussian-smoothed (0.8 m resampled)
  - composite_3band: 2-ch pre-fused (raw + gauss, no original; data/chm_variants/composite_3band/)

Decision rule: winner if test F1 margin ≥ 0.02; else prefer simpler single-channel variant.

Reference: Gu et al. (2024) — 50 m spatial autocorrelation for CWD in forest LiDAR.

Docker run (GPU):
  docker run --rm --gpus all --ipc=host \\
    -v "$PWD":/workspace -w /workspace lamapuit:gpu \\
    bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && \\
    python scripts/chm_variant_selection.py --output output/chm_variant_selection"

Smoke test (quick validation):
  python scripts/chm_variant_selection.py --smoke-test \\
    --modes original,composite_3band --models convnext_small \\
    --output output/chm_variant_selection_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from importlib.util import spec_from_file_location
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ============================================================================
# SETUP
# ============================================================================

_REPO_ROOT = Path(__file__).parent.parent
_V4_DIR = _REPO_ROOT / "scripts" / "model_search_v4"

import sys

# Set up sys.path for proper imports
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Create __init__.py for model_search_v4 to make it a proper package
init_file = _V4_DIR / "__init__.py"
if not init_file.exists():
    init_file.write_text("")

# Now import from model_search_v4 package
from model_search_v4.model_search_v4 import (
    _adapt_first_conv_to_nch,
    _CHMInputNorm,
    _binary_metrics,
    DEFAULT_SOURCE_WEIGHTS,
    write_curated_labels_drop_only,
    write_spatial_block_test_split,
    _best_threshold_from_probs,
)
from model_search_v4._labels import parse_raster_identity

# Import from base model_search.py via importlib
def _load_model_search_module():
    spec = spec_from_file_location("model_search_base", _REPO_ROOT / "scripts" / "model_search.py")
    if spec.loader is None:
        raise RuntimeError("Could not load model_search.py")
    mod = spec.loader.load_module()
    return mod

base_mod = _load_model_search_module()
_load_records_with_probs = base_mod._load_records_with_probs
_load_test_keys = base_mod._load_test_keys
_evaluate_classifier = base_mod._evaluate_classifier
Record = base_mod.Record

# Import from fine_tune_cnn.py
def _load_norm_tile():
    spec = spec_from_file_location("fine_tune_cnn", _REPO_ROOT / "scripts" / "fine_tune_cnn.py")
    if spec.loader is None:
        raise RuntimeError("Could not load fine_tune_cnn.py")
    mod = spec.loader.load_module()
    return mod._norm_tile

_norm_tile = _load_norm_tile()

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("chm_variant_selection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # File handler
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RunRecord:
    """One fold-run result."""
    mode: str
    model: str
    fold: int
    val_f1: float
    test_f1: float
    val_auc: float
    test_auc: float
    threshold: float
    n_train: int
    n_val: int
    n_test: int
    time_sec: float

# ============================================================================
# PATH RESOLUTION
# ============================================================================

def _chip_path_for_mode(
    raster_name: str,
    mode: str,
    original_chm_dir: Path,
    harmonized_root: Path,
    composite_dir: Path,
) -> Optional[Path]:
    """Resolve physical file path for a raster in a given mode."""
    stem = Path(raster_name).stem

    if mode == "original":
        return original_chm_dir / raster_name

    elif mode == "raw":
        # Replace suffix: chm_max_hag_20cm → harmonized_dem_last_raw_chm
        new_name = stem.replace("chm_max_hag_20cm", "harmonized_dem_last_raw_chm") + ".tif"
        return harmonized_root / "chm_raw" / new_name

    elif mode == "gauss":
        # Replace suffix: chm_max_hag_20cm → harmonized_dem_last_gauss_chm
        new_name = stem.replace("chm_max_hag_20cm", "harmonized_dem_last_gauss_chm") + ".tif"
        return harmonized_root / "chm_gauss" / new_name

    elif mode == "composite_3band":
        # Extract tile + year: e.g., "436646_2018_madal_..." → "436646_2018_3band.tif"
        m = re.match(r"^(\d+)_(\d{4})_", stem)
        if m:
            return composite_dir / f"{m.group(1)}_{m.group(2)}_3band.tif"
        return None

    else:
        raise ValueError(f"Unknown mode: {mode}")

# ============================================================================
# CHIP LOADING
# ============================================================================

def load_chips_for_mode(
    records: list[Record],
    mode: str,
    chm_dirs: dict[str, Path],
    canonical_size: int = 128,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load chip arrays for a given mode.

    Returns:
        X: (N, C, H, W) float32 array, values in [0, 1]
        y: (N,) int array, 0 or 1
        w: (N,) float array, sample weights
        meta: list[dict] with raster, row_off, col_off, label per record
    """
    import rasterio
    from rasterio.windows import Window

    n_channels = {"original": 1, "raw": 1, "gauss": 1, "composite_3band": 2}[mode]

    X_list = []
    y_list = []
    w_list = []
    meta_list = []
    skipped = 0

    for rec in records:
        path = _chip_path_for_mode(
            rec.raster,
            mode,
            chm_dirs["original"],
            chm_dirs["harmonized_root"],
            chm_dirs["composite"],
        )

        if path is None or not path.exists():
            skipped += 1
            continue

        try:
            with rasterio.open(path) as src:
                window = Window(rec.col_off, rec.row_off, canonical_size, canonical_size)

                if mode == "composite_3band":
                    # 2-band composite
                    band1 = src.read(1, window=window, out_dtype=np.float32)
                    band2 = src.read(2, window=window, out_dtype=np.float32)
                    chip = np.stack([_norm_tile(band1), _norm_tile(band2)], axis=0)
                else:
                    # Single-band modes
                    band = src.read(1, window=window, out_dtype=np.float32)
                    chip = _norm_tile(band)[np.newaxis, :, :]  # (1, H, W)

                X_list.append(chip)
                y_list.append(rec.label)
                w_list.append(rec.weight)
                meta_list.append({
                    "raster": rec.raster,
                    "row_off": rec.row_off,
                    "col_off": rec.col_off,
                    "label": rec.label,
                })
        except Exception as e:
            logging.warning(f"Failed to load {path}: {e}")
            skipped += 1
            continue

    if skipped > 0:
        logging.info(f"Mode {mode}: skipped {skipped} records (file not found or read error)")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    w = np.array(w_list, dtype=np.float32)

    assert X.shape[1] == n_channels, f"Expected {n_channels} channels, got {X.shape[1]}"

    return X, y, w, meta_list

# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model_for_mode(
    model_name: str,
    n_channels: int,
    device: torch.device,
) -> nn.Module:
    """Build a model adapted for the given channel count."""
    import torchvision.models as tv_models

    # Load pretrained 3-ch backbone
    if model_name == "convnext_small":
        backbone = tv_models.convnext_small(weights=tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    elif model_name == "efficientnet_b2":
        backbone = tv_models.efficientnet_b2(weights=tv_models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Adapt first conv
    if n_channels != 3:
        _adapt_first_conv_to_nch(backbone, n_channels)

    # Wrap with learnable input normalization (especially important for multi-channel)
    if n_channels > 1:
        backbone = _CHMInputNorm.wrap(backbone, n_channels=n_channels)

    # Replace final classification layer (ImageNet has 1000 classes; we need 2)
    if hasattr(backbone, "fc"):
        # EfficientNet
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, 2)
    elif hasattr(backbone, "classifier"):
        # ConvNeXt
        if isinstance(backbone.classifier[-1], nn.Linear):
            num_ftrs = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(num_ftrs, 2)
        else:
            # ConvNeXt has classifier as Sequential with AdaptiveAvgPool + Flatten + Linear
            for i, layer in enumerate(backbone.classifier):
                if isinstance(layer, nn.Linear):
                    num_ftrs = layer.in_features
                    backbone.classifier[i] = nn.Linear(num_ftrs, 2)
                    break
    else:
        raise ValueError(f"Cannot find classification head in {model_name}")

    backbone.to(device)
    return backbone

# ============================================================================
# CV FOLD GENERATION
# ============================================================================

def make_cv_folds(
    records: list[Record],
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Generate CV fold indices using StratifiedGroupKFold."""
    y = np.array([r.label for r in records], dtype=int)

    # Groups: place_key (tile + site, year-agnostic) to keep all years together
    groups = []
    for r in records:
        identity = parse_raster_identity(r.raster)
        if identity and 'place_key' in identity:
            place_key = identity['place_key']
        elif identity and 'tile' in identity and 'site' in identity:
            place_key = f"{identity['tile']}_{identity['site']}"
        else:
            place_key = Path(r.raster).stem
        groups.append(place_key)
    groups = np.array(groups)

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(y)), y, groups))

    return y, folds

# ============================================================================
# TRAINING
# ============================================================================

def train_one_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Train one fold, return best val metrics."""

    # Compute class weights for loss
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    w_pos = float(n_neg / max(n_pos, 1))
    class_weight = torch.tensor([1.0, w_pos], device=device)

    # Build dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(w_train),
    )
    sampler = WeightedRandomSampler(
        weights=w_train,
        num_samples=len(w_train),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    # Training loop
    best_val_f1 = -np.inf
    best_val_auc = -np.inf
    best_state_dict = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for X_batch, y_batch, w_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_preds = []
        val_probs = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.softmax(logits, dim=1)[:, 1]  # P(class=1)
                val_probs.append(probs.cpu().numpy())

        val_probs = np.concatenate(val_probs)

        # Compute metrics at best threshold
        try:
            threshold = _best_threshold_from_probs(y_val, val_probs, metric="f1")
        except Exception:
            threshold = 0.5

        metrics = _binary_metrics(y_val, val_probs, threshold)
        val_f1 = metrics["f1"]
        val_auc = metrics["auc"]

        # Early stopping
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_auc > best_val_auc):
            best_val_f1 = val_f1
            best_val_auc = val_auc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        scheduler.step()

    # Load best state
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "val_f1": best_val_f1,
        "val_auc": best_val_auc,
        "threshold": threshold,
        "best_state_dict": best_state_dict,
    }

# ============================================================================
# ABLATION
# ============================================================================

def run_mode_ablation(
    mode: str,
    models: list[str],
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    output_dir: Path,
    logger: logging.Logger,
) -> list[RunRecord]:
    """Run ablation for one mode across all models and folds."""

    records = []

    for model_name in models:
        logger.info(f"Mode {mode}, model {model_name}: starting ablation ({len(fold_splits)} folds)")

        for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits):
            fold_start = time.time()

            X_tr = X_train[tr_idx]
            y_tr = y_train[tr_idx]
            w_tr = w_train[tr_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]

            # Build model
            model = build_model_for_mode(model_name, X_train.shape[1], device)

            # Train
            train_result = train_one_fold(
                model, X_tr, y_tr, w_tr, X_val, y_val,
                device, epochs, batch_size, patience,
            )
            val_f1 = train_result["val_f1"]
            val_auc = train_result["val_auc"]
            threshold = train_result["threshold"]

            # Evaluate on test
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(X_test).to(device).float())
                test_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            test_metrics = _binary_metrics(y_test, test_probs, threshold)
            test_f1 = test_metrics["f1"]
            test_auc = test_metrics["auc"]

            fold_time = time.time() - fold_start

            rec = RunRecord(
                mode=mode,
                model=model_name,
                fold=fold_idx,
                val_f1=float(val_f1),
                test_f1=float(test_f1),
                val_auc=float(val_auc),
                test_auc=float(test_auc),
                threshold=float(threshold),
                n_train=len(X_tr),
                n_val=len(X_val),
                n_test=len(X_test),
                time_sec=fold_time,
            )
            records.append(rec)

            logger.info(
                f"  fold {fold_idx}: val_f1={val_f1:.4f}, test_f1={test_f1:.4f}, time={fold_time:.1f}s"
            )

        logger.info(f"Mode {mode}, model {model_name}: completed {len(fold_splits)} folds")

    # Write results incrementally
    out_csv = output_dir / "results.csv"
    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(
        out_csv,
        mode="a",
        header=not out_csv.exists(),
        index=False,
    )

    return records

# ============================================================================
# PREPARE
# ============================================================================

def prepare_labels_and_split(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    """Prepare labels and split; reuse if they exist."""
    prepared_dir = Path(args.prepared_dir)
    labels_dir = prepared_dir / "labels_curated_v4"
    split_path = prepared_dir / "cnn_test_split_v4.json"

    if (labels_dir.exists() and split_path.exists() and
        any(labels_dir.glob("*_labels.csv"))):
        logger.info(f"Reusing existing prepared data: {prepared_dir}")
        return labels_dir, split_path

    logger.info("Preparing labels and split...")

    # Curate labels
    drop_labels_dir = Path(args.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stats, all_candidates = write_curated_labels_drop_only(
        drop_labels_dir=drop_labels_dir,
        curated_labels_dir=labels_dir,
        t_high=args.t_high,
        t_low=args.t_low,
    )
    logger.info(f"Label curation: {stats}")

    # Spatial split
    split_meta = write_spatial_block_test_split(
        all_candidates=all_candidates,
        output_test_split=split_path,
        seed=args.seed,
        test_fraction=0.20,
        split_block_size_places=2,
        neighbor_buffer_blocks=1,
    )
    logger.info(f"Spatial split: {split_meta}")

    return labels_dir, split_path

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/chm_variant_selection", help="Output directory")
    parser.add_argument("--prepared-dir", default="output/model_search_v4/prepared", help="Prepared labels/split dir")
    parser.add_argument("--labels-dir", default="output/onboarding_labels_v2_drop13", help="Source labels dir")
    parser.add_argument("--original-chm-dir", default="data/lamapuit/chm_max_hag_13_drop", help="Original CHM dir")
    parser.add_argument("--harmonized-root", default="output/chm_dataset_harmonized_0p8m_raw_gauss", help="Harmonized root dir")
    parser.add_argument("--composite-dir", default="data/chm_variants/composite_3band", help="Composite 3band dir")
    parser.add_argument("--modes", default="original,raw,gauss,composite_3band", help="Comma-sep list of modes")
    parser.add_argument("--models", default="convnext_small,efficientnet_b2", help="Comma-sep list of models")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per fold")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--t-high", type=float, default=0.9995, help="Auto→cdw threshold")
    parser.add_argument("--t-low", type=float, default=0.0698, help="Auto→no_cdw threshold")
    parser.add_argument("--smoke-test", action="store_true", help="Quick validation mode")

    args = parser.parse_args()

    # Adjust for smoke test
    if args.smoke_test:
        args.n_folds = 3
        args.epochs = 3
        args.batch_size = 8
        args.patience = 2
        args.modes = args.modes.split(",")[:2]  # First 2 modes
        args.models = args.models.split(",")[:1]  # First model

    output_dir = Path(args.output)
    logger = setup_logging(output_dir)

    logger.info(f"Args: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Prepare
    labels_dir, split_path = prepare_labels_and_split(args, logger)

    # Load records
    logger.info("Loading records...")
    records = _load_records_with_probs(labels_dir, DEFAULT_SOURCE_WEIGHTS)
    logger.info(f"Loaded {len(records)} records")

    # Load test keys
    test_keys = _load_test_keys(split_path)
    logger.info(f"Test set size: {len(test_keys)}")

    # Partition train/test
    rec_test = [r for r in records if r.key in test_keys]
    rec_train = [r for r in records if r.key not in test_keys]
    logger.info(f"Train: {len(rec_train)}, Test: {len(rec_test)}")

    # For smoke test, subsample
    if args.smoke_test:
        import random
        rng = random.Random(args.seed)
        cdw_idx = [i for i, r in enumerate(rec_train) if r.label == 1]
        no_cdw_idx = [i for i, r in enumerate(rec_train) if r.label == 0]
        rng.shuffle(cdw_idx)
        rng.shuffle(no_cdw_idx)
        n = max(50, len(rec_train) // 10)
        half = n // 2
        keep = sorted(cdw_idx[:half] + no_cdw_idx[:half])
        rec_train = [rec_train[i] for i in keep]

        test_idx = rng.sample(range(len(rec_test)), k=max(20, len(rec_test) // 10))
        rec_test = [rec_test[i] for i in test_idx]
        logger.info(f"Smoke test: train subsampled to {len(rec_train)}, test to {len(rec_test)}")

    # CV folds (computed once, shared across modes)
    logger.info("Computing CV folds...")
    y_train, fold_splits = make_cv_folds(rec_train, args.n_folds, args.seed)
    logger.info(f"CV folds: {len(fold_splits)} folds")

    # Resolve directories
    chm_dirs = {
        "original": Path(args.original_chm_dir),
        "harmonized_root": Path(args.harmonized_root),
        "composite": Path(args.composite_dir),
    }

    # Run ablation per mode
    all_records = []
    modes = [m.strip() for m in args.modes.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    for mode in modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Mode: {mode}")
        logger.info(f"{'='*60}")

        # Load chips
        logger.info(f"Loading chips for mode {mode}...")
        X_train, y_train_check, w_train, _ = load_chips_for_mode(
            rec_train, mode, chm_dirs, device=str(device),
        )
        X_test, y_test, w_test, _ = load_chips_for_mode(
            rec_test, mode, chm_dirs, device=str(device),
        )
        logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # Sanity checks
        assert X_train.shape[1] > 0, f"No channels in X_train"
        assert len(np.unique(y_test)) == 2, f"Test set is not balanced"

        # Run ablation
        mode_records = run_mode_ablation(
            mode,
            models,
            fold_splits,
            X_train,
            y_train_check,
            w_train,
            X_test,
            y_test,
            device,
            args.epochs,
            args.batch_size,
            args.patience,
            output_dir,
            logger,
        )
        all_records.extend(mode_records)

    # Write summary
    logger.info("\nWriting summary...")
    results_df = pd.DataFrame([asdict(r) for r in all_records])

    mode_stats = results_df.groupby("mode")["test_f1"].agg(["mean", "std", "count"])
    mode_stats.columns = ["mean_f1", "std_f1", "n_runs"]

    # 95% CI via t-distribution
    from scipy import stats as scipy_stats
    for mode in mode_stats.index:
        n = mode_stats.loc[mode, "n_runs"]
        std = mode_stats.loc[mode, "std_f1"]
        if n > 1:
            ci_half = scipy_stats.t.ppf(0.975, df=int(n-1)) * std / np.sqrt(n)
        else:
            ci_half = 0.0
        mode_stats.loc[mode, "ci95_lower"] = mode_stats.loc[mode, "mean_f1"] - ci_half
        mode_stats.loc[mode, "ci95_upper"] = mode_stats.loc[mode, "mean_f1"] + ci_half

    summary_json = {
        "created_at": pd.Timestamp.now().isoformat(),
        "seed": args.seed,
        "modes": mode_stats.to_dict(orient="index"),
        "ranking": mode_stats["mean_f1"].sort_values(ascending=False).index.tolist(),
        "smoke_test": args.smoke_test,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))
    logger.info(f"Summary: {summary_json}")

    logger.info("\n✓ Experiment completed")
    logger.info(f"Results: {output_dir / 'results.csv'}")
    logger.info(f"Summary: {output_dir / 'summary.json'}")

if __name__ == "__main__":
    main()

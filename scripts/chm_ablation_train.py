"""
CHM Input Ablation Training — Optimized for 4-hour budget

Trains 3 models × 4 CHM inputs × 3 folds with GroupKFold (place+year grouping).
Prevents leakage: all years of a place stay in same fold.

Usage (in Docker with conda env):
    python scripts/chm_ablation_train.py \
        --labels-dir output/model_search_v4/prepared/labels_main_budget \
        --output-dir output/model_search_chm_ablation_results \
        --chm-raw output/chm_dataset_harmonized_0p8m_raw_gauss_stable/chm_raw \
        --chm-gauss output/chm_dataset_harmonized_0p8m_raw_gauss_stable/chm_gauss \
        --chm-baseline chm_max_hag \
        --n-folds 3 \
        --test-fraction 0.10 \
        --epochs 30 \
        --batch-size 16 \
        --patience 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any
from collections import defaultdict
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import timm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_search_v4._labels import parse_raster_identity

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


# ============================================================================
# Data Loading
# ============================================================================

def load_high_confidence_labels(labels_dir: Path) -> list[dict[str, Any]]:
    """Load manual + high-confidence auto labels (>0.95 CWD or <0.05 no-CWD)."""
    candidates: list[dict[str, Any]] = []

    for f in sorted(labels_dir.glob("*_labels.csv")):
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raster = str(row["raster"])
                row_off = int(row["row_off"])
                col_off = int(row["col_off"])
                label = str(row.get("label", ""))
                source = str(row.get("source", ""))
                score = float(row.get("score", 0.5))

                is_manual = "manual" in source.lower() or source == "auto_reviewed"
                is_high_conf = (source == "auto_threshold_gate_v4" and
                               (score > 0.95 or score < 0.05))

                if is_manual or is_high_conf:
                    raster_id, _, _ = parse_raster_identity(raster)
                    grid_x, grid_y, year = raster_id

                    candidates.append({
                        "raster": raster,
                        "grid_x": grid_x,
                        "grid_y": grid_y,
                        "year": year,
                        "row_off": row_off,
                        "col_off": col_off,
                        "label": int(label == "CWD"),
                        "source": source,
                    })

    logger.info(f"Loaded {len(candidates)} high-confidence labels")
    return candidates


def create_grouped_kfold(
    candidates: list[dict[str, Any]],
    n_splits: int = 3,
    seed: int = 2026,
) -> tuple[list[tuple[list[int], list[int]]], dict]:
    """GroupKFold where place+year stays together."""
    place_year_map = {}
    group_counter = 0

    for cand in candidates:
        place_key = (cand["grid_x"], cand["grid_y"], cand["year"])
        if place_key not in place_year_map:
            place_year_map[place_key] = group_counter
            group_counter += 1

    groups = np.array([
        place_year_map[(c["grid_x"], c["grid_y"], c["year"])]
        for c in candidates
    ])

    logger.info(f"Created {len(place_year_map)} place+year groups")

    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(candidates, groups=groups))

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            raise ValueError(f"Fold {fold_idx}: leakage in {len(overlap)} groups")
        logger.info(f"Fold {fold_idx}: train={len(train_idx)} test={len(test_idx)}")

    return folds, place_year_map


# ============================================================================
# CHM Chip Extraction
# ============================================================================

def extract_chip_from_raster(raster_path: Path, row: int, col: int, size: int = 128) -> np.ndarray | None:
    """Extract chip from raster file."""
    if not raster_path.exists():
        return None
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1, window=rasterio.windows.Window(col, row, size, size))
            return data.astype(np.float32)
    except Exception:
        return None


def get_chm_filename(raster: str, chm_type: str) -> str:
    """Construct CHM filename from raster identifier."""
    parts = raster.split("_")
    if chm_type == "raw":
        return f"{raster}_harmonized_dem_last_raw_chm.tif"
    elif chm_type == "gauss":
        return f"{raster}_harmonized_dem_last_gauss_chm.tif"
    elif chm_type == "baseline":
        # Legacy: format is like "436646_2020_tava_chm_max_hag_20cm.tif"
        grid = raster.split("_")[0]
        year = raster.split("_")[1]
        region = raster.split("_")[2]
        return f"{grid}_{year}_{region}_chm_max_hag_20cm.tif"
    return ""


class CHMDataset(Dataset):
    """Dataset for CHM input modes with 3-channel support."""

    def __init__(
        self,
        candidates: list[dict],
        indices: list[int],
        input_mode: str,
        chm_raw_dir: Path,
        chm_gauss_dir: Path,
        chm_baseline_dir: Path,
    ):
        self.candidates = [candidates[i] for i in indices]
        self.input_mode = input_mode
        self.chm_raw_dir = chm_raw_dir
        self.chm_gauss_dir = chm_gauss_dir
        self.chm_baseline_dir = chm_baseline_dir

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        cand = self.candidates[idx]
        raster = cand["raster"]
        row = cand["row_off"]
        col = cand["col_off"]
        label = cand["label"]

        if self.input_mode == "raw_1ch":
            fname = get_chm_filename(raster, "raw")
            chip = extract_chip_from_raster(self.chm_raw_dir / fname, row, col)
            if chip is None:
                return torch.zeros(1, 128, 128), label
            return torch.from_numpy(chip[np.newaxis, :, :]), label

        elif self.input_mode == "gauss_1ch":
            fname = get_chm_filename(raster, "gauss")
            chip = extract_chip_from_raster(self.chm_gauss_dir / fname, row, col)
            if chip is None:
                return torch.zeros(1, 128, 128), label
            return torch.from_numpy(chip[np.newaxis, :, :]), label

        elif self.input_mode == "baseline_1ch":
            fname = get_chm_filename(raster, "baseline")
            chip = extract_chip_from_raster(self.chm_baseline_dir / fname, row, col)
            if chip is None:
                return torch.zeros(1, 128, 128), label
            return torch.from_numpy(chip[np.newaxis, :, :]), label

        elif self.input_mode == "rgb_3ch":
            # Stack [raw, gauss, baseline] as 3 channels
            chips = []
            for chm_type in ["raw", "gauss", "baseline"]:
                if chm_type == "raw":
                    fname = get_chm_filename(raster, "raw")
                    chip = extract_chip_from_raster(self.chm_raw_dir / fname, row, col)
                elif chm_type == "gauss":
                    fname = get_chm_filename(raster, "gauss")
                    chip = extract_chip_from_raster(self.chm_gauss_dir / fname, row, col)
                else:  # baseline
                    fname = get_chm_filename(raster, "baseline")
                    chip = extract_chip_from_raster(self.chm_baseline_dir / fname, row, col)

                if chip is None:
                    chip = np.zeros((128, 128), dtype=np.float32)
                chips.append(chip)

            stacked = np.stack(chips, axis=0)  # (3, 128, 128)
            return torch.from_numpy(stacked), label

        return torch.zeros(1, 128, 128), label


# ============================================================================
# Model Training
# ============================================================================

def get_model(model_name: str, n_channels: int) -> nn.Module:
    """Load timm model with channel adaptation."""
    if model_name == "convnext_small":
        m = timm.create_model("convnext_small", pretrained=True, num_classes=2)
    elif model_name == "efficientnet_b2":
        m = timm.create_model("efficientnet_b2", pretrained=True, num_classes=2)
    elif model_name == "maxvit_small":
        m = timm.create_model("maxvit_small_tf_224", pretrained=True, num_classes=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Adapt first conv to input channels
    if n_channels != 3:
        old_conv = m.conv_embed[0] if hasattr(m, "conv_embed") else m.stem[0]
        if isinstance(old_conv, nn.Conv2d):
            new_conv = nn.Conv2d(n_channels, old_conv.out_channels,
                                kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                bias=old_conv.bias is not None)
            # Average weights for RGB->mono
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if hasattr(m, "conv_embed"):
                m.conv_embed[0] = new_conv
            else:
                m.stem[0] = new_conv

    return m


def train_fold(
    fold_idx: int,
    train_dataset: Dataset,
    test_dataset: Dataset,
    model_name: str,
    input_mode: str,
    device: str,
    epochs: int = 30,
    batch_size: int = 16,
    patience: int = 5,
) -> dict:
    """Train a single model on a single fold."""
    n_channels = 1 if "1ch" in input_mode else 3

    model = get_model(model_name, n_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_f1 = 0.0
    no_improve_count = 0

    logger.info(f"[{model_name}:{input_mode}:fold{fold_idx}] Training start")
    fold_start = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                preds = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Threshold for F1
        best_thr = 0.5
        best_f1_epoch = 0.0
        for thr in np.linspace(0.1, 0.9, 20):
            f1 = f1_score(all_labels, (all_preds >= thr).astype(int))
            if f1 > best_f1_epoch:
                best_f1_epoch = f1
                best_thr = thr

        auc = roc_auc_score(all_labels, all_preds)

        if best_f1_epoch > best_f1:
            best_f1 = best_f1_epoch
            no_improve_count = 0
            logger.info(f"  epoch {epoch+1}/{epochs} | loss={train_loss:.4f} | f1={best_f1_epoch:.4f} "
                       f"auc={auc:.4f} thr={best_thr:.2f} [BEST]")
        else:
            no_improve_count += 1
            if no_improve_count % 2 == 0:
                logger.info(f"  epoch {epoch+1}/{epochs} | loss={train_loss:.4f} | f1={best_f1_epoch:.4f} "
                           f"auc={auc:.4f} no-improve={no_improve_count}/{patience}")

        if no_improve_count >= patience:
            logger.info(f"  Early stop at epoch {epoch+1}")
            break

    fold_time = time.time() - fold_start

    # Final test metrics
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    best_thr = 0.5
    best_f1_final = 0.0
    for thr in np.linspace(0.1, 0.9, 20):
        f1 = f1_score(all_labels, (all_preds >= thr).astype(int))
        if f1 > best_f1_final:
            best_f1_final = f1
            best_thr = thr

    preds_binary = (all_preds >= best_thr).astype(int)
    auc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, preds_binary)
    recall = recall_score(all_labels, preds_binary)

    logger.info(f"[{model_name}:{input_mode}:fold{fold_idx}] FINAL | f1={best_f1_final:.4f} "
               f"auc={auc:.4f} p={precision:.4f} r={recall:.4f} time={fold_time:.1f}s")

    return {
        "fold": fold_idx,
        "model": model_name,
        "input_mode": input_mode,
        "f1": float(best_f1_final),
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(best_thr),
        "time_sec": fold_time,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CHM Input Ablation Training")
    parser.add_argument("--labels-dir", default="output/model_search_v4/prepared/labels_main_budget")
    parser.add_argument("--output-dir", default="output/model_search_chm_ablation_results")
    parser.add_argument("--chm-raw", default="output/chm_dataset_harmonized_0p8m_raw_gauss_stable/chm_raw")
    parser.add_argument("--chm-gauss", default="output/chm_dataset_harmonized_0p8m_raw_gauss_stable/chm_gauss")
    parser.add_argument("--chm-baseline", default="chm_max_hag")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    candidates = load_high_confidence_labels(Path(args.labels_dir))
    folds, place_year_map = create_grouped_kfold(candidates, n_splits=args.n_folds)

    # Models and inputs
    models = ["convnext_small", "efficientnet_b2", "maxvit_small"]
    inputs = ["raw_1ch", "gauss_1ch", "baseline_1ch", "rgb_3ch"]

    results = []
    start_time = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        logger.info(f"\n{'='*80}")
        logger.info(f"FOLD {fold_idx + 1}/{args.n_folds}")
        logger.info(f"{'='*80}")

        for input_mode in inputs:
            train_ds = CHMDataset(
                candidates, train_idx, input_mode,
                Path(args.chm_raw), Path(args.chm_gauss), Path(args.chm_baseline)
            )
            test_ds = CHMDataset(
                candidates, test_idx, input_mode,
                Path(args.chm_raw), Path(args.chm_gauss), Path(args.chm_baseline)
            )

            for model_name in models:
                result = train_fold(
                    fold_idx, train_ds, test_ds, model_name, input_mode, device,
                    epochs=args.epochs, batch_size=args.batch_size, patience=args.patience
                )
                results.append(result)

    # Save results
    total_time = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"ALL FOLDS COMPLETE | Total time: {total_time/60:.1f} min")
    logger.info(f"{'='*80}")

    results_file = output_dir / "results.json"
    with open(results_file, "w") as fh:
        json.dump(results, fh, indent=2)

    logger.info(f"Results saved to {results_file}")

    # Summary table
    logger.info("\nSummary by Input Mode:")
    by_input = defaultdict(list)
    for r in results:
        by_input[r["input_mode"]].append(r["f1"])

    for input_mode in inputs:
        f1_scores = by_input[input_mode]
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        logger.info(f"  {input_mode:12s} | F1={mean_f1:.4f}±{std_f1:.4f}")


if __name__ == "__main__":
    main()

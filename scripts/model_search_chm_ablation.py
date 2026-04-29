"""
CHM Input Ablation Experiment — Grouped K-Fold, 5 inputs × 3 models

Determines which CHM preprocessing (raw vs gauss vs baseline) trains best
for CWD detection, preventing place+year leakage via GroupKFold.

Usage:
    python scripts/model_search_chm_ablation.py \
        --labels-dir output/model_search_v4/prepared/labels_main_budget \
        --output-dir output/model_search_chm_ablation_grouped_kfold \
        --chm-raw output/chm_dataset_harmonized_0p8m_raw_gauss_stable/chm_raw \
        --chm-gauss output/chm_dataset_harmonized_0p8m_raw_gauss_stable/chm_gauss \
        --chm-baseline chm_max_hag \
        --high-conf-only
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

import numpy as np
import rasterio
from sklearn.model_selection import GroupKFold

# Allow running from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_search_v4._labels import parse_raster_identity

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def load_high_confidence_labels(labels_dir: Path) -> list[dict[str, Any]]:
    """Load manual labels + auto-detected with high confidence.

    High confidence: auto_source with score > 0.95 (CWD) or < 0.05 (no-CWD).
    """
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

                # Keep manual labels always
                is_manual = "manual" in source.lower() or source == "auto_reviewed"

                # Keep auto labels only if high confidence
                is_high_conf_auto = (source == "auto_threshold_gate_v4" and
                                     (score > 0.95 or score < 0.05))

                if is_manual or is_high_conf_auto:
                    raster_id, row_off_rs, col_off_rs = parse_raster_identity(raster)
                    grid_x, grid_y, year = raster_id

                    candidates.append({
                        "key": (raster, row_off, col_off),
                        "raster": raster,
                        "grid_x": grid_x,
                        "grid_y": grid_y,
                        "year": year,
                        "row_off": row_off,
                        "col_off": col_off,
                        "label": label,
                        "source": source,
                        "score": score,
                    })

    logger.info(f"Loaded {len(candidates)} high-confidence labels")
    return candidates


def create_grouped_kfold_split(
    candidates: list[dict[str, Any]],
    n_splits: int = 5,
    seed: int = 2026,
) -> list[tuple[list[int], list[int]]]:
    """Create GroupKFold split where place+year groups stay together.

    This prevents leakage: if (place=436, year=2020) is in test fold,
    NO other chip from (436, 2020) can be in training.
    """
    # Create place+year groups
    place_year_map = {}  # (grid_x, grid_y, year) → group_id
    group_counter = 0

    for cand in candidates:
        place_key = (cand["grid_x"], cand["grid_y"], cand["year"])
        if place_key not in place_year_map:
            place_year_map[place_key] = group_counter
            group_counter += 1

    # Assign groups to each candidate
    groups = np.array([
        place_year_map[(c["grid_x"], c["grid_y"], c["year"])]
        for c in candidates
    ])

    logger.info(f"Created {len(place_year_map)} place+year groups from {len(candidates)} candidates")

    # Create GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(candidates, groups=groups))

    logger.info(f"GroupKFold split: {n_splits} folds")
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            logger.error(f"Fold {fold_idx}: leakage detected! {len(overlap)} groups in both train and test")
            raise ValueError(f"GroupKFold produced leakage in fold {fold_idx}")
        logger.info(f"Fold {fold_idx}: train={len(train_idx)} test={len(test_idx)}")

    return folds, place_year_map


def extract_chip(
    raster_path: Path,
    row_off: int,
    col_off: int,
    chip_size: int = 128,
) -> np.ndarray | None:
    """Extract 128×128 chip from raster file."""
    if not raster_path.exists():
        return None

    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1, window=rasterio.windows.Window(
                col_off, row_off, chip_size, chip_size
            ))
            return data.astype(np.float32)
    except Exception:
        return None


def prepare_split_manifests(
    candidates: list[dict[str, Any]],
    folds: list[tuple[list[int], list[int]]],
    chm_raw_dir: Path,
    chm_gauss_dir: Path,
    chm_baseline_dir: Path,
    output_dir: Path,
) -> None:
    """Create JSON manifests for each fold with all 5 input modes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Input mode configurations
    input_modes = {
        "raw_1ch": {
            "type": "single_channel",
            "source_dir": chm_raw_dir,
            "channel": 0,
        },
        "gauss_1ch": {
            "type": "single_channel",
            "source_dir": chm_gauss_dir,
            "channel": 0,
        },
        "baseline_1ch": {
            "type": "single_channel",
            "source_dir": chm_baseline_dir,
            "channel": 0,
        },
        "rgb_3ch": {
            "type": "three_channel",
            "sources": [
                ("raw", chm_raw_dir),
                ("gauss", chm_gauss_dir),
                ("baseline", chm_baseline_dir),
            ],
        },
        "gf_24feat": {
            "type": "classical_features",
            "n_features": 24,
        },
    }

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        manifest = {
            "fold": fold_idx,
            "input_modes": input_modes,
            "train": {
                "candidates": [
                    {
                        "key": candidates[i]["key"],
                        "raster": candidates[i]["raster"],
                        "row_off": candidates[i]["row_off"],
                        "col_off": candidates[i]["col_off"],
                        "label": candidates[i]["label"],
                        "place_year": (candidates[i]["grid_x"], candidates[i]["grid_y"], candidates[i]["year"]),
                    }
                    for i in train_idx
                ],
                "size": len(train_idx),
            },
            "test": {
                "candidates": [
                    {
                        "key": candidates[i]["key"],
                        "raster": candidates[i]["raster"],
                        "row_off": candidates[i]["row_off"],
                        "col_off": candidates[i]["col_off"],
                        "label": candidates[i]["label"],
                        "place_year": (candidates[i]["grid_x"], candidates[i]["grid_y"], candidates[i]["year"]),
                    }
                    for i in test_idx
                ],
                "size": len(test_idx),
            },
        }

        manifest_path = fold_dir / "manifest.json"
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

        logger.info(f"Fold {fold_idx}: manifest saved to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CHM Input Ablation — Grouped K-Fold with 5 inputs × 3 models"
    )
    parser.add_argument(
        "--labels-dir",
        default="output/model_search_v4/prepared/labels_main_budget",
        help="Directory with *_labels.csv files",
    )
    parser.add_argument(
        "--output-dir",
        default="output/model_search_chm_ablation_grouped_kfold",
        help="Output directory for experiment setup",
    )
    parser.add_argument(
        "--chm-raw",
        default="output/chm_dataset_harmonized_0p8m_raw_gauss_stable/chm_raw",
        help="Directory with raw CHM tifs",
    )
    parser.add_argument(
        "--chm-gauss",
        default="output/chm_dataset_harmonized_0p8m_raw_gauss_stable/chm_gauss",
        help="Directory with Gaussian-smoothed CHM tifs",
    )
    parser.add_argument(
        "--chm-baseline",
        default="chm_max_hag",
        help="Directory with baseline (legacy 0.2m) CHM tifs",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of K-Fold splits",
    )
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    chm_raw_dir = Path(args.chm_raw)
    chm_gauss_dir = Path(args.chm_gauss)
    chm_baseline_dir = Path(args.chm_baseline)

    logger.info(f"CHM sources:")
    logger.info(f"  raw:      {chm_raw_dir}")
    logger.info(f"  gauss:    {chm_gauss_dir}")
    logger.info(f"  baseline: {chm_baseline_dir}")

    # Load high-confidence labels
    candidates = load_high_confidence_labels(labels_dir)

    # Create Grouped K-Fold split
    folds, place_year_map = create_grouped_kfold_split(candidates, n_splits=args.n_splits)

    # Prepare split manifests
    prepare_split_manifests(
        candidates, folds,
        chm_raw_dir, chm_gauss_dir, chm_baseline_dir,
        output_dir,
    )

    logger.info(f"Experiment setup complete: {output_dir}")
    logger.info(f"Ready to train {len(place_year_map)} place+year groups × 5 inputs × 3 models × 5 folds")


if __name__ == "__main__":
    main()

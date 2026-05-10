#!/usr/bin/env python3
"""Phase II: Adaptive Tiling & Dataset Construction.

Builds a patch index (CSV) from the 4-band composite CHM with 5-fold
vertical-stripe spatial cross-validation, computes per-band statistics,
and provides the CWDSegDataset PyTorch Dataset used by Phase III.

5-fold spatial CV strategy:
    The 5000-column raster is divided into 5 vertical stripes of 1000 cols each
    (= 200m east-west ground extent per stripe at 0.2m/px).
    Stripe 0 (cols 0-999, westernmost) is the permanently held-out test set.
    Folds 0-4 rotate stripes 1-4 as validation; remaining stripes = train.
    A 64-px buffer between train and val stripes is excluded from loss (valid=0).

Usage:
    python phase2_dataset.py                   # build patch index + band stats
    python phase2_dataset.py --validate        # print per-fold stats and exit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.raster_io import read_multiband_window, normalize_bands, compute_band_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_WIDTH = 5000          # CHM raster total columns
STRIPE_WIDTH = 1000        # columns per stripe (5 stripes total)
N_STRIPES = 5
TEST_STRIPE = 0            # permanently held-out (westernmost)
BUFFER_PX = 64             # exclusion buffer between train and val stripes
PATCH_SIZE = 256
STRIDE = 192               # 25% overlap  (PATCH_SIZE - PATCH_SIZE // 4)
MIN_VALID_PX = 328         # 0.05 × 256² — skip near-empty patches
BINARY_BANDS = [3]         # 0-indexed; Band 4 (validity mask) is passed as-is
EPS = 1e-8


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PatchEntry:
    row_off: int
    col_off: int
    stripe_id: int     # 0-4 (0 = test)
    fold_id: int       # which CV fold this is validation for (-1 if test)
    n_valid: int       # pixels where valid_mask == 1 in phase1 mask
    n_positive: int    # pixels where target == 1 AND valid == 1


# ---------------------------------------------------------------------------
# 5-fold vertical-stripe spatial CV splitter
# ---------------------------------------------------------------------------


class SpatialCVSplitter:
    """Assigns fold IDs based on vertical stripe position.

    Stripe 0 (cols 0-999)  → test set (fold_id = -1)
    Stripes 1-4             → folds 0-3 respectively as the validation stripe
    """

    def __init__(
        self,
        stripe_width: int = STRIPE_WIDTH,
        n_stripes: int = N_STRIPES,
        test_stripe: int = TEST_STRIPE,
        buffer_px: int = BUFFER_PX,
    ) -> None:
        self.stripe_width = stripe_width
        self.n_stripes = n_stripes
        self.test_stripe = test_stripe
        self.buffer_px = buffer_px
        # Stripes 1..N-1 become folds 0..N-2
        self._val_stripes = [s for s in range(n_stripes) if s != test_stripe]

    def stripe_of(self, col_off: int, patch_size: int = PATCH_SIZE) -> int:
        """Return the stripe index for a patch starting at col_off."""
        center_col = col_off + patch_size // 2
        return min(center_col // self.stripe_width, self.n_stripes - 1)

    def fold_id_of(self, stripe: int) -> int:
        """Return fold_id for the stripe (-1 for test stripe)."""
        if stripe == self.test_stripe:
            return -1
        try:
            return self._val_stripes.index(stripe)
        except ValueError:
            return -1

    def is_in_buffer(self, col_off: int, patch_size: int = PATCH_SIZE) -> bool:
        """True if the patch overlaps any inter-stripe buffer zone."""
        stripe = self.stripe_of(col_off, patch_size)
        stripe_start = stripe * self.stripe_width
        stripe_end = stripe_start + self.stripe_width

        patch_left = col_off
        patch_right = col_off + patch_size

        left_gap = patch_left - stripe_start
        right_gap = stripe_end - patch_right

        return left_gap < self.buffer_px or right_gap < self.buffer_px

    def train_val_split(
        self, entries: list[PatchEntry], val_fold: int
    ) -> tuple[list[PatchEntry], list[PatchEntry]]:
        """Return (train, val) patch lists for a given CV fold.

        val_fold: 0..3 — the fold index that is used as validation.
        The test stripe (fold_id == -1) is excluded from both.
        """
        val_stripe = self._val_stripes[val_fold]
        train: list[PatchEntry] = []
        val: list[PatchEntry] = []
        for e in entries:
            if e.fold_id == -1:
                continue  # test stripe — excluded
            if e.stripe_id == val_stripe:
                val.append(e)
            else:
                train.append(e)
        return train, val

    def test_entries(self, entries: list[PatchEntry]) -> list[PatchEntry]:
        return [e for e in entries if e.fold_id == -1]


# ---------------------------------------------------------------------------
# Patch index construction
# ---------------------------------------------------------------------------


def build_patch_index(
    composite_tif: Path,
    mask_tif: Path,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    min_valid_px: int = MIN_VALID_PX,
    splitter: SpatialCVSplitter | None = None,
) -> list[PatchEntry]:
    """Scan the raster in a dense grid and collect usable patch entries."""
    if splitter is None:
        splitter = SpatialCVSplitter()

    with rasterio.open(composite_tif) as src:
        H, W = src.height, src.width

    entries: list[PatchEntry] = []

    ys = _grid_positions(H, patch_size, stride)
    xs = _grid_positions(W, patch_size, stride)

    with rasterio.open(mask_tif) as msrc:
        for y0 in ys:
            for x0 in xs:
                window = Window(x0, y0, patch_size, patch_size)
                # Band 2 = valid_mask, Band 1 = target
                raw = msrc.read([1, 2], window=window, boundless=True, fill_value=0.0)
                target_p = raw[0].astype(np.float32)
                valid_p = raw[1].astype(np.float32)

                n_valid = int((valid_p > 0.5).sum())
                if n_valid < min_valid_px:
                    continue

                n_pos = int(((target_p > 0.5) & (valid_p > 0.5)).sum())
                stripe = splitter.stripe_of(x0, patch_size)
                fold_id = splitter.fold_id_of(stripe)

                entries.append(
                    PatchEntry(
                        row_off=y0,
                        col_off=x0,
                        stripe_id=stripe,
                        fold_id=fold_id,
                        n_valid=n_valid,
                        n_positive=n_pos,
                    )
                )

    return entries


def _grid_positions(dim: int, patch_size: int, stride: int) -> list[int]:
    positions = list(range(0, max(1, dim - patch_size + 1), stride))
    last = max(0, dim - patch_size)
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def save_patch_index(entries: list[PatchEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(entries[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(e) for e in entries)


def load_patch_index(path: Path) -> list[PatchEntry]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [
            PatchEntry(
                row_off=int(r["row_off"]),
                col_off=int(r["col_off"]),
                stripe_id=int(r["stripe_id"]),
                fold_id=int(r["fold_id"]),
                n_valid=int(r["n_valid"]),
                n_positive=int(r["n_positive"]),
            )
            for r in reader
        ]


# ---------------------------------------------------------------------------
# CWDSegDataset — PyTorch Dataset
# ---------------------------------------------------------------------------


class CWDSegDataset(Dataset):
    """Patch dataset for CWD semantic segmentation.

    Reads 256×256 patches from the 4-band composite CHM and the Phase I
    true-mask raster. Applies joint geometric+radiometric augmentation.

    Returns:
        image:  Tensor (C, 256, 256) float32, normalized
        target: Tensor (1, 256, 256) float32, {0, 1}
        valid:  Tensor (1, 256, 256) float32, {0, 1}  (buffer pixels zeroed)
    """

    def __init__(
        self,
        entries: list[PatchEntry],
        composite_tif: Path,
        mask_tif: Path,
        band_stats: dict,
        patch_size: int = PATCH_SIZE,
        in_channels: int = 4,
        augment: bool = False,
        buffer_px: int = BUFFER_PX,
        stripe_width: int = STRIPE_WIDTH,
        val_stripe: int | None = None,
    ) -> None:
        self.entries = entries
        self.composite_tif = composite_tif
        self.mask_tif = mask_tif
        self.band_stats = band_stats
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.augment = augment
        self.buffer_px = buffer_px
        self.stripe_width = stripe_width
        self.val_stripe = val_stripe

        self._aug = None
        if augment:
            from common.augmentation import get_full_aug
            self._aug = get_full_aug()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        ps = self.patch_size

        # Read composite CHM
        img = read_multiband_window(self.composite_tif, e.row_off, e.col_off, ps)
        if img.shape[0] > self.in_channels:
            img = img[: self.in_channels]
        elif img.shape[0] < self.in_channels:
            pad = np.zeros((self.in_channels - img.shape[0], ps, ps), dtype=np.float32)
            img = np.concatenate([img, pad], axis=0)

        img = normalize_bands(img, self.band_stats, binary_bands=BINARY_BANDS)

        # Read phase1 mask (Band1=target, Band2=valid_mask)
        with rasterio.open(self.mask_tif) as msrc:
            raw = msrc.read(
                [1, 2],
                window=Window(e.col_off, e.row_off, ps, ps),
                boundless=True,
                fill_value=0.0,
            ).astype(np.float32)
        target = raw[0]
        valid = raw[1]

        # Apply buffer zone: zero out valid_mask near stripe boundaries
        if self.val_stripe is not None:
            valid = self._apply_buffer(valid, e.col_off, ps)

        # Augmentation (jointly applied to image + target + valid)
        if self.augment and self._aug is not None:
            # albumentations expects HWC for image; bands first for us
            img_hwc = img.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
            result = self._aug(image=img_hwc, target=target, valid=valid)
            img = result["image"].transpose(2, 0, 1)
            target = result["target"]
            valid = result["valid"]

        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)).float(),
            "target": torch.from_numpy(target[np.newaxis]).float(),
            "valid": torch.from_numpy(valid[np.newaxis]).float(),
            "row_off": e.row_off,
            "col_off": e.col_off,
        }

    def _apply_buffer(self, valid: np.ndarray, col_off: int, ps: int) -> np.ndarray:
        """Zero valid_mask pixels within buffer_px of the val_stripe boundary."""
        valid = valid.copy()
        assert self.val_stripe is not None

        left_boundary = self.val_stripe * self.stripe_width
        right_boundary = left_boundary + self.stripe_width

        # Left boundary of val stripe
        left_buf_start = left_boundary - self.buffer_px
        left_buf_end = left_boundary + self.buffer_px
        for col in range(max(0, left_buf_start - col_off), min(ps, left_buf_end - col_off)):
            if 0 <= col < ps:
                valid[:, col] = 0.0

        # Right boundary of val stripe
        right_buf_start = right_boundary - self.buffer_px
        right_buf_end = right_boundary + self.buffer_px
        for col in range(max(0, right_buf_start - col_off), min(ps, right_buf_end - col_off)):
            if 0 <= col < ps:
                valid[:, col] = 0.0

        return valid


def make_weighted_sampler(entries: list[PatchEntry], pos_weight: float = 3.0) -> WeightedRandomSampler:
    """Oversample patches with at least one positive pixel by pos_weight factor."""
    weights = [pos_weight if e.n_positive > 0 else 1.0 for e in entries]
    return WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase II: Dataset construction")
    p.add_argument(
        "--composite-tif",
        type=Path,
        default=ROOT / "seg_pipeline" / "input" / "composite_4band.tif",
    )
    p.add_argument(
        "--mask-tif",
        type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase1_masks" / "406455_2021_tava_truemask.tif",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase2_dataset",
    )
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--min-valid-px", type=int, default=MIN_VALID_PX)
    p.add_argument("--validate", action="store_true", help="Print fold stats and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splitter = SpatialCVSplitter()

    index_path = args.output_dir / "patch_index.csv"
    stats_path = args.output_dir / "band_stats.json"

    if index_path.exists() and not args.validate:
        print(f"Loading existing patch index: {index_path}")
        entries = load_patch_index(index_path)
    else:
        print("Building patch index…")
        entries = build_patch_index(
            composite_tif=args.composite_tif,
            mask_tif=args.mask_tif,
            patch_size=args.patch_size,
            stride=args.stride,
            min_valid_px=args.min_valid_px,
            splitter=splitter,
        )
        save_patch_index(entries, index_path)
        print(f"Saved {len(entries):,} patches → {index_path}")

    # Per-fold statistics
    test_e = splitter.test_entries(entries)
    print(f"\nTotal patches:  {len(entries):,}")
    print(f"Test (stripe 0): {len(test_e):,}")
    n_folds = N_STRIPES - 1
    for fold in range(n_folds):
        train_e, val_e = splitter.train_val_split(entries, fold)
        n_pos_tr = sum(1 for e in train_e if e.n_positive > 0)
        n_pos_va = sum(1 for e in val_e if e.n_positive > 0)
        print(
            f"Fold {fold}: train={len(train_e):,} (pos={n_pos_tr:,})  "
            f"val={len(val_e):,} (pos={n_pos_va:,})"
        )

    if args.validate:
        return

    # Band statistics from valid pixels in the composite
    if stats_path.exists():
        print(f"\nLoading existing band stats: {stats_path}")
        band_stats = json.loads(stats_path.read_text())
    else:
        print("\nComputing band statistics…")
        with rasterio.open(args.mask_tif) as msrc:
            valid_band = msrc.read(2).astype(bool)
        band_stats = compute_band_stats(args.composite_tif, valid_mask=valid_band)
        stats_path.write_text(json.dumps(band_stats, indent=2))
        print(f"Saved band stats → {stats_path}")
        for i, s in band_stats.items():
            print(f"  Band {i}: mean={s['mean']:.4f}  std={s['std']:.4f}  p2={s['p2']:.4f}  p98={s['p98']:.4f}")


if __name__ == "__main__":
    main()

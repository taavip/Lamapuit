#!/usr/bin/env python3
"""Phase II V3: Adaptive Tiling & Dataset Construction — 5-fold CV + V2 improvements.

V3 restores the 5-fold vertical-stripe CV structure from V1 (stripe width = 1000 cols,
5 stripes total) while retaining all V2 improvements: conflict zone masking, nodata
dropout augmentation, and composite CHM as default input.

Design rationale (V3 vs V2):
    V2's root failure was training data starvation.  With N_STRIPES=3, each training fold
    saw only 29–45 positive patches (vs. 80–95 in V1's best folds).  CWD morphology is
    heterogeneous — short dead snags, elongated fallen logs, partial canopy occlusions —
    and exposure to only 30% of available positive examples leaves the model unable to
    learn the full appearance distribution.  Restoring N_STRIPES=5 recovers training set
    size per fold at the cost of smaller validation stripes (1000-col vs 1667-col), which
    is an acceptable trade-off because the held-out test stripe is fixed regardless.

5-fold spatial CV strategy (V3):
    The 5000-column raster is divided into 5 vertical stripes of 1000 cols each.
    Stripe 0 (cols 0–999, westernmost) is the permanently held-out test set.
    Folds 0–3 rotate stripes 1–4 as validation; the remaining 3 stripes = train.
    A 64-px buffer between train and val stripes is excluded from loss (valid=0).

Conflict zone masking (retained from V2):
    Reads Band 3 (ensemble_prob) from the Phase I mask TIF.
    Excludes pixels where ensemble_prob ≥ 0.15 AND outside GPKG polygons.
    Rationale: these "conflict zones" represent high model confidence for CWD presence
    that is not corroborated by manual annotation, indicating either annotation
    omission or a false positive from the Phase I ensemble.  Training on such
    pixels introduces contradictory supervision signals.

Usage:
    python phase2_dataset_v3.py                          # default: composite variant
    python phase2_dataset_v3.py --chm-variant gauss      # override variant
    python phase2_dataset_v3.py --validate               # fold stats + exit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.raster_io import read_multiband_window, normalize_bands, compute_band_stats

# ---------------------------------------------------------------------------
# Constants — 5-fold CV (matching V1 geography, V2 conflict masking)
# ---------------------------------------------------------------------------

TILE_WIDTH = 5000
N_STRIPES = 5                          # restored from V1
TEST_STRIPE = 0                        # permanently held-out westernmost stripe
STRIPE_WIDTH = TILE_WIDTH // N_STRIPES  # 1000 columns per stripe (same as V1)
BUFFER_PX = 64
PATCH_SIZE = 256
STRIDE = 192
MIN_VALID_PX = 328
EPS = 1e-8

CONFLICT_ENSEMBLE_THRESHOLD = 0.15   # retained from V2

DEFAULT_VARIANT = "composite"         # V2 ablation found composite wins decisively


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PatchEntry:
    row_off: int
    col_off: int
    stripe_id: int
    fold_id: int    # which CV fold this is validation for (-1 if test)
    n_valid: int
    n_positive: int


# ---------------------------------------------------------------------------
# 5-fold vertical-stripe spatial CV splitter (V3)
# ---------------------------------------------------------------------------


class SpatialCVSplitterV3:
    """5-fold vertical-stripe spatial CV, restoring V1 fold structure.

    DEPRECATED: Creates severely imbalanced folds (fold 0: 44% train, fold 3: 92% train).
    Use SpatialCVSplitterV4 for balanced 2-fold CV instead.

    Stripe 0 (cols 0–999)    → test set (fold_id = -1, never trained/validated)
    Stripe 1 (cols 1000–1999) → fold 0 as validation, folds 1–3 as training
    Stripe 2 (cols 2000–2999) → fold 1 as validation, folds 0,2,3 as training
    Stripe 3 (cols 3000–3999) → fold 2 as validation, folds 0,1,3 as training
    Stripe 4 (cols 4000–4999) → fold 3 as validation, folds 0,1,2 as training

    Each training fold has ~3 × 1000 = 3000 training columns, recovering the
    training data volume of V1 (390–416 patches, 80–95 positives per fold).
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
        self._val_stripes = [s for s in range(n_stripes) if s != test_stripe]

    def stripe_of(self, col_off: int, patch_size: int = PATCH_SIZE) -> int:
        center_col = col_off + patch_size // 2
        return min(center_col // self.stripe_width, self.n_stripes - 1)

    def fold_id_of(self, stripe: int) -> int:
        if stripe == self.test_stripe:
            return -1
        try:
            return self._val_stripes.index(stripe)
        except ValueError:
            return -1

    def is_in_buffer(self, col_off: int, patch_size: int = PATCH_SIZE) -> bool:
        stripe = self.stripe_of(col_off, patch_size)
        stripe_start = stripe * self.stripe_width
        stripe_end = min(stripe_start + self.stripe_width, TILE_WIDTH)
        left_gap = col_off - stripe_start
        right_gap = stripe_end - (col_off + patch_size)
        return left_gap < self.buffer_px or right_gap < self.buffer_px

    def train_val_split(
        self, entries: list[PatchEntry], val_fold: int
    ) -> tuple[list[PatchEntry], list[PatchEntry]]:
        """Return (train, val) for a given fold.

        Train = all non-test, non-val stripes (3 stripes = ~3000 cols, matching V1).
        Val = the single stripe assigned to val_fold.
        """
        val_stripe = self._val_stripes[val_fold]
        train: list[PatchEntry] = []
        val: list[PatchEntry] = []
        for e in entries:
            if e.fold_id == -1:
                continue
            if e.stripe_id == val_stripe:
                val.append(e)
            else:
                train.append(e)
        return train, val

    def test_entries(self, entries: list[PatchEntry]) -> list[PatchEntry]:
        return [e for e in entries if e.fold_id == -1]


class SpatialCVSplitterV4:
    """2-fold balanced vertical-stripe spatial CV with single-stripe validation.

    Maximizes training data while keeping folds more balanced. Stripe 1 is the
    largest non-test stripe (118 patches), so use it alone for one fold's validation.

    Stripe 0 (cols 0–999)    → test set (fold_id = -1, never trained/validated)
    Stripe 1 (cols 1000–1999) → Fold 0 validation (118 patches, 34% of training data)
    Stripes 2,3,4 (cols 2000–4999) → Fold 0 training (95 patches)

    Stripe 1 (cols 1000–1999) → Fold 1 training (118 patches, 55% of training data)
    Stripes 2,3,4 (cols 2000–4999) → Fold 1 validation (95 patches)

    Benefits:
    ✓ More balanced train/val ratio (95:118 = 1.24× vs 35:178 = 5.09×)
    ✓ Maximizes training data (use 3 stripes for training, not 2)
    ✓ Geographic separation prevents leakage
    ✓ Symmetric CV (each fold trains ~50% of data)
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

    def stripe_of(self, col_off: int, patch_size: int = PATCH_SIZE) -> int:
        center_col = col_off + patch_size // 2
        return min(center_col // self.stripe_width, self.n_stripes - 1)

    def fold_id_of(self, stripe: int) -> int:
        """Assign fold based on stripe.

        Stripe 1 is validation for fold 0 (and training for fold 1).
        Stripes 2,3,4 are training for fold 0 (and validation for fold 1).
        """
        if stripe == self.test_stripe:
            return -1
        elif stripe == 1:
            return 0  # Stripe 1 is validation for fold 0
        else:
            return 1  # Stripes 2,3,4 are validation for fold 1

    def is_in_buffer(self, col_off: int, patch_size: int = PATCH_SIZE) -> bool:
        """Check if patch is in buffer zone at stripe boundaries."""
        stripe = self.stripe_of(col_off, patch_size)
        stripe_start = stripe * self.stripe_width
        stripe_end = min(stripe_start + self.stripe_width, TILE_WIDTH)
        left_gap = col_off - stripe_start
        right_gap = stripe_end - (col_off + patch_size)
        return left_gap < self.buffer_px or right_gap < self.buffer_px

    def train_val_split(
        self, entries: list[PatchEntry], val_fold: int
    ) -> tuple[list[PatchEntry], list[PatchEntry]]:
        """Return (train, val) for a given fold.

        Fold 0: Train on stripes 2,3,4 (95 patches), Val on stripe 1 (118 patches)
        Fold 1: Train on stripe 1 (118 patches), Val on stripes 2,3,4 (95 patches)
        """
        if val_fold == 0:
            val_stripes = {1}
            train_stripes = {2, 3, 4}
        elif val_fold == 1:
            val_stripes = {2, 3, 4}
            train_stripes = {1}
        else:
            raise ValueError(f"Invalid fold: {val_fold}. Must be 0 or 1.")

        train: list[PatchEntry] = []
        val: list[PatchEntry] = []
        for e in entries:
            if e.fold_id == -1:
                continue
            if e.stripe_id in val_stripes:
                val.append(e)
            elif e.stripe_id in train_stripes:
                train.append(e)
        return train, val

    def test_entries(self, entries: list[PatchEntry]) -> list[PatchEntry]:
        return [e for e in entries if e.fold_id == -1]


# ---------------------------------------------------------------------------
# CHM variant helpers (shared with V2 — kept in sync)
# ---------------------------------------------------------------------------


def _read_chm_path(variant: str, root: Path) -> Path:
    variant_map = {
        "baseline": "baseline_chm.tif",
        "raw": "raw_chm.tif",
        "gauss": "gauss_chm.tif",
        "masked": "masked_chm.tif",
        "composite": "composite_4band.tif",
    }
    if variant not in variant_map:
        raise ValueError(f"Unknown CHM variant: {variant}. Choices: {list(variant_map.keys())}")
    return root / "seg_pipeline" / "input" / variant_map[variant]


def _get_in_channels(variant: str) -> int:
    if variant in ("baseline", "raw", "gauss"):
        return 1
    elif variant == "masked":
        return 2
    elif variant == "composite":
        return 4
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _get_binary_bands(variant: str) -> list[int]:
    if variant in ("baseline", "raw", "gauss"):
        return []
    elif variant == "masked":
        return [1]
    elif variant == "composite":
        return [3]
    else:
        return []


# ---------------------------------------------------------------------------
# Patch index construction with conflict zone masking (V2 method, retained)
# ---------------------------------------------------------------------------


def build_patch_index(
    chm_tif: Path,
    mask_tif: Path,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    min_valid_px: int = MIN_VALID_PX,
    splitter: SpatialCVSplitterV3 | None = None,
    conflict_threshold: float = CONFLICT_ENSEMBLE_THRESHOLD,
) -> list[PatchEntry]:
    """Scan the raster in a dense grid and collect usable patch entries.

    Conflict zone masking is applied identically to V2: pixels where
    ensemble_prob ≥ threshold AND target=0 are excluded from loss (valid=0).
    This prevents training on areas where the ensemble predicts CWD but no
    manual annotation exists, avoiding contradictory supervision signals.
    """
    if splitter is None:
        splitter = SpatialCVSplitterV3()

    with rasterio.open(chm_tif) as src:
        H, W = src.height, src.width

    entries: list[PatchEntry] = []
    ys = _grid_positions(H, patch_size, stride)
    xs = _grid_positions(W, patch_size, stride)

    with rasterio.open(mask_tif) as msrc:
        for y0 in ys:
            for x0 in xs:
                window = Window(x0, y0, patch_size, patch_size)
                raw = msrc.read(
                    [1, 2, 3], window=window, boundless=True, fill_value=0.0
                )
                target_p = raw[0].astype(np.float32)
                valid_base = raw[1].astype(np.float32)
                ensemble_prob = raw[2].astype(np.float32)

                conflict_mask = (ensemble_prob >= conflict_threshold) & (target_p < 0.5)
                valid_after_conflict = valid_base * (~conflict_mask).astype(np.float32)

                n_valid = int((valid_after_conflict > 0.5).sum())
                if n_valid < min_valid_px:
                    continue

                n_pos = int(((target_p > 0.5) & (valid_after_conflict > 0.5)).sum())
                stripe = splitter.stripe_of(x0, patch_size)
                fold_id = splitter.fold_id_of(stripe)

                entries.append(PatchEntry(
                    row_off=y0, col_off=x0,
                    stripe_id=stripe, fold_id=fold_id,
                    n_valid=n_valid, n_positive=n_pos,
                ))

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
                row_off=int(r["row_off"]), col_off=int(r["col_off"]),
                stripe_id=int(r["stripe_id"]), fold_id=int(r["fold_id"]),
                n_valid=int(r["n_valid"]), n_positive=int(r["n_positive"]),
            )
            for r in reader
        ]


# ---------------------------------------------------------------------------
# CWDSegDataset — PyTorch Dataset
# ---------------------------------------------------------------------------


class CWDSegDataset(Dataset):
    """Patch dataset for CWD semantic segmentation (V3).

    Identical to V2 CWDSegDataset except it uses the V3 splitter constants.
    The NodataDropout augmentation is applied during training (p=0.4, 5–15% drop).

    Returns:
        image:  Tensor (C, 256, 256) float32, normalized
        target: Tensor (1, 256, 256) float32, {0, 1}
        valid:  Tensor (1, 256, 256) float32, {0, 1}
    """

    def __init__(
        self,
        entries: list[PatchEntry],
        chm_tif: Path,
        mask_tif: Path,
        band_stats: dict,
        patch_size: int = PATCH_SIZE,
        in_channels: int = 4,
        augment: bool = False,
        aug_mode: str = "full",
        buffer_px: int = BUFFER_PX,
        stripe_width: int = STRIPE_WIDTH,
        val_stripe: int | None = None,
        variant: str = DEFAULT_VARIANT,
    ) -> None:
        self.entries = entries
        self.chm_tif = chm_tif
        self.mask_tif = mask_tif
        self.band_stats = band_stats
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.augment = augment
        self.aug_mode = aug_mode
        self.buffer_px = buffer_px
        self.stripe_width = stripe_width
        self.val_stripe = val_stripe
        self.variant = variant
        self.binary_bands = _get_binary_bands(variant)

        self._aug = None
        if augment:
            if aug_mode == "none":
                self._aug = None
            elif aug_mode == "geometric":
                from common.augmentation import get_geometric_aug
                self._aug = get_geometric_aug()
            elif aug_mode == "full":
                from common.augmentation import get_full_aug
                self._aug = get_full_aug()
            else:
                raise ValueError(f"Unknown aug_mode '{aug_mode}'. Expected one of: none, geometric, full")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        ps = self.patch_size

        img = read_multiband_window(self.chm_tif, e.row_off, e.col_off, ps)

        # For masked variant: replace Band 2 (constant 255 from CHM) with actual validity mask from phase1
        if self.variant == "masked" and img.shape[0] >= 2:
            with rasterio.open(self.mask_tif) as msrc:
                mask_band2 = msrc.read(
                    2, window=Window(e.col_off, e.row_off, ps, ps),
                    boundless=True, fill_value=0.0
                ).astype(np.float32)
            img[1] = mask_band2  # Replace Band 2 with actual validity from phase1

        # For composite variant: replace Band 4 (constant 255 from CHM) with actual validity mask from phase1
        if self.variant == "composite" and img.shape[0] >= 4:
            with rasterio.open(self.mask_tif) as msrc:
                mask_band2 = msrc.read(
                    2, window=Window(e.col_off, e.row_off, ps, ps),
                    boundless=True, fill_value=0.0
                ).astype(np.float32)
            img[3] = mask_band2  # Replace Band 4 with actual validity from phase1

        if img.shape[0] > self.in_channels:
            img = img[: self.in_channels]
        elif img.shape[0] < self.in_channels:
            pad = np.zeros((self.in_channels - img.shape[0], ps, ps), dtype=np.float32)
            img = np.concatenate([img, pad], axis=0)

        img = normalize_bands(img, self.band_stats, binary_bands=self.binary_bands)

        with rasterio.open(self.mask_tif) as msrc:
            raw = msrc.read(
                [1, 2, 3],
                window=Window(e.col_off, e.row_off, ps, ps),
                boundless=True, fill_value=0.0,
            ).astype(np.float32)
        target = raw[0]
        valid = raw[1]
        ensemble_prob = raw[2]

        conflict_mask = (ensemble_prob >= CONFLICT_ENSEMBLE_THRESHOLD) & (target < 0.5)
        valid = valid * (~conflict_mask).astype(np.float32)

        if self.val_stripe is not None:
            valid = self._apply_buffer(valid, e.col_off, ps)

        if self.augment and self._aug is not None:
            img_hwc = img.transpose(1, 2, 0)
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
        valid = valid.copy()
        assert self.val_stripe is not None
        left_boundary = self.val_stripe * self.stripe_width
        right_boundary = left_boundary + self.stripe_width
        for boundary in (left_boundary, right_boundary):
            for col in range(max(0, boundary - self.buffer_px - col_off),
                             min(ps, boundary + self.buffer_px - col_off)):
                if 0 <= col < ps:
                    valid[:, col] = 0.0
        return valid


def make_weighted_sampler(entries: list[PatchEntry], pos_weight: float = 3.0) -> WeightedRandomSampler:
    weights = [pos_weight if e.n_positive > 0 else 1.0 for e in entries]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase II: Dataset with conflict masking and balanced CV")
    p.add_argument(
        "--chm-variant", type=str, default=DEFAULT_VARIANT,
        choices=["baseline", "raw", "gauss", "masked", "composite"],
    )
    p.add_argument(
        "--mask-tif", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase1_masks" / "406455_2021_tava_truemask.tif",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase2_dataset_v3",
    )
    p.add_argument(
        "--cv-version", type=int, default=4, choices=[3, 4],
        help="Cross-validation strategy: 3=V3 (imbalanced 4-fold), 4=V4 (balanced 2-fold, recommended)"
    )
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--min-valid-px", type=int, default=MIN_VALID_PX)
    p.add_argument("--validate", action="store_true", help="Print fold stats and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chm_tif = _read_chm_path(args.chm_variant, ROOT)
    if not chm_tif.exists():
        print(f"ERROR: CHM file not found: {chm_tif}", file=sys.stderr)
        sys.exit(1)

    # Select CV strategy
    if args.cv_version == 4:
        splitter = SpatialCVSplitterV4()
        n_folds = 2
        print(f"Using balanced 2-fold CV (V4)")
    else:
        splitter = SpatialCVSplitterV3()
        n_folds = N_STRIPES - 1
        print(f"Using imbalanced 4-fold CV (V3)")

    in_channels = _get_in_channels(args.chm_variant)

    index_path = args.output_dir / f"patch_index_{args.chm_variant}.csv"
    stats_path = args.output_dir / f"band_stats_{args.chm_variant}.json"

    if index_path.exists() and not args.validate:
        print(f"Loading existing patch index: {index_path}")
        entries = load_patch_index(index_path)
    else:
        print(f"Building patch index for variant '{args.chm_variant}'…")
        entries = build_patch_index(
            chm_tif=chm_tif, mask_tif=args.mask_tif,
            patch_size=args.patch_size, stride=args.stride,
            min_valid_px=args.min_valid_px, splitter=splitter,
        )
        save_patch_index(entries, index_path)
        print(f"Saved {len(entries):,} patches → {index_path}")

    test_e = splitter.test_entries(entries)
    print(f"\nVariant: {args.chm_variant}  (in_channels={in_channels})")
    print(f"Total patches:  {len(entries):,}")
    print(f"Test (stripe 0): {len(test_e):,} ({sum(1 for e in test_e if e.n_positive>0):,} positive)")
    for fold in range(n_folds):
        train_e, val_e = splitter.train_val_split(entries, fold)
        n_pos_tr = sum(1 for e in train_e if e.n_positive > 0)
        n_pos_va = sum(1 for e in val_e if e.n_positive > 0)
        ratio = len(train_e) / len(val_e) if len(val_e) > 0 else 0
        print(
            f"Fold {fold}: train={len(train_e):,} (pos={n_pos_tr:,})  "
            f"val={len(val_e):,} (pos={n_pos_va:,})  ratio={ratio:.2f}"
        )

    if args.validate:
        return

    if stats_path.exists():
        print(f"\nLoading existing band stats: {stats_path}")
        band_stats = json.loads(stats_path.read_text())
    else:
        print("\nComputing band statistics (excluding conflict zones)…")
        with rasterio.open(args.mask_tif) as msrc:
            raw = msrc.read([1, 2, 3])
            target_band = raw[0].astype(np.float32)
            valid_band = raw[1].astype(bool)
            ensemble_prob = raw[2].astype(np.float32)
            conflict_mask = (ensemble_prob >= CONFLICT_ENSEMBLE_THRESHOLD) & (target_band < 0.5)
            valid_after_conflict = valid_band & ~conflict_mask

        # For composite variant, exclude band 4 (mask) from stats computation
        # The mask band is constant [0, 255] and handled separately (clipped to [0, 1] as binary)
        if args.chm_variant == "composite":
            band_stats = compute_band_stats(chm_tif, valid_mask=valid_after_conflict, bands=[1, 2, 3])
            # Add dummy stats for band 4 (0-indexed 3) - will be clipped to [0, 1] via binary_bands
            band_stats["3"] = {"mean": 127.5, "std": 1.0, "p2": 0.0, "p98": 255.0}
        else:
            band_stats = compute_band_stats(chm_tif, valid_mask=valid_after_conflict)

        stats_path.write_text(json.dumps(band_stats, indent=2))
        print(f"Saved band stats → {stats_path}")
        for i, s in band_stats.items():
            print(f"  Band {i}: mean={s['mean']:.4f}  std={s['std']:.4f}")


if __name__ == "__main__":
    main()

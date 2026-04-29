#!/usr/bin/env python3
"""Complete object-oriented CWD pipeline with strict geospatial splits.

This module provides a single-file implementation for training and inference of
a Partial Convolution U-Net that reconstructs CHM heights and segments coarse
woody debris (CWD). The implementation is designed for sparse ALS-derived CHM
inputs and uses pseudo labels from legacy high-F1 tile classifiers.

Sections
--------
1. Data loading, strict spatial-temporal splits, and preprocessing.
2. PartialConv U-Net architecture and legacy hotspot integration.
3. Reconstruction/segmentation losses and joint objective.
4. Training loop with curriculum masking and checkpointing.
5. Inference with MC Dropout uncertainty quantification.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tvm

try:
    from scipy.ndimage import binary_erosion
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    _HAS_BOUNDARY_METRIC_DEPS = True
except Exception:  # pragma: no cover - optional dependency fallback
    binary_erosion = None
    distance_transform_edt = None
    skeletonize = None
    _HAS_BOUNDARY_METRIC_DEPS = False


# ============================================================================
# Module constants
# ============================================================================

TILE_SIZE = 256
CHUNK_SIZE = 128
OVERLAP = 0.5
NODATA_THRESHOLD = 0.0
LOG_EVERY_STEPS = 10
DEFAULT_BUFFER_METERS = 50.0
EPS = 1e-8

DEFAULT_DATA_DIR = Path("output/chm_dataset_harmonized_0p8m_raw_gauss")
DEFAULT_REGISTRY_DIR = Path("registry/chm_dataset_harmonized_0p8m_raw_gauss")
DEFAULT_OUTPUT_DIR = Path("output/cwd_partialconv")
DEFAULT_LEGACY_CHECKPOINT = Path(
    "output/model_search_v3_academic_leakage26/final_models/convnext_small_full_ce_mixup.pt"
)

_MAP_YEAR_RE = re.compile(r"(?P<mapsheet>\d{6})_(?P<year>\d{4})")
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


# ============================================================================
# Data containers
# ============================================================================


@dataclass(slots=True)
class SampleRecord:
    """Container for one CHM raster variant entry.

    Parameters
    ----------
    sample_id : str
        Canonical sample identifier such as ``436646_2022_madal``.
    mapsheet : str
        Six-digit mapsheet code.
    year : int
        Acquisition year.
    variant : str
        CHM variant name, typically ``raw`` or ``gauss``.
    raster_path : Path
        Path to the CHM GeoTIFF used as model input.
    label_path : Path | None
        Path to tile-level label CSV if available.
    has_label : bool
        Whether labels are expected for this sample.
    """

    sample_id: str
    mapsheet: str
    year: int
    variant: str
    raster_path: Path
    label_path: Optional[Path]
    has_label: bool

    def label_raster_name(self) -> str:
        """Return canonical label CSV ``raster`` field value.

        Returns
        -------
        str
            Baseline CHM raster name used by tile label CSV rows.
        """
        return f"{self.sample_id}_chm_max_hag_20cm.tif"


@dataclass(slots=True)
class TileRecord:
    """Container for one tile sample extracted from a raster.

    Parameters
    ----------
    tile_id : str
        Unique tile identifier.
    sample_id : str
        Parent sample identifier.
    mapsheet : str
        Six-digit mapsheet code.
    year : int
        Acquisition year.
    variant : str
        CHM variant name.
    raster_path : Path
        CHM GeoTIFF path.
    label_path : Path | None
        Label CSV path.
    label_raster_name : str
        Raster key used in tile label CSV.
    row_off : int
        Top-left row offset in source raster.
    col_off : int
        Top-left column offset in source raster.
    chunk_size : int
        Native tile size before padding.
    center_x : float
        Tile center x coordinate in map CRS.
    center_y : float
        Tile center y coordinate in map CRS.
    place_key : str
        Spatial identity key shared across years to prevent temporal leakage.
    """

    tile_id: str
    sample_id: str
    mapsheet: str
    year: int
    variant: str
    raster_path: Path
    label_path: Optional[Path]
    label_raster_name: str
    row_off: int
    col_off: int
    chunk_size: int
    center_x: float
    center_y: float
    place_key: str


@dataclass(slots=True)
class SplitResult:
    """Return object for strict train/val/test split generation.

    Parameters
    ----------
    train : list[TileRecord]
        Training tiles.
    val : list[TileRecord]
        Validation tiles.
    test : list[TileRecord]
        Test tiles.
    discarded : list[TileRecord]
        Tiles removed by split buffer.
    metadata : dict[str, Any]
        Split diagnostics and leakage-prevention statistics.
    """

    train: list[TileRecord]
    val: list[TileRecord]
    test: list[TileRecord]
    discarded: list[TileRecord]
    metadata: dict[str, Any]


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for training loop and optimization.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    num_workers : int
        DataLoader worker count.
    lr : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    eta_min : float
        Minimum cosine scheduler learning rate.
    recon_weight : float
        Weight for reconstruction loss term.
    seg_weight : float
        Weight for segmentation loss term.
    max_grad_norm : float
        Gradient clipping norm.
    eval_threshold_sweep_min : float
        Minimum threshold for validation F1 sweep.
    eval_threshold_sweep_max : float
        Maximum threshold for validation F1 sweep.
    eval_threshold_sweep_step : float
        Step size for validation F1 threshold sweep.
    eval_threshold_sweep_enabled : bool
        Whether to compute best-threshold validation F1 each epoch.
    early_stopping_patience : int
        Stop if monitored metric fails to improve for this many epochs.
    early_stopping_min_delta : float
        Minimum improvement required to reset early-stopping patience.
    monitor_metric : str
        Validation metric used for model selection and early stopping.
    eval_metric_threshold : float
        Probability threshold used for Dice/IoU/clDice/HD95 binarization.
    boundary_metric_max_samples : int
        Maximum labeled validation tiles used per epoch for boundary metrics.
    seg_target_source : str
        Segmentation supervision source. One of ``labels``, ``hotspot``,
        or ``labels_or_hotspot``.
    hotspot_conf_min : float
        Minimum confidence floor used for hotspot-derived supervision.
    hotspot_conf_gamma : float
        Exponent controlling hotspot-confidence sharpness.
    """

    epochs: int = 40
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    eta_min: float = 1e-6
    recon_weight: float = 1.0
    seg_weight: float = 5.0
    max_grad_norm: float = 5.0
    eval_threshold_sweep_min: float = 0.1
    eval_threshold_sweep_max: float = 0.5
    eval_threshold_sweep_step: float = 0.05
    eval_threshold_sweep_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    monitor_metric: str = "sota_score"
    eval_metric_threshold: float = 0.5
    boundary_metric_max_samples: int = 128
    seg_target_source: str = "labels"
    hotspot_conf_min: float = 0.05
    hotspot_conf_gamma: float = 1.0


# ============================================================================
# Utilities
# ============================================================================


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging for training and CLI execution.

    Parameters
    ----------
    verbose : bool, default=False
        If ``True``, uses debug logging level.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s: %(message)s")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for deterministic runs.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_mapsheet_year(text: str) -> tuple[str, int]:
    """Parse mapsheet and year from a filename-like string.

    Parameters
    ----------
    text : str
        Input text containing ``<mapsheet>_<year>``.

    Returns
    -------
    tuple[str, int]
        Parsed mapsheet and year.

    Raises
    ------
    ValueError
        If no mapsheet-year pair can be extracted.
    """
    match = _MAP_YEAR_RE.search(Path(text).name)
    if match is None:
        raise ValueError(f"Cannot parse mapsheet/year from: {text}")
    return match.group("mapsheet"), int(match.group("year"))


def centered_pad(array: np.ndarray, target_size: int = TILE_SIZE) -> np.ndarray:
    """Zero-pad a 2-D array to a centered square of ``target_size``.

    Parameters
    ----------
    array : np.ndarray
        Input 2-D array.
    target_size : int, default=TILE_SIZE
        Output side length.

    Returns
    -------
    np.ndarray
        Zero-padded output array.
    """
    h, w = array.shape
    out = np.zeros((target_size, target_size), dtype=array.dtype)
    if h > target_size or w > target_size:
        return array[:target_size, :target_size].copy()
    top = (target_size - h) // 2
    left = (target_size - w) // 2
    out[top : top + h, left : left + w] = array
    return out


def tile_id_to_artifact_stem(tile_id: str) -> str:
    """Convert a tile id to a filesystem-safe artifact stem.

    Parameters
    ----------
    tile_id : str
        Tile id value.

    Returns
    -------
    str
        Safe stem for filenames.
    """
    text = _SAFE_STEM_RE.sub("_", str(tile_id)).strip("_")
    return text or "tile"


def cam_to_binary_mask(cam_map: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert a normalized CAM map to a binary mask via Otsu thresholding.

    Parameters
    ----------
    cam_map : np.ndarray
        Continuous CAM map expected in ``[0,1]``.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(binary_mask, threshold_01)``.
    """
    cam = np.clip(np.asarray(cam_map, dtype=np.float32), 0.0, 1.0)
    cam_u8 = np.clip(np.rint(cam * 255.0), 0.0, 255.0).astype(np.uint8)

    # Otsu auto-threshold for each tile.
    threshold_u8, binary_u8 = cv2.threshold(
        cam_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    binary = (binary_u8 > 0).astype(np.float32)
    return binary, float(threshold_u8 / 255.0)


def hotspot_to_confidence_map(
    hotspot: torch.Tensor,
    min_conf: float = 0.05,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Convert hotspot activations into confidence weights for loss.

    Parameters
    ----------
    hotspot : torch.Tensor
        Hotspot map in ``[0,1]``.
    min_conf : float, default=0.05
        Minimum confidence floor.
    gamma : float, default=1.0
        Certainty curve exponent.

    Returns
    -------
    torch.Tensor
        Confidence map in ``[min_conf, 1]``.

    Notes
    -----
    Confidence is highest for values near 0 or 1 (more certain negative/positive)
    and lowest near 0.5 (ambiguous hotspots).
    """
    min_conf_f = float(np.clip(min_conf, 0.0, 1.0))
    gamma_f = float(max(gamma, EPS))

    hotspot_01 = torch.clamp(hotspot, 0.0, 1.0)
    certainty = torch.abs(hotspot_01 - 0.5) * 2.0
    confidence = min_conf_f + (1.0 - min_conf_f) * torch.pow(certainty, gamma_f)
    return torch.clamp(confidence, min_conf_f, 1.0)


def _build_cam_artifact_index(cam_mask_dir: Path) -> set[str]:
    """Index available CAM mask stems in a directory.

    Parameters
    ----------
    cam_mask_dir : Path
        Directory containing ``*_mask.npy`` artifacts.

    Returns
    -------
    set[str]
        Filename stems (without ``_mask.npy`` suffix).
    """
    stems: set[str] = set()
    try:
        for path in cam_mask_dir.glob("*_mask.npy"):
            name = path.name
            if name.endswith("_mask.npy"):
                stems.add(name[: -len("_mask.npy")])
    except Exception as exc:
        logging.warning("Failed to index CAM masks in %s (%s)", cam_mask_dir, exc)
    return stems


def _iter_offsets(size: int, chunk_size: int, overlap: float) -> list[int]:
    """Generate tile offsets covering one raster axis.

    Parameters
    ----------
    size : int
        Raster height or width in pixels.
    chunk_size : int
        Tile side length in pixels.
    overlap : float
        Fractional overlap in ``[0, 1)``.

    Returns
    -------
    list[int]
        Monotonic list of offsets that covers full axis extents.
    """
    if size <= chunk_size:
        return [0]
    stride = max(1, int(chunk_size * (1.0 - overlap)))
    offsets = list(range(0, max(1, size - chunk_size + 1), stride))
    border = size - chunk_size
    if offsets[-1] != border:
        offsets.append(border)
    return offsets


def _resolve_path(path_like: str, roots: Iterable[Path]) -> Path:
    """Resolve a possibly relative path against candidate roots.

    Parameters
    ----------
    path_like : str
        Input path from CSV or CLI.
    roots : Iterable[Path]
        Candidate root folders.

    Returns
    -------
    Path
        Resolved path. Existing path is preferred when available.
    """
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    for root in roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return Path.cwd() / candidate


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely parse float values with a default fallback.

    Parameters
    ----------
    value : Any
        Input value.
    default : float, default=0.0
        Returned fallback when parsing fails.

    Returns
    -------
    float
        Parsed float or fallback.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely parse integer values with a default fallback.

    Parameters
    ----------
    value : Any
        Input value.
    default : int, default=0
        Returned fallback when parsing fails.

    Returns
    -------
    int
        Parsed integer or fallback.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ============================================================================
# Section 1: Data loading, split generation, preprocessing
# ============================================================================


def load_sample_records(
    data_dir: str | Path,
    registry_dir: str | Path | None = None,
    include_variants: tuple[str, ...] = ("raw", "gauss"),
    include_unlabeled: bool = True,
) -> list[SampleRecord]:
    """Load sample records from registry CSVs or local debug manifests.

    Parameters
    ----------
    data_dir : str | Path
        CHM dataset root.
    registry_dir : str | Path | None, default=None
        Registry folder with ``ml_variants.csv`` / ``ml_samples.csv``.
    include_variants : tuple[str, ...], default=("raw", "gauss")
        Variants to load.
    include_unlabeled : bool, default=True
        Whether to include records without tile labels.

    Returns
    -------
    list[SampleRecord]
        Loaded sample records.
    """
    root = Path(data_dir)
    reg_root = Path(registry_dir) if registry_dir is not None else None
    roots = [Path.cwd(), root]
    if reg_root is not None:
        roots.append(reg_root)

    records: list[SampleRecord] = []

    if reg_root is not None and (reg_root / "ml_variants.csv").exists():
        with (reg_root / "ml_variants.csv").open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                variant = str(row.get("variant", "")).strip().lower()
                if variant not in include_variants:
                    continue
                if _safe_int(row.get("tif_exists"), 0) != 1:
                    continue

                sample_id = str(row.get("sample_id", "")).strip()
                if not sample_id:
                    continue

                mapsheet = str(row.get("mapsheet", "")).strip() or parse_mapsheet_year(sample_id)[0]
                year = _safe_int(row.get("year"), parse_mapsheet_year(sample_id)[1])
                raster_path = _resolve_path(str(row.get("tif_path", "")), roots)

                label_path_raw = str(row.get("label_path", "")).strip()
                label_path = _resolve_path(label_path_raw, roots) if label_path_raw else None
                has_label = _safe_int(row.get("label_exists"), 0) == 1 and label_path is not None

                if has_label or include_unlabeled:
                    records.append(
                        SampleRecord(
                            sample_id=sample_id,
                            mapsheet=mapsheet,
                            year=year,
                            variant=variant,
                            raster_path=raster_path,
                            label_path=label_path,
                            has_label=has_label,
                        )
                    )

    elif (root / "reports" / "dataset_manifest.csv").exists():
        manifest = root / "reports" / "dataset_manifest.csv"
        with manifest.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                for variant in include_variants:
                    tif_col = "raw_chm" if variant == "raw" else "gauss_chm"
                    tif_path = str(row.get(tif_col, "")).strip()
                    if not tif_path:
                        continue
                    raster_path = _resolve_path(tif_path, roots)
                    sample_id = f"{row.get('tile', '').strip()}_{row.get('year', '').strip()}_{row.get('campaign', '').strip()}"
                    if not sample_id:
                        continue
                    mapsheet, year = parse_mapsheet_year(sample_id)
                    records.append(
                        SampleRecord(
                            sample_id=sample_id,
                            mapsheet=mapsheet,
                            year=year,
                            variant=variant,
                            raster_path=raster_path,
                            label_path=None,
                            has_label=False,
                        )
                    )

    else:
        for variant in include_variants:
            variant_dir = root / f"chm_{variant}"
            if not variant_dir.exists():
                continue
            for tif_path in sorted(variant_dir.glob("*.tif")):
                try:
                    mapsheet, year = parse_mapsheet_year(tif_path.name)
                except ValueError:
                    continue
                sample_id = tif_path.name.split("_harmonized")[0]
                records.append(
                    SampleRecord(
                        sample_id=sample_id,
                        mapsheet=mapsheet,
                        year=year,
                        variant=variant,
                        raster_path=tif_path,
                        label_path=None,
                        has_label=False,
                    )
                )

    records = [r for r in records if r.raster_path.exists()]
    logging.info("Loaded %d sample records", len(records))
    return records


def _load_label_offsets(label_path: Path, default_chunk_size: int = CHUNK_SIZE) -> list[tuple[int, int, int]]:
    """Read tile offsets from one label CSV.

    Parameters
    ----------
    label_path : Path
        Path to label CSV.
    default_chunk_size : int, default=CHUNK_SIZE
        Fallback chunk size when field is missing.

    Returns
    -------
    list[tuple[int, int, int]]
        Unique ``(row_off, col_off, chunk_size)`` tuples.
    """
    if not label_path.exists():
        return []
    offsets: set[tuple[int, int, int]] = set()
    with label_path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            row_off = _safe_int(row.get("row_off"), 0)
            col_off = _safe_int(row.get("col_off"), 0)
            chunk = _safe_int(row.get("chunk_size"), default_chunk_size)
            offsets.add((row_off, col_off, chunk))
    return sorted(offsets)


def build_tile_index(
    sample_records: list[SampleRecord],
    chunk_size: int = CHUNK_SIZE,
    overlap: float = OVERLAP,
    max_tiles_per_raster: int | None = 2500,
    unlabeled_stride_multiplier: int = 4,
    seed: int = 42,
) -> list[TileRecord]:
    """Build tile-level index used by strict splits and datasets.

    Parameters
    ----------
    sample_records : list[SampleRecord]
        Raster-level sample records.
    chunk_size : int, default=CHUNK_SIZE
        Native tile side length before padding.
    overlap : float, default=OVERLAP
        Tile overlap fraction for unlabeled raster sampling.
    max_tiles_per_raster : int | None, default=2500
        Optional cap to keep indexing/training memory bounded.
    unlabeled_stride_multiplier : int, default=4
        Stride multiplier applied when no labels are available.
    seed : int, default=42
        Random seed for deterministic sub-sampling.

    Returns
    -------
    list[TileRecord]
        Tile index for all selected rasters.
    """
    rng = random.Random(seed)
    tiles: list[TileRecord] = []

    for rec in sample_records:
        if not rec.raster_path.exists():
            continue

        with rasterio.open(rec.raster_path) as src:
            height, width = src.height, src.width
            offsets: list[tuple[int, int, int]] = []

            if rec.has_label and rec.label_path is not None and rec.label_path.exists():
                offsets = _load_label_offsets(rec.label_path, default_chunk_size=chunk_size)

            if not offsets:
                stride = max(1, int(chunk_size * (1.0 - overlap)))
                stride *= max(1, unlabeled_stride_multiplier)
                row_offsets = _iter_offsets(height, chunk_size, overlap=1.0 - (stride / chunk_size))
                col_offsets = _iter_offsets(width, chunk_size, overlap=1.0 - (stride / chunk_size))
                offsets = [(r, c, chunk_size) for r in row_offsets for c in col_offsets]

            if max_tiles_per_raster is not None and len(offsets) > max_tiles_per_raster:
                offsets = rng.sample(offsets, max_tiles_per_raster)

            max_row = max(0, height - chunk_size)
            max_col = max(0, width - chunk_size)

            for row_off, col_off, csize in sorted(offsets):
                row_off = min(max(0, row_off), max_row)
                col_off = min(max(0, col_off), max_col)

                center_row = row_off + (csize / 2.0)
                center_col = col_off + (csize / 2.0)
                center_x, center_y = rasterio.transform.xy(
                    src.transform,
                    center_row,
                    center_col,
                    offset="center",
                )
                cx = float(center_x)
                cy = float(center_y)
                place_key = f"{rec.mapsheet}:{round(cx, 1)}:{round(cy, 1)}"

                tile_id = f"{rec.sample_id}:{rec.variant}:{row_off}:{col_off}"
                tiles.append(
                    TileRecord(
                        tile_id=tile_id,
                        sample_id=rec.sample_id,
                        mapsheet=rec.mapsheet,
                        year=rec.year,
                        variant=rec.variant,
                        raster_path=rec.raster_path,
                        label_path=rec.label_path,
                        label_raster_name=rec.label_raster_name(),
                        row_off=row_off,
                        col_off=col_off,
                        chunk_size=csize,
                        center_x=cx,
                        center_y=cy,
                        place_key=place_key,
                    )
                )

    logging.info("Indexed %d tiles", len(tiles))
    return tiles


def _strict_split_tile_records(
    tile_records: list[TileRecord],
    test_size: float = 0.2,
    val_size: float = 0.1,
    buffer_meters: float = DEFAULT_BUFFER_METERS,
    seed: int = 42,
) -> SplitResult:
    """Split tiles with spatial bisection + temporal leakage prevention.

    Parameters
    ----------
    tile_records : list[TileRecord]
        Tile-level records to split.
    test_size : float, default=0.2
        Target global test proportion.
    val_size : float, default=0.1
        Validation proportion sampled from train by place key.
    buffer_meters : float, default=DEFAULT_BUFFER_METERS
        Spatial buffer around split lines where tiles are discarded.
    seed : int, default=42
        Random seed.

    Returns
    -------
    SplitResult
        Split object with diagnostics.
    """
    if not tile_records:
        raise ValueError("No tile records to split")

    rng = random.Random(seed)
    by_mapsheet: dict[str, list[TileRecord]] = {}
    for tile in tile_records:
        by_mapsheet.setdefault(tile.mapsheet, []).append(tile)

    orientation_by_mapsheet: dict[str, str] = {}
    splitline_by_mapsheet: dict[str, float] = {}
    side_by_tile: dict[str, str] = {}

    discarded: list[TileRecord] = []
    non_buffer_tiles: list[TileRecord] = []

    for mapsheet, group in by_mapsheet.items():
        xs = [tile.center_x for tile in group]
        ys = [tile.center_y for tile in group]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)

        orientation = "x" if x_span >= y_span else "y"
        split_line = (max(xs) + min(xs)) / 2.0 if orientation == "x" else (max(ys) + min(ys)) / 2.0

        orientation_by_mapsheet[mapsheet] = orientation
        splitline_by_mapsheet[mapsheet] = split_line

        for tile in group:
            coord = tile.center_x if orientation == "x" else tile.center_y
            if abs(coord - split_line) <= buffer_meters:
                discarded.append(tile)
                continue
            side = "low" if coord < split_line else "high"
            side_by_tile[tile.tile_id] = side
            non_buffer_tiles.append(tile)

    counts_by_mapsheet_side: dict[tuple[str, str], int] = {}
    for tile in non_buffer_tiles:
        side = side_by_tile[tile.tile_id]
        key = (tile.mapsheet, side)
        counts_by_mapsheet_side[key] = counts_by_mapsheet_side.get(key, 0) + 1

    candidate_side_by_mapsheet: dict[str, str] = {}
    candidate_count_by_mapsheet: dict[str, int] = {}
    for mapsheet in by_mapsheet.keys():
        low_count = counts_by_mapsheet_side.get((mapsheet, "low"), 0)
        high_count = counts_by_mapsheet_side.get((mapsheet, "high"), 0)
        if low_count == 0 and high_count == 0:
            continue
        if low_count == 0:
            chosen = "high"
        elif high_count == 0:
            chosen = "low"
        else:
            chosen = "low" if low_count <= high_count else "high"
        candidate_side_by_mapsheet[mapsheet] = chosen
        candidate_count_by_mapsheet[mapsheet] = counts_by_mapsheet_side.get((mapsheet, chosen), 0)

    target_test = max(1, int(len(non_buffer_tiles) * test_size))
    mapsheets = list(candidate_side_by_mapsheet.keys())
    rng.shuffle(mapsheets)
    mapsheets.sort(key=lambda m: candidate_count_by_mapsheet[m])

    selected_test_mapsheets: set[str] = set()
    selected_count = 0
    for mapsheet in mapsheets:
        if selected_count >= target_test and selected_test_mapsheets:
            break
        selected_test_mapsheets.add(mapsheet)
        selected_count += candidate_count_by_mapsheet[mapsheet]

    train_base: list[TileRecord] = []
    test: list[TileRecord] = []
    for tile in non_buffer_tiles:
        mapsheet = tile.mapsheet
        side = side_by_tile[tile.tile_id]
        if mapsheet in selected_test_mapsheets and side == candidate_side_by_mapsheet[mapsheet]:
            test.append(tile)
        else:
            train_base.append(tile)

    test_place_keys = {tile.place_key for tile in test}
    train_no_temporal_leak = [tile for tile in train_base if tile.place_key not in test_place_keys]
    temporal_pruned = len(train_base) - len(train_no_temporal_leak)

    train_keys = sorted({tile.place_key for tile in train_no_temporal_leak})
    rng.shuffle(train_keys)
    n_val_keys = max(1, int(len(train_keys) * val_size)) if train_keys else 0
    val_keys = set(train_keys[:n_val_keys])

    val = [tile for tile in train_no_temporal_leak if tile.place_key in val_keys]
    train = [tile for tile in train_no_temporal_leak if tile.place_key not in val_keys]

    metadata = {
        "total_tiles": len(tile_records),
        "non_buffer_tiles": len(non_buffer_tiles),
        "discarded_by_buffer": len(discarded),
        "target_test": target_test,
        "actual_test": len(test),
        "temporal_pruned_from_train": temporal_pruned,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "selected_test_mapsheets": sorted(selected_test_mapsheets),
        "buffer_meters": buffer_meters,
    }

    return SplitResult(train=train, val=val, test=test, discarded=discarded, metadata=metadata)


def create_strict_splits(
    data_dir: str | Path,
    test_size: float = 0.2,
    buffer_meters: float = DEFAULT_BUFFER_METERS,
    registry_dir: str | Path | None = None,
    val_size: float = 0.1,
    max_tiles_per_raster: int | None = 2500,
    seed: int = 42,
) -> SplitResult:
    """Create strict spatial-temporal train/val/test splits from CHM data.

    Parameters
    ----------
    data_dir : str | Path
        CHM dataset root directory.
    test_size : float, default=0.2
        Target global test proportion.
    buffer_meters : float, default=DEFAULT_BUFFER_METERS
        Buffer around split line to avoid overlap leakage.
    registry_dir : str | Path | None, default=None
        Optional registry folder with ``ml_variants.csv``.
    val_size : float, default=0.1
        Validation proportion sampled from train split.
    max_tiles_per_raster : int | None, default=2500
        Optional per-raster cap to limit tile indexing volume.
    seed : int, default=42
        Random seed.

    Returns
    -------
    SplitResult
        Strict split results.
    """
    samples = load_sample_records(
        data_dir=data_dir,
        registry_dir=registry_dir,
        include_variants=("raw", "gauss"),
        include_unlabeled=True,
    )
    tiles = build_tile_index(
        sample_records=samples,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        max_tiles_per_raster=max_tiles_per_raster,
        unlabeled_stride_multiplier=4,
        seed=seed,
    )
    split = _strict_split_tile_records(
        tiles,
        test_size=test_size,
        val_size=val_size,
        buffer_meters=buffer_meters,
        seed=seed,
    )
    logging.info("Strict split metadata: %s", json.dumps(split.metadata, indent=2))
    return split


class PseudoLabelStore:
    """Tile-level pseudo-label and confidence lookup.

    The loader supports existing CSV schema with fields:
    ``raster,row_off,col_off,chunk_size,label,source,model_prob,...``.

    Parameters
    ----------
    default_confidence : float, default=0.5
        Fallback confidence when explicit model probability is missing.
    """

    def __init__(self, default_confidence: float = 0.5) -> None:
        """Initialize an empty pseudo-label store.

        Parameters
        ----------
        default_confidence : float, default=0.5
            Fallback confidence.
        """
        self.default_confidence = float(default_confidence)
        self._labels: dict[tuple[str, int, int, int], tuple[float, float, str]] = {}

    @staticmethod
    def _source_rank(source: str) -> int:
        """Map label source tags to deterministic priority ranks.

        Parameters
        ----------
        source : str
            Source name.

        Returns
        -------
        int
            Rank where higher means preferred.
        """
        if source.startswith("self_train"):
            return 2
        lookup = {
            "manual": 4,
            "auto_reviewed": 3,
            "auto": 2,
            "auto_skip": 1,
            "": 0,
        }
        return lookup.get(source, 0)

    @staticmethod
    def _to_binary_label(label_text: str) -> Optional[float]:
        """Convert label text to binary value.

        Parameters
        ----------
        label_text : str
            Label string from CSV.

        Returns
        -------
        float | None
            ``1.0`` for CWD, ``0.0`` for non-CWD, ``None`` for unknown labels.
        """
        text = label_text.strip().lower()
        if text in {"cdw", "1", "true", "yes"}:
            return 1.0
        if text in {"no_cdw", "0", "false", "no"}:
            return 0.0
        return None

    def _estimate_confidence(self, row: dict[str, str], label_value: float, source: str) -> float:
        """Estimate confidence for one tile label row.

        Parameters
        ----------
        row : dict[str, str]
            CSV row dictionary.
        label_value : float
            Binary label value.
        source : str
            Label source tag.

        Returns
        -------
        float
            Confidence in ``[0, 1]``.
        """
        model_prob = row.get("model_prob")
        if model_prob not in (None, ""):
            p = float(np.clip(_safe_float(model_prob, self.default_confidence), 0.0, 1.0))
            conf = p if label_value >= 0.5 else 1.0 - p
            return float(np.clip(conf, 0.05, 1.0))

        source_defaults = {
            "manual": 1.0,
            "auto_reviewed": 0.9,
            "auto": 0.75,
            "auto_skip": 0.2,
        }
        return float(source_defaults.get(source, self.default_confidence))

    def add_label_csv(self, label_path: Path) -> None:
        """Load one label CSV into the store.

        Parameters
        ----------
        label_path : Path
            Path to label CSV.
        """
        if not label_path.exists():
            return

        with label_path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                label_value = self._to_binary_label(str(row.get("label", "")))
                if label_value is None:
                    continue

                source = str(row.get("source", "")).strip().lower()
                raster_name = str(row.get("raster", "")).strip()
                row_off = _safe_int(row.get("row_off"), 0)
                col_off = _safe_int(row.get("col_off"), 0)
                chunk = _safe_int(row.get("chunk_size"), CHUNK_SIZE)
                conf = self._estimate_confidence(row, label_value, source)

                key = (raster_name, row_off, col_off, chunk)
                current = self._labels.get(key)
                if current is None:
                    self._labels[key] = (label_value, conf, source)
                    continue

                _, _, current_source = current
                if self._source_rank(source) >= self._source_rank(current_source):
                    self._labels[key] = (label_value, conf, source)

    @classmethod
    def from_tiles(cls, tiles: list[TileRecord], default_confidence: float = 0.5) -> "PseudoLabelStore":
        """Create label store from tile records.

        Parameters
        ----------
        tiles : list[TileRecord]
            Tiles containing label CSV paths.
        default_confidence : float, default=0.5
            Fallback confidence.

        Returns
        -------
        PseudoLabelStore
            Populated store.
        """
        store = cls(default_confidence=default_confidence)
        unique_paths = sorted({tile.label_path for tile in tiles if tile.label_path is not None})
        for path in unique_paths:
            if path is not None:
                store.add_label_csv(path)
        logging.info("Loaded pseudo-label entries: %d", len(store._labels))
        return store

    def get_tile_targets(self, tile: TileRecord, target_size: int = TILE_SIZE) -> tuple[np.ndarray, np.ndarray, bool]:
        """Get padded label and confidence maps for one tile.

        Parameters
        ----------
        tile : TileRecord
            Tile descriptor.
        target_size : int, default=TILE_SIZE
            Output target map size.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, bool]
            ``(label_map, confidence_map, has_label)`` each shaped ``(1,H,W)``.
        """
        key = (tile.label_raster_name, tile.row_off, tile.col_off, tile.chunk_size)
        value = self._labels.get(key)
        if value is None:
            zeros = np.zeros((1, target_size, target_size), dtype=np.float32)
            return zeros, zeros.copy(), False

        label_value, conf_value, _ = value
        label_patch = np.full((tile.chunk_size, tile.chunk_size), label_value, dtype=np.float32)
        conf_patch = np.full((tile.chunk_size, tile.chunk_size), conf_value, dtype=np.float32)

        label_map = centered_pad(label_patch, target_size=target_size)[None, ...]
        conf_map = centered_pad(conf_patch, target_size=target_size)[None, ...]
        return label_map, conf_map, True

    def has_tile_label(self, tile: TileRecord) -> bool:
        """Check whether a tile already has a label entry.

        Parameters
        ----------
        tile : TileRecord
            Tile descriptor.

        Returns
        -------
        bool
            ``True`` when the tile key exists in the store.
        """
        key = (tile.label_raster_name, tile.row_off, tile.col_off, tile.chunk_size)
        return key in self._labels

    def get_tile_label(self, tile: TileRecord) -> Optional[tuple[float, float, str]]:
        """Return raw label tuple for one tile when available.

        Parameters
        ----------
        tile : TileRecord
            Tile descriptor.

        Returns
        -------
        tuple[float, float, str] | None
            ``(label_value, confidence, source)`` when present.
        """
        key = (tile.label_raster_name, tile.row_off, tile.col_off, tile.chunk_size)
        return self._labels.get(key)

    def set_tile_label(
        self,
        tile: TileRecord,
        label_value: float,
        confidence: float,
        source: str,
    ) -> None:
        """Insert or update one tile label entry.

        Parameters
        ----------
        tile : TileRecord
            Tile descriptor.
        label_value : float
            Binary label value where values >= 0.5 map to ``1.0``.
        confidence : float
            Confidence score in ``[0, 1]``.
        source : str
            Source tag used for deterministic source-priority replacement.
        """
        key = (tile.label_raster_name, tile.row_off, tile.col_off, tile.chunk_size)
        label = 1.0 if float(label_value) >= 0.5 else 0.0
        conf = float(np.clip(confidence, 0.0, 1.0))
        src = str(source).strip().lower()

        current = self._labels.get(key)
        if current is None:
            self._labels[key] = (label, conf, src)
            return

        _, _, current_source = current
        if self._source_rank(src) >= self._source_rank(current_source):
            self._labels[key] = (label, conf, src)


class CHMDataset(Dataset):
    """Dataset that returns CHM + valid-mask inputs with pseudo-label targets.

    Parameters
    ----------
    tiles : list[TileRecord]
        Tile index to expose.
    label_store : PseudoLabelStore
        Label and confidence lookup.
    nodata_threshold : float, default=NODATA_THRESHOLD
        Values lower than threshold are treated as nodata.
    augment : bool, default=False
        Whether to apply geometric augmentations.
    cam_mask_dir : Path | None, default=None
        Directory containing offline CAM supervision artifacts.
    cam_mask_index : set[str] | None, default=None
        Optional pre-indexed CAM mask stems to avoid repeated directory scans.
    """

    def __init__(
        self,
        tiles: list[TileRecord],
        label_store: PseudoLabelStore,
        nodata_threshold: float = NODATA_THRESHOLD,
        augment: bool = False,
        cam_mask_dir: Optional[Path] = None,
        cam_mask_index: Optional[set[str]] = None,
    ) -> None:
        """Initialize dataset.

        Parameters
        ----------
        tiles : list[TileRecord]
            Tile list.
        label_store : PseudoLabelStore
            Label store.
        nodata_threshold : float, default=NODATA_THRESHOLD
            Nodata threshold.
        augment : bool, default=False
            Augmentation flag.
        cam_mask_dir : Path | None, default=None
            Optional directory with ``*_mask.npy`` and ``*_cam.npy`` artifacts
            generated from offline Grad-CAM++ + Otsu.
        cam_mask_index : set[str] | None, default=None
            Optional pre-indexed CAM mask stems.
        """
        self.tiles = tiles
        self.label_store = label_store
        self.nodata_threshold = float(nodata_threshold)
        self.augment = bool(augment)
        self.cam_mask_dir = Path(cam_mask_dir) if cam_mask_dir is not None else None
        self._cam_mask_stems: Optional[set[str]] = None
        if self.cam_mask_dir is not None:
            if cam_mask_index is not None:
                self._cam_mask_stems = set(cam_mask_index)
            else:
                self._cam_mask_stems = _build_cam_artifact_index(self.cam_mask_dir)

    def __len__(self) -> int:
        """Return dataset length.

        Returns
        -------
        int
            Number of tiles.
        """
        return len(self.tiles)

    def _read_tile(self, tile: TileRecord) -> tuple[np.ndarray, np.ndarray]:
        """Read one CHM chunk and valid mask from raster.

        Parameters
        ----------
        tile : TileRecord
            Tile descriptor.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(chm_padded, valid_mask_padded)`` each shaped ``(H, W)``.
        """
        with rasterio.open(tile.raster_path) as src:
            window = Window(tile.col_off, tile.row_off, tile.chunk_size, tile.chunk_size)
            arr = src.read(1, window=window, boundless=True, fill_value=np.nan).astype(np.float32)
            if arr.shape != (tile.chunk_size, tile.chunk_size):
                fixed = np.full((tile.chunk_size, tile.chunk_size), np.nan, dtype=np.float32)
                h = min(tile.chunk_size, arr.shape[0])
                w = min(tile.chunk_size, arr.shape[1])
                fixed[:h, :w] = arr[:h, :w]
                arr = fixed

            invalid = ~np.isfinite(arr)
            if src.nodata is not None:
                invalid |= np.isclose(arr, float(src.nodata))
            invalid |= arr < self.nodata_threshold

            chm = arr.copy()
            chm[invalid] = 0.0
            valid = (~invalid).astype(np.float32)

        chm_padded = centered_pad(chm, target_size=TILE_SIZE)
        valid_padded = centered_pad(valid, target_size=TILE_SIZE)
        return chm_padded, valid_padded

    @staticmethod
    def _augment_pair(
        input_map: np.ndarray,
        label_map: np.ndarray,
        conf_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply synchronized geometric augmentations.

        Parameters
        ----------
        input_map : np.ndarray
            Input tensor map shaped ``(2,H,W)``.
        label_map : np.ndarray
            Label map shaped ``(1,H,W)``.
        conf_map : np.ndarray
            Confidence map shaped ``(1,H,W)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Augmented arrays in the same shape as inputs.
        """
        if random.random() < 0.5:
            input_map = input_map[:, :, ::-1].copy()
            label_map = label_map[:, :, ::-1].copy()
            conf_map = conf_map[:, :, ::-1].copy()
        if random.random() < 0.5:
            input_map = input_map[:, ::-1, :].copy()
            label_map = label_map[:, ::-1, :].copy()
            conf_map = conf_map[:, ::-1, :].copy()
        if random.random() < 0.75:
            k = random.randint(1, 3)
            input_map = np.rot90(input_map, k=k, axes=(1, 2)).copy()
            label_map = np.rot90(label_map, k=k, axes=(1, 2)).copy()
            conf_map = np.rot90(conf_map, k=k, axes=(1, 2)).copy()
        return input_map, label_map, conf_map

    @staticmethod
    def _resize_supervision_map(
        arr: np.ndarray,
        tile_size: int,
        interpolation: int,
    ) -> np.ndarray:
        """Resize or pad a supervision map to tile output resolution.

        Parameters
        ----------
        arr : np.ndarray
            Input map.
        tile_size : int
            Original tile size before centered padding.
        interpolation : int
            OpenCV interpolation flag.

        Returns
        -------
        np.ndarray
            Resized map shaped ``(TILE_SIZE, TILE_SIZE)``.
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D supervision map, got shape={arr.shape}")
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        if arr.shape == (TILE_SIZE, TILE_SIZE):
            return arr.astype(np.float32)
        if arr.shape == (tile_size, tile_size):
            return centered_pad(arr.astype(np.float32), target_size=TILE_SIZE)
        resized = cv2.resize(arr, (TILE_SIZE, TILE_SIZE), interpolation=interpolation)
        return resized.astype(np.float32)

    def _load_cam_supervision(self, tile: TileRecord) -> Optional[tuple[np.ndarray, np.ndarray, bool]]:
        """Load offline CAM/Otsu supervision for a tile when available.

        Parameters
        ----------
        tile : TileRecord
            Tile descriptor.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, bool] | None
            ``(label_map, confidence_map, has_label)`` or ``None`` when missing.
        """
        if self.cam_mask_dir is None:
            return None

        stem = tile_id_to_artifact_stem(tile.tile_id)
        if self._cam_mask_stems is not None and stem not in self._cam_mask_stems:
            return None

        mask_path = self.cam_mask_dir / f"{stem}_mask.npy"
        cam_path = self.cam_mask_dir / f"{stem}_cam.npy"
        if not mask_path.exists():
            return None

        try:
            mask = np.load(mask_path, allow_pickle=False)
            mask = self._resize_supervision_map(mask, tile.chunk_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask >= 0.5).astype(np.float32)
        except Exception as exc:
            logging.warning("Failed to load CAM mask for %s from %s (%s)", tile.tile_id, mask_path, exc)
            return None

        conf = mask.copy()
        if cam_path.exists():
            try:
                conf = np.load(cam_path, allow_pickle=False)
                conf = self._resize_supervision_map(conf, tile.chunk_size, interpolation=cv2.INTER_LINEAR)
                conf = np.clip(conf, 0.0, 1.0).astype(np.float32)
            except Exception as exc:
                logging.warning(
                    "Failed to load CAM confidence for %s from %s (%s); falling back to mask confidence",
                    tile.tile_id,
                    cam_path,
                    exc,
                )
                conf = mask.copy()

        return mask[None, ...], conf[None, ...], True

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Fetch one training sample.

        Parameters
        ----------
        index : int
            Item index.

        Returns
        -------
        dict[str, Any]
            Batch dictionary for model/training code.
        """
        tile = self.tiles[index]
        chm, valid = self._read_tile(tile)

        input_map = np.stack([chm, valid], axis=0).astype(np.float32)
        chm_target = chm[None, ...].astype(np.float32)

        label_map, conf_map, has_label = self.label_store.get_tile_targets(tile, target_size=TILE_SIZE)
        cam_supervision = self._load_cam_supervision(tile)
        if cam_supervision is not None:
            label_map, conf_map, has_label = cam_supervision

        if self.augment:
            input_map, label_map, conf_map = self._augment_pair(input_map, label_map, conf_map)

        sample = {
            "input": torch.from_numpy(input_map),
            "chm_target": torch.from_numpy(chm_target),
            "seg_target": torch.from_numpy(label_map.astype(np.float32)),
            "confidence": torch.from_numpy(conf_map.astype(np.float32)),
            "has_label": torch.tensor([1.0 if has_label else 0.0], dtype=torch.float32),
            "tile_id": tile.tile_id,
            "sample_id": tile.sample_id,
            "mapsheet": tile.mapsheet,
            "year": tile.year,
            "variant": tile.variant,
        }
        return sample


def apply_curriculum_masking(
    chm_tensor: torch.Tensor,
    epoch: int,
    max_epochs: int,
) -> torch.Tensor:
    """Apply progressive random and structured masking to CHM inputs.

    Parameters
    ----------
    chm_tensor : torch.Tensor
        Input tensor shaped ``(B,2,H,W)`` where channel 0 is CHM and channel 1
        is valid-data mask.
    epoch : int
        Current epoch (1-based).
    max_epochs : int
        Total number of epochs.

    Returns
    -------
    torch.Tensor
        Masked input tensor with same shape as input.
    """
    out = chm_tensor.clone()
    chm = out[:, 0:1]
    valid = out[:, 1:2]

    progress = float(np.clip(epoch / max(1, max_epochs), 0.0, 1.0))
    # Keep masking moderate; extreme dropout collapses sparse-object learning.
    dropout_rate = 0.35 * (progress**0.8)

    random_keep = (torch.rand_like(valid) > dropout_rate).float()
    valid = valid * random_keep

    batch, _, height, width = valid.shape
    n_blocks = int(1 + 8 * progress)
    for b in range(batch):
        for _ in range(n_blocks):
            block = random.randint(4, 16)
            if block >= min(height, width):
                continue
            row = random.randint(0, height - block)
            col = random.randint(0, width - block)

            if random.random() < 0.5:
                valid[b, :, row : row + block, col : col + block] = 0.0
            else:
                rr = torch.arange(block, device=valid.device).view(-1, 1)
                cc = torch.arange(block, device=valid.device).view(1, -1)
                chess = ((rr + cc) % 2 == 0).float()[None, None, ...]
                patch = valid[b : b + 1, :, row : row + block, col : col + block]
                valid[b : b + 1, :, row : row + block, col : col + block] = patch * (1.0 - chess)

    chm = chm * valid
    out[:, 0:1] = chm
    out[:, 1:2] = valid
    return out


# ============================================================================
# Section 2: Model architecture + hotspot integration
# ============================================================================


class PartialConv2d(nn.Module):
    """Partial convolution layer with dynamic valid-mask normalization.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size.
    stride : int, default=1
        Convolution stride.
    padding : int, default=0
        Convolution padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        """Initialize PartialConv2d layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Kernel size.
        stride : int, default=1
            Stride.
        padding : int, default=0
            Padding.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.mask_conv = nn.Conv2d(
            1,
            1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        with torch.no_grad():
            self.mask_conv.weight.fill_(1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply partial convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input features shaped ``(B,C,H,W)``.
        mask : torch.Tensor
            Binary valid mask shaped ``(B,1,H,W)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(output, new_mask)`` with updated valid support.
        """
        masked_x = x * mask
        conv_out = self.conv(masked_x)

        with torch.no_grad():
            mask_sum = self.mask_conv(mask)

        w_total = float(self.kernel_size * self.kernel_size * self.in_channels)
        scaled = conv_out * (w_total / (mask_sum + EPS))
        new_mask = (mask_sum > 0).float()
        scaled = scaled * new_mask
        return scaled, new_mask


class PartialEncoderBlock(nn.Module):
    """Encoder block: PartialConv -> optional BN -> ReLU.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels.
    stride : int, default=2
        PartialConv stride.
    use_bn : bool, default=True
        Whether to apply batch normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        use_bn: bool = True,
    ) -> None:
        """Initialize encoder block.

        Parameters
        ----------
        in_channels : int
            Input channels.
        out_channels : int
            Output channels.
        stride : int, default=2
            Stride.
        use_bn : bool, default=True
            Batch norm flag.
        """
        super().__init__()
        self.pconv = PartialConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        mask : torch.Tensor
            Input valid mask.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(features, mask)``.
        """
        x, mask = self.pconv(x, mask)
        x = self.bn(x)
        x = self.act(x)
        return x, mask


class PartialDecoderBlock(nn.Module):
    """Decoder block with upsample, skip concat, PartialConv, BN, ReLU.

    Parameters
    ----------
    in_channels : int
        Input channels after concatenation.
    out_channels : int
        Output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize decoder block.

        Parameters
        ----------
        in_channels : int
            Input channels.
        out_channels : int
            Output channels.
        """
        super().__init__()
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        skip_x: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with skip connections.

        Parameters
        ----------
        x : torch.Tensor
            Low-resolution decoder features.
        mask : torch.Tensor
            Decoder mask.
        skip_x : torch.Tensor
            Skip-connection feature map.
        skip_mask : torch.Tensor
            Skip-connection mask.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Decoded features and updated mask.
        """
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, scale_factor=2.0, mode="nearest")
        x = torch.cat([x, skip_x], dim=1)
        mask = torch.clamp(mask + skip_mask, 0.0, 1.0)
        x, mask = self.pconv(x, mask)
        x = self.bn(x)
        x = self.act(x)
        return x, mask


class PartialConvUNet(nn.Module):
    """Dual-head PartialConv U-Net for reconstruction + segmentation.

    Parameters
    ----------
    dropout_p : float, default=0.3
        Dropout probability used in segmentation head.
    """

    def __init__(self, dropout_p: float = 0.3) -> None:
        """Initialize network architecture.

        Parameters
        ----------
        dropout_p : float, default=0.3
            Segmentation head dropout probability.
        """
        super().__init__()

        self.enc1 = PartialEncoderBlock(2, 32, stride=2, use_bn=False)
        self.enc2 = PartialEncoderBlock(32, 64, stride=2, use_bn=True)
        self.enc3 = PartialEncoderBlock(64, 128, stride=2, use_bn=True)
        self.enc4 = PartialEncoderBlock(128, 256, stride=2, use_bn=True)
        self.enc5 = PartialEncoderBlock(256, 512, stride=2, use_bn=True)

        self.bottleneck_pconv = PartialConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bottleneck_bn = nn.BatchNorm2d(512)
        self.bottleneck_act = nn.ReLU(inplace=True)

        self.rec_dec1 = PartialDecoderBlock(512 + 256, 256)
        self.rec_dec2 = PartialDecoderBlock(256 + 128, 128)
        self.rec_dec3 = PartialDecoderBlock(128 + 64, 64)
        self.rec_dec4 = PartialDecoderBlock(64 + 32, 32)

        self.seg_dec1 = PartialDecoderBlock(512 + 256, 256)
        self.seg_dec2 = PartialDecoderBlock(256 + 128, 128)
        self.seg_dec3 = PartialDecoderBlock(128 + 64, 64)
        self.seg_dec4 = PartialDecoderBlock(64 + 32, 32)

        self.recon_head = nn.Conv2d(32, 1, kernel_size=1)
        self.seg_dropout = nn.Dropout2d(p=dropout_p)
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for dual-head network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor shaped ``(B,2,H,W)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(reconstruction, segmentation_probability)``.
        """
        mask = x[:, 1:2]

        e1, m1 = self.enc1(x, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)
        e5, m5 = self.enc5(e4, m4)

        bottleneck, m6 = self.bottleneck_pconv(e5, m5)
        bottleneck = self.bottleneck_bn(bottleneck)
        bottleneck = self.bottleneck_act(bottleneck)

        r1, rm1 = self.rec_dec1(bottleneck, m6, e4, m4)
        r2, rm2 = self.rec_dec2(r1, rm1, e3, m3)
        r3, rm3 = self.rec_dec3(r2, rm2, e2, m2)
        r4, _ = self.rec_dec4(r3, rm3, e1, m1)
        recon = self.recon_head(r4)

        s1, sm1 = self.seg_dec1(bottleneck, m6, e4, m4)
        s2, sm2 = self.seg_dec2(s1, sm1, e3, m3)
        s3, sm3 = self.seg_dec3(s2, sm2, e2, m2)
        s4, _ = self.seg_dec4(s3, sm3, e1, m1)
        seg = self.seg_head(self.seg_dropout(s4))
        seg = torch.sigmoid(seg)

        if recon.shape[-1] != TILE_SIZE or recon.shape[-2] != TILE_SIZE:
            recon = F.interpolate(recon, size=(TILE_SIZE, TILE_SIZE), mode="bilinear", align_corners=False)
        if seg.shape[-1] != TILE_SIZE or seg.shape[-2] != TILE_SIZE:
            seg = F.interpolate(seg, size=(TILE_SIZE, TILE_SIZE), mode="bilinear", align_corners=False)
        return recon, seg


def _adapt_first_conv_to_1ch(model: nn.Module) -> None:
    """Adapt first 3-channel convolution in a model to accept 1 channel.

    Parameters
    ----------
    model : nn.Module
        Input model modified in place.
    """
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
            )
            with torch.no_grad():
                new_conv.weight.copy_(module.weight.mean(dim=1, keepdim=True))
                if module.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(module.bias)

            parent = model
            parts = module_name.split(".")
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            leaf = parts[-1]
            if leaf.isdigit():
                parent[int(leaf)] = new_conv
            else:
                setattr(parent, leaf, new_conv)
            return


def _replace_classifier_head(model: nn.Module, num_classes: int = 2) -> None:
    """Replace classification head with a new ``num_classes`` output layer.

    Parameters
    ----------
    model : nn.Module
        Classifier model.
    num_classes : int, default=2
        Number of classes.
    """
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
        return
    if hasattr(model, "classifier"):
        cls = model.classifier
        if isinstance(cls, nn.Linear):
            model.classifier = nn.Linear(cls.in_features, num_classes)
            return
        if isinstance(cls, nn.Sequential):
            for idx in range(len(cls) - 1, -1, -1):
                if isinstance(cls[idx], nn.Linear):
                    cls[idx] = nn.Linear(cls[idx].in_features, num_classes)
                    return
    raise RuntimeError("Unsupported classifier head replacement path")


def _build_convnext_small_1ch() -> nn.Module:
    """Build one-channel ConvNeXt Small classifier.

    Returns
    -------
    nn.Module
        ConvNeXt Small with 1-channel input and 2-class output.
    """
    model = tvm.convnext_small(weights=None)
    _adapt_first_conv_to_1ch(model)
    _replace_classifier_head(model, num_classes=2)
    return model


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Clean state dict keys from common wrappers.

    Parameters
    ----------
    state_dict : dict[str, Any]
        Raw state dict.

    Returns
    -------
    dict[str, Any]
        Cleaned state dict.
    """
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key in {"n_averaged", "param_averages"}:
            continue
        out_key = key
        if out_key.startswith("module."):
            out_key = out_key[len("module.") :]
        if out_key.startswith("model."):
            out_key = out_key[len("model.") :]
        cleaned[out_key] = value
    return cleaned


def load_legacy_model(checkpoint_path: str | Path, device: torch.device) -> Optional[nn.Module]:
    """Load ConvNeXt Small legacy classifier for hotspot generation.

    Parameters
    ----------
    checkpoint_path : str | Path
        Legacy classifier checkpoint path.
    device : torch.device
        Target device.

    Returns
    -------
    nn.Module | None
        Loaded model in eval mode, or ``None`` when loading fails.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        logging.warning("Legacy checkpoint not found: %s", path)
        return None

    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = _build_convnext_small_1ch().to(device)

        if isinstance(payload, dict):
            state = payload.get("state_dict")
            if state is None:
                state = payload.get("model_state_dict")
            if state is None:
                state = payload
        else:
            state = payload

        cleaned = _clean_state_dict(state)
        model.load_state_dict(cleaned, strict=False)
        model.eval()
        logging.info("Loaded legacy hotspot model: %s", path)
        return model
    except Exception as exc:  # pragma: no cover - best effort optional dependency path
        logging.warning("Failed to load legacy model from %s (%s)", path, exc)
        return None


def _last_spatial_conv(model: nn.Module) -> Optional[nn.Conv2d]:
    """Return last Conv2d layer with kernel size greater than 1.

    Parameters
    ----------
    model : nn.Module
        Input model.

    Returns
    -------
    nn.Conv2d | None
        Last spatial convolution layer.
    """
    last: Optional[nn.Conv2d] = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
            last = module
    return last


def _extract_positive_score(logits: torch.Tensor) -> torch.Tensor:
    """Extract positive-class confidence from classifier output.

    Parameters
    ----------
    logits : torch.Tensor
        Classifier output.

    Returns
    -------
    torch.Tensor
        Positive-class score per sample.
    """
    if logits.ndim == 2 and logits.shape[1] >= 2:
        return torch.softmax(logits, dim=1)[:, 1]
    if logits.ndim == 2 and logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    return torch.sigmoid(logits.view(logits.shape[0], -1).mean(dim=1))


def _normalize_maps(cam: torch.Tensor) -> torch.Tensor:
    """Normalize saliency maps to ``[0,1]`` per batch item.

    Parameters
    ----------
    cam : torch.Tensor
        Saliency tensor shaped ``(B,1,H,W)``.

    Returns
    -------
    torch.Tensor
        Normalized saliency maps.
    """
    flat = cam.view(cam.shape[0], -1)
    mn = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    mx = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return (cam - mn) / (mx - mn + EPS)


def compute_gradcam_hotspots(legacy_model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    """Compute Grad-CAM++ hotspot maps from legacy classifier.

    Parameters
    ----------
    legacy_model : nn.Module
        Legacy classifier.
    input_tensor : torch.Tensor
        CHM tensor shaped ``(B,1,H,W)``.

    Returns
    -------
    torch.Tensor
        Hotspot maps in ``[0,1]`` with shape ``(B,1,H,W)``.
    """
    model = legacy_model
    model.eval()
    target_layer = _last_spatial_conv(model)
    if target_layer is None:
        return torch.zeros_like(input_tensor)

    x = torch.clamp(input_tensor, 0.0, 20.0) / 20.0
    orig_size = x.shape[-2:]
    x_resized = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
    x_resized = x_resized.requires_grad_(True)

    feats: list[torch.Tensor] = []

    def _hook(_: nn.Module, __: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        """Capture forward activations for Grad-CAM++.

        Parameters
        ----------
        _ : nn.Module
            Hooked module.
        __ : tuple[torch.Tensor, ...]
            Forward input tuple.
        out : torch.Tensor
            Forward output tensor.
        """
        feats.append(out)

    handle = target_layer.register_forward_hook(_hook)
    try:
        logits = model(x_resized)
        score = _extract_positive_score(logits)
        grads = torch.autograd.grad(score.sum(), feats[0], retain_graph=False, create_graph=False)[0]

        activ = feats[0]
        grads2 = grads.pow(2)
        grads3 = grads.pow(3)
        denom = 2.0 * grads2 + (activ * grads3).sum(dim=(2, 3), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.full_like(denom, EPS))
        alpha = grads2 / (denom + EPS)
        weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activ).sum(dim=1, keepdim=True))
        cam = _normalize_maps(cam)
        cam = F.interpolate(cam, size=orig_size, mode="bilinear", align_corners=False)
        return cam.detach()
    except Exception:  # pragma: no cover - fallback path for unusual checkpoints
        return torch.zeros_like(input_tensor)
    finally:
        handle.remove()


def compute_hirescam_hotspots(legacy_model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    """Compute HiResCAM hotspots for SOTA ablation benchmarking.

    Parameters
    ----------
    legacy_model : nn.Module
        Legacy classifier.
    input_tensor : torch.Tensor
        CHM tensor shaped ``(B,1,H,W)``.

    Returns
    -------
    torch.Tensor
        HiResCAM maps in ``[0,1]``.
    """
    model = legacy_model
    model.eval()
    target_layer = _last_spatial_conv(model)
    if target_layer is None:
        return torch.zeros_like(input_tensor)

    x = torch.clamp(input_tensor, 0.0, 20.0) / 20.0
    orig_size = x.shape[-2:]
    x_resized = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
    x_resized = x_resized.requires_grad_(True)

    feats: list[torch.Tensor] = []

    def _hook(_: nn.Module, __: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        """Capture forward activations for HiResCAM.

        Parameters
        ----------
        _ : nn.Module
            Hooked module.
        __ : tuple[torch.Tensor, ...]
            Forward input tuple.
        out : torch.Tensor
            Forward output tensor.
        """
        feats.append(out)

    handle = target_layer.register_forward_hook(_hook)
    try:
        logits = model(x_resized)
        score = _extract_positive_score(logits)
        grads = torch.autograd.grad(score.sum(), feats[0], retain_graph=False, create_graph=False)[0]
        cam = torch.relu((grads * feats[0]).sum(dim=1, keepdim=True))
        cam = _normalize_maps(cam)
        cam = F.interpolate(cam, size=orig_size, mode="bilinear", align_corners=False)
        return cam.detach()
    except Exception:
        return torch.zeros_like(input_tensor)
    finally:
        handle.remove()


def compute_integrated_gradients_hotspots(
    legacy_model: nn.Module,
    input_tensor: torch.Tensor,
    steps: int = 24,
) -> torch.Tensor:
    """Compute Integrated Gradients hotspot maps.

    Parameters
    ----------
    legacy_model : nn.Module
        Legacy classifier.
    input_tensor : torch.Tensor
        CHM tensor shaped ``(B,1,H,W)``.
    steps : int, default=24
        Number of Riemann steps.

    Returns
    -------
    torch.Tensor
        Integrated Gradients saliency maps in ``[0,1]``.
    """
    model = legacy_model
    model.eval()

    x = torch.clamp(input_tensor, 0.0, 20.0) / 20.0
    orig_size = x.shape[-2:]
    x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)

    baseline = torch.zeros_like(x)
    acc = torch.zeros_like(x)
    for idx in range(1, steps + 1):
        alpha = float(idx) / float(steps)
        interp = (baseline + alpha * (x - baseline)).requires_grad_(True)
        logits = model(interp)
        score = _extract_positive_score(logits)
        grads = torch.autograd.grad(score.sum(), interp, retain_graph=False, create_graph=False)[0]
        acc = acc + grads

    attrs = (x - baseline) * (acc / float(steps))
    cam = torch.relu(attrs.abs().mean(dim=1, keepdim=True))
    cam = _normalize_maps(cam)
    cam = F.interpolate(cam, size=orig_size, mode="bilinear", align_corners=False)
    return cam.detach()


def compute_rise_hotspots(
    legacy_model: nn.Module,
    input_tensor: torch.Tensor,
    n_masks: int = 128,
    mask_size: int = 8,
    p_keep: float = 0.5,
) -> torch.Tensor:
    """Compute simplified RISE saliency maps.

    Parameters
    ----------
    legacy_model : nn.Module
        Legacy classifier.
    input_tensor : torch.Tensor
        CHM tensor shaped ``(B,1,H,W)``.
    n_masks : int, default=128
        Number of random masks.
    mask_size : int, default=8
        Low-resolution mask dimension.
    p_keep : float, default=0.5
        Bernoulli keep probability.

    Returns
    -------
    torch.Tensor
        RISE maps in ``[0,1]``.
    """
    model = legacy_model
    model.eval()

    x = torch.clamp(input_tensor, 0.0, 20.0) / 20.0
    bsz, _, height, width = x.shape
    sal = torch.zeros((bsz, 1, height, width), device=x.device, dtype=x.dtype)
    cnt = torch.zeros_like(sal)

    for _ in range(n_masks):
        low = (torch.rand((bsz, 1, mask_size, mask_size), device=x.device) < p_keep).float()
        mask = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
        mask = (mask > 0.5).float()
        masked = x * mask
        logits = model(F.interpolate(masked, size=(128, 128), mode="bilinear", align_corners=False))
        score = _extract_positive_score(logits).view(bsz, 1, 1, 1)
        sal = sal + score * mask
        cnt = cnt + mask

    sal = torch.where(cnt > 0, sal / (cnt + EPS), torch.zeros_like(sal))
    sal = _normalize_maps(sal)
    return sal.detach()


# ============================================================================
# Section 3: Losses
# ============================================================================


class ReconstructionLoss(nn.Module):
    """Reconstruction loss with valid/hole and Sobel edge consistency terms.

    Parameters
    ----------
    hole_weight : float, default=6.0
        Relative weight for hole-region reconstruction loss.
    edge_weight : float, default=0.5
        Relative weight for edge-consistency loss.
    """

    def __init__(self, hole_weight: float = 6.0, edge_weight: float = 0.5) -> None:
        """Initialize reconstruction loss.

        Parameters
        ----------
        hole_weight : float, default=6.0
            Hole-region weight.
        edge_weight : float, default=0.5
            Edge weight.
        """
        super().__init__()
        self.hole_weight = float(hole_weight)
        self.edge_weight = float(edge_weight)

        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _sobel_mag(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient magnitude.

        Parameters
        ----------
        x : torch.Tensor
            Input map ``(B,1,H,W)``.

        Returns
        -------
        torch.Tensor
            Gradient magnitude map.
        """
        gx = F.conv2d(x, self.sobel_x.detach(), padding=1)
        gy = F.conv2d(x, self.sobel_y.detach(), padding=1)
        return torch.sqrt(gx * gx + gy * gy + EPS)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted CHM ``(B,1,H,W)``.
        target : torch.Tensor
            Target CHM ``(B,1,H,W)``.
        valid_mask : torch.Tensor
            Input validity mask ``(B,1,H,W)``.

        Returns
        -------
        torch.Tensor
            Scalar reconstruction loss.
        """
        hole_mask = 1.0 - valid_mask
        abs_err = torch.abs(pred - target)

        valid_loss = (abs_err * valid_mask).sum() / (valid_mask.sum() + EPS)
        hole_loss = (abs_err * hole_mask).sum() / (hole_mask.sum() + EPS)

        pred_edges = self._sobel_mag(pred)
        tgt_edges = self._sobel_mag(target)
        edge_loss = torch.mean(torch.abs(pred_edges - tgt_edges))

        total = valid_loss + self.hole_weight * hole_loss + self.edge_weight * edge_loss
        return total


class SegmentationLoss(nn.Module):
    """Weighted Dice + Focal loss with confidence and hotspot modulation.

    Parameters
    ----------
    alpha : float, default=0.25
        Focal alpha parameter.
    gamma : float, default=2.0
        Focal gamma parameter.
    smooth : float, default=1.0
        Dice smoothing factor.
    hotspot_boost : float, default=0.5
        Additive hotspot emphasis multiplier.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smooth: float = 1.0,
        hotspot_boost: float = 0.5,
    ) -> None:
        """Initialize segmentation loss.

        Parameters
        ----------
        alpha : float, default=0.25
            Focal alpha.
        gamma : float, default=2.0
            Focal gamma.
        smooth : float, default=1.0
            Dice smooth.
        hotspot_boost : float, default=0.5
            Hotspot multiplier.
        """
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.hotspot_boost = float(hotspot_boost)

    def _weighted_dice(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Compute weighted Dice loss.

        Parameters
        ----------
        pred : torch.Tensor
            Probability predictions.
        target : torch.Tensor
            Binary targets.
        weight : torch.Tensor
            Pixel weights.

        Returns
        -------
        torch.Tensor
            Dice loss value.
        """
        p = pred.view(pred.shape[0], -1)
        t = target.view(target.shape[0], -1)
        w = weight.view(weight.shape[0], -1)
        inter = (w * p * t).sum(dim=1)
        den = (w * p).sum(dim=1) + (w * t).sum(dim=1)
        dice = 1.0 - (2.0 * inter + self.smooth) / (den + self.smooth)
        return dice

    def _weighted_focal(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Compute weighted focal loss.

        Parameters
        ----------
        pred : torch.Tensor
            Probability predictions.
        target : torch.Tensor
            Binary targets.
        weight : torch.Tensor
            Pixel weights.

        Returns
        -------
        torch.Tensor
            Focal loss value per sample.
        """
        p = torch.clamp(pred, EPS, 1.0 - EPS)
        bce = -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
        pt = target * p + (1.0 - target) * (1.0 - p)
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        focal = alpha_t * torch.pow(1.0 - pt, self.gamma) * bce
        focal = focal * weight
        return focal.view(focal.shape[0], -1).mean(dim=1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor,
        has_label: torch.Tensor,
        hotspot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute confidence-modulated segmentation loss.

        Parameters
        ----------
        pred : torch.Tensor
            Segmentation probabilities.
        target : torch.Tensor
            Binary pseudo-label map.
        confidence : torch.Tensor
            Confidence map in ``[0,1]``.
        has_label : torch.Tensor
            Per-sample label availability mask shaped ``(B,1)``.
        hotspot : torch.Tensor | None, default=None
            Optional hotspot weighting map.

        Returns
        -------
        torch.Tensor
            Scalar segmentation loss.
        """
        base_weight = torch.clamp(confidence, 0.05, 1.0)
        if hotspot is not None:
            base_weight = base_weight * (1.0 + self.hotspot_boost * torch.clamp(hotspot, 0.0, 1.0))

        dice = self._weighted_dice(pred, target, base_weight)
        focal = self._weighted_focal(pred, target, base_weight)
        seg = 0.5 * dice + 0.5 * focal

        labeled = has_label.view(-1)
        if torch.sum(labeled) <= 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        seg = seg * labeled
        return seg.sum() / (labeled.sum() + EPS)


class JointLoss(nn.Module):
    """Combined objective ``L = 1.0*L_recon + 5.0*L_seg``.

    Parameters
    ----------
    recon_weight : float, default=1.0
        Reconstruction loss weight.
    seg_weight : float, default=5.0
        Segmentation loss weight.
    """

    def __init__(self, recon_weight: float = 1.0, seg_weight: float = 5.0) -> None:
        """Initialize joint loss.

        Parameters
        ----------
        recon_weight : float, default=1.0
            Reconstruction weight.
        seg_weight : float, default=5.0
            Segmentation weight.
        """
        super().__init__()
        self.recon_weight = float(recon_weight)
        self.seg_weight = float(seg_weight)
        self.recon_loss = ReconstructionLoss()
        self.seg_loss = SegmentationLoss(alpha=0.25, gamma=2.0, smooth=1.0)

    def forward(
        self,
        recon_pred: torch.Tensor,
        recon_target: torch.Tensor,
        valid_mask: torch.Tensor,
        seg_pred: torch.Tensor,
        seg_target: torch.Tensor,
        confidence: torch.Tensor,
        has_label: torch.Tensor,
        hotspot: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute joint loss and component diagnostics.

        Parameters
        ----------
        recon_pred : torch.Tensor
            Reconstructed CHM prediction.
        recon_target : torch.Tensor
            CHM reconstruction target.
        valid_mask : torch.Tensor
            Input validity mask.
        seg_pred : torch.Tensor
            Segmentation probabilities.
        seg_target : torch.Tensor
            Segmentation pseudo-labels.
        confidence : torch.Tensor
            Pseudo-label confidence map.
        has_label : torch.Tensor
            Label availability indicator.
        hotspot : torch.Tensor | None, default=None
            Optional hotspot map.

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            ``(total_loss, component_logs)``.
        """
        l_recon = self.recon_loss(recon_pred, recon_target, valid_mask)
        l_seg = self.seg_loss(seg_pred, seg_target, confidence, has_label, hotspot=hotspot)
        total = self.recon_weight * l_recon + self.seg_weight * l_seg
        logs = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_recon": float(l_recon.detach().cpu().item()),
            "loss_seg": float(l_seg.detach().cpu().item()),
        }
        return total, logs


# ============================================================================
# Section 4: Training loop
# ============================================================================


def _tile_level_vectors(
    prob: torch.Tensor,
    target: torch.Tensor,
    has_label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract labeled tile-level probabilities and binary targets.

    Parameters
    ----------
    prob : torch.Tensor
        Predicted probability map.
    target : torch.Tensor
        Target binary map.
    has_label : torch.Tensor
        Label availability flags.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(tile_prob, tile_target)`` vectors on CPU for labeled tiles only.

    Notes
    -----
    Presence-style tile targets use max pooling rather than mean pooling to
    avoid bias from centered padding and sparse positive masks.
    """
    labeled = has_label.view(-1) > 0.5
    if torch.sum(labeled) == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty

    tile_prob = torch.amax(prob, dim=(1, 2, 3)).detach().float()[labeled].cpu()
    tile_target = (torch.amax(target, dim=(1, 2, 3)) > 0.5).detach().float()[labeled].cpu()
    return tile_prob, tile_target


def _f1_score_from_tile_vectors(
    tile_prob: torch.Tensor,
    tile_target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute tile-level F1 from vectors of probabilities and targets.

    Parameters
    ----------
    tile_prob : torch.Tensor
        Tile-level probabilities.
    tile_target : torch.Tensor
        Tile-level binary targets.
    threshold : float, default=0.5
        Decision threshold.

    Returns
    -------
    float
        F1 score.
    """
    if tile_prob.numel() == 0:
        return 0.0

    thr = float(np.clip(threshold, 0.0, 1.0))
    pred = (tile_prob >= thr).float()
    truth = tile_target.float()

    tp = torch.sum((pred == 1.0) & (truth == 1.0)).float()
    fp = torch.sum((pred == 1.0) & (truth == 0.0)).float()
    fn = torch.sum((pred == 0.0) & (truth == 1.0)).float()

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    return float(f1.detach().cpu().item())


def _best_f1_over_thresholds(
    tile_prob: torch.Tensor,
    tile_target: torch.Tensor,
    thresholds: Sequence[float],
) -> tuple[float, float]:
    """Find the best tile-level F1 across a threshold grid.

    Parameters
    ----------
    tile_prob : torch.Tensor
        Tile-level probabilities.
    tile_target : torch.Tensor
        Tile-level binary targets.
    thresholds : Sequence[float]
        Threshold grid to evaluate.

    Returns
    -------
    tuple[float, float]
        ``(best_f1, best_threshold)``.
    """
    if tile_prob.numel() == 0:
        return 0.0, 0.5

    grid = sorted({float(np.clip(thr, 0.0, 1.0)) for thr in thresholds})
    if not grid:
        grid = [0.5]

    best_threshold = grid[0]
    best_f1 = -1.0
    for thr in grid:
        score = _f1_score_from_tile_vectors(tile_prob, tile_target, threshold=thr)
        if score > best_f1 + 1e-12:
            best_f1 = score
            best_threshold = thr

    return float(best_f1), float(best_threshold)


def _f1_score_from_maps(
    prob: torch.Tensor,
    target: torch.Tensor,
    has_label: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute tile-level F1 using thresholded mean predictions.

    Parameters
    ----------
    prob : torch.Tensor
        Predicted probability map.
    target : torch.Tensor
        Target binary map.
    has_label : torch.Tensor
        Label availability flags.
    threshold : float, default=0.5
        Decision threshold.

    Returns
    -------
    float
        F1 score for labeled tiles.
    """
    tile_prob, tile_target = _tile_level_vectors(prob, target, has_label)
    return _f1_score_from_tile_vectors(tile_prob, tile_target, threshold=threshold)


def _dice_iou_from_confusion(tp: float, fp: float, fn: float) -> tuple[float, float]:
    """Compute Dice and IoU from scalar confusion components.

    Parameters
    ----------
    tp : float
        True-positive count.
    fp : float
        False-positive count.
    fn : float
        False-negative count.

    Returns
    -------
    tuple[float, float]
        ``(dice, iou)``.
    """
    dice = (2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS)
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    return float(dice), float(iou)


def _extract_labeled_binary_maps(
    prob: torch.Tensor,
    target: torch.Tensor,
    has_label: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract labeled binary prediction/target maps.

    Parameters
    ----------
    prob : torch.Tensor
        Predicted probabilities ``(B,1,H,W)``.
    target : torch.Tensor
        Binary targets ``(B,1,H,W)``.
    has_label : torch.Tensor
        Label availability flags.
    threshold : float
        Probability threshold.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(pred_bin, target_bin)`` for labeled samples only.
    """
    labeled = has_label.view(-1) > 0.5
    if torch.sum(labeled) <= 0:
        shape = (0, 1, target.shape[-2], target.shape[-1])
        empty = torch.empty(shape, dtype=torch.bool, device=target.device)
        return empty, empty

    thr = float(np.clip(threshold, 0.0, 1.0))
    pred_bin = (prob[labeled] >= thr).detach()
    target_bin = (target[labeled] > 0.5).detach()
    return pred_bin.bool(), target_bin.bool()


def _boundary_map(mask: np.ndarray) -> np.ndarray:
    """Extract a binary boundary map from a binary mask."""
    if mask.size == 0 or not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    if binary_erosion is None:
        return mask.astype(bool, copy=False)

    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def _hd95_single_mask(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """Compute symmetric HD95 between one predicted and one target mask."""
    pred_mask = pred_mask.astype(bool, copy=False)
    target_mask = target_mask.astype(bool, copy=False)

    if not np.any(pred_mask) and not np.any(target_mask):
        return 0.0

    if not np.any(pred_mask) or not np.any(target_mask):
        height, width = pred_mask.shape
        return float(np.hypot(float(height), float(width)))

    if distance_transform_edt is None:
        height, width = pred_mask.shape
        return float(np.hypot(float(height), float(width)))

    pred_boundary = _boundary_map(pred_mask)
    target_boundary = _boundary_map(target_mask)
    if not np.any(pred_boundary):
        pred_boundary = pred_mask
    if not np.any(target_boundary):
        target_boundary = target_mask

    dist_to_pred = distance_transform_edt(np.logical_not(pred_boundary))
    dist_to_target = distance_transform_edt(np.logical_not(target_boundary))
    d_target_to_pred = dist_to_pred[target_boundary]
    d_pred_to_target = dist_to_target[pred_boundary]
    if d_target_to_pred.size == 0 and d_pred_to_target.size == 0:
        return 0.0

    all_distances = np.concatenate((d_target_to_pred, d_pred_to_target), axis=0)
    return float(np.percentile(all_distances, 95.0))


def _cldice_single_mask(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """Compute clDice between one predicted and one target mask."""
    pred_mask = pred_mask.astype(bool, copy=False)
    target_mask = target_mask.astype(bool, copy=False)

    if not np.any(pred_mask) and not np.any(target_mask):
        return 1.0

    if skeletonize is None:
        inter = float(np.logical_and(pred_mask, target_mask).sum())
        union = float(np.logical_or(pred_mask, target_mask).sum())
        return float((2.0 * inter + EPS) / (inter + union + EPS))

    pred_skel = skeletonize(pred_mask)
    target_skel = skeletonize(target_mask)

    topology_precision = float(np.logical_and(pred_skel, target_mask).sum()) / (float(pred_skel.sum()) + EPS)
    topology_sensitivity = float(np.logical_and(target_skel, pred_mask).sum()) / (float(target_skel.sum()) + EPS)
    return float((2.0 * topology_precision * topology_sensitivity) / (topology_precision + topology_sensitivity + EPS))


def _boundary_metrics_from_binary_maps(
    pred_bin: torch.Tensor,
    target_bin: torch.Tensor,
) -> tuple[list[float], list[float]]:
    """Compute per-sample clDice and HD95 from binary maps."""
    if pred_bin.numel() == 0 or target_bin.numel() == 0:
        return [], []

    pred_np = pred_bin.detach().cpu().numpy().astype(bool)
    target_np = target_bin.detach().cpu().numpy().astype(bool)
    cldice_values: list[float] = []
    hd95_values: list[float] = []

    for idx in range(pred_np.shape[0]):
        pred_mask = pred_np[idx, 0]
        target_mask = target_np[idx, 0]
        cldice_values.append(_cldice_single_mask(pred_mask, target_mask))
        hd95_values.append(_hd95_single_mask(pred_mask, target_mask))

    return cldice_values, hd95_values


def _hd95_to_score(hd95: float) -> float:
    """Map HD95 to a bounded higher-is-better score."""
    if not math.isfinite(hd95):
        return 0.0
    return float(1.0 / (1.0 + max(0.0, hd95)))


def _is_metric_improved(
    value: float,
    best_value: float,
    mode: str,
    min_delta: float,
) -> bool:
    """Return whether a monitored metric improved enough to reset patience."""
    if not math.isfinite(value):
        return False

    if mode == "min":
        return value < (best_value - min_delta)
    return value > (best_value + min_delta)


def dense_crf_refine(prob_map: torch.Tensor, chm_map: torch.Tensor) -> torch.Tensor:
    """Apply DenseCRF-style refinement with optional CRF backend.

    Parameters
    ----------
    prob_map : torch.Tensor
        Probability map ``(B,1,H,W)``.
    chm_map : torch.Tensor
        CHM map ``(B,1,H,W)``.

    Returns
    -------
    torch.Tensor
        Refined probability map.
    """
    _ = chm_map
    # Lightweight fallback when pydensecrf is not installed.
    return torch.clamp(F.avg_pool2d(prob_map, kernel_size=3, stride=1, padding=1), 0.0, 1.0)


def irn_lite_refine(prob_map: torch.Tensor, chm_map: torch.Tensor, beta: float = 7.5) -> torch.Tensor:
    """Apply IRN-lite affinity smoothing guided by CHM gradients.

    Parameters
    ----------
    prob_map : torch.Tensor
        Probability map.
    chm_map : torch.Tensor
        CHM map.
    beta : float, default=7.5
        Affinity sharpness.

    Returns
    -------
    torch.Tensor
        Refined map.
    """
    gx = torch.abs(chm_map[:, :, :, 1:] - chm_map[:, :, :, :-1])
    gy = torch.abs(chm_map[:, :, 1:, :] - chm_map[:, :, :-1, :])
    gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
    gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
    affinity = torch.exp(-beta * (gx + gy))
    smooth = F.avg_pool2d(prob_map * affinity, kernel_size=3, stride=1, padding=1)
    norm = F.avg_pool2d(affinity, kernel_size=3, stride=1, padding=1)
    return torch.clamp(smooth / (norm + EPS), 0.0, 1.0)


def sam_style_refine(prob_map: torch.Tensor, chm_map: torch.Tensor) -> torch.Tensor:
    """Apply SAM-style seed-and-grow refinement.

    Parameters
    ----------
    prob_map : torch.Tensor
        Probability map.
    chm_map : torch.Tensor
        CHM map.

    Returns
    -------
    torch.Tensor
        Refined map.
    """
    seeds = (prob_map > 0.7).float()
    support = (prob_map > 0.35).float() * (chm_map > 0.05).float()
    grown = F.max_pool2d(seeds, kernel_size=5, stride=1, padding=2)
    refined = torch.clamp(torch.maximum(prob_map, grown * support), 0.0, 1.0)
    return refined


def benchmark_hotspot_methods(
    legacy_model: Optional[nn.Module],
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 8,
) -> dict[str, float]:
    """Benchmark hotspot/post-processing methods on validation batches.

    Parameters
    ----------
    legacy_model : nn.Module | None
        Legacy classifier for CAM-based methods.
    model : nn.Module
        Segmentation model.
    dataloader : DataLoader
        Validation loader.
    device : torch.device
        Execution device.
    max_batches : int, default=8
        Maximum batches to evaluate.

    Returns
    -------
    dict[str, float]
        Method -> mean tile-level F1.
    """
    methods = {
        "dense_crf": lambda p, x: dense_crf_refine(p, x),
        "irn_lite": lambda p, x: irn_lite_refine(p, x),
        "sam_style": lambda p, x: sam_style_refine(p, x),
    }

    score_lists: dict[str, list[float]] = {name: [] for name in methods}
    if legacy_model is not None:
        score_lists.update({"gradcam_pp": [], "hirescam": [], "intgrad": [], "rise": []})

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            inputs = batch["input"].to(device)
            seg_target = batch["seg_target"].to(device)
            has_label = batch["has_label"].to(device)

            _, base_seg = model(inputs)
            chm = inputs[:, 0:1]

            for name, fn in methods.items():
                refined = fn(base_seg, chm)
                score_lists[name].append(_f1_score_from_maps(refined, seg_target, has_label))

    if legacy_model is not None:
        # CAM methods require gradients; run without no_grad block.
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            inputs = batch["input"].to(device)
            seg_target = batch["seg_target"].to(device)
            has_label = batch["has_label"].to(device)
            chm = inputs[:, 0:1]

            with torch.no_grad():
                _, base_seg = model(inputs)

            for method_name, fn in {
                "gradcam_pp": compute_gradcam_hotspots,
                "hirescam": compute_hirescam_hotspots,
                "intgrad": compute_integrated_gradients_hotspots,
                "rise": compute_rise_hotspots,
            }.items():
                hotspots = fn(legacy_model, chm)
                combined = torch.clamp(torch.maximum(base_seg, hotspots), 0.0, 1.0)
                score_lists[method_name].append(_f1_score_from_maps(combined, seg_target, has_label))

    out: dict[str, float] = {}
    for name, values in score_lists.items():
        out[name] = float(np.mean(values)) if values else 0.0
    return out


class Trainer:
    """Trainer class implementing optimization, logging, and checkpointing.

    Parameters
    ----------
    model : nn.Module
        Trainable segmentation model.
    criterion : JointLoss
        Joint objective function.
    config : TrainingConfig
        Hyperparameter configuration.
    device : torch.device
        Training device.
    output_dir : Path
        Checkpoint and log output directory.
    legacy_model : nn.Module | None, default=None
        Optional legacy classifier for Grad-CAM++ hotspot weighting.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: JointLoss,
        config: TrainingConfig,
        device: torch.device,
        output_dir: Path,
        legacy_model: Optional[nn.Module] = None,
    ) -> None:
        """Initialize trainer state.

        Parameters
        ----------
        model : nn.Module
            Model.
        criterion : JointLoss
            Criterion.
        config : TrainingConfig
            Config.
        device : torch.device
            Device.
        output_dir : Path
            Output directory.
        legacy_model : nn.Module | None, default=None
            Optional hotspot model.
        """
        self.model = model
        self.criterion = criterion.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.legacy_model = legacy_model

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler: Optional[CosineAnnealingLR] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.output_dir / "best_partialconv_unet.pt"
        self.last_path = self.output_dir / "last_partialconv_unet.pt"
        self._warned_hotspot_target_fallback = False

        monitor_metric = str(self.config.monitor_metric).strip().lower()
        allowed_monitor_metrics = {"sota_score", "dice", "iou", "cldice", "hd95", "f1", "loss"}
        if monitor_metric not in allowed_monitor_metrics:
            monitor_metric = "sota_score"
        if monitor_metric in {"sota_score", "cldice", "hd95"} and not _HAS_BOUNDARY_METRIC_DEPS:
            logging.warning(
                "Boundary metrics dependencies are unavailable; monitor metric '%s' falls back to 'dice'",
                monitor_metric,
            )
            monitor_metric = "dice"
        self.monitor_metric = monitor_metric

    def _compute_hotspots(self, inputs: torch.Tensor, epoch: int, force: bool = False) -> Optional[torch.Tensor]:
        """Compute Grad-CAM++ hotspots from legacy ConvNeXt model.

        Parameters
        ----------
        inputs : torch.Tensor
            Model input tensor.
        epoch : int
            Current epoch.
        force : bool, default=False
            If ``True``, compute hotspots regardless of epoch schedule.

        Returns
        -------
        torch.Tensor | None
            Hotspot map or ``None``.
        """
        if self.legacy_model is None:
            return None
        # For weighting-only mode, hotspots are most valuable in early training.
        if not force and epoch > max(3, self.config.epochs // 2):
            return None
        chm = inputs[:, 0:1]
        with torch.enable_grad():
            hotspots = compute_gradcam_hotspots(self.legacy_model, chm)
        return hotspots.detach()

    def _resolve_seg_supervision(
        self,
        inputs_used: torch.Tensor,
        seg_target_labels: torch.Tensor,
        confidence_labels: torch.Tensor,
        has_label_labels: torch.Tensor,
        epoch: int,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Resolve loss supervision tensors from configured segmentation source.

        Parameters
        ----------
        inputs_used : torch.Tensor
            Possibly masked model inputs.
        seg_target_labels : torch.Tensor
            Label-derived segmentation targets.
        confidence_labels : torch.Tensor
            Label-derived confidence maps.
        has_label_labels : torch.Tensor
            Label-availability indicator.
        epoch : int
            Current epoch.
        training : bool
            Train/val phase flag.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
            ``(seg_target_loss, confidence_loss, has_label_loss, hotspot_weight)``.
        """
        mode = str(self.config.seg_target_source).strip().lower()
        if mode not in {"labels", "hotspot", "labels_or_hotspot"}:
            mode = "labels"

        seg_target_loss = seg_target_labels
        confidence_loss = confidence_labels
        has_label_loss = has_label_labels
        hotspot_weight: Optional[torch.Tensor] = None

        needs_hotspot_target = mode in {"hotspot", "labels_or_hotspot"}
        hotspots: Optional[torch.Tensor] = None

        if needs_hotspot_target:
            hotspots = self._compute_hotspots(inputs_used, epoch=epoch, force=True)
            if hotspots is None:
                if not self._warned_hotspot_target_fallback:
                    logging.warning(
                        "seg_target_source=%s requested but hotspots are unavailable; falling back to label targets",
                        mode,
                    )
                    self._warned_hotspot_target_fallback = True
                return seg_target_loss, confidence_loss, has_label_loss, None

            hotspot_target = torch.clamp(hotspots, 0.0, 1.0)
            hotspot_conf = hotspot_to_confidence_map(
                hotspot_target,
                min_conf=self.config.hotspot_conf_min,
                gamma=self.config.hotspot_conf_gamma,
            )

            if mode == "hotspot":
                seg_target_loss = hotspot_target
                confidence_loss = hotspot_conf
                has_label_loss = torch.ones_like(has_label_labels)
            else:  # labels_or_hotspot
                labeled = (has_label_labels > 0.5).float().view(-1, 1, 1, 1)
                seg_target_loss = labeled * seg_target_labels + (1.0 - labeled) * hotspot_target
                confidence_loss = labeled * confidence_labels + (1.0 - labeled) * hotspot_conf
                has_label_loss = torch.ones_like(has_label_labels)

            return seg_target_loss, confidence_loss, has_label_loss, None

        if training:
            hotspot_weight = self._compute_hotspots(inputs_used, epoch=epoch, force=False)

        return seg_target_loss, confidence_loss, has_label_loss, hotspot_weight

    def _save_checkpoint(
        self,
        epoch: int,
        best_monitor_value: float,
        best_val_loss: float,
        is_best: bool,
    ) -> None:
        """Save last and optional best checkpoints.

        Parameters
        ----------
        epoch : int
            Epoch number.
        best_monitor_value : float
            Best monitored metric value so far.
        best_val_loss : float
            Best validation loss so far.
        is_best : bool
            Whether current checkpoint is best.
        """
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_monitor_value": best_monitor_value,
            "monitor_metric": self.monitor_metric,
            "config": asdict(self.config),
        }
        torch.save(payload, self.last_path)
        if is_best:
            torch.save(payload, self.best_path)

    def _build_eval_thresholds(self) -> list[float]:
        """Build threshold grid used for validation F1 sweeping.

        Returns
        -------
        list[float]
            Sorted threshold values.
        """
        if not self.config.eval_threshold_sweep_enabled:
            return [0.5]

        low = float(np.clip(self.config.eval_threshold_sweep_min, 0.0, 1.0))
        high = float(np.clip(self.config.eval_threshold_sweep_max, 0.0, 1.0))
        if high < low:
            low, high = high, low

        step = float(self.config.eval_threshold_sweep_step)
        if step <= 0.0:
            step = 0.05

        thresholds = np.arange(low, high + 0.5 * step, step, dtype=np.float32)
        if thresholds.size == 0:
            return [0.5]

        unique = sorted({float(np.clip(value, 0.0, 1.0)) for value in thresholds.tolist()})
        return unique if unique else [0.5]

    def _run_epoch(self, dataloader: DataLoader, epoch: int, training: bool) -> dict[str, float]:
        """Run one epoch for train or validation phase.

        Parameters
        ----------
        dataloader : DataLoader
            Dataloader.
        epoch : int
            Current epoch.
        training : bool
            Training phase flag.

        Returns
        -------
        dict[str, float]
            Aggregated metrics.
        """
        if training:
            self.model.train()
        else:
            self.model.eval()
        phase = "train" if training else "val"

        losses: list[float] = []
        recon_losses: list[float] = []
        seg_losses: list[float] = []
        tile_prob_chunks: list[torch.Tensor] = []
        tile_target_chunks: list[torch.Tensor] = []
        total_tp_px = 0.0
        total_fp_px = 0.0
        total_fn_px = 0.0
        cldice_values: list[float] = []
        hd95_values: list[float] = []
        skipped_batches = 0

        for step, batch in enumerate(dataloader, start=1):
            inputs = torch.nan_to_num(batch["input"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
            recon_target = torch.nan_to_num(batch["chm_target"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
            seg_target_labels = torch.nan_to_num(batch["seg_target"].to(self.device), nan=0.0, posinf=1.0, neginf=0.0)
            confidence_labels = torch.clamp(
                torch.nan_to_num(batch["confidence"].to(self.device), nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )
            has_label_labels = torch.clamp(
                torch.nan_to_num(batch["has_label"].to(self.device), nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )

            if training:
                inputs_used = apply_curriculum_masking(inputs, epoch=epoch, max_epochs=self.config.epochs)
            else:
                inputs_used = inputs

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                recon_pred, seg_pred = self.model(inputs_used)
                if not torch.isfinite(recon_pred).all() or not torch.isfinite(seg_pred).all():
                    skipped_batches += 1
                    logging.warning(
                        "epoch=%d phase=%s step=%d skipped due to non-finite model outputs",
                        epoch,
                        phase,
                        step,
                    )
                    continue

                seg_target_loss, confidence_loss, has_label_loss, hotspot_weight = self._resolve_seg_supervision(
                    inputs_used=inputs_used,
                    seg_target_labels=seg_target_labels,
                    confidence_labels=confidence_labels,
                    has_label_labels=has_label_labels,
                    epoch=epoch,
                    training=training,
                )

                total_loss, logs = self.criterion(
                    recon_pred=recon_pred,
                    recon_target=recon_target,
                    valid_mask=inputs_used[:, 1:2],
                    seg_pred=seg_pred,
                    seg_target=seg_target_loss,
                    confidence=confidence_loss,
                    has_label=has_label_loss,
                    hotspot=hotspot_weight,
                )

                if not torch.isfinite(total_loss):
                    skipped_batches += 1
                    logging.warning(
                        "epoch=%d phase=%s step=%d skipped due to non-finite loss",
                        epoch,
                        phase,
                        step,
                    )
                    continue

                if training:
                    total_loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    grad_norm_value = (
                        float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                    )
                    if not math.isfinite(grad_norm_value):
                        skipped_batches += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        logging.warning(
                            "epoch=%d phase=%s step=%d skipped due to non-finite gradient norm",
                            epoch,
                            phase,
                            step,
                        )
                        continue
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

            losses.append(logs["loss_total"])
            recon_losses.append(logs["loss_recon"])
            seg_losses.append(logs["loss_seg"])
            step_f1 = _f1_score_from_maps(seg_pred.detach(), seg_target_labels, has_label_labels)

            tile_prob, tile_target = _tile_level_vectors(seg_pred.detach(), seg_target_labels, has_label_labels)
            if tile_prob.numel() > 0:
                tile_prob_chunks.append(tile_prob)
                tile_target_chunks.append(tile_target)

            pred_bin, target_bin = _extract_labeled_binary_maps(
                prob=seg_pred.detach(),
                target=seg_target_labels,
                has_label=has_label_labels,
                threshold=self.config.eval_metric_threshold,
            )
            if pred_bin.numel() > 0:
                total_tp_px += float(torch.sum(pred_bin & target_bin).detach().cpu().item())
                total_fp_px += float(torch.sum(pred_bin & (~target_bin)).detach().cpu().item())
                total_fn_px += float(torch.sum((~pred_bin) & target_bin).detach().cpu().item())

                if not training and _HAS_BOUNDARY_METRIC_DEPS:
                    max_samples = max(0, int(self.config.boundary_metric_max_samples))
                    if max_samples > 0 and len(cldice_values) < max_samples:
                        remaining = max_samples - len(cldice_values)
                        take = min(remaining, int(pred_bin.shape[0]))
                        if take > 0:
                            batch_cldice, batch_hd95 = _boundary_metrics_from_binary_maps(
                                pred_bin=pred_bin[:take],
                                target_bin=target_bin[:take],
                            )
                            cldice_values.extend(batch_cldice)
                            hd95_values.extend(batch_hd95)

            if step % LOG_EVERY_STEPS == 0:
                logging.info(
                    "epoch=%d phase=%s step=%d loss=%.4f recon=%.4f seg=%.4f f1=%.4f",
                    epoch,
                    phase,
                    step,
                    logs["loss_total"],
                    logs["loss_recon"],
                    logs["loss_seg"],
                    step_f1,
                )

        if tile_prob_chunks:
            all_tile_prob = torch.cat(tile_prob_chunks)
            all_tile_target = torch.cat(tile_target_chunks)
        else:
            all_tile_prob = torch.empty(0, dtype=torch.float32)
            all_tile_target = torch.empty(0, dtype=torch.float32)

        if not losses:
            logging.warning(
                "epoch=%d phase=%s produced no valid batches after filtering non-finite values",
                epoch,
                phase,
            )
            mean_loss = 1e9
            mean_recon = 1e9
            mean_seg = 1e9
        else:
            mean_loss = float(np.mean(losses))
            mean_recon = float(np.mean(recon_losses))
            mean_seg = float(np.mean(seg_losses))

        metrics: dict[str, float] = {
            "loss": mean_loss,
            "loss_recon": mean_recon,
            "loss_seg": mean_seg,
            "f1": _f1_score_from_tile_vectors(all_tile_prob, all_tile_target, threshold=0.5),
            "dice": _dice_iou_from_confusion(total_tp_px, total_fp_px, total_fn_px)[0],
            "iou": _dice_iou_from_confusion(total_tp_px, total_fp_px, total_fn_px)[1],
            "skipped_batches": float(skipped_batches),
        }

        if not training:
            if cldice_values:
                val_cldice = float(np.mean(cldice_values))
                val_hd95 = float(np.mean(hd95_values))
            else:
                val_cldice = 0.0
                val_hd95 = float("inf")

            val_hd95_score = _hd95_to_score(val_hd95)
            val_sota_score = float((metrics["dice"] + metrics["iou"] + val_cldice + val_hd95_score) / 4.0)

            metrics["cldice"] = val_cldice
            metrics["hd95"] = val_hd95
            metrics["hd95_score"] = val_hd95_score
            metrics["sota_score"] = val_sota_score

        if not training and self.config.eval_threshold_sweep_enabled:
            best_f1, best_thr = _best_f1_over_thresholds(
                all_tile_prob,
                all_tile_target,
                self._build_eval_thresholds(),
            )
            metrics["f1_best"] = best_f1
            metrics["f1_best_threshold"] = best_thr

        if skipped_batches > 0:
            logging.warning(
                "epoch=%d phase=%s skipped %d/%d batches due to non-finite values",
                epoch,
                phase,
                skipped_batches,
                len(dataloader),
            )

        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict[str, float]]:
        """Train model and save best checkpoint.

        Parameters
        ----------
        train_loader : DataLoader
            Training dataloader.
        val_loader : DataLoader
            Validation dataloader.

        Returns
        -------
        list[dict[str, float]]
            Epoch-wise metrics history.
        """
        total_steps = max(1, self.config.epochs * max(1, len(train_loader)))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=self.config.eta_min)

        best_val_loss = float("inf")
        if self.monitor_metric in {"loss", "hd95"}:
            best_monitor_value = float("inf")
            monitor_mode = "min"
        else:
            best_monitor_value = float("-inf")
            monitor_mode = "max"

        patience = max(0, int(self.config.early_stopping_patience))
        min_delta = max(0.0, float(self.config.early_stopping_min_delta))
        epochs_without_improvement = 0

        history: list[dict[str, float]] = []

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, epoch=epoch, training=True)
            val_metrics = self._run_epoch(val_loader, epoch=epoch, training=False)

            monitor_value = float(val_metrics.get(self.monitor_metric, float("nan")))
            if not math.isfinite(monitor_value):
                if monitor_mode == "min":
                    monitor_value = float("inf")
                else:
                    monitor_value = float("-inf")

            is_best = _is_metric_improved(
                value=monitor_value,
                best_value=best_monitor_value,
                mode=monitor_mode,
                min_delta=min_delta,
            ) or (not self.best_path.exists())

            if is_best and math.isfinite(monitor_value):
                best_monitor_value = monitor_value
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            row = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_f1": train_metrics["f1"],
                "train_dice": train_metrics.get("dice", 0.0),
                "train_iou": train_metrics.get("iou", 0.0),
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
                "val_dice": val_metrics.get("dice", 0.0),
                "val_iou": val_metrics.get("iou", 0.0),
                "val_cldice": val_metrics.get("cldice", 0.0),
                "val_hd95": val_metrics.get("hd95", float("inf")),
                "val_hd95_score": val_metrics.get("hd95_score", 0.0),
                "val_sota_score": val_metrics.get("sota_score", 0.0),
                "val_monitor_metric_name": self.monitor_metric,
                "val_monitor_metric": monitor_value,
                "best_monitor_metric": best_monitor_value,
                "epochs_without_improvement": float(epochs_without_improvement),
                "train_skipped_batches": train_metrics.get("skipped_batches", 0.0),
                "val_skipped_batches": val_metrics.get("skipped_batches", 0.0),
            }
            if "f1_best" in val_metrics and "f1_best_threshold" in val_metrics:
                row["val_f1_best_thresholded"] = val_metrics["f1_best"]
                row["val_f1_best_threshold"] = val_metrics["f1_best_threshold"]
            history.append(row)

            val_loss_for_compare = float(val_metrics["loss"])
            if not math.isfinite(val_loss_for_compare):
                val_loss_for_compare = float("inf")
            if val_loss_for_compare < best_val_loss:
                best_val_loss = val_loss_for_compare

            self._save_checkpoint(
                epoch=epoch,
                best_monitor_value=best_monitor_value,
                best_val_loss=best_val_loss,
                is_best=is_best,
            )

            if "f1_best" in val_metrics and "f1_best_threshold" in val_metrics:
                logging.info(
                    "epoch=%d train_loss=%.4f val_loss=%.4f val_dice=%.4f val_iou=%.4f val_cldice=%.4f val_hd95=%.4f val_sota=%.4f monitor[%s]=%.4f no_improve=%d val_f1_best=%.4f@thr=%.2f best=%s",
                    epoch,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics.get("dice", 0.0),
                    val_metrics.get("iou", 0.0),
                    val_metrics.get("cldice", 0.0),
                    val_metrics.get("hd95", float("inf")),
                    val_metrics.get("sota_score", 0.0),
                    self.monitor_metric,
                    monitor_value,
                    epochs_without_improvement,
                    val_metrics["f1_best"],
                    val_metrics["f1_best_threshold"],
                    is_best,
                )
            else:
                logging.info(
                    "epoch=%d train_loss=%.4f val_loss=%.4f val_dice=%.4f val_iou=%.4f val_cldice=%.4f val_hd95=%.4f val_sota=%.4f monitor[%s]=%.4f no_improve=%d best=%s",
                    epoch,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics.get("dice", 0.0),
                    val_metrics.get("iou", 0.0),
                    val_metrics.get("cldice", 0.0),
                    val_metrics.get("hd95", float("inf")),
                    val_metrics.get("sota_score", 0.0),
                    self.monitor_metric,
                    monitor_value,
                    epochs_without_improvement,
                    is_best,
                )

            if patience > 0 and epochs_without_improvement >= patience:
                history[-1]["early_stopped"] = 1.0
                logging.info(
                    "Early stopping at epoch=%d monitor[%s]=%.4f best=%.4f patience=%d",
                    epoch,
                    self.monitor_metric,
                    monitor_value,
                    best_monitor_value,
                    patience,
                )
                break

        return history


# ============================================================================
# Section 5: Inference + uncertainty quantification
# ============================================================================


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable dropout layers during evaluation for MC Dropout inference.

    Parameters
    ----------
    model : nn.Module
        Model containing dropout modules.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def preprocess_tile_for_inference(
    tile_array: np.ndarray,
    nodata_threshold: float = NODATA_THRESHOLD,
) -> torch.Tensor:
    """Convert raw tile to model input tensor ``(1,2,TILE_SIZE,TILE_SIZE)``.

    Parameters
    ----------
    tile_array : np.ndarray
        Raw CHM tile array.
    nodata_threshold : float, default=NODATA_THRESHOLD
        Nodata threshold.

    Returns
    -------
    torch.Tensor
        Model-ready tensor.
    """
    arr = tile_array.astype(np.float32)
    invalid = ~np.isfinite(arr)
    invalid |= arr < nodata_threshold
    chm = arr.copy()
    chm[invalid] = 0.0
    valid = (~invalid).astype(np.float32)

    chm_padded = centered_pad(chm, target_size=TILE_SIZE)
    valid_padded = centered_pad(valid, target_size=TILE_SIZE)
    x = np.stack([chm_padded, valid_padded], axis=0)[None, ...]
    return torch.from_numpy(x.astype(np.float32))


def infer(
    model: nn.Module,
    tile_array: np.ndarray,
    device: torch.device,
    n_passes: int = 10,
    uncertainty_threshold: float = 0.05,
) -> dict[str, np.ndarray]:
    """Run MC Dropout inference and return prediction uncertainty maps.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model.
    tile_array : np.ndarray
        Raw CHM tile.
    device : torch.device
        Inference device.
    n_passes : int, default=10
        Number of stochastic dropout passes.
    uncertainty_threshold : float, default=0.05
        Variance threshold used to suppress uncertain predictions.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys ``prediction``, ``uncertainty``, and ``raw_passes``.

    Notes
    -----
    On Bernoulli-like output probabilities in ``[0,1]``, variance values above
    roughly ``0.05`` indicate unstable predictions and are commonly linked to
    hallucination in sparse or nodata-heavy regions. These pixels are set to 0.
    """
    x = preprocess_tile_for_inference(tile_array).to(device)
    model.eval()
    enable_mc_dropout(model)

    passes: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_passes):
            _, seg = model(x)
            passes.append(seg[0, 0].detach().cpu().numpy())

    raw = np.stack(passes, axis=0)
    mean_pred = np.mean(raw, axis=0)
    var_pred = np.var(raw, axis=0)

    filtered = mean_pred.copy()
    filtered[var_pred > uncertainty_threshold] = 0.0
    return {
        "prediction": filtered.astype(np.float32),
        "uncertainty": var_pred.astype(np.float32),
        "raw_passes": raw.astype(np.float32),
    }


def read_raster_tile(raster_path: str | Path, row_off: int, col_off: int, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    """Read one tile from a GeoTIFF raster.

    Parameters
    ----------
    raster_path : str | Path
        Input raster path.
    row_off : int
        Row offset.
    col_off : int
        Column offset.
    chunk_size : int, default=CHUNK_SIZE
        Tile side length.

    Returns
    -------
    np.ndarray
        Tile array shaped ``(chunk_size, chunk_size)``.
    """
    with rasterio.open(raster_path) as src:
        window = Window(col_off, row_off, chunk_size, chunk_size)
        tile = src.read(1, window=window, boundless=True, fill_value=np.nan)
    if tile.shape != (chunk_size, chunk_size):
        out = np.full((chunk_size, chunk_size), np.nan, dtype=np.float32)
        h = min(chunk_size, tile.shape[0])
        w = min(chunk_size, tile.shape[1])
        out[:h, :w] = tile[:h, :w]
        return out.astype(np.float32)
    return tile.astype(np.float32)


def load_partialconv_checkpoint(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    """Load PartialConvUNet checkpoint.

    Parameters
    ----------
    checkpoint_path : str | Path
        Checkpoint path.
    device : torch.device
        Target device.

    Returns
    -------
    nn.Module
        Loaded model.
    """
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PartialConvUNet().to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ============================================================================
# CLI entry points
# ============================================================================


def _build_dataloaders(
    split: SplitResult,
    batch_size: int,
    num_workers: int,
    label_store: Optional[PseudoLabelStore] = None,
    cam_mask_dir: Optional[Path] = None,
    cam_mask_index: Optional[set[str]] = None,
) -> tuple[DataLoader, DataLoader, PseudoLabelStore]:
    """Create train and validation dataloaders from split object.

    Parameters
    ----------
    split : SplitResult
        Strict split results.
    batch_size : int
        Batch size.
    num_workers : int
        DataLoader workers.
    label_store : PseudoLabelStore | None, default=None
        Existing label store. When ``None``, a store is built from split tiles.
    cam_mask_dir : Path | None, default=None
        Optional directory with offline CAM supervision artifacts.
    cam_mask_index : set[str] | None, default=None
        Optional pre-indexed CAM mask stems.

    Returns
    -------
    tuple[DataLoader, DataLoader, PseudoLabelStore]
        Train loader, validation loader, and label store.
    """
    if label_store is None:
        label_store = PseudoLabelStore.from_tiles(split.train + split.val + split.test)

    train_ds = CHMDataset(
        split.train,
        label_store=label_store,
        augment=True,
        cam_mask_dir=cam_mask_dir,
        cam_mask_index=cam_mask_index,
    )
    val_ds = CHMDataset(
        split.val,
        label_store=label_store,
        augment=False,
        cam_mask_dir=cam_mask_dir,
        cam_mask_index=cam_mask_index,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, label_store


def _count_labeled_tiles(tiles: list[TileRecord], label_store: PseudoLabelStore) -> int:
    """Count how many tiles currently have label entries.

    Parameters
    ----------
    tiles : list[TileRecord]
        Tile list to inspect.
    label_store : PseudoLabelStore
        Label lookup store.

    Returns
    -------
    int
        Number of tiles with existing labels.
    """
    return sum(1 for tile in tiles if label_store.has_tile_label(tile))


def _cam_mask_path_for_tile(tile: TileRecord, cam_mask_dir: Path) -> Path:
    """Build expected CAM mask path for one tile.

    Parameters
    ----------
    tile : TileRecord
        Tile descriptor.
    cam_mask_dir : Path
        Artifact directory.

    Returns
    -------
    Path
        Expected mask file path.
    """
    return cam_mask_dir / f"{tile_id_to_artifact_stem(tile.tile_id)}_mask.npy"


def _count_cam_mask_tiles(
    tiles: list[TileRecord],
    cam_mask_dir: Path,
    mask_index: Optional[set[str]] = None,
) -> int:
    """Count tiles with available offline CAM masks.

    Parameters
    ----------
    tiles : list[TileRecord]
        Tile list to inspect.
    cam_mask_dir : Path
        Directory containing ``*_mask.npy`` artifacts.
    mask_index : set[str] | None, default=None
        Optional pre-indexed CAM mask stems.

    Returns
    -------
    int
        Number of tiles with available CAM masks.
    """
    if mask_index is not None:
        return sum(1 for tile in tiles if tile_id_to_artifact_stem(tile.tile_id) in mask_index)
    return sum(1 for tile in tiles if _cam_mask_path_for_tile(tile, cam_mask_dir).exists())


def _summarize_tile_label_stats(tiles: list[TileRecord], label_store: PseudoLabelStore) -> dict[str, float]:
    """Summarize label availability and class balance for a tile list.

    Parameters
    ----------
    tiles : list[TileRecord]
        Tile list to inspect.
    label_store : PseudoLabelStore
        Label lookup store.

    Returns
    -------
    dict[str, float]
        Summary including coverage, class counts, and mean confidence.
    """
    total_tiles = len(tiles)
    labeled_tiles = 0
    positive_tiles = 0
    negative_tiles = 0
    confidences: list[float] = []
    labeled_sample_ids: set[str] = set()

    for tile in tiles:
        label_entry = label_store.get_tile_label(tile)
        if label_entry is None:
            continue

        label_value, confidence, _ = label_entry
        labeled_tiles += 1
        labeled_sample_ids.add(tile.sample_id)
        confidences.append(float(confidence))

        if float(label_value) >= 0.5:
            positive_tiles += 1
        else:
            negative_tiles += 1

    unlabeled_tiles = max(0, total_tiles - labeled_tiles)
    coverage = float(labeled_tiles / total_tiles) if total_tiles > 0 else 0.0
    pos_ratio = float(positive_tiles / labeled_tiles) if labeled_tiles > 0 else 0.0
    neg_ratio = float(negative_tiles / labeled_tiles) if labeled_tiles > 0 else 0.0
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0

    return {
        "total_tiles": float(total_tiles),
        "labeled_tiles": float(labeled_tiles),
        "unlabeled_tiles": float(unlabeled_tiles),
        "positive_tiles": float(positive_tiles),
        "negative_tiles": float(negative_tiles),
        "coverage": coverage,
        "positive_ratio": pos_ratio,
        "negative_ratio": neg_ratio,
        "mean_confidence": mean_confidence,
        "labeled_samples": float(len(labeled_sample_ids)),
    }


def _log_round_data_sufficiency(
    round_idx: int,
    split: SplitResult,
    label_store: PseudoLabelStore,
    batch_size: int,
) -> None:
    """Log data sufficiency diagnostics for one training round.

    Parameters
    ----------
    round_idx : int
        Current 1-based self-training round.
    split : SplitResult
        Active train/val/test split.
    label_store : PseudoLabelStore
        Current label store.
    batch_size : int
        Training batch size used for practical warning thresholds.
    """
    train_stats = _summarize_tile_label_stats(split.train, label_store)
    val_stats = _summarize_tile_label_stats(split.val, label_store)

    logging.info(
        (
            "Round %d data coverage: "
            "train labeled=%d/%d (%.1f%%, pos=%d, neg=%d, samples=%d, mean_conf=%.3f); "
            "val labeled=%d/%d (%.1f%%, pos=%d, neg=%d, samples=%d, mean_conf=%.3f)"
        ),
        round_idx,
        int(train_stats["labeled_tiles"]),
        int(train_stats["total_tiles"]),
        100.0 * train_stats["coverage"],
        int(train_stats["positive_tiles"]),
        int(train_stats["negative_tiles"]),
        int(train_stats["labeled_samples"]),
        train_stats["mean_confidence"],
        int(val_stats["labeled_tiles"]),
        int(val_stats["total_tiles"]),
        100.0 * val_stats["coverage"],
        int(val_stats["positive_tiles"]),
        int(val_stats["negative_tiles"]),
        int(val_stats["labeled_samples"]),
        val_stats["mean_confidence"],
    )

    min_train_labeled = max(256, batch_size * 20)
    if int(train_stats["labeled_tiles"]) < min_train_labeled:
        logging.warning(
            "Round %d may be data-limited: labeled train tiles=%d (recommended >= %d)",
            round_idx,
            int(train_stats["labeled_tiles"]),
            min_train_labeled,
        )

    if int(train_stats["positive_tiles"]) == 0 or int(train_stats["negative_tiles"]) == 0:
        logging.warning(
            "Round %d train labels miss a class: positives=%d negatives=%d",
            round_idx,
            int(train_stats["positive_tiles"]),
            int(train_stats["negative_tiles"]),
        )
    else:
        minority_ratio = min(train_stats["positive_ratio"], train_stats["negative_ratio"])
        if minority_ratio < 0.1:
            logging.warning(
                "Round %d severe train class imbalance: pos_ratio=%.3f neg_ratio=%.3f",
                round_idx,
                train_stats["positive_ratio"],
                train_stats["negative_ratio"],
            )

    min_val_labeled = max(32, batch_size * 4)
    if int(val_stats["labeled_tiles"]) < min_val_labeled:
        logging.warning(
            "Round %d validation may be noisy: labeled val tiles=%d (recommended >= %d)",
            round_idx,
            int(val_stats["labeled_tiles"]),
            min_val_labeled,
        )


def generate_self_training_pseudo_labels(
    model: nn.Module,
    tiles: list[TileRecord],
    label_store: PseudoLabelStore,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pos_threshold: float,
    neg_threshold: float,
    max_new_labels: int,
    round_index: int,
) -> dict[str, int]:
    """Generate and add high-confidence pseudo labels for unlabeled train tiles.

    Parameters
    ----------
    model : nn.Module
        Current segmentation model.
    tiles : list[TileRecord]
        Candidate training tiles.
    label_store : PseudoLabelStore
        Label store to be updated.
    device : torch.device
        Inference device.
    batch_size : int
        Batch size used during pseudo-label generation.
    num_workers : int
        Dataloader workers.
    pos_threshold : float
        Minimum mean tile probability for assigning positive pseudo labels.
    neg_threshold : float
        Maximum mean tile probability for assigning negative pseudo labels.
    max_new_labels : int
        Maximum number of labels to add in this round.
    round_index : int
        1-based self-training round index.

    Returns
    -------
    dict[str, int]
        Summary with keys ``unlabeled_pool``, ``candidate_tiles``, and ``added``.
    """
    pos_threshold = float(np.clip(pos_threshold, 0.5, 1.0))
    neg_threshold = float(np.clip(neg_threshold, 0.0, 0.5))
    if neg_threshold >= pos_threshold:
        neg_threshold = max(0.0, pos_threshold - 0.01)

    unlabeled_tiles = [tile for tile in tiles if not label_store.has_tile_label(tile)]
    if not unlabeled_tiles:
        return {"unlabeled_pool": 0, "candidate_tiles": 0, "added": 0}

    eval_ds = CHMDataset(unlabeled_tiles, label_store=label_store, augment=False)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    by_tile_id = {tile.tile_id: tile for tile in unlabeled_tiles}
    candidates: list[tuple[float, float, TileRecord]] = []

    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            inputs = batch["input"].to(device)
            _, seg_pred = model(inputs)
            tile_scores = seg_pred.mean(dim=(1, 2, 3)).detach().cpu().tolist()
            tile_ids = [str(tid) for tid in batch["tile_id"]]

            for tile_id, score in zip(tile_ids, tile_scores):
                tile = by_tile_id.get(tile_id)
                if tile is None:
                    continue

                if score >= pos_threshold:
                    label = 1.0
                    conf = float(score)
                elif score <= neg_threshold:
                    label = 0.0
                    conf = float(1.0 - score)
                else:
                    continue
                candidates.append((conf, label, tile))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if max_new_labels > 0:
        candidates = candidates[:max_new_labels]

    source = f"self_train_round_{round_index}"
    added = 0
    for conf, label, tile in candidates:
        if label_store.has_tile_label(tile):
            continue
        label_store.set_tile_label(tile, label_value=label, confidence=conf, source=source)
        added += 1

    return {
        "unlabeled_pool": len(unlabeled_tiles),
        "candidate_tiles": len(candidates),
        "added": added,
    }


def run_training(args: argparse.Namespace) -> None:
    """Run full training routine from CLI args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI args.
    """
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    split = create_strict_splits(
        data_dir=args.data_dir,
        test_size=args.test_size,
        buffer_meters=args.buffer_meters,
        registry_dir=args.registry_dir,
        val_size=args.val_size,
        max_tiles_per_raster=args.max_tiles_per_raster,
        seed=args.seed,
    )

    model = PartialConvUNet(dropout_p=0.3).to(device)
    legacy_model = load_legacy_model(args.legacy_checkpoint, device=device)

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eta_min=args.eta_min,
        recon_weight=1.0,
        seg_weight=5.0,
        max_grad_norm=args.max_grad_norm,
        eval_threshold_sweep_min=args.eval_threshold_sweep_min,
        eval_threshold_sweep_max=args.eval_threshold_sweep_max,
        eval_threshold_sweep_step=args.eval_threshold_sweep_step,
        eval_threshold_sweep_enabled=not args.disable_eval_threshold_sweep,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        monitor_metric=str(args.monitor_metric).strip().lower(),
        eval_metric_threshold=args.eval_metric_threshold,
        boundary_metric_max_samples=args.boundary_metric_max_samples,
        seg_target_source=str(args.seg_target_source).strip().lower(),
        hotspot_conf_min=float(np.clip(args.hotspot_conf_min, 0.0, 1.0)),
        hotspot_conf_gamma=float(max(args.hotspot_conf_gamma, EPS)),
    )
    criterion = JointLoss(recon_weight=config.recon_weight, seg_weight=config.seg_weight)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_rounds = int(args.self_train_rounds)
    rounds = max(1, min(3, requested_rounds))
    if requested_rounds != rounds:
        logging.info("Capped self-training rounds from %d to %d", requested_rounds, rounds)

    label_store = PseudoLabelStore.from_tiles(split.train + split.val + split.test)
    cam_mask_dir: Optional[Path] = None
    cam_mask_index: Optional[set[str]] = None
    if str(args.cam_mask_dir).strip():
        candidate = Path(args.cam_mask_dir)
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"CAM mask directory does not exist or is not a directory: {candidate}")
        cam_mask_dir = candidate
        cam_mask_index = _build_cam_artifact_index(cam_mask_dir)
        train_cam = _count_cam_mask_tiles(split.train, cam_mask_dir, mask_index=cam_mask_index)
        val_cam = _count_cam_mask_tiles(split.val, cam_mask_dir, mask_index=cam_mask_index)
        train_ratio = (100.0 * train_cam / max(1, len(split.train)))
        val_ratio = (100.0 * val_cam / max(1, len(split.val)))
        logging.info(
            "Using offline CAM masks from %s (indexed=%d, train=%d/%d %.1f%%, val=%d/%d %.1f%%)",
            cam_mask_dir,
            len(cam_mask_index),
            train_cam,
            len(split.train),
            train_ratio,
            val_cam,
            len(split.val),
            val_ratio,
        )

    history: list[dict[str, Any]] = []
    self_training_summary: list[dict[str, Any]] = []
    final_model = model
    final_val_loader: Optional[DataLoader] = None

    for round_idx in range(1, rounds + 1):
        round_out_dir = out_dir / f"round_{round_idx}" if rounds > 1 else out_dir
        labeled_before = _count_labeled_tiles(split.train, label_store)
        _log_round_data_sufficiency(
            round_idx=round_idx,
            split=split,
            label_store=label_store,
            batch_size=args.batch_size,
        )

        train_loader, val_loader, _ = _build_dataloaders(
            split=split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label_store=label_store,
            cam_mask_dir=cam_mask_dir,
            cam_mask_index=cam_mask_index,
        )

        trainer = Trainer(
            model=final_model,
            criterion=criterion,
            config=config,
            device=device,
            output_dir=round_out_dir,
            legacy_model=legacy_model,
        )
        round_history = trainer.fit(train_loader=train_loader, val_loader=val_loader)
        for row in round_history:
            row_with_round = dict(row)
            row_with_round["round"] = float(round_idx)
            history.append(row_with_round)

        final_model = trainer.model
        final_val_loader = val_loader

        round_summary: dict[str, Any] = {
            "round": round_idx,
            "train_tiles": len(split.train),
            "val_tiles": len(split.val),
            "labeled_train_tiles_before": labeled_before,
            "best_checkpoint": str(trainer.best_path),
            "last_checkpoint": str(trainer.last_path),
        }

        if round_idx < rounds:
            pseudo_stats = generate_self_training_pseudo_labels(
                model=final_model,
                tiles=split.train,
                label_store=label_store,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pos_threshold=args.self_train_pos_threshold,
                neg_threshold=args.self_train_neg_threshold,
                max_new_labels=args.self_train_max_new_labels,
                round_index=round_idx,
            )
            round_summary.update(
                {
                    "self_training_unlabeled_pool": pseudo_stats["unlabeled_pool"],
                    "self_training_candidates": pseudo_stats["candidate_tiles"],
                    "self_training_added": pseudo_stats["added"],
                }
            )
            if pseudo_stats["added"] <= 0:
                round_summary["stopped_early"] = True
                round_summary["labeled_train_tiles_after"] = _count_labeled_tiles(split.train, label_store)
                self_training_summary.append(round_summary)
                logging.info(
                    "Stopping iterative self-training at round %d: no new pseudo labels added",
                    round_idx,
                )
                break

        round_summary["labeled_train_tiles_after"] = _count_labeled_tiles(split.train, label_store)
        self_training_summary.append(round_summary)

    (out_dir / "split_metadata.json").write_text(json.dumps(split.metadata, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "self_training_summary.json").write_text(
        json.dumps(self_training_summary, indent=2),
        encoding="utf-8",
    )

    if args.benchmark_hotspots:
        if final_val_loader is None:
            raise RuntimeError("Validation dataloader is unavailable for hotspot benchmarking")
        bench = benchmark_hotspot_methods(
            legacy_model=legacy_model,
            model=final_model,
            dataloader=final_val_loader,
            device=device,
            max_batches=args.benchmark_batches,
        )
        (out_dir / "hotspot_benchmark.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")
        logging.info("Hotspot benchmark: %s", json.dumps(bench, indent=2))


def run_inference_cli(args: argparse.Namespace) -> None:
    """Run tile-level inference from CLI args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI args.
    """
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_partialconv_checkpoint(args.checkpoint, device=device)

    tile = read_raster_tile(
        raster_path=args.raster,
        row_off=args.row_off,
        col_off=args.col_off,
        chunk_size=args.chunk_size,
    )

    result = infer(
        model=model,
        tile_array=tile,
        device=device,
        n_passes=args.n_passes,
        uncertainty_threshold=args.uncertainty_threshold,
    )

    summary = {
        "prediction_mean": float(result["prediction"].mean()),
        "prediction_max": float(result["prediction"].max()),
        "uncertainty_mean": float(result["uncertainty"].mean()),
        "uncertainty_max": float(result["uncertainty"].max()),
    }
    logging.info("Inference summary: %s", json.dumps(summary, indent=2))

    if args.output_npz:
        output_path = Path(args.output_npz)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **result)
        logging.info("Saved inference outputs to %s", output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for training and inference.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser(description="PartialConv U-Net CWD pipeline")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train PartialConv U-Net")
    train_p.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    train_p.add_argument("--registry-dir", type=str, default=str(DEFAULT_REGISTRY_DIR))
    train_p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    train_p.add_argument("--legacy-checkpoint", type=str, default=str(DEFAULT_LEGACY_CHECKPOINT))
    train_p.add_argument("--epochs", type=int, default=40)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--num-workers", type=int, default=4)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--eta-min", type=float, default=1e-6)
    train_p.add_argument("--max-grad-norm", type=float, default=5.0)
    train_p.add_argument("--test-size", type=float, default=0.2)
    train_p.add_argument("--val-size", type=float, default=0.1)
    train_p.add_argument("--buffer-meters", type=float, default=50.0)
    train_p.add_argument("--max-tiles-per-raster", type=int, default=2500)
    train_p.add_argument("--benchmark-hotspots", action="store_true")
    train_p.add_argument("--benchmark-batches", type=int, default=8)
    train_p.add_argument("--self-train-rounds", type=int, default=1)
    train_p.add_argument("--self-train-pos-threshold", type=float, default=0.9)
    train_p.add_argument("--self-train-neg-threshold", type=float, default=0.1)
    train_p.add_argument("--self-train-max-new-labels", type=int, default=2000)
    train_p.add_argument("--eval-threshold-sweep-min", type=float, default=0.1)
    train_p.add_argument("--eval-threshold-sweep-max", type=float, default=0.5)
    train_p.add_argument("--eval-threshold-sweep-step", type=float, default=0.05)
    train_p.add_argument("--disable-eval-threshold-sweep", action="store_true")
    train_p.add_argument("--early-stopping-patience", type=int, default=5)
    train_p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    train_p.add_argument(
        "--monitor-metric",
        type=str,
        default="sota_score",
        choices=["sota_score", "dice", "iou", "cldice", "hd95", "f1", "loss"],
        help="Validation metric used for best checkpoint selection and early stopping",
    )
    train_p.add_argument(
        "--eval-metric-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to binarize segmentation for Dice/IoU/clDice/HD95",
    )
    train_p.add_argument(
        "--boundary-metric-max-samples",
        type=int,
        default=128,
        help="Maximum labeled validation tiles per epoch used for clDice and HD95",
    )
    train_p.add_argument(
        "--cam-mask-dir",
        type=str,
        default="",
        help="Optional directory with offline Grad-CAM++ + Otsu mask artifacts (*.npy)",
    )
    train_p.add_argument(
        "--seg-target-source",
        type=str,
        default="labels",
        choices=["labels", "hotspot", "labels_or_hotspot"],
        help=(
            "Segmentation supervision source: labels (default), hotspot (use legacy ConvNeXt Grad-CAM++), "
            "or labels_or_hotspot (fallback to hotspot for unlabeled tiles)"
        ),
    )
    train_p.add_argument(
        "--hotspot-conf-min",
        type=float,
        default=0.05,
        help="Minimum confidence floor for hotspot-derived supervision maps",
    )
    train_p.add_argument(
        "--hotspot-conf-gamma",
        type=float,
        default=1.0,
        help="Hotspot confidence exponent: >1 sharper borders, <1 flatter confidence",
    )
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--device", type=str, default="")

    infer_p = subparsers.add_parser("infer", help="Inference with MC Dropout")
    infer_p.add_argument("--checkpoint", type=str, required=True)
    infer_p.add_argument("--raster", type=str, required=True)
    infer_p.add_argument("--row-off", type=int, required=True)
    infer_p.add_argument("--col-off", type=int, required=True)
    infer_p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    infer_p.add_argument("--n-passes", type=int, default=10)
    infer_p.add_argument("--uncertainty-threshold", type=float, default=0.05)
    infer_p.add_argument("--output-npz", type=str, default="")
    infer_p.add_argument("--device", type=str, default="")

    return parser


def main() -> None:
    """CLI main function.

    Returns
    -------
    None
        This function is executed for side effects.
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if args.command == "train":
        run_training(args)
    elif args.command == "infer":
        run_inference_cli(args)
    else:  # pragma: no cover - argparse enforces known commands
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

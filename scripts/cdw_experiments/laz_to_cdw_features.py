#!/usr/bin/env python3
"""Standalone CDW feature experiments for tile-based multi-year LAZ inputs.

This script implements a separate experimental pipeline (no edits to existing
production scripts) to compare LAZ -> CHM feature engineering ideas for CDW.

Implemented experiment matrix
- Phase 1 DTM variants:
  - `median_ground`: median Z from class-2 ground points across years.
  - `lowest_all`: lowest Z from all points across years.
- Local fallback windows:
  - 1-2 m^2 neighborhood gating for year-local ground fallback when
    |year-local - multi-year reference| > threshold.
- Phase 2 filtering:
  - Exclude only LAS classes 6 (building) and 9 (water).
  - Keep classes 7 and 18 included.
- Phase 3/4/5 per-output layers:
    - Single CHM 0.0-1.3m (for direct comparison against original baseline)
    - Optional RGB split bands:
        - Band 1: max HAG in [0.0, 0.4)
        - Band 2: max HAG in [0.4, 0.7)
        - Band 3: max HAG in [0.7, 1.3]
    - Intensity layer: average normalized intensity in [0.4, 1.3]
    - Density layer: point density in [0.0, 1.3]
- Return modes:
  - `all`, `last`, `last2`

Outputs are GeoTIFF rasters suitable for QGIS comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence

import laspy
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from scipy.ndimage import maximum_filter


NODATA_FLOAT = -9999.0


@dataclass(frozen=True)
class RasterGrid:
    origin_x: float
    max_y: float
    resolution: float
    width: int
    height: int

    @property
    def size(self) -> int:
        return self.width * self.height

    def xy_to_row_col(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = ((x - self.origin_x) / self.resolution).astype(np.int64)
        row = ((self.max_y - y) / self.resolution).astype(np.int64)
        return row, col


@dataclass(frozen=True)
class LocalFallbackConfig:
    threshold_m: float
    min_window_m2: float
    max_window_m2: float


def setup_logger(log_path: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("cdw_experiments")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def snap_origin(min_x: float, min_y: float, resolution: float) -> tuple[float, float]:
    return (
        math.floor(min_x / resolution) * resolution,
        math.floor(min_y / resolution) * resolution,
    )


def discover_tile_laz_files(input_dir: Path, tile_id: str, years: Sequence[int] | None) -> list[Path]:
    pattern = f"{tile_id}_*_madal.laz"
    files = sorted(input_dir.glob(pattern))
    if years is None:
        return files

    year_set = set(years)
    selected: list[Path] = []
    for path in files:
        year = extract_year(path.name)
        if year is not None and year in year_set:
            selected.append(path)
    return selected


def extract_year(name: str) -> int | None:
    m = re.search(r"_(\d{4})_", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def build_common_grid(laz_paths: Sequence[Path], resolution: float) -> RasterGrid:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for path in laz_paths:
        with laspy.open(path) as fh:
            hdr = fh.header
            min_x = min(min_x, float(hdr.mins[0]))
            min_y = min(min_y, float(hdr.mins[1]))
            max_x = max(max_x, float(hdr.maxs[0]))
            max_y = max(max_y, float(hdr.maxs[1]))

    ox, oy = snap_origin(min_x, min_y, resolution)
    width = int(math.ceil((max_x - ox) / resolution))
    height = int(math.ceil((max_y - oy) / resolution))
    return RasterGrid(origin_x=ox, max_y=max_y, resolution=resolution, width=width, height=height)


def _read_chunk_arrays(points: laspy.ScaleAwarePointRecord) -> dict[str, np.ndarray | None]:
    arr: dict[str, np.ndarray | None] = {
        "x": np.asarray(points.x, dtype=np.float64),
        "y": np.asarray(points.y, dtype=np.float64),
        "z": np.asarray(points.z, dtype=np.float32),
        "classification": None,
        "return_number": None,
        "number_of_returns": None,
        "intensity": None,
    }

    try:
        arr["classification"] = np.asarray(points.classification, dtype=np.int16)
    except Exception:
        pass

    try:
        arr["return_number"] = np.asarray(points.return_number, dtype=np.int16)
        arr["number_of_returns"] = np.asarray(points.number_of_returns, dtype=np.int16)
    except Exception:
        pass

    try:
        arr["intensity"] = np.asarray(points.intensity, dtype=np.float32)
    except Exception:
        pass

    return arr


def _class_mask(classification: np.ndarray | None, n: int, exclude_classes: set[int]) -> np.ndarray:
    if classification is None or not exclude_classes:
        return np.ones(n, dtype=bool)
    mask = np.ones(n, dtype=bool)
    for cls in exclude_classes:
        mask &= classification != cls
    return mask


def _return_mask(
    return_number: np.ndarray | None,
    number_of_returns: np.ndarray | None,
    n: int,
    return_mode: str,
) -> np.ndarray:
    if return_mode == "all":
        return np.ones(n, dtype=bool)

    if return_number is None or number_of_returns is None:
        return np.ones(n, dtype=bool)

    if return_mode == "last":
        return return_number == number_of_returns

    if return_mode == "last2":
        cutoff = np.maximum(1, number_of_returns - 1)
        return return_number >= cutoff

    raise ValueError(f"Unsupported return mode: {return_mode}")


def _valid_rc_mask(row: np.ndarray, col: np.ndarray, grid: RasterGrid) -> np.ndarray:
    return (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)


def _to_flat_index(row: np.ndarray, col: np.ndarray, grid: RasterGrid) -> np.ndarray:
    return row * grid.width + col


def _collect_cell_z_for_median(
    laz_paths: Sequence[Path],
    grid: RasterGrid,
    chunk_size: int,
    use_ground_only: bool,
    exclude_classes: set[int],
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    flat_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []

    for path in laz_paths:
        logger.info("Collecting points for median surface: %s", path.name)
        with laspy.open(path) as fh:
            for points in fh.chunk_iterator(chunk_size):
                d = _read_chunk_arrays(points)
                x = d["x"]
                y = d["y"]
                z = d["z"]
                cls = d["classification"]
                n = len(x)

                keep = _class_mask(cls, n, exclude_classes)
                if use_ground_only and cls is not None:
                    keep &= cls == 2

                if not np.any(keep):
                    continue

                xk = x[keep]
                yk = y[keep]
                zk = z[keep]

                row, col = grid.xy_to_row_col(xk, yk)
                valid = _valid_rc_mask(row, col, grid)
                if not np.any(valid):
                    continue

                flat_parts.append(_to_flat_index(row[valid], col[valid], grid).astype(np.int64))
                z_parts.append(zk[valid].astype(np.float32))

    if not flat_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    flat_idx = np.concatenate(flat_parts)
    z_vals = np.concatenate(z_parts)
    return flat_idx, z_vals


def _median_raster_from_points(flat_idx: np.ndarray, z_vals: np.ndarray, grid: RasterGrid) -> np.ndarray:
    out = np.full(grid.size, NODATA_FLOAT, dtype=np.float32)
    if flat_idx.size == 0:
        return out.reshape((grid.height, grid.width))

    # Sort by cell index then z so per-cell medians can be selected by midpoint.
    order = np.lexsort((z_vals, flat_idx))
    idx_sorted = flat_idx[order]
    z_sorted = z_vals[order]

    unique_idx, starts, counts = np.unique(idx_sorted, return_index=True, return_counts=True)
    mid_lo = starts + (counts - 1) // 2
    mid_hi = starts + counts // 2
    med = (z_sorted[mid_lo] + z_sorted[mid_hi]) * 0.5

    out[unique_idx] = med.astype(np.float32)
    return out.reshape((grid.height, grid.width))


def build_reference_surface(
    laz_paths: Sequence[Path],
    grid: RasterGrid,
    method: str,
    chunk_size: int,
    exclude_classes: set[int],
    logger: logging.Logger,
) -> np.ndarray:
    if method == "median_ground":
        flat_idx, z_vals = _collect_cell_z_for_median(
            laz_paths=laz_paths,
            grid=grid,
            chunk_size=chunk_size,
            use_ground_only=True,
            exclude_classes=exclude_classes,
            logger=logger,
        )

        if flat_idx.size == 0:
            logger.warning("No class-2 points found; fallback to all-points median for reference DTM")
            flat_idx, z_vals = _collect_cell_z_for_median(
                laz_paths=laz_paths,
                grid=grid,
                chunk_size=chunk_size,
                use_ground_only=False,
                exclude_classes=exclude_classes,
                logger=logger,
            )

        return _median_raster_from_points(flat_idx, z_vals, grid)

    if method == "median_all":
        flat_idx, z_vals = _collect_cell_z_for_median(
            laz_paths=laz_paths,
            grid=grid,
            chunk_size=chunk_size,
            use_ground_only=False,
            exclude_classes=exclude_classes,
            logger=logger,
        )
        return _median_raster_from_points(flat_idx, z_vals, grid)

    if method == "lowest_all":
        lowest = np.full(grid.size, np.inf, dtype=np.float32)
        for path in laz_paths:
            logger.info("Collecting points for lowest-all surface: %s", path.name)
            with laspy.open(path) as fh:
                for points in fh.chunk_iterator(chunk_size):
                    d = _read_chunk_arrays(points)
                    x = d["x"]
                    y = d["y"]
                    z = d["z"]
                    cls = d["classification"]
                    n = len(x)

                    keep = _class_mask(cls, n, exclude_classes)
                    if not np.any(keep):
                        continue

                    xk = x[keep]
                    yk = y[keep]
                    zk = z[keep]

                    row, col = grid.xy_to_row_col(xk, yk)
                    valid = _valid_rc_mask(row, col, grid)
                    if not np.any(valid):
                        continue

                    flat = _to_flat_index(row[valid], col[valid], grid)
                    np.minimum.at(lowest, flat, zk[valid])

        out = np.full(grid.size, NODATA_FLOAT, dtype=np.float32)
        valid = np.isfinite(lowest)
        out[valid] = lowest[valid]
        return out.reshape((grid.height, grid.width))

    if method == "lowest_ground":
        lowest = np.full(grid.size, np.inf, dtype=np.float32)
        for path in laz_paths:
            logger.info("Collecting ground points for lowest-ground surface: %s", path.name)
            with laspy.open(path) as fh:
                for points in fh.chunk_iterator(chunk_size):
                    d = _read_chunk_arrays(points)
                    x = d["x"]
                    y = d["y"]
                    z = d["z"]
                    cls = d["classification"]
                    n = len(x)

                    # First remove explicitly excluded classes, then require ground class (2)
                    keep = _class_mask(cls, n, exclude_classes)
                    if cls is None:
                        continue
                    keep &= cls == 2
                    if not np.any(keep):
                        continue

                    xk = x[keep]
                    yk = y[keep]
                    zk = z[keep]

                    row, col = grid.xy_to_row_col(xk, yk)
                    valid = _valid_rc_mask(row, col, grid)
                    if not np.any(valid):
                        continue

                    flat = _to_flat_index(row[valid], col[valid], grid)
                    np.minimum.at(lowest, flat, zk[valid])

        out = np.full(grid.size, NODATA_FLOAT, dtype=np.float32)
        valid = np.isfinite(lowest)
        out[valid] = lowest[valid]
        return out.reshape((grid.height, grid.width))

    raise ValueError(f"Unsupported reference method: {method}")


def build_year_local_surface(
    laz_path: Path,
    grid: RasterGrid,
    method: str,
    chunk_size: int,
    exclude_classes: set[int],
    logger: logging.Logger,
) -> np.ndarray:
    if method in ("median_ground", "median_all"):
        if method == "median_ground":
            flat_idx, z_vals = _collect_cell_z_for_median(
                laz_paths=[laz_path],
                grid=grid,
                chunk_size=chunk_size,
                use_ground_only=True,
                exclude_classes=exclude_classes,
                logger=logger,
            )
            if flat_idx.size == 0:
                logger.warning("No class-2 points in %s; fallback to all-points median", laz_path.name)
                flat_idx, z_vals = _collect_cell_z_for_median(
                    laz_paths=[laz_path],
                    grid=grid,
                    chunk_size=chunk_size,
                    use_ground_only=False,
                    exclude_classes=exclude_classes,
                    logger=logger,
                )
        else:
            flat_idx, z_vals = _collect_cell_z_for_median(
                laz_paths=[laz_path],
                grid=grid,
                chunk_size=chunk_size,
                use_ground_only=False,
                exclude_classes=exclude_classes,
                logger=logger,
            )
        return _median_raster_from_points(flat_idx, z_vals, grid)

    if method in ("lowest_all", "lowest_ground"):
        return build_reference_surface(
            laz_paths=[laz_path],
            grid=grid,
            method=method,
            chunk_size=chunk_size,
            exclude_classes=exclude_classes,
            logger=logger,
        )

    raise ValueError(f"Unsupported local surface method: {method}")


def apply_local_fallback(
    reference_surface: np.ndarray,
    local_surface: np.ndarray,
    fallback: LocalFallbackConfig,
    grid: RasterGrid,
) -> tuple[np.ndarray, np.ndarray]:
    ref_valid = reference_surface != NODATA_FLOAT
    local_valid = local_surface != NODATA_FLOAT

    diff = np.zeros_like(reference_surface, dtype=np.float32)
    both = ref_valid & local_valid
    diff[both] = np.abs(local_surface[both] - reference_surface[both])

    # Convert requested 1-2 m^2 windows into odd kernel sizes for local gating.
    side_min = max(grid.resolution, math.sqrt(fallback.min_window_m2))
    side_max = max(grid.resolution, math.sqrt(fallback.max_window_m2))
    k_min = max(1, int(round(side_min / grid.resolution)))
    k_max = max(1, int(round(side_max / grid.resolution)))
    if k_min % 2 == 0:
        k_min += 1
    if k_max % 2 == 0:
        k_max += 1

    local_max = np.maximum(
        maximum_filter(diff, size=k_min, mode="nearest"),
        maximum_filter(diff, size=k_max, mode="nearest"),
    )

    fallback_mask = (local_max > fallback.threshold_m) & local_valid

    fused = reference_surface.copy()
    # Where reference is missing and local exists, always use local.
    fused[(~ref_valid) & local_valid] = local_surface[(~ref_valid) & local_valid]
    # Where difference exceeds threshold in local 1-2 m^2 neighborhoods, use local.
    fused[fallback_mask] = local_surface[fallback_mask]

    return fused, fallback_mask.astype(np.float32)


def _initialize_feature_arrays(grid: RasterGrid) -> dict[str, np.ndarray]:
    return {
        "b1_max_hag": np.full(grid.size, NODATA_FLOAT, dtype=np.float32),
        "b2_max_hag": np.full(grid.size, NODATA_FLOAT, dtype=np.float32),
        "b3_max_hag": np.full(grid.size, NODATA_FLOAT, dtype=np.float32),
        "intensity_sum": np.zeros(grid.size, dtype=np.float32),
        "intensity_count": np.zeros(grid.size, dtype=np.uint32),
        "density_count": np.zeros(grid.size, dtype=np.uint32),
    }


def _compute_intensity_minmax(
    laz_path: Path,
    grid: RasterGrid,
    ground_surface: np.ndarray,
    return_mode: str,
    hag_min: float,
    hag_max: float,
    chunk_size: int,
    exclude_classes: set[int],
) -> tuple[float | None, float | None, int]:
    i_min = float("inf")
    i_max = float("-inf")
    used = 0

    with laspy.open(laz_path) as fh:
        for points in fh.chunk_iterator(chunk_size):
            d = _read_chunk_arrays(points)
            x = d["x"]
            y = d["y"]
            z = d["z"]
            cls = d["classification"]
            rn = d["return_number"]
            nr = d["number_of_returns"]
            intensity = d["intensity"]

            n = len(x)
            keep = _class_mask(cls, n, exclude_classes)
            keep &= _return_mask(rn, nr, n, return_mode)
            if not np.any(keep):
                continue

            xk = x[keep]
            yk = y[keep]
            zk = z[keep]
            ik = intensity[keep] if intensity is not None else None

            if ik is None:
                continue

            row, col = grid.xy_to_row_col(xk, yk)
            valid_rc = _valid_rc_mask(row, col, grid)
            if not np.any(valid_rc):
                continue

            row = row[valid_rc]
            col = col[valid_rc]
            zk = zk[valid_rc]
            ik = ik[valid_rc]

            g = ground_surface[row, col]
            valid_ground = g != NODATA_FLOAT
            if not np.any(valid_ground):
                continue

            hag = zk[valid_ground] - g[valid_ground]
            hk = (hag >= hag_min) & (hag <= hag_max)
            if not np.any(hk):
                continue

            vals = ik[valid_ground][hk]
            i_min = min(i_min, float(vals.min()))
            i_max = max(i_max, float(vals.max()))
            used += int(vals.size)

    if used == 0:
        return None, None, 0
    return i_min, i_max, used


def compute_feature_bands(
    laz_path: Path,
    grid: RasterGrid,
    ground_surface: np.ndarray,
    return_mode: str,
    hag_max: float,
    chunk_size: int,
    exclude_classes: set[int],
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    arr = _initialize_feature_arrays(grid)

    i_min, i_max, i_used = _compute_intensity_minmax(
        laz_path=laz_path,
        grid=grid,
        ground_surface=ground_surface,
        return_mode=return_mode,
        hag_min=0.4,
        hag_max=hag_max,
        chunk_size=chunk_size,
        exclude_classes=exclude_classes,
    )

    with laspy.open(laz_path) as fh:
        for points in fh.chunk_iterator(chunk_size):
            d = _read_chunk_arrays(points)
            x = d["x"]
            y = d["y"]
            z = d["z"]
            cls = d["classification"]
            rn = d["return_number"]
            nr = d["number_of_returns"]
            intensity = d["intensity"]

            n = len(x)
            keep = _class_mask(cls, n, exclude_classes)
            keep &= _return_mask(rn, nr, n, return_mode)
            if not np.any(keep):
                continue

            xk = x[keep]
            yk = y[keep]
            zk = z[keep]
            ik = intensity[keep] if intensity is not None else None

            row, col = grid.xy_to_row_col(xk, yk)
            valid_rc = _valid_rc_mask(row, col, grid)
            if not np.any(valid_rc):
                continue

            row = row[valid_rc]
            col = col[valid_rc]
            zk = zk[valid_rc]
            if ik is not None:
                ik = ik[valid_rc]

            g = ground_surface[row, col]
            valid_ground = g != NODATA_FLOAT
            if not np.any(valid_ground):
                continue

            row = row[valid_ground]
            col = col[valid_ground]
            zk = zk[valid_ground]
            if ik is not None:
                ik = ik[valid_ground]
            hag = zk - g[valid_ground]

            in_hag = (hag >= 0.0) & (hag <= hag_max)
            if not np.any(in_hag):
                continue

            row = row[in_hag]
            col = col[in_hag]
            hag = hag[in_hag]
            if ik is not None:
                ik = ik[in_hag]

            flat = _to_flat_index(row, col, grid)

            # Band 1: [0.0, 0.4)
            m1 = hag < 0.4
            if np.any(m1):
                np.maximum.at(arr["b1_max_hag"], flat[m1], hag[m1])

            # Band 2: [0.4, 0.7)
            m2 = (hag >= 0.4) & (hag < 0.7)
            if np.any(m2):
                np.maximum.at(arr["b2_max_hag"], flat[m2], hag[m2])

            # Band 3: [0.7, 1.3]
            m3 = hag >= 0.7
            if np.any(m3):
                np.maximum.at(arr["b3_max_hag"], flat[m3], hag[m3])

            # Band 5: density count in [0.0, 1.3]
            np.add.at(arr["density_count"], flat, 1)

            # Band 4: normalized intensity in [0.4, 1.3]
            if ik is not None and i_min is not None and i_max is not None:
                mi = hag >= 0.4
                if np.any(mi):
                    denom = i_max - i_min
                    if denom <= 1e-8:
                        norm_i = np.full(np.count_nonzero(mi), 0.5, dtype=np.float32)
                    else:
                        norm_i = (ik[mi] - i_min) / denom
                        norm_i = np.clip(norm_i.astype(np.float32), 0.0, 1.0)

                    flat_i = flat[mi]
                    np.add.at(arr["intensity_sum"], flat_i, norm_i)
                    np.add.at(arr["intensity_count"], flat_i, 1)

    intensity = np.full(grid.size, NODATA_FLOAT, dtype=np.float32)
    has_intensity = arr["intensity_count"] > 0
    intensity[has_intensity] = arr["intensity_sum"][has_intensity] / arr["intensity_count"][has_intensity]

    density = np.full(grid.size, NODATA_FLOAT, dtype=np.float32)
    has_density = arr["density_count"] > 0
    density[has_density] = arr["density_count"][has_density].astype(np.float32)

    bands = [
        arr["b1_max_hag"].reshape((grid.height, grid.width)),
        arr["b2_max_hag"].reshape((grid.height, grid.width)),
        arr["b3_max_hag"].reshape((grid.height, grid.width)),
        intensity.reshape((grid.height, grid.width)),
        density.reshape((grid.height, grid.width)),
    ]

    stats: dict[str, float | int] = {
        "intensity_points_used": int(i_used),
        "intensity_min": float(i_min) if i_min is not None else float("nan"),
        "intensity_max": float(i_max) if i_max is not None else float("nan"),
        "band1_valid_px": int(np.count_nonzero(bands[0] != NODATA_FLOAT)),
        "band2_valid_px": int(np.count_nonzero(bands[1] != NODATA_FLOAT)),
        "band3_valid_px": int(np.count_nonzero(bands[2] != NODATA_FLOAT)),
        "band4_valid_px": int(np.count_nonzero(bands[3] != NODATA_FLOAT)),
        "band5_valid_px": int(np.count_nonzero(bands[4] != NODATA_FLOAT)),
    }
    return bands, stats


def _combine_split_bands_to_single_chm(b1: np.ndarray, b2: np.ndarray, b3: np.ndarray) -> np.ndarray:
    stack = np.stack([b1, b2, b3], axis=0)
    valid_any = np.any(stack != NODATA_FLOAT, axis=0)
    stack_for_max = np.where(stack == NODATA_FLOAT, -np.inf, stack)
    merged = np.max(stack_for_max, axis=0).astype(np.float32)
    out = np.full_like(b1, NODATA_FLOAT, dtype=np.float32)
    out[valid_any] = merged[valid_any]
    return out


def _prepare_rgb_split_for_visualization(
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare 3 split bands for RGB display.

    Rule requested by user:
    - If all 3 bands are nodata at a pixel, keep nodata in all 3.
    - If at least one band has a value, set nodata bands at that pixel to 0.
    """
    r = b1.copy()
    g = b2.copy()
    b = b3.copy()

    valid_any = (r != NODATA_FLOAT) | (g != NODATA_FLOAT) | (b != NODATA_FLOAT)

    r[(r == NODATA_FLOAT) & valid_any] = 0.0
    g[(g == NODATA_FLOAT) & valid_any] = 0.0
    b[(b == NODATA_FLOAT) & valid_any] = 0.0
    return r, g, b


def _compute_delta_vs_baseline(
    experiment_chm: np.ndarray,
    baseline_tif: Path,
    grid: RasterGrid,
    logger: logging.Logger,
) -> np.ndarray | None:
    if not baseline_tif.exists():
        return None

    expected_transform = from_origin(grid.origin_x, grid.max_y, grid.resolution, grid.resolution)

    with rasterio.open(baseline_tif) as src:
        if src.width != grid.width or src.height != grid.height:
            logger.warning(
                "Baseline size mismatch for %s (expected %dx%d, got %dx%d); skipping delta",
                baseline_tif.name,
                grid.width,
                grid.height,
                src.width,
                src.height,
            )
            return None

        if src.transform != expected_transform:
            logger.warning("Baseline transform mismatch for %s; skipping delta", baseline_tif.name)
            return None

        base = src.read(1).astype(np.float32)
        valid_base = np.isfinite(base)
        if src.nodata is not None:
            valid_base &= base != float(src.nodata)

    valid_exp = experiment_chm != NODATA_FLOAT
    valid = valid_exp & valid_base

    delta = np.full_like(experiment_chm, NODATA_FLOAT, dtype=np.float32)
    delta[valid] = experiment_chm[valid] - base[valid]
    return delta


def write_single_band_tif(
    out_path: Path,
    band: np.ndarray,
    grid: RasterGrid,
    tags: dict[str, str],
    description: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": "float32",
        "transform": from_origin(grid.origin_x, grid.max_y, grid.resolution, grid.resolution),
        "nodata": NODATA_FLOAT,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(band.astype(np.float32), 1)
        dst.set_band_description(1, description)
        dst.update_tags(**tags)
        dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")


def write_multiband_tif(
    out_path: Path,
    bands: Sequence[np.ndarray],
    grid: RasterGrid,
    tags: dict[str, str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": len(bands),
        "dtype": "float32",
        "transform": from_origin(grid.origin_x, grid.max_y, grid.resolution, grid.resolution),
        "nodata": NODATA_FLOAT,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band.astype(np.float32), i)

        dst.set_band_description(1, "max_hag_0.0_0.4m")
        dst.set_band_description(2, "max_hag_0.4_0.7m")
        dst.set_band_description(3, "max_hag_0.7_1.3m")
        dst.set_band_description(4, "avg_norm_intensity_0.4_1.3m")
        dst.set_band_description(5, "point_density_0.0_1.3m")

        dst.update_tags(**tags)
        dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")


def write_rgb_split_tif(
    out_path: Path,
    r_band: np.ndarray,
    g_band: np.ndarray,
    b_band: np.ndarray,
    grid: RasterGrid,
    tags: dict[str, str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 3,
        "dtype": "float32",
        "transform": from_origin(grid.origin_x, grid.max_y, grid.resolution, grid.resolution),
        "nodata": NODATA_FLOAT,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(r_band.astype(np.float32), 1)
        dst.write(g_band.astype(np.float32), 2)
        dst.write(b_band.astype(np.float32), 3)
        dst.set_band_description(1, "max_hag_0.0_0.4m")
        dst.set_band_description(2, "max_hag_0.4_0.7m")
        dst.set_band_description(3, "max_hag_0.7_1.3m")
        dst.update_tags(**tags)
        dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")


def write_report(report_rows: Sequence[dict[str, str | int | float]], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": list(report_rows),
        "total_rows": len(report_rows),
        "ok_rows": sum(1 for r in report_rows if r.get("status") == "ok"),
        "error_rows": sum(1 for r in report_rows if r.get("status") == "error"),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    all_keys: set[str] = set()
    for row in report_rows:
        all_keys.update(row.keys())

    ordered = [
        "status",
        "tile_id",
        "year",
        "dtm_variant",
        "return_mode",
        "input_laz",
        "single_chm_tif",
        "intensity_tif",
        "density_tif",
        "split_rgb_tif",
        "baseline_tif",
        "delta_vs_original_tif",
        "reference_surface_tif",
        "fused_surface_tif",
        "fallback_mask_tif",
        "fallback_trigger_fraction",
        "single_chm_valid_px",
        "band1_valid_px",
        "band2_valid_px",
        "band3_valid_px",
        "band4_valid_px",
        "band5_valid_px",
        "intensity_points_used",
        "intensity_min",
        "intensity_max",
        "error",
    ]
    fields = ordered + sorted(k for k in all_keys if k not in ordered)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in report_rows:
            w.writerow(row)


def parse_years(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    years: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        years.append(int(chunk))
    return years


def parse_modes(raw: str) -> list[str]:
    modes = [x.strip().lower() for x in raw.split(",") if x.strip()]
    allowed = {"all", "last", "last2"}
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise ValueError(f"Unsupported return mode(s): {bad}. Allowed: {sorted(allowed)}")
    if not modes:
        raise ValueError("At least one return mode is required")
    return modes


def main() -> int:
    p = argparse.ArgumentParser(description="Standalone CDW feature experiments for LAZ tile")
    p.add_argument("--input-dir", type=Path, default=Path("data/lamapuit/laz"))
    p.add_argument("--output-dir", type=Path, default=Path("data/lamapuit/cdw_experiments_436646"))
    p.add_argument("--tile-id", default="436646")
    p.add_argument("--years", default="2018,2020,2022,2024")
    p.add_argument("--resolution", type=float, default=0.2)
    p.add_argument("--chunk-size", type=int, default=2_000_000)
    p.add_argument("--hag-max", type=float, default=1.3)
    p.add_argument("--fallback-threshold-m", type=float, default=0.3)
    p.add_argument("--fallback-window-min-m2", type=float, default=1.0)
    p.add_argument("--fallback-window-max-m2", type=float, default=2.0)
    p.add_argument("--exclude-classes", default="6,9")
    p.add_argument("--return-modes", default="all,last,last2")
    p.add_argument(
        "--dtm-variants",
        default=None,
        help="Comma-separated DTM variants to run (overrides defaults). Examples: 'median_all,lowest_all'",
    )
    p.add_argument(
        "--baseline-chm-dir",
        type=Path,
        default=Path("data/lamapuit/chm_max_hag_13_drop"),
        help="Directory of original baseline CHM files for delta comparison.",
    )
    p.add_argument(
        "--write-split-rgb",
        action="store_true",
        help="Also write optional 3-band split RGB rasters (off by default).",
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--only-chm",
        action="store_true",
        help="Only write the combined CHM per experiment; skip intensity/density/split-rgb/delta outputs.",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--report-json",
        type=Path,
        default=Path("analysis/cdw_experiments_436646/cdw_experiment_report.json"),
    )
    p.add_argument(
        "--report-csv",
        type=Path,
        default=Path("analysis/cdw_experiments_436646/cdw_experiment_report.csv"),
    )
    args = p.parse_args()

    if args.fallback_window_min_m2 <= 0 or args.fallback_window_max_m2 <= 0:
        raise ValueError("Fallback window areas must be > 0")
    if args.fallback_window_min_m2 > args.fallback_window_max_m2:
        raise ValueError("fallback-window-min-m2 must be <= fallback-window-max-m2")

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_root / "cdw_experiments.log", args.verbose)

    years = parse_years(args.years)
    modes = parse_modes(args.return_modes)
    exclude_classes = {int(x.strip()) for x in args.exclude_classes.split(",") if x.strip()}

    laz_files = discover_tile_laz_files(args.input_dir, str(args.tile_id), years)
    if not laz_files:
        logger.error("No LAZ files found for tile %s in %s", args.tile_id, args.input_dir)
        return 2

    grid = build_common_grid(laz_files, args.resolution)
    logger.info(
        "Grid: width=%d height=%d res=%.3f origin_x=%.3f max_y=%.3f",
        grid.width,
        grid.height,
        grid.resolution,
        grid.origin_x,
        grid.max_y,
    )

    fallback_cfg = LocalFallbackConfig(
        threshold_m=args.fallback_threshold_m,
        min_window_m2=args.fallback_window_min_m2,
        max_window_m2=args.fallback_window_max_m2,
    )

    # Determine which DTM variants to run. Allow overriding via CLI for focused tests.
    if args.dtm_variants is not None and str(args.dtm_variants).strip() != "":
        dtm_variants = [x.strip() for x in str(args.dtm_variants).split(",") if x.strip()]
    else:
        dtm_variants = ["median_all", "lowest_all", "lowest_ground"]
    report_rows: list[dict[str, str | int | float]] = []

    for dtm_variant in dtm_variants:
        logger.info("Building multi-year reference surface: variant=%s", dtm_variant)
        ref_surface = build_reference_surface(
            laz_paths=laz_files,
            grid=grid,
            method=dtm_variant,
            chunk_size=args.chunk_size,
            exclude_classes=exclude_classes,
            logger=logger,
        )

        variant_dir = out_root / dtm_variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        ref_tif = variant_dir / f"{args.tile_id}_reference_surface_{dtm_variant}_{int(args.resolution * 100)}cm.tif"
        if not args.dry_run and not (args.skip_existing and ref_tif.exists()):
            write_single_band_tif(
                out_path=ref_tif,
                band=ref_surface,
                grid=grid,
                tags={
                    "TILE_ID": str(args.tile_id),
                    "DTM_VARIANT": dtm_variant,
                    "RESOLUTION": str(args.resolution),
                    "PROC_DATE": datetime.now(UTC).isoformat(timespec="seconds"),
                    "TYPE": "reference_surface",
                },
                description=f"reference_surface_{dtm_variant}",
            )

        for laz_path in laz_files:
            year = extract_year(laz_path.name)
            year_txt = str(year) if year is not None else "unknown"
            stem = laz_path.stem

            logger.info("Year-local surface and fallback: %s | variant=%s", laz_path.name, dtm_variant)
            local_surface = build_year_local_surface(
                laz_path=laz_path,
                grid=grid,
                method=dtm_variant,
                chunk_size=args.chunk_size,
                exclude_classes=exclude_classes,
                logger=logger,
            )
            fused_surface, fallback_mask = apply_local_fallback(
                reference_surface=ref_surface,
                local_surface=local_surface,
                fallback=fallback_cfg,
                grid=grid,
            )

            fused_tif = variant_dir / f"{stem}_fused_surface_{dtm_variant}_{int(args.resolution * 100)}cm.tif"
            fallback_tif = variant_dir / f"{stem}_fallback_mask_{dtm_variant}_{int(args.resolution * 100)}cm.tif"

            if not args.dry_run and not (args.skip_existing and fused_tif.exists()):
                write_single_band_tif(
                    out_path=fused_tif,
                    band=fused_surface,
                    grid=grid,
                    tags={
                        "TILE_ID": str(args.tile_id),
                        "SOURCE_LAZ": laz_path.name,
                        "YEAR": year_txt,
                        "DTM_VARIANT": dtm_variant,
                        "TYPE": "fused_surface",
                        "FALLBACK_THRESHOLD_M": str(args.fallback_threshold_m),
                        "FALLBACK_MIN_M2": str(args.fallback_window_min_m2),
                        "FALLBACK_MAX_M2": str(args.fallback_window_max_m2),
                    },
                    description="fused_ground_surface",
                )

            if not args.dry_run and not (args.skip_existing and fallback_tif.exists()):
                write_single_band_tif(
                    out_path=fallback_tif,
                    band=fallback_mask,
                    grid=grid,
                    tags={
                        "TILE_ID": str(args.tile_id),
                        "SOURCE_LAZ": laz_path.name,
                        "YEAR": year_txt,
                        "DTM_VARIANT": dtm_variant,
                        "TYPE": "fallback_mask",
                        "VALUE_NOTE": "1.0=fallback_to_year_local_surface",
                    },
                    description="fallback_mask",
                )

            fallback_fraction = float(np.count_nonzero(fallback_mask > 0.5) / fallback_mask.size)

            for mode in modes:
                res_cm = int(args.resolution * 100)
                single_chm_tif = variant_dir / f"{stem}_exp_return_chm13_{dtm_variant}_{mode}_{res_cm}cm.tif"
                intensity_tif = variant_dir / f"{stem}_exp_intensity_04_13_{dtm_variant}_{mode}_{res_cm}cm.tif"
                density_tif = variant_dir / f"{stem}_exp_density_00_13_{dtm_variant}_{mode}_{res_cm}cm.tif"
                split_rgb_tif = (
                    variant_dir / f"{stem}_exp_split_rgb_{dtm_variant}_{mode}_{res_cm}cm.tif"
                    if args.write_split_rgb
                    else None
                )

                baseline_tif = args.baseline_chm_dir / f"{stem}_chm_max_hag_{res_cm}cm.tif"
                delta_tif = variant_dir / f"{stem}_delta_vs_original_{dtm_variant}_{mode}_{res_cm}cm.tif"

                if args.only_chm:
                    required_outputs = [single_chm_tif]
                else:
                    required_outputs = [single_chm_tif, intensity_tif, density_tif]
                    if split_rgb_tif is not None:
                        required_outputs.append(split_rgb_tif)
                    if baseline_tif.exists():
                        required_outputs.append(delta_tif)

                if args.skip_existing and all(path.exists() for path in required_outputs):
                    logger.info("Skip existing outputs for %s | %s | %s", stem, dtm_variant, mode)
                    report_rows.append(
                        {
                            "status": "skipped_existing",
                            "tile_id": str(args.tile_id),
                            "year": year_txt,
                            "dtm_variant": dtm_variant,
                            "return_mode": mode,
                            "input_laz": str(laz_path),
                            "single_chm_tif": str(single_chm_tif),
                            "intensity_tif": str(intensity_tif),
                            "density_tif": str(density_tif),
                            "split_rgb_tif": str(split_rgb_tif) if split_rgb_tif is not None else "",
                            "baseline_tif": str(baseline_tif) if baseline_tif.exists() else "",
                            "delta_vs_original_tif": str(delta_tif) if delta_tif.exists() else "",
                            "reference_surface_tif": str(ref_tif),
                            "fused_surface_tif": str(fused_tif),
                            "fallback_mask_tif": str(fallback_tif),
                            "fallback_trigger_fraction": fallback_fraction,
                            "error": "",
                        }
                    )
                    continue

                if args.dry_run:
                    logger.info("Dry-run planned output: %s", single_chm_tif)
                    if not args.only_chm:
                        logger.info("Dry-run planned output: %s", intensity_tif)
                        logger.info("Dry-run planned output: %s", density_tif)
                        if split_rgb_tif is not None:
                            logger.info("Dry-run planned output: %s", split_rgb_tif)
                        if baseline_tif.exists():
                            logger.info("Dry-run planned output: %s", delta_tif)

                    report_rows.append(
                        {
                            "status": "dry_run",
                            "tile_id": str(args.tile_id),
                            "year": year_txt,
                            "dtm_variant": dtm_variant,
                            "return_mode": mode,
                            "input_laz": str(laz_path),
                            "single_chm_tif": str(single_chm_tif),
                            "intensity_tif": str(intensity_tif),
                            "density_tif": str(density_tif),
                            "split_rgb_tif": str(split_rgb_tif) if split_rgb_tif is not None else "",
                            "baseline_tif": str(baseline_tif) if baseline_tif.exists() else "",
                            "delta_vs_original_tif": str(delta_tif) if baseline_tif.exists() else "",
                            "reference_surface_tif": str(ref_tif),
                            "fused_surface_tif": str(fused_tif),
                            "fallback_mask_tif": str(fallback_tif),
                            "fallback_trigger_fraction": fallback_fraction,
                            "error": "",
                        }
                    )
                    continue

                try:
                    bands, stats = compute_feature_bands(
                        laz_path=laz_path,
                        grid=grid,
                        ground_surface=fused_surface,
                        return_mode=mode,
                        hag_max=args.hag_max,
                        chunk_size=args.chunk_size,
                        exclude_classes=exclude_classes,
                    )

                    b1, b2, b3, intensity_band, density_band = bands
                    single_chm = _combine_split_bands_to_single_chm(b1, b2, b3)
                    stats["single_chm_valid_px"] = int(np.count_nonzero(single_chm != NODATA_FLOAT))

                    write_single_band_tif(
                        out_path=single_chm_tif,
                        band=single_chm,
                        grid=grid,
                        tags={
                            "TILE_ID": str(args.tile_id),
                            "SOURCE_LAZ": laz_path.name,
                            "YEAR": year_txt,
                            "DTM_VARIANT": dtm_variant,
                            "RETURN_MODE": mode,
                            "TYPE": "experiment_return_chm13",
                            "HAG_MAX": str(args.hag_max),
                            "EXCLUDE_CLASSES": ",".join(str(c) for c in sorted(exclude_classes)),
                        },
                        description="chm_max_hag_0.0_1.3m",
                    )

                    split_rgb_tif_str = ""
                    if not args.only_chm:
                        write_single_band_tif(
                            out_path=intensity_tif,
                            band=intensity_band,
                            grid=grid,
                            tags={
                                "TILE_ID": str(args.tile_id),
                                "SOURCE_LAZ": laz_path.name,
                                "YEAR": year_txt,
                                "DTM_VARIANT": dtm_variant,
                                "RETURN_MODE": mode,
                                "TYPE": "experiment_intensity_0.4_1.3",
                            },
                            description="avg_norm_intensity_0.4_1.3m",
                        )

                        write_single_band_tif(
                            out_path=density_tif,
                            band=density_band,
                            grid=grid,
                            tags={
                                "TILE_ID": str(args.tile_id),
                                "SOURCE_LAZ": laz_path.name,
                                "YEAR": year_txt,
                                "DTM_VARIANT": dtm_variant,
                                "RETURN_MODE": mode,
                                "TYPE": "experiment_density_0.0_1.3",
                            },
                            description="point_density_0.0_1.3m",
                        )

                        if split_rgb_tif is not None:
                            r_band, g_band, b_band = _prepare_rgb_split_for_visualization(b1, b2, b3)
                            write_rgb_split_tif(
                                out_path=split_rgb_tif,
                                r_band=r_band,
                                g_band=g_band,
                                b_band=b_band,
                                grid=grid,
                                tags={
                                    "TILE_ID": str(args.tile_id),
                                    "SOURCE_LAZ": laz_path.name,
                                    "YEAR": year_txt,
                                    "DTM_VARIANT": dtm_variant,
                                    "RETURN_MODE": mode,
                                    "TYPE": "optional_split_rgb",
                                    "RGB_FILL_RULE": "if_any_band_valid_set_other_nodata_to_0",
                                },
                            )
                            split_rgb_tif_str = str(split_rgb_tif)

                    baseline_tif_str = str(baseline_tif) if baseline_tif.exists() else ""
                    delta_tif_str = ""
                    if not args.only_chm and baseline_tif.exists():
                        delta = _compute_delta_vs_baseline(
                            experiment_chm=single_chm,
                            baseline_tif=baseline_tif,
                            grid=grid,
                            logger=logger,
                        )
                        if delta is not None:
                            write_single_band_tif(
                                out_path=delta_tif,
                                band=delta,
                                grid=grid,
                                tags={
                                    "TILE_ID": str(args.tile_id),
                                    "SOURCE_LAZ": laz_path.name,
                                    "YEAR": year_txt,
                                    "DTM_VARIANT": dtm_variant,
                                    "RETURN_MODE": mode,
                                    "TYPE": "delta_vs_original",
                                    "DELTA_FORMULA": "experiment_chm13 - original_chm13",
                                    "ORIGINAL_TIF": str(baseline_tif),
                                },
                                description="delta_vs_original_chm13",
                            )
                            delta_tif_str = str(delta_tif)

                    row: dict[str, str | int | float] = {
                        "status": "ok",
                        "tile_id": str(args.tile_id),
                        "year": year_txt,
                        "dtm_variant": dtm_variant,
                        "return_mode": mode,
                        "input_laz": str(laz_path),
                        "single_chm_tif": str(single_chm_tif),
                        "intensity_tif": str(intensity_tif) if not args.only_chm else "",
                        "density_tif": str(density_tif) if not args.only_chm else "",
                        "split_rgb_tif": split_rgb_tif_str,
                        "baseline_tif": baseline_tif_str,
                        "delta_vs_original_tif": delta_tif_str,
                        "reference_surface_tif": str(ref_tif),
                        "fused_surface_tif": str(fused_tif),
                        "fallback_mask_tif": str(fallback_tif),
                        "fallback_trigger_fraction": fallback_fraction,
                        "error": "",
                    }
                    row.update(stats)
                    report_rows.append(row)
                    if args.only_chm:
                        logger.info("Wrote CHM output: %s", single_chm_tif.name)
                    else:
                        logger.info(
                            "Wrote separate experiment outputs: %s | %s | %s",
                            single_chm_tif.name,
                            intensity_tif.name,
                            density_tif.name,
                        )
                except Exception as exc:
                    logger.exception("Failed output for %s | variant=%s mode=%s", laz_path.name, dtm_variant, mode)
                    report_rows.append(
                        {
                            "status": "error",
                            "tile_id": str(args.tile_id),
                            "year": year_txt,
                            "dtm_variant": dtm_variant,
                            "return_mode": mode,
                            "input_laz": str(laz_path),
                            "single_chm_tif": str(single_chm_tif),
                            "intensity_tif": str(intensity_tif),
                            "density_tif": str(density_tif),
                            "split_rgb_tif": str(split_rgb_tif) if split_rgb_tif is not None else "",
                            "baseline_tif": str(baseline_tif) if baseline_tif.exists() else "",
                            "delta_vs_original_tif": str(delta_tif),
                            "reference_surface_tif": str(ref_tif),
                            "fused_surface_tif": str(fused_tif),
                            "fallback_mask_tif": str(fallback_tif),
                            "fallback_trigger_fraction": fallback_fraction,
                            "error": str(exc),
                        }
                    )

    write_report(report_rows, args.report_json, args.report_csv)
    ok = sum(1 for r in report_rows if r.get("status") == "ok")
    err = sum(1 for r in report_rows if r.get("status") == "error")
    dry = sum(1 for r in report_rows if r.get("status") == "dry_run")
    skip = sum(1 for r in report_rows if r.get("status") == "skipped_existing")
    logger.info("Experiment run complete | ok=%d error=%d dry_run=%d skipped=%d", ok, err, dry, skip)
    logger.info("Reports: %s | %s", args.report_json, args.report_csv)

    return 0 if err == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())

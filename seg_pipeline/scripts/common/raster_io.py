"""Raster I/O helpers: multi-band reading, normalization, writing.

Consolidates patterns from train_deeplabv3plus_manual_masks.py read_chm_chip()
and train_partialconv_manual_masks.py make_model_input().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.windows import Window


def read_multiband_window(
    tif_path: Path,
    row_off: int,
    col_off: int,
    size: int,
    bands: Sequence[int] | None = None,
) -> np.ndarray:
    """Read a square window from a multi-band GeoTIFF.

    Returns:
        Float32 array of shape (C, size, size). Nodata/NaN regions are set to 0.
    """
    with rasterio.open(tif_path) as src:
        band_ids = list(bands) if bands else list(range(1, src.count + 1))
        arr = src.read(
            band_ids,
            window=Window(col_off, row_off, size, size),
            boundless=True,
            fill_value=np.nan,
        ).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


def read_full_band(tif_path: Path, band: int = 1) -> tuple[np.ndarray, "rasterio.profiles.Profile"]:
    """Read a single band from a GeoTIFF, returning (array, profile)."""
    with rasterio.open(tif_path) as src:
        arr = src.read(band).astype(np.float32)
        nodata = src.nodata
        profile = src.profile.copy()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, profile


def normalize_bands(
    arr: np.ndarray,
    band_stats: dict,
    binary_bands: Sequence[int] | None = None,
) -> np.ndarray:
    """Z-score normalize each band using pre-computed stats.

    band_stats: dict with keys '0', '1', ... mapping to {'mean', 'std'}.
    binary_bands: 0-indexed band indices to pass through without normalization.
    Bands with values in [0, 255] range are automatically scaled to [0, 1].
    """
    out = arr.copy()
    binary_set = set(binary_bands or [])
    for i in range(out.shape[0]):
        out[i] = np.nan_to_num(out[i], nan=0.0)

        if i in binary_set:
            # If band is in [0, 255] range (common for rasterized masks), scale to [0, 1]
            if out[i].max() > 1.5:  # Likely [0, 255] encoding
                out[i] = out[i] / 255.0
            out[i] = np.clip(out[i], 0.0, 1.0)
            continue

        key = str(i)
        stats = band_stats.get(key, {})
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 1.0)) or 1.0
        band = out[i]
        band = (band - mean) / std
        band = np.clip(band, -3.0, 3.0)
        out[i] = band
    return out


def compute_band_stats(
    tif_path: Path,
    valid_mask: np.ndarray,
    bands: Sequence[int] | None = None,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> dict:
    """Compute per-band mean/std/p2/p98 on valid pixels for normalization.

    valid_mask: 2D boolean array (H, W); stats computed only where True.
    Returns dict keyed by 0-indexed band index.
    """
    with rasterio.open(tif_path) as src:
        band_ids = list(bands) if bands else list(range(1, src.count + 1))
        data = src.read(band_ids).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        data[data == nodata] = np.nan

    stats: dict = {}
    for i, _bid in enumerate(band_ids):
        band = data[i]
        mask = valid_mask & np.isfinite(band)
        vals = band[mask]
        if len(vals) == 0:
            stats[str(i)] = {"mean": 0.0, "std": 1.0, "p2": 0.0, "p98": 1.0}
            continue
        lo, hi = float(np.percentile(vals, p_lo)), float(np.percentile(vals, p_hi))
        clipped = np.clip(vals, lo, hi)
        stats[str(i)] = {
            "mean": float(clipped.mean()),
            "std": float(clipped.std()) or 1.0,
            "p2": lo,
            "p98": hi,
        }
    return stats


def write_raster_like(
    arr: np.ndarray,
    reference_tif: Path,
    output_path: Path,
    nodata: float | None = np.nan,
    dtype: str = "float32",
) -> None:
    """Write a raster using the CRS/transform/shape from a reference GeoTIFF.

    arr: shape (H, W) or (C, H, W). If 2D, written as single-band.
    """
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    c, h, w = arr.shape

    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()

    profile.update(
        count=c,
        dtype=dtype,
        nodata=nodata,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr.astype(dtype))

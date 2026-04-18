#!/usr/bin/env python3
"""Ground-method benchmark for 436646_2024_madal LAZ -> CHM.

Runs multiple ground estimation strategies when LAS ground tags are unreliable,
then evaluates CDW visibility using existing tile labels.

Outputs are written under a dedicated temp folder.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, grey_opening, median_filter, sobel, laplace
from scipy.stats import rankdata

from process_laz_to_chm_improved import compute_hag_raster_streamed


@dataclass
class MethodResult:
    name: str
    chm_path: Path
    notes: str


def _run(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _write_pdal_pipeline(pipeline: list, path: Path) -> None:
    path.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")


def _pdal_reclassify(input_laz: Path, output_laz: Path, method: str, work_dir: Path) -> None:
    output_laz.parent.mkdir(parents=True, exist_ok=True)
    pipe_path = work_dir / f"pdal_{method}.json"

    if method == "smrf":
        filters = [
            {"type": "filters.assign", "value": "Classification = 1"},
            {
                "type": "filters.smrf",
                "scalar": 1.2,
                "slope": 0.2,
                "threshold": 0.45,
                "window": 16.0,
            },
        ]
    elif method == "csf":
        filters = [
            {"type": "filters.assign", "value": "Classification = 1"},
            {
                "type": "filters.csf",
                "resolution": 1.0,
                "rigidness": 2,
                "step": 0.65,
                "threshold": 0.5,
                "hdiff": 0.3,
            },
        ]
    else:
        raise ValueError(f"Unsupported PDAL method: {method}")

    pipeline = [
        {
            "type": "readers.las",
            "filename": str(input_laz),
            "override_srs": "EPSG:3067",
        },
        *filters,
        {"type": "writers.las", "filename": str(output_laz), "compression": "true"},
    ]

    _write_pdal_pipeline(pipeline, pipe_path)
    _run(["pdal", "pipeline", str(pipe_path)])


def _snap_origin(minx: float, miny: float, resolution: float) -> Tuple[float, float]:
    ox = math.floor(minx / resolution) * resolution
    oy = math.floor(miny / resolution) * resolution
    return ox, oy


def _bilinear_sample(dem: np.ndarray, x: np.ndarray, y: np.ndarray, ox: float, maxy: float, res: float) -> np.ndarray:
    col = (x - ox) / res
    row = (maxy - y) / res

    c0 = np.floor(col).astype(int)
    r0 = np.floor(row).astype(int)
    c1 = c0 + 1
    r1 = r0 + 1

    h, w = dem.shape
    c0 = np.clip(c0, 0, w - 1)
    c1 = np.clip(c1, 0, w - 1)
    r0 = np.clip(r0, 0, h - 1)
    r1 = np.clip(r1, 0, h - 1)

    dc = np.clip(col - c0, 0.0, 1.0)
    dr = np.clip(row - r0, 0.0, 1.0)

    z00 = dem[r0, c0]
    z10 = dem[r0, c1]
    z01 = dem[r1, c0]
    z11 = dem[r1, c1]

    z0 = z00 * (1.0 - dc) + z10 * dc
    z1 = z01 * (1.0 - dc) + z11 * dc
    return z0 * (1.0 - dr) + z1 * dr


def _compute_quantile_surface_chm(
    laz_path: Path,
    out_tif: Path,
    resolution: float,
    hag_max: float,
    nodata: float,
    chunk_size: int,
    dem_resolution: float,
    filter_mode: str,
) -> Path:
    """Compute CHM from a robust ground surface without using class tags.

    Strategy:
    1) Build coarse DEM from per-cell minimum Z over all points.
    2) Apply opening + Gaussian smoothing to suppress low object imprint.
    3) Interpolate DEM at each point and compute HAG.
    4) Rasterize max HAG at target resolution.
    """
    with laspy.open(str(laz_path)) as fh:
        hdr = fh.header
        minx, miny = float(hdr.mins[0]), float(hdr.mins[1])
        maxx, maxy = float(hdr.maxs[0]), float(hdr.maxs[1])

    ox, oy = _snap_origin(minx, miny, dem_resolution)
    dem_w = int(math.ceil((maxx - ox) / dem_resolution))
    dem_h = int(math.ceil((maxy - oy) / dem_resolution))

    dem = np.full((dem_h, dem_w), np.nan, dtype=np.float32)

    with laspy.open(str(laz_path)) as fh:
        for pts in fh.chunk_iterator(chunk_size):
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float32)

            c = ((x - ox) / dem_resolution).astype(np.int32)
            r = ((maxy - y) / dem_resolution).astype(np.int32)
            valid = (r >= 0) & (r < dem_h) & (c >= 0) & (c < dem_w)
            if not np.any(valid):
                continue

            rr = r[valid]
            cc = c[valid]
            zz = z[valid]
            flat = rr * dem_w + cc
            order = np.argsort(flat)
            flat = flat[order]
            zz = zz[order]

            uniq, idx = np.unique(flat, return_index=True)
            mins = np.minimum.reduceat(zz, idx)

            current = dem.ravel()[uniq]
            merged = np.where(np.isfinite(current), np.minimum(current, mins), mins)
            dem.ravel()[uniq] = merged

    if not np.isfinite(dem).any():
        raise RuntimeError("Quantile-surface DEM is empty")

    finite = np.isfinite(dem)
    fill_val = float(np.nanmedian(dem[finite]))
    dem_filled = np.where(finite, dem, fill_val)

    opened = grey_opening(dem_filled, size=(5, 5))
    smooth = gaussian_filter(opened, sigma=1.2)

    ox2, oy2 = _snap_origin(minx, miny, resolution)
    out_w = int(math.ceil((maxx - ox2) / resolution))
    out_h = int(math.ceil((maxy - oy2) / resolution))
    raster = np.full((out_h, out_w), nodata, dtype=np.float32)

    with laspy.open(str(laz_path)) as fh:
        for pts in fh.chunk_iterator(chunk_size):
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float32)

            g = _bilinear_sample(smooth, x, y, ox, maxy, dem_resolution)
            hag_raw = z - g
            if filter_mode == "drop":
                keep = (hag_raw >= 0.0) & (hag_raw <= hag_max)
                if not np.any(keep):
                    continue
                x = x[keep]
                y = y[keep]
                hag = hag_raw[keep]
            elif filter_mode == "clip":
                hag = np.clip(hag_raw, 0.0, hag_max)
            elif filter_mode == "raw":
                hag = np.clip(hag_raw, 0.0, None)
            else:
                raise ValueError(f"Unsupported filter_mode: {filter_mode}")

            c = ((x - ox2) / resolution).astype(np.int32)
            r = ((maxy - y) / resolution).astype(np.int32)
            valid = (r >= 0) & (r < out_h) & (c >= 0) & (c < out_w)
            if not np.any(valid):
                continue
            flat = r[valid] * out_w + c[valid]
            np.maximum.at(raster.ravel(), flat, hag[valid].astype(np.float32))

    transform = from_origin(ox2, maxy, resolution, resolution)
    profile = {
        "driver": "GTiff",
        "height": out_h,
        "width": out_w,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(raster, 1)
        dst.update_tags(
            SOURCE_LAZ=laz_path.name,
            METHOD="allpoints_min_opening_gaussian",
            HAG_MAX=str(hag_max),
            DEM_RES=str(dem_resolution),
            FILTER_MODE=filter_mode,
        )

    return out_tif


def _build_min_dem_from_points(
    laz_path: Path,
    dem_resolution: float,
    chunk_size: int,
    class2_only: bool,
) -> tuple[np.ndarray, float, float, float, float]:
    with laspy.open(str(laz_path)) as fh:
        hdr = fh.header
        minx, miny = float(hdr.mins[0]), float(hdr.mins[1])
        maxx, maxy = float(hdr.maxs[0]), float(hdr.maxs[1])

    ox, oy = _snap_origin(minx, miny, dem_resolution)
    dem_w = int(math.ceil((maxx - ox) / dem_resolution))
    dem_h = int(math.ceil((maxy - oy) / dem_resolution))
    dem = np.full((dem_h, dem_w), np.nan, dtype=np.float32)

    with laspy.open(str(laz_path)) as fh:
        for pts in fh.chunk_iterator(chunk_size):
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float32)

            if class2_only:
                try:
                    cls = np.asarray(pts.classification)
                    keep = cls == 2
                except Exception:
                    keep = np.zeros(len(x), dtype=bool)
                if not np.any(keep):
                    continue
                x = x[keep]
                y = y[keep]
                z = z[keep]

            c = ((x - ox) / dem_resolution).astype(np.int32)
            r = ((maxy - y) / dem_resolution).astype(np.int32)
            valid = (r >= 0) & (r < dem_h) & (c >= 0) & (c < dem_w)
            if not np.any(valid):
                continue

            rr = r[valid]
            cc = c[valid]
            zz = z[valid]
            flat = rr * dem_w + cc
            order = np.argsort(flat)
            flat = flat[order]
            zz = zz[order]
            uniq, idx = np.unique(flat, return_index=True)
            mins = np.minimum.reduceat(zz, idx)

            current = dem.ravel()[uniq]
            merged = np.where(np.isfinite(current), np.minimum(current, mins), mins)
            dem.ravel()[uniq] = merged

    if not np.isfinite(dem).any():
        if class2_only:
            # Fallback: if class 2 is unavailable, fallback to all points to avoid failure.
            return _build_min_dem_from_points(laz_path, dem_resolution, chunk_size, class2_only=False)
        raise RuntimeError("Ground DEM is empty")

    return dem, ox, oy, maxx, maxy


def _compute_class2_surface_chm(
    laz_path: Path,
    out_tif: Path,
    resolution: float,
    hag_max: float,
    nodata: float,
    chunk_size: int,
    dem_resolution: float,
    filter_mode: str,
    smooth_mode: str,
    sigma_low: float = 0.8,
    sigma_high: float = 2.4,
) -> Path:
    dem, ox, _oy, _maxx, maxy = _build_min_dem_from_points(
        laz_path=laz_path,
        dem_resolution=dem_resolution,
        chunk_size=chunk_size,
        class2_only=True,
    )

    finite = np.isfinite(dem)
    fill_val = float(np.nanmedian(dem[finite]))
    base = np.where(finite, dem, fill_val)

    if smooth_mode == "fixed_low":
        ground = gaussian_filter(base, sigma=sigma_low)
    elif smooth_mode == "fixed_high":
        ground = gaussian_filter(base, sigma=sigma_high)
    elif smooth_mode == "adaptive":
        low = gaussian_filter(base, sigma=sigma_low)
        high = gaussian_filter(base, sigma=sigma_high)
        trend = gaussian_filter(base, sigma=1.0)
        rough = gaussian_filter(np.abs(base - trend), sigma=1.0)
        r10 = float(np.percentile(rough, 10))
        r90 = float(np.percentile(rough, 90))
        denom = (r90 - r10) if (r90 - r10) > 1e-6 else 1.0
        w = np.clip((rough - r10) / denom, 0.0, 1.0)
        # High roughness -> lower smoothing, low roughness -> stronger smoothing.
        ground = w * low + (1.0 - w) * high
    else:
        raise ValueError(f"Unsupported smooth_mode: {smooth_mode}")

    with laspy.open(str(laz_path)) as fh:
        hdr = fh.header
        minx, miny = float(hdr.mins[0]), float(hdr.mins[1])
        maxx, maxy = float(hdr.maxs[0]), float(hdr.maxs[1])

    ox2, _oy2 = _snap_origin(minx, miny, resolution)
    out_w = int(math.ceil((maxx - ox2) / resolution))
    out_h = int(math.ceil((maxy - _oy2) / resolution))
    raster = np.full((out_h, out_w), nodata, dtype=np.float32)

    with laspy.open(str(laz_path)) as fh:
        for pts in fh.chunk_iterator(chunk_size):
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float32)
            g = _bilinear_sample(ground, x, y, ox, maxy, dem_resolution)
            hag_raw = z - g

            if filter_mode == "drop":
                keep = (hag_raw >= 0.0) & (hag_raw <= hag_max)
                if not np.any(keep):
                    continue
                x = x[keep]
                y = y[keep]
                hag = hag_raw[keep]
            elif filter_mode == "clip":
                hag = np.clip(hag_raw, 0.0, hag_max)
            elif filter_mode == "raw":
                hag = np.clip(hag_raw, 0.0, None)
            else:
                raise ValueError(f"Unsupported filter_mode: {filter_mode}")

            c = ((x - ox2) / resolution).astype(np.int32)
            r = ((maxy - y) / resolution).astype(np.int32)
            valid = (r >= 0) & (r < out_h) & (c >= 0) & (c < out_w)
            if not np.any(valid):
                continue
            flat = r[valid] * out_w + c[valid]
            np.maximum.at(raster.ravel(), flat, hag[valid].astype(np.float32))

    transform = from_origin(ox2, maxy, resolution, resolution)
    profile = {
        "driver": "GTiff",
        "height": out_h,
        "width": out_w,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(raster, 1)
        dst.update_tags(
            SOURCE_LAZ=laz_path.name,
            METHOD=f"class2_surface_{smooth_mode}",
            HAG_MAX=str(hag_max),
            DEM_RES=str(dem_resolution),
            FILTER_MODE=filter_mode,
        )

    return out_tif


def _smooth_ground_surface(
    base: np.ndarray,
    smooth_mode: str,
    sigma_low: float = 0.8,
    sigma_high: float = 2.4,
) -> np.ndarray:
    if smooth_mode == "fixed_low":
        return gaussian_filter(base, sigma=sigma_low)
    if smooth_mode == "fixed_high":
        return gaussian_filter(base, sigma=sigma_high)
    if smooth_mode == "adaptive":
        low = gaussian_filter(base, sigma=sigma_low)
        high = gaussian_filter(base, sigma=sigma_high)
        trend = gaussian_filter(base, sigma=1.0)
        rough = gaussian_filter(np.abs(base - trend), sigma=1.0)
        r10 = float(np.percentile(rough, 10))
        r90 = float(np.percentile(rough, 90))
        denom = (r90 - r10) if (r90 - r10) > 1e-6 else 1.0
        w = np.clip((rough - r10) / denom, 0.0, 1.0)
        return w * low + (1.0 - w) * high
    raise ValueError(f"Unsupported smooth_mode: {smooth_mode}")


def _compute_class2_outlier_mask(
    dem: np.ndarray,
    method: str,
    neighborhood_sz: int = 5,
    threshold: float = 2.0,
) -> np.ndarray:
    valid = np.isfinite(dem)
    fill = float(np.nanmedian(dem[valid]))
    base = np.where(valid, dem, fill)

    if method == "local_residual":
        med = median_filter(base, size=neighborhood_sz, mode="reflect")
        resid = base - med
        mad_local = median_filter(np.abs(resid), size=neighborhood_sz, mode="reflect") * 1.4826
        mask = valid & (resid > (threshold * (mad_local + 1e-6)))
        return mask

    if method == "mad":
        med = median_filter(base, size=neighborhood_sz, mode="reflect")
        dev = np.abs(base - med)
        mad_local = median_filter(dev, size=neighborhood_sz, mode="reflect") * 1.4826
        # positive-side only outliers (likely elevated objects mislabeled as ground)
        mask = valid & ((base - med) > (threshold * (mad_local + 1e-6)))
        return mask

    if method == "slope_curvature":
        gx = sobel(base, axis=1, mode="reflect") / 8.0
        gy = sobel(base, axis=0, mode="reflect") / 8.0
        slope = np.hypot(gx, gy)
        curv = laplace(base, mode="reflect")
        local_med = median_filter(base, size=neighborhood_sz, mode="reflect")
        positive_resid = (base - local_med) > 0.12
        # Peak-like bumps usually have negative Laplacian.
        mask = valid & positive_resid & (slope > threshold) & (curv < -0.03)
        return mask

    raise ValueError(f"Unsupported class2 filter method: {method}")


def _rebuild_class2_dem_with_reject_mask(
    laz_path: Path,
    dem_resolution: float,
    chunk_size: int,
    ox: float,
    maxy: float,
    dem_h: int,
    dem_w: int,
    reject_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    dem = np.full((dem_h, dem_w), np.nan, dtype=np.float32)
    total_class2 = 0
    rejected_class2 = 0

    with laspy.open(str(laz_path)) as fh:
        for pts in fh.chunk_iterator(chunk_size):
            try:
                cls = np.asarray(pts.classification)
            except Exception:
                continue
            keep_cls = cls == 2
            if not np.any(keep_cls):
                continue

            x = np.asarray(pts.x[keep_cls], dtype=np.float64)
            y = np.asarray(pts.y[keep_cls], dtype=np.float64)
            z = np.asarray(pts.z[keep_cls], dtype=np.float32)
            total_class2 += int(len(x))

            c = ((x - ox) / dem_resolution).astype(np.int32)
            r = ((maxy - y) / dem_resolution).astype(np.int32)
            valid = (r >= 0) & (r < dem_h) & (c >= 0) & (c < dem_w)
            if not np.any(valid):
                continue

            rv = r[valid]
            cv = c[valid]
            zv = z[valid]
            rej = reject_mask[rv, cv]
            rejected_class2 += int(np.count_nonzero(rej))
            keep = ~rej
            if not np.any(keep):
                continue

            rr = rv[keep]
            cc = cv[keep]
            zz = zv[keep]
            flat = rr * dem_w + cc
            order = np.argsort(flat)
            flat = flat[order]
            zz = zz[order]
            uniq, idx = np.unique(flat, return_index=True)
            mins = np.minimum.reduceat(zz, idx)

            current = dem.ravel()[uniq]
            merged = np.where(np.isfinite(current), np.minimum(current, mins), mins)
            dem.ravel()[uniq] = merged

    stats = {
        "n_class2_input": int(total_class2),
        "n_class2_filtered": int(rejected_class2),
        "filter_pct": float(100.0 * rejected_class2 / total_class2) if total_class2 else 0.0,
    }
    return dem, stats


def _rasterize_chm_from_ground(
    laz_path: Path,
    out_tif: Path,
    ground: np.ndarray,
    ox: float,
    maxy: float,
    dem_resolution: float,
    resolution: float,
    hag_max: float,
    nodata: float,
    chunk_size: int,
    filter_mode: str,
    method_tag: str,
) -> Path:
    with laspy.open(str(laz_path)) as fh:
        hdr = fh.header
        minx, miny = float(hdr.mins[0]), float(hdr.mins[1])
        maxx, maxy = float(hdr.maxs[0]), float(hdr.maxs[1])

    ox2, oy2 = _snap_origin(minx, miny, resolution)
    out_w = int(math.ceil((maxx - ox2) / resolution))
    out_h = int(math.ceil((maxy - oy2) / resolution))
    raster = np.full((out_h, out_w), nodata, dtype=np.float32)

    with laspy.open(str(laz_path)) as fh:
        for pts in fh.chunk_iterator(chunk_size):
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float32)
            g = _bilinear_sample(ground, x, y, ox, maxy, dem_resolution)
            hag_raw = z - g

            if filter_mode == "drop":
                keep = (hag_raw >= 0.0) & (hag_raw <= hag_max)
                if not np.any(keep):
                    continue
                x = x[keep]
                y = y[keep]
                hag = hag_raw[keep]
            elif filter_mode == "clip":
                hag = np.clip(hag_raw, 0.0, hag_max)
            elif filter_mode == "raw":
                hag = np.clip(hag_raw, 0.0, None)
            else:
                raise ValueError(f"Unsupported filter_mode: {filter_mode}")

            c = ((x - ox2) / resolution).astype(np.int32)
            r = ((maxy - y) / resolution).astype(np.int32)
            valid = (r >= 0) & (r < out_h) & (c >= 0) & (c < out_w)
            if not np.any(valid):
                continue
            flat = r[valid] * out_w + c[valid]
            np.maximum.at(raster.ravel(), flat, hag[valid].astype(np.float32))

    transform = from_origin(ox2, maxy, resolution, resolution)
    profile = {
        "driver": "GTiff",
        "height": out_h,
        "width": out_w,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(raster, 1)
        dst.update_tags(
            SOURCE_LAZ=laz_path.name,
            METHOD=method_tag,
            HAG_MAX=str(hag_max),
            DEM_RES=str(dem_resolution),
            FILTER_MODE=filter_mode,
        )
    return out_tif


def _compute_class2_surface_chm_with_ground_filter(
    laz_path: Path,
    out_tif: Path,
    resolution: float,
    hag_max: float,
    nodata: float,
    chunk_size: int,
    dem_resolution: float,
    filter_mode: str,
    smooth_mode: str,
    class2_filter_method: str,
    neighborhood_sz: int = 5,
    threshold: float = 2.0,
) -> tuple[Path, dict]:
    dem0, ox, _oy, _maxx, maxy = _build_min_dem_from_points(
        laz_path=laz_path,
        dem_resolution=dem_resolution,
        chunk_size=chunk_size,
        class2_only=True,
    )
    reject_mask = _compute_class2_outlier_mask(
        dem=dem0,
        method=class2_filter_method,
        neighborhood_sz=neighborhood_sz,
        threshold=threshold,
    )

    dem1, fstats = _rebuild_class2_dem_with_reject_mask(
        laz_path=laz_path,
        dem_resolution=dem_resolution,
        chunk_size=chunk_size,
        ox=ox,
        maxy=maxy,
        dem_h=dem0.shape[0],
        dem_w=dem0.shape[1],
        reject_mask=reject_mask,
    )

    finite0 = np.isfinite(dem0)
    base0 = np.where(finite0, dem0, float(np.nanmedian(dem0[finite0])))
    rough0 = float(np.median(np.abs(base0 - median_filter(base0, size=5, mode="reflect"))))

    finite1 = np.isfinite(dem1)
    if not np.any(finite1):
        # Safety fallback if filtering was too strict.
        dem1 = dem0.copy()
        finite1 = np.isfinite(dem1)
    base1 = np.where(finite1, dem1, float(np.nanmedian(dem1[finite1])))
    rough1 = float(np.median(np.abs(base1 - median_filter(base1, size=5, mode="reflect"))))
    ground = _smooth_ground_surface(base1, smooth_mode=smooth_mode)

    _rasterize_chm_from_ground(
        laz_path=laz_path,
        out_tif=out_tif,
        ground=ground,
        ox=ox,
        maxy=maxy,
        dem_resolution=dem_resolution,
        resolution=resolution,
        hag_max=hag_max,
        nodata=nodata,
        chunk_size=chunk_size,
        filter_mode=filter_mode,
        method_tag=f"class2_filter_{class2_filter_method}_{smooth_mode}",
    )

    fstats.update(
        {
            "n_reject_cells": int(np.count_nonzero(reject_mask)),
            "reject_cell_pct": float(100.0 * np.count_nonzero(reject_mask) / reject_mask.size),
            "dem_roughness_before": rough0,
            "dem_roughness_after": rough1,
            "dem_roughness_reduction_pct": float(100.0 * (rough0 - rough1) / rough0) if rough0 > 1e-9 else 0.0,
            "filter_method": class2_filter_method,
            "neighborhood_sz": int(neighborhood_sz),
            "threshold": float(threshold),
        }
    )
    return out_tif, fstats


def _read_valid(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        return arr.compressed().astype(np.float32)


def _raster_stats(path: Path) -> dict:
    vals = _read_valid(path)
    if vals.size == 0:
        return {"count": 0}
    return {
        "count": int(vals.size),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
    }


def _quicklook_png(chm_path: Path, out_png: Path, title: str) -> None:
    with rasterio.open(chm_path) as src:
        arr = src.read(1, masked=True).astype(np.float32)
        data = arr.filled(np.nan)

    vmax = np.nanpercentile(data, 99.5)
    vmax = max(vmax, 0.3)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("HAG (m)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _hist_png(chm_path: Path, out_png: Path, title: str) -> None:
    vals = _read_valid(chm_path)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.hist(vals, bins=150, range=(0, max(1.3, float(np.percentile(vals, 99.9)))), color="#1f77b4", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("HAG (m)")
    ax.set_ylabel("Pixel count")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _diff_png(ref_path: Path, other_path: Path, out_png: Path, title: str) -> None:
    with rasterio.open(ref_path) as a, rasterio.open(other_path) as b:
        ar = a.read(1, masked=True).astype(np.float32)
        br = b.read(1, masked=True).astype(np.float32)
        if ar.shape != br.shape or tuple(a.transform) != tuple(b.transform):
            raise RuntimeError("Diff requires aligned rasters")
        valid = (~ar.mask) & (~br.mask)
        diff = np.full(ar.shape, np.nan, dtype=np.float32)
        diff[valid] = br.data[valid] - ar.data[valid]

    lim = float(np.nanpercentile(np.abs(diff), 99)) if np.isfinite(diff).any() else 0.5
    lim = max(lim, 0.2)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title(title)
    ax.axis("off")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Delta HAG (m)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _tile_max_feature(chm_path: Path, labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df = df[df["label"].isin(["cdw", "no_cdw"])].copy()
    if df.empty:
        raise RuntimeError("No binary labels found")

    with rasterio.open(chm_path) as src:
        arr = src.read(1)
        nodata = src.nodata

    feats = []
    h, w = arr.shape
    for row in df.itertuples(index=False):
        rr = int(row.row_off)
        cc = int(row.col_off)
        cs = int(row.chunk_size)
        r2 = min(rr + cs, h)
        c2 = min(cc + cs, w)
        tile = arr[rr:r2, cc:c2].astype(np.float32)
        if nodata is not None:
            tile = tile[tile != nodata]
        if tile.size == 0:
            tmax = 0.0
            tmean = 0.0
            tabove = 0.0
        else:
            tile = np.clip(tile, 0.0, None)
            tmax = float(np.max(tile))
            tmean = float(np.mean(tile))
            tabove = float(np.mean(tile >= 0.15))
        feats.append((row.label, tmax, tmean, tabove))

    out = pd.DataFrame(feats, columns=["label", "tile_max", "tile_mean", "tile_frac_above_15cm"])
    out["y"] = (out["label"] == "cdw").astype(int)
    return out


def _threshold_metrics(feat: pd.DataFrame, feature: str) -> dict:
    y = feat["y"].to_numpy()
    s = feat[feature].to_numpy()
    if s.size == 0:
        return {}

    thresholds = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 120)))
    best = None
    for thr in thresholds:
        pred = (s >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tpr
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        j = tpr - fpr

        item = {
            "threshold": float(thr),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "precision": float(prec),
            "f1": float(f1),
            "youden_j": float(j),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        if best is None or item["youden_j"] > best["youden_j"]:
            best = item

    return best or {}


def _auc_score(y: np.ndarray, s: np.ndarray) -> float | None:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(s)
    rank_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    na = int(a.size)
    nb = int(b.size)
    pooled = ((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1)
    if pooled <= 1e-12:
        return None
    return float((float(np.mean(a)) - float(np.mean(b))) / math.sqrt(pooled))


def _method_eval(chm_path: Path, labels_csv: Path) -> dict:
    feats = _tile_max_feature(chm_path, labels_csv)
    cdw = feats[feats["label"] == "cdw"]
    no = feats[feats["label"] == "no_cdw"]
    y = feats["y"].to_numpy()

    out = {
        "n_tiles": int(len(feats)),
        "n_cdw": int(len(cdw)),
        "n_no_cdw": int(len(no)),
        "cdw_tile_max_mean": float(cdw["tile_max"].mean()) if len(cdw) else None,
        "no_tile_max_mean": float(no["tile_max"].mean()) if len(no) else None,
        "cdw_detect_rate_15cm": float((cdw["tile_max"] >= 0.15).mean()) if len(cdw) else None,
        "no_false_high_rate_15cm": float((no["tile_max"] >= 0.15).mean()) if len(no) else None,
        "best_youden_tile_max": _threshold_metrics(feats, "tile_max"),
        "best_youden_frac_above_15cm": _threshold_metrics(feats, "tile_frac_above_15cm"),
        "auc_tile_max": _auc_score(y, feats["tile_max"].to_numpy()),
        "auc_frac_above_15cm": _auc_score(y, feats["tile_frac_above_15cm"].to_numpy()),
        "cohens_d_tile_max": _cohens_d(cdw["tile_max"].to_numpy(), no["tile_max"].to_numpy()),
        "cohens_d_frac_above_15cm": _cohens_d(
            cdw["tile_frac_above_15cm"].to_numpy(),
            no["tile_frac_above_15cm"].to_numpy(),
        ),
    }
    return out


def _choose_best_method(results: Dict[str, dict]) -> str:
    def score(name: str) -> float:
        r = results[name]
        j1 = float(r.get("best_youden_tile_max", {}).get("youden_j", 0.0))
        j2 = float(r.get("best_youden_frac_above_15cm", {}).get("youden_j", 0.0))
        cdw = float(r.get("cdw_detect_rate_15cm", 0.0) or 0.0)
        no_fp = float(r.get("no_false_high_rate_15cm", 1.0) or 1.0)
        return 0.45 * j1 + 0.25 * j2 + 0.2 * cdw + 0.1 * (1.0 - no_fp)

    best = max(results.keys(), key=score)
    return best


def _write_markdown_report(report_path: Path, payload: dict) -> None:
    m = []
    m.append("# Ground Estimation Study: 436646_2024_madal")
    m.append("")
    m.append("## Objective")
    m.append("Assess alternatives to unreliable LAS ground tags for LAZ->CHM generation and select the best method for CDW visibility.")
    m.append("")
    m.append(f"Configured CHM mode: **{payload.get('filter_mode', 'unknown')}**")
    m.append("")
    m.append("## Methods")
    for row in payload["methods"]:
        m.append(f"- {row['name']}: {row['notes']}")
    m.append("")
    m.append("## Raster Statistics")
    for name, st in payload["raster_stats"].items():
        m.append(f"- {name}: max={st.get('max')} p95={st.get('p95')} mean={st.get('mean')} std={st.get('std')}")
    m.append("")
    m.append("## CDW Validation (tile labels)")
    for name, ev in payload["evaluation"].items():
        best1 = ev.get("best_youden_tile_max", {})
        best2 = ev.get("best_youden_frac_above_15cm", {})
        m.append(
            f"- {name}: cdw_detect_rate_15cm={ev.get('cdw_detect_rate_15cm'):.4f}, "
            f"no_false_high_rate_15cm={ev.get('no_false_high_rate_15cm'):.4f}, "
            f"best_tile_max_j={best1.get('youden_j', 0.0):.4f} at thr={best1.get('threshold', 0.0):.4f}, "
            f"best_frac15_j={best2.get('youden_j', 0.0):.4f} at thr={best2.get('threshold', 0.0):.4f}"
        )
        m.append(
            f"  auc_tile_max={ev.get('auc_tile_max', None)} auc_frac15={ev.get('auc_frac_above_15cm', None)} "
            f"cohens_d_tile_max={ev.get('cohens_d_tile_max', None)}"
        )
    if payload.get("filter_diagnostics"):
        m.append("")
        m.append("## Ground Filter Diagnostics")
        for k, v in payload["filter_diagnostics"].items():
            m.append(
                f"- {k}: n_class2_input={v.get('n_class2_input')} n_class2_filtered={v.get('n_class2_filtered')} "
                f"filter_pct={v.get('filter_pct'):.3f} reject_cell_pct={v.get('reject_cell_pct'):.3f} "
                f"roughness_reduction_pct={v.get('dem_roughness_reduction_pct'):.3f}"
            )
    m.append("")
    m.append("## Recommended Method")
    m.append(f"**{payload['best_method']}**")
    m.append("")
    m.append("## Thesis-Oriented Interpretation")
    m.append("- Ground tags (class 2) can suppress CDW when misclassified points define the local ground surface.")
    m.append("- Recomputing ground with physical filters (SMRF or CSF) reduces dependence on delivered class tags.")
    m.append("- The selected method is the one with strongest CDW/no-CDW separability on independent tile labels and acceptable false-high behavior.")
    m.append("- For thesis reproducibility, report filter parameters, HAG cap, grid resolution, and label-source version.")

    report_path.write_text("\n".join(m) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Ground-method study for one LAZ tile")
    ap.add_argument("--laz", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/ground_method_study_436646"))
    ap.add_argument("--resolution", type=float, default=0.2)
    ap.add_argument("--hag-max", type=float, default=1.3)
    ap.add_argument(
        "--filter-mode",
        choices=["drop", "clip", "raw"],
        default="drop",
        help="drop: keep 0..hag_max only; clip: clamp to hag_max; raw: no upper clip",
    )
    ap.add_argument("--chunk-size", type=int, default=2_000_000)
    ap.add_argument("--dem-resolution", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = args.out_dir
    chm_dir = out_dir / "chm"
    viz_dir = out_dir / "visuals"
    scratch = out_dir / "scratch"
    for d in (chm_dir, viz_dir, scratch):
        d.mkdir(parents=True, exist_ok=True)

    methods: List[MethodResult] = []
    filter_diagnostics: Dict[str, dict] = {}

    drop_above = args.filter_mode == "drop"
    if args.filter_mode == "raw":
        hag_max_effective = 100.0
    else:
        hag_max_effective = args.hag_max

    # 1) Baseline from existing class tags
    m1 = chm_dir / "436646_2024_madal_class2_idw_chm_max_hag_20cm.tif"
    compute_hag_raster_streamed(
        laz_path=args.laz,
        out_tif=m1,
        resolution=args.resolution,
        hag_max=hag_max_effective,
        nodata=-9999.0,
        chunk_size=args.chunk_size,
        drop_above_hag_max=drop_above,
    )
    methods.append(
        MethodResult(
            "class2_idw_baseline",
            m1,
            f"Existing LAS class 2 ground points + IDW interpolation ({args.filter_mode} mode).",
        )
    )

    # 2) SMRF reclassification
    smrf_laz = scratch / "436646_2024_madal_smrf_reclass.laz"
    _pdal_reclassify(args.laz, smrf_laz, "smrf", scratch)
    m2 = chm_dir / "436646_2024_madal_smrf_idw_chm_max_hag_20cm.tif"
    compute_hag_raster_streamed(
        laz_path=smrf_laz,
        out_tif=m2,
        resolution=args.resolution,
        hag_max=hag_max_effective,
        nodata=-9999.0,
        chunk_size=args.chunk_size,
        drop_above_hag_max=drop_above,
    )
    methods.append(
        MethodResult(
            "smrf_idw",
            m2,
            f"PDAL SMRF ground reclassification, then same IDW CHM logic ({args.filter_mode} mode).",
        )
    )

    # 3) CSF reclassification
    csf_laz = scratch / "436646_2024_madal_csf_reclass.laz"
    _pdal_reclassify(args.laz, csf_laz, "csf", scratch)
    m3 = chm_dir / "436646_2024_madal_csf_idw_chm_max_hag_20cm.tif"
    compute_hag_raster_streamed(
        laz_path=csf_laz,
        out_tif=m3,
        resolution=args.resolution,
        hag_max=hag_max_effective,
        nodata=-9999.0,
        chunk_size=args.chunk_size,
        drop_above_hag_max=drop_above,
    )
    methods.append(
        MethodResult(
            "csf_idw",
            m3,
            f"PDAL CSF ground reclassification, then same IDW CHM logic ({args.filter_mode} mode).",
        )
    )

    # 4) No-tag robust all-points ground surface
    m4 = chm_dir / "436646_2024_madal_allpoints_surface_chm_max_hag_20cm.tif"
    _compute_quantile_surface_chm(
        laz_path=args.laz,
        out_tif=m4,
        resolution=args.resolution,
        hag_max=args.hag_max,
        nodata=-9999.0,
        chunk_size=args.chunk_size,
        dem_resolution=args.dem_resolution,
        filter_mode=args.filter_mode,
    )
    methods.append(
        MethodResult(
            "allpoints_surface",
            m4,
            f"No class tags: coarse minimum surface + morphological opening + smoothing, then CHM ({args.filter_mode} mode).",
        )
    )

    # 5-7) Class2-ground surface variants with explicit smoothness control
    m5 = chm_dir / "436646_2024_madal_class2_surface_fixed_low_chm_max_hag_20cm.tif"
    _compute_class2_surface_chm(
        laz_path=args.laz,
        out_tif=m5,
        resolution=args.resolution,
        hag_max=args.hag_max,
        nodata=-9999.0,
        chunk_size=args.chunk_size,
        dem_resolution=args.dem_resolution,
        filter_mode=args.filter_mode,
        smooth_mode="fixed_low",
    )
    methods.append(
        MethodResult(
            "class2_surface_fixed_low",
            m5,
            f"Class2 ground DEM with fixed low smoothing (sigma=0.8), then CHM ({args.filter_mode} mode).",
        )
    )

    m6 = chm_dir / "436646_2024_madal_class2_surface_fixed_high_chm_max_hag_20cm.tif"
    _compute_class2_surface_chm(
        laz_path=args.laz,
        out_tif=m6,
        resolution=args.resolution,
        hag_max=args.hag_max,
        nodata=-9999.0,
        chunk_size=args.chunk_size,
        dem_resolution=args.dem_resolution,
        filter_mode=args.filter_mode,
        smooth_mode="fixed_high",
    )
    methods.append(
        MethodResult(
            "class2_surface_fixed_high",
            m6,
            f"Class2 ground DEM with fixed high smoothing (sigma=2.4), then CHM ({args.filter_mode} mode).",
        )
    )

    m7 = chm_dir / "436646_2024_madal_class2_surface_adaptive_chm_max_hag_20cm.tif"
    _compute_class2_surface_chm(
        laz_path=args.laz,
        out_tif=m7,
        resolution=args.resolution,
        hag_max=args.hag_max,
        nodata=-9999.0,
        chunk_size=args.chunk_size,
        dem_resolution=args.dem_resolution,
        filter_mode=args.filter_mode,
        smooth_mode="adaptive",
    )
    methods.append(
        MethodResult(
            "class2_surface_adaptive",
            m7,
            f"Class2 ground DEM with adaptive roughness-based smoothing, then CHM ({args.filter_mode} mode).",
        )
    )

    # 8-10) Option C: selective class-2 filtering methods all at once
    m8 = chm_dir / "436646_2024_madal_class2_filter_local_residual_chm_max_hag_20cm.tif"
    _, d8 = _compute_class2_surface_chm_with_ground_filter(
        laz_path=args.laz,
        out_tif=m8,
        resolution=args.resolution,
        hag_max=args.hag_max,
        nodata=-9999.0,
        chunk_size=args.chunk_size,
        dem_resolution=args.dem_resolution,
        filter_mode=args.filter_mode,
        smooth_mode="fixed_high",
        class2_filter_method="local_residual",
        neighborhood_sz=5,
        threshold=2.0,
    )
    methods.append(
        MethodResult(
            "class2_filter_local_residual",
            m8,
            f"Class2 filter local residual (n=5, thr=2.0) + fixed_high surface smoothing, then CHM ({args.filter_mode} mode).",
        )
    )
    filter_diagnostics["class2_filter_local_residual"] = d8

    m9 = chm_dir / "436646_2024_madal_class2_filter_mad_chm_max_hag_20cm.tif"
    _, d9 = _compute_class2_surface_chm_with_ground_filter(
        laz_path=args.laz,
        out_tif=m9,
        resolution=args.resolution,
        hag_max=args.hag_max,
        nodata=-9999.0,
        chunk_size=args.chunk_size,
        dem_resolution=args.dem_resolution,
        filter_mode=args.filter_mode,
        smooth_mode="fixed_high",
        class2_filter_method="mad",
        neighborhood_sz=5,
        threshold=2.5,
    )
    methods.append(
        MethodResult(
            "class2_filter_mad",
            m9,
            f"Class2 filter MAD (n=5, thr=2.5) + fixed_high surface smoothing, then CHM ({args.filter_mode} mode).",
        )
    )
    filter_diagnostics["class2_filter_mad"] = d9

    m10 = chm_dir / "436646_2024_madal_class2_filter_slope_curvature_chm_max_hag_20cm.tif"
    _, d10 = _compute_class2_surface_chm_with_ground_filter(
        laz_path=args.laz,
        out_tif=m10,
        resolution=args.resolution,
        hag_max=args.hag_max,
        nodata=-9999.0,
        chunk_size=args.chunk_size,
        dem_resolution=args.dem_resolution,
        filter_mode=args.filter_mode,
        smooth_mode="fixed_high",
        class2_filter_method="slope_curvature",
        neighborhood_sz=5,
        threshold=0.6,
    )
    methods.append(
        MethodResult(
            "class2_filter_slope_curvature",
            m10,
            f"Class2 filter slope-curvature (slope_thr=0.6) + fixed_high surface smoothing, then CHM ({args.filter_mode} mode).",
        )
    )
    filter_diagnostics["class2_filter_slope_curvature"] = d10

    raster_stats: Dict[str, dict] = {}
    evaluation: Dict[str, dict] = {}

    base = methods[0]
    for m in methods:
        raster_stats[m.name] = _raster_stats(m.chm_path)
        evaluation[m.name] = _method_eval(m.chm_path, args.labels)

        _quicklook_png(m.chm_path, viz_dir / f"{m.name}_quicklook.png", f"{m.name} CHM")
        _hist_png(m.chm_path, viz_dir / f"{m.name}_hist.png", f"{m.name} histogram")

        if m.name != base.name:
            _diff_png(base.chm_path, m.chm_path, viz_dir / f"diff_{base.name}_vs_{m.name}.png", f"Delta {m.name} - {base.name}")

    best_method = _choose_best_method(evaluation)

    payload = {
        "input_laz": str(args.laz),
        "labels": str(args.labels),
        "filter_mode": args.filter_mode,
        "hag_max_requested": args.hag_max,
        "hag_max_effective": hag_max_effective,
        "methods": [{"name": x.name, "chm": str(x.chm_path), "notes": x.notes} for x in methods],
        "raster_stats": raster_stats,
        "evaluation": evaluation,
        "filter_diagnostics": filter_diagnostics,
        "best_method": best_method,
    }

    out_json = out_dir / "ground_method_report.json"
    out_md = out_dir / "ground_method_report.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown_report(out_md, payload)

    print(f"report_json={out_json}")
    print(f"report_md={out_md}")
    print(f"best_method={best_method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

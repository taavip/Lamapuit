#!/usr/bin/env python3
"""SOTA-style DTM/HAG experiment for tile 436646 across all years.

This experiment implements a practical pipeline aligned with the requested method:
1) Ground classification with PDAL SMRF.
2) Multi-temporal harmonization via stacked ground points + SOR + per-cell 10th percentile.
3) Interpolation via TIN linear and Natural Neighbor-style linear triangulation.
4) Edge-preserving bilateral filtering variants on DTM.
5) CHM creation at 0.2 m using configurable return policy/class exclusions and
    configurable HAG/CHM clipping bounds.

Outputs are written into a dated experiment directory.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import laspy
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata
from scipy.spatial import cKDTree
from scipy.stats import rankdata

try:
    import cv2

    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

NODATA = -9999.0


@dataclass
class YearInput:
    year: int
    laz_path: Path
    labels_csv: Path
    baseline_chm: Path


def _run(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_pdal_pipeline(pipeline: list, path: Path) -> None:
    path.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")


def _snap_origin(minx: float, miny: float, resolution: float) -> Tuple[float, float]:
    ox = math.floor(minx / resolution) * resolution
    oy = math.floor(miny / resolution) * resolution
    return ox, oy


def _pdal_smrf_reclassify(
    input_laz: Path,
    output_laz: Path,
    work_dir: Path,
    epsg: int,
    scalar: float,
    slope: float,
    threshold: float,
    window: float,
) -> None:
    output_laz.parent.mkdir(parents=True, exist_ok=True)
    pipe_path = work_dir / f"pdal_smrf_{input_laz.stem}.json"

    pipeline = [
        {
            "type": "readers.las",
            "filename": str(input_laz),
            "override_srs": f"EPSG:{epsg}",
        },
        {"type": "filters.assign", "value": "Classification = 1"},
        {
            "type": "filters.smrf",
            "scalar": float(scalar),
            "slope": float(slope),
            "threshold": float(threshold),
            "window": float(window),
        },
        {"type": "writers.las", "filename": str(output_laz), "compression": "true"},
    ]

    _write_pdal_pipeline(pipeline, pipe_path)
    _run(["pdal", "pipeline", str(pipe_path)])


def _laz_bounds(laz_path: Path) -> Tuple[float, float, float, float]:
    with laspy.open(str(laz_path)) as fh:
        hdr = fh.header
        return float(hdr.mins[0]), float(hdr.mins[1]), float(hdr.maxs[0]), float(hdr.maxs[1])


def _global_bounds(paths: Iterable[Path]) -> Tuple[float, float, float, float]:
    mins = []
    maxs = []
    for p in paths:
        minx, miny, maxx, maxy = _laz_bounds(p)
        mins.append((minx, miny))
        maxs.append((maxx, maxy))
    minx = float(min(v[0] for v in mins))
    miny = float(min(v[1] for v in mins))
    maxx = float(max(v[0] for v in maxs))
    maxy = float(max(v[1] for v in maxs))
    return minx, miny, maxx, maxy


def _build_year_ground_dem(
    smrf_laz: Path,
    ox: float,
    maxy: float,
    dem_resolution: float,
    dem_h: int,
    dem_w: int,
    chunk_size: int,
) -> Tuple[np.ndarray, dict]:
    """Per-year DEM from class-2 minimum Z per cell."""
    dem = np.full((dem_h, dem_w), np.nan, dtype=np.float32)
    n_points = 0
    n_ground = 0

    with laspy.open(str(smrf_laz)) as fh:
        for pts in fh.chunk_iterator(chunk_size):
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float32)
            n_points += int(len(x))

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
            n_ground += int(len(x))

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

    valid_cells = int(np.isfinite(dem).sum())
    stats = {
        "points_total": n_points,
        "points_ground_class2": n_ground,
        "ground_ratio_pct": float(100.0 * n_ground / n_points) if n_points else 0.0,
        "dem_valid_cells": valid_cells,
        "dem_valid_pct": float(100.0 * valid_cells / dem.size) if dem.size else 0.0,
    }
    return dem, stats


def _estimate_vertical_offsets(
    year_dems: Dict[int, np.ndarray],
    reference_year: int,
    stable_slope_quantile: float,
) -> Tuple[Dict[int, float], dict]:
    """Estimate robust vertical offsets using low-slope overlapping surfaces."""
    ref = year_dems[reference_year]
    finite_ref = np.isfinite(ref)
    if not np.any(finite_ref):
        offsets = {y: 0.0 for y in year_dems}
        return offsets, {"reference_year": reference_year, "notes": "reference DEM has no finite cells"}

    fill = np.where(finite_ref, ref, np.nanmedian(ref[finite_ref])).astype(np.float32)
    gx, gy = np.gradient(fill)
    slope = np.hypot(gx, gy)
    slope_thr = float(np.nanpercentile(slope[finite_ref], stable_slope_quantile))
    stable = finite_ref & (slope <= slope_thr)

    offsets: Dict[int, float] = {reference_year: 0.0}
    diag = {
        "reference_year": reference_year,
        "stable_slope_quantile": float(stable_slope_quantile),
        "stable_slope_threshold": slope_thr,
        "pairwise": {},
    }

    for year, dem in sorted(year_dems.items()):
        if year == reference_year:
            continue
        mask = stable & np.isfinite(dem)
        if int(mask.sum()) < 1000:
            mask = finite_ref & np.isfinite(dem)
        if int(mask.sum()) == 0:
            off = 0.0
        else:
            delta = ref[mask].astype(np.float64) - dem[mask].astype(np.float64)
            off = float(np.median(delta))
        offsets[year] = off
        diag["pairwise"][str(year)] = {
            "overlap_cells": int(mask.sum()),
            "vertical_shift_to_reference_m": off,
        }

    return offsets, diag


def _collect_smrf_ground_points(
    smrf_laz: Path,
    z_offset: float,
    chunk_size: int,
    max_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks: List[np.ndarray] = []
    per_chunk_cap = max(5000, max_points // 4)

    with laspy.open(str(smrf_laz)) as fh:
        for pts in fh.chunk_iterator(chunk_size):
            try:
                cls = np.asarray(pts.classification)
                keep = cls == 2
            except Exception:
                keep = np.zeros(len(pts.x), dtype=bool)

            if not np.any(keep):
                continue

            x = np.asarray(pts.x[keep], dtype=np.float64)
            y = np.asarray(pts.y[keep], dtype=np.float64)
            z = np.asarray(pts.z[keep], dtype=np.float64) + float(z_offset)
            arr = np.column_stack((x, y, z))
            if arr.shape[0] > per_chunk_cap:
                idx = rng.choice(arr.shape[0], size=per_chunk_cap, replace=False)
                arr = arr[idx]
            chunks.append(arr)

    if not chunks:
        return np.empty((0, 3), dtype=np.float64)

    all_pts = np.vstack(chunks)
    if all_pts.shape[0] > max_points:
        idx = rng.choice(all_pts.shape[0], size=max_points, replace=False)
        all_pts = all_pts[idx]
    return all_pts


def _sor_filter_points(
    xyz: np.ndarray,
    neighbors: int,
    std_multiplier: float,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, dict]:
    """Statistical Outlier Removal (SOR) on stacked ground super-cloud."""
    if xyz.shape[0] == 0:
        return xyz, {"input_points": 0, "output_points": 0}

    rng = np.random.default_rng(seed)
    in_count = int(xyz.shape[0])
    work = xyz
    sampled = False
    if work.shape[0] > max_points:
        idx = rng.choice(work.shape[0], size=max_points, replace=False)
        work = work[idx]
        sampled = True

    k = int(max(2, min(neighbors + 1, work.shape[0])))
    tree = cKDTree(work[:, :3].astype(np.float64))
    try:
        dists, _ = tree.query(work[:, :3], k=k, workers=-1)
    except TypeError:
        dists, _ = tree.query(work[:, :3], k=k)

    if dists.ndim == 1:
        mean_dist = dists.astype(np.float64)
    else:
        mean_dist = np.mean(dists[:, 1:], axis=1)

    mu = float(np.mean(mean_dist))
    sigma = float(np.std(mean_dist))
    thr = mu + float(std_multiplier) * sigma
    keep = mean_dist <= thr
    filtered = work[keep]

    stats = {
        "input_points": in_count,
        "sor_working_points": int(work.shape[0]),
        "sampled_for_sor": bool(sampled),
        "neighbors": int(neighbors),
        "std_multiplier": float(std_multiplier),
        "mean_neighbor_distance": mu,
        "std_neighbor_distance": sigma,
        "distance_threshold": thr,
        "output_points": int(filtered.shape[0]),
        "kept_pct": float(100.0 * filtered.shape[0] / max(work.shape[0], 1)),
    }
    return filtered, stats


def _temporal_percentile_dem(
    xyz: np.ndarray,
    ox: float,
    maxy: float,
    dem_resolution: float,
    dem_h: int,
    dem_w: int,
    percentile: float,
) -> Tuple[np.ndarray, dict]:
    """Build harmonized DTM grid as per-cell temporal percentile (e.g. 10th)."""
    dem = np.full((dem_h, dem_w), np.nan, dtype=np.float32)
    if xyz.shape[0] == 0:
        return dem, {"valid_cells": 0, "valid_pct": 0.0, "percentile": float(percentile)}

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    c = ((x - ox) / dem_resolution).astype(np.int32)
    r = ((maxy - y) / dem_resolution).astype(np.int32)
    valid = (r >= 0) & (r < dem_h) & (c >= 0) & (c < dem_w)
    if not np.any(valid):
        return dem, {"valid_cells": 0, "valid_pct": 0.0, "percentile": float(percentile)}

    rr = r[valid]
    cc = c[valid]
    zz = z[valid]
    flat = rr * dem_w + cc

    order = np.argsort(flat)
    flat_sorted = flat[order]
    z_sorted = zz[order]

    uniq, idx, counts = np.unique(flat_sorted, return_index=True, return_counts=True)
    out_flat = dem.ravel()

    for i in range(len(uniq)):
        s = idx[i]
        e = s + counts[i]
        vals = z_sorted[s:e]
        if vals.size == 1:
            qv = float(vals[0])
        else:
            qv = float(np.percentile(vals, percentile))
        out_flat[uniq[i]] = qv

    valid_cells = int(np.isfinite(dem).sum())
    stats = {
        "percentile": float(percentile),
        "input_points": int(xyz.shape[0]),
        "valid_cells": valid_cells,
        "valid_pct": float(100.0 * valid_cells / dem.size) if dem.size else 0.0,
    }
    return dem, stats


def _dem_to_anchors(dem: np.ndarray, ox: float, maxy: float, dem_resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    rr, cc = np.where(np.isfinite(dem))
    if rr.size == 0:
        raise RuntimeError("Harmonized DEM has no finite cells")
    x = ox + (cc.astype(np.float64) + 0.5) * dem_resolution
    y = maxy - (rr.astype(np.float64) + 0.5) * dem_resolution
    z = dem[rr, cc].astype(np.float64)
    return np.column_stack((x, y)), z


class IDWInterpolator:
    def __init__(self, xy: np.ndarray, z: np.ndarray, k: int = 6, power: float = 2.0, eps: float = 1e-8):
        self.z = z.astype(np.float64)
        self.tree = cKDTree(xy.astype(np.float64))
        self.k = int(max(1, min(k, len(self.z))))
        self.power = float(power)
        self.eps = float(eps)

    def __call__(self, xyq: np.ndarray) -> np.ndarray:
        if self.k == 1:
            _, idx = self.tree.query(xyq, k=1)
            return self.z[idx].astype(np.float64)

        try:
            dists, idx = self.tree.query(xyq, k=self.k, workers=-1)
        except TypeError:
            dists, idx = self.tree.query(xyq, k=self.k)

        with np.errstate(divide="ignore", invalid="ignore"):
            weights = 1.0 / np.power(dists + self.eps, self.power)
            denom = np.sum(weights, axis=1)
            denom[denom == 0] = 1.0
            vals = self.z[idx]
            out = np.sum(weights * vals, axis=1) / denom
        return out.astype(np.float64)


class TINInterpolator:
    def __init__(self, xy: np.ndarray, z: np.ndarray):
        self.linear = LinearNDInterpolator(xy.astype(np.float64), z.astype(np.float64), fill_value=np.nan)
        self.nearest = NearestNDInterpolator(xy.astype(np.float64), z.astype(np.float64))

    def __call__(self, xyq: np.ndarray) -> np.ndarray:
        out = np.asarray(self.linear(xyq), dtype=np.float64).reshape(-1)
        miss = ~np.isfinite(out)
        if np.any(miss):
            out[miss] = np.asarray(self.nearest(xyq[miss]), dtype=np.float64).reshape(-1)
        return out


def _interpolate_grid_model(
    model,
    ox: float,
    maxy: float,
    dem_resolution: float,
    dem_h: int,
    dem_w: int,
    batch_size: int = 200_000,
) -> np.ndarray:
    out = np.empty((dem_h, dem_w), dtype=np.float32)
    total = dem_h * dem_w
    flat = out.ravel()

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        idx = np.arange(start, end, dtype=np.int64)
        rr = idx // dem_w
        cc = idx % dem_w
        x = ox + (cc.astype(np.float64) + 0.5) * dem_resolution
        y = maxy - (rr.astype(np.float64) + 0.5) * dem_resolution
        q = np.column_stack((x, y))
        flat[start:end] = model(q).astype(np.float32)

    return out


def _interpolate_natural_neighbor_grid(
    xy: np.ndarray,
    z: np.ndarray,
    ox: float,
    maxy: float,
    dem_resolution: float,
    dem_h: int,
    dem_w: int,
) -> np.ndarray:
    cols = ox + (np.arange(dem_w, dtype=np.float64) + 0.5) * dem_resolution
    rows = maxy - (np.arange(dem_h, dtype=np.float64) + 0.5) * dem_resolution
    gx, gy = np.meshgrid(cols, rows)

    dtm = griddata(xy, z, (gx, gy), method="linear")
    miss = ~np.isfinite(dtm)
    if np.any(miss):
        nearest_vals = griddata(xy, z, (gx[miss], gy[miss]), method="nearest")
        dtm[miss] = nearest_vals
    return dtm.astype(np.float32)


def _bilateral_filter_dem(
    dem: np.ndarray,
    diameter: int,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    if not HAVE_CV2:
        return dem.copy()
    valid = np.isfinite(dem)
    if not np.any(valid):
        return dem.copy()
    fill_val = float(np.nanmedian(dem[valid]))
    work = np.where(valid, dem, fill_val).astype(np.float32)
    filtered = cv2.bilateralFilter(
        work,
        d=int(max(3, diameter)),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
        borderType=cv2.BORDER_REFLECT,
    )
    filtered = np.asarray(filtered, dtype=np.float32)
    filtered[~valid] = np.nan
    return filtered


def _slope_adaptive_merge_dem(
    dem_base: np.ndarray,
    dem_steep: np.ndarray,
    dem_resolution: float,
    slope_quantile: float,
) -> Tuple[np.ndarray, dict]:
    """Blend DEMs: keep base on gentle terrain, use alternate DEM on steep cells."""
    base = dem_base.astype(np.float32)
    alt = dem_steep.astype(np.float32)
    valid = np.isfinite(base) & np.isfinite(alt)
    if not np.any(valid):
        return base.copy(), {
            "slope_quantile": float(slope_quantile),
            "slope_threshold_deg": None,
            "steep_cells": 0,
            "steep_pct": 0.0,
        }

    fill = np.where(np.isfinite(base), base, np.nanmedian(base[np.isfinite(base)])).astype(np.float32)
    dzdy, dzdx = np.gradient(fill, float(dem_resolution), float(dem_resolution))
    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)

    thr = float(np.nanpercentile(slope_deg[valid], slope_quantile))
    steep = valid & (slope_deg >= thr)

    merged = base.copy()
    merged[steep] = alt[steep]

    stats = {
        "slope_quantile": float(slope_quantile),
        "slope_threshold_deg": thr,
        "steep_cells": int(np.sum(steep)),
        "steep_pct": float(100.0 * np.sum(steep) / np.sum(valid)),
    }
    return merged, stats


def _smooth_masked_chm(
    arr: np.ndarray,
    nodata: float,
    sigma: float,
    hag_max: float,
    chm_clip_min: float,
) -> np.ndarray:
    if sigma <= 0:
        out = arr.copy()
        valid = out != nodata
        out[valid] = np.clip(out[valid], chm_clip_min, hag_max)
        return out

    valid = arr != nodata
    if not np.any(valid):
        return arr.copy()

    data = np.where(valid, arr, 0.0).astype(np.float32)
    weight = valid.astype(np.float32)
    smooth_data = gaussian_filter(data, sigma=sigma, mode="nearest")
    smooth_weight = gaussian_filter(weight, sigma=sigma, mode="nearest")

    out = np.full(arr.shape, nodata, dtype=np.float32)
    keep = smooth_weight > 1e-6
    out[keep] = smooth_data[keep] / smooth_weight[keep]

    valid_out = out != nodata
    out[valid_out] = np.clip(out[valid_out], chm_clip_min, hag_max)
    return out


def _write_gtiff(path: Path, arr: np.ndarray, transform, crs, nodata: float, tags: dict) -> None:
    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


def _read_baseline_grid_spec(chm_path: Path) -> dict:
    with rasterio.open(chm_path) as src:
        return {
            "width": int(src.width),
            "height": int(src.height),
            "transform": src.transform,
            "crs": src.crs,
            "nodata": float(src.nodata) if src.nodata is not None else NODATA,
            "ox": float(src.transform.c),
            "maxy": float(src.transform.f),
            "res": float(abs(src.transform.a)),
        }


def _generate_year_chms(
    year_input: YearInput,
    method_dtms: Dict[str, np.ndarray],
    dtm_transform,
    dtm_crs,
    out_dir: Path,
    hag_min: float,
    hag_max: float,
    chm_clip_min: float,
    chunk_size: int,
    return_mode: str,
    exclude_classes: set[int],
    gaussian_sigma: float,
    seed: int,
) -> Dict[str, Path]:
    spec = _read_baseline_grid_spec(year_input.baseline_chm)
    width = spec["width"]
    height = spec["height"]
    ox = spec["ox"]
    maxy = spec["maxy"]
    res = spec["res"]
    nodata = spec["nodata"]
    out_crs = spec["crs"] if spec["crs"] is not None else dtm_crs

    method_chms = {m: np.full((height, width), nodata, dtype=np.float32) for m in method_dtms}
    dtm_resampled_dict = {}

    for method_name, dtm in method_dtms.items():
        dtm_resampled = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=dtm.astype(np.float32),
            destination=dtm_resampled,
            src_transform=dtm_transform,
            src_crs=dtm_crs,
            dst_transform=spec["transform"],
            dst_crs=out_crs,
            resampling=Resampling.bilinear,
        )
        dtm_resampled_dict[method_name] = dtm_resampled

    rng = np.random.default_rng(seed + year_input.year)

    with laspy.open(str(year_input.laz_path)) as fh:
        for pts in fh.chunk_iterator(chunk_size):
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float32)
            if x.size == 0:
                continue

            keep = np.ones(x.size, dtype=bool)
            try:
                cls = np.asarray(pts.classification)
                if exclude_classes:
                    keep &= ~np.isin(cls, list(exclude_classes))
            except Exception:
                pass

            if return_mode != "all":
                try:
                    rn = np.asarray(pts.return_number)
                    nr = np.asarray(pts.number_of_returns)
                    if return_mode == "last":
                        keep &= rn == nr
                    elif return_mode == "last2":
                        cutoff = np.maximum(1, nr - 1)
                        keep &= rn >= cutoff
                    else:
                        raise ValueError(f"Unsupported return_mode: {return_mode}")
                except Exception:
                    # If return attributes are missing, keep behavior deterministic.
                    keep &= False

            if not np.any(keep):
                continue

            x = x[keep]
            y = y[keep]
            z = z[keep]

            # Tiny random tie-break to avoid deterministic striping when many points share same z.
            z = z + rng.normal(0.0, 1e-6, size=z.shape[0]).astype(np.float32)

            col = ((x - ox) / res).astype(np.int32)
            row = ((maxy - y) / res).astype(np.int32)
            valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
            if not np.any(valid):
                continue

            x = x[valid]
            y = y[valid]
            z = z[valid]
            row = row[valid]
            col = col[valid]
            flat = row * width + col

            for method_name, chm_arr in method_chms.items():
                dtm_z = dtm_resampled_dict[method_name][row, col]
                valid_dtm = np.isfinite(dtm_z)
                if not np.any(valid_dtm):
                    continue

                hag = z[valid_dtm] - dtm_z[valid_dtm]
                keep_hag = (hag >= hag_min) & (hag <= hag_max)
                if not np.any(keep_hag):
                    continue

                valid_flat = flat[valid_dtm][keep_hag]
                valid_hag = np.clip(hag[keep_hag], chm_clip_min, hag_max).astype(np.float32)
                np.maximum.at(chm_arr.ravel(), valid_flat, valid_hag)

    outputs: Dict[str, Path] = {}
    for method_name, chm_arr in method_chms.items():
        out_path = out_dir / f"{year_input.year}_{method_name}_chm.tif"
        _write_gtiff(
            out_path,
            chm_arr,
            spec["transform"],
            out_crs,
            nodata,
            {
                "SOURCE_LAZ": year_input.laz_path.name,
                "YEAR": year_input.year,
                "METHOD": method_name,
                "HAG_MIN": hag_min,
                "HAG_MAX": hag_max,
                "CHM_CLIP_MIN": chm_clip_min,
                "RETURN_MODE": return_mode,
                "EXCLUDE_CLASSES": ",".join(str(v) for v in sorted(exclude_classes)),
                "POST_FILTER": "none",
            },
        )
        outputs[method_name] = out_path

        if gaussian_sigma > 0:
            gauss_arr = _smooth_masked_chm(
                chm_arr,
                nodata=nodata,
                sigma=gaussian_sigma,
                hag_max=hag_max,
                chm_clip_min=chm_clip_min,
            )
            gauss_name = f"{method_name}_gauss"
            gauss_path = out_dir / f"{year_input.year}_{gauss_name}_chm.tif"
            _write_gtiff(
                gauss_path,
                gauss_arr,
                spec["transform"],
                out_crs,
                nodata,
                {
                    "SOURCE_LAZ": year_input.laz_path.name,
                    "YEAR": year_input.year,
                    "METHOD": gauss_name,
                    "HAG_MIN": hag_min,
                    "HAG_MAX": hag_max,
                    "CHM_CLIP_MIN": chm_clip_min,
                    "RETURN_MODE": return_mode,
                    "EXCLUDE_CLASSES": ",".join(str(v) for v in sorted(exclude_classes)),
                    "POST_FILTER": f"gaussian_sigma_{gaussian_sigma}",
                },
            )
            outputs[gauss_name] = gauss_path

    return outputs


def _tile_features(chm_path: Path, labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df = df[df["label"].isin(["cdw", "no_cdw"])].copy()
    if df.empty:
        raise RuntimeError(f"No binary labels in {labels_csv}")

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


def _eval_from_features(feats: pd.DataFrame) -> dict:
    cdw = feats[feats["label"] == "cdw"]
    no = feats[feats["label"] == "no_cdw"]
    y = feats["y"].to_numpy()

    return {
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


def _choose_best_method(aggregate_eval: Dict[str, dict]) -> str:
    def score(name: str) -> float:
        r = aggregate_eval[name]
        j1 = float(r.get("best_youden_tile_max", {}).get("youden_j", 0.0))
        auc = float(r.get("auc_tile_max", 0.0) or 0.0)
        cdw = float(r.get("cdw_detect_rate_15cm", 0.0) or 0.0)
        no_fp = float(r.get("no_false_high_rate_15cm", 1.0) or 1.0)
        # Reward separability and low false-high rate.
        return 0.4 * j1 + 0.25 * auc + 0.2 * cdw + 0.15 * (1.0 - no_fp)

    return max(aggregate_eval.keys(), key=score)


def _write_markdown_report(path: Path, payload: dict) -> None:
    lines: List[str] = []
    lines.append("# SOTA DTM/HAG Experiment for Tile 436646")
    lines.append("")
    lines.append("## Abstract")
    lines.append(
        "This experiment benchmarks a practical SOTA-inspired terrain workflow for CHM generation: "
        "SMRF ground classification, temporal super-cloud harmonization with SOR and 10th-percentile aggregation, "
        "TIN/Natural-Neighbor interpolation, and bilateral edge-preserving DTM filtering. "
        "CHM is generated at 0.2 m using last returns only, with class exclusions and strict HAG range 0-1.3 m."
    )
    lines.append("")

    lines.append("## Methods")
    lines.append("1. Ground reclassification with PDAL SMRF for each year.")
    lines.append("2. Vertical alignment to a reference year on stable low-slope surfaces.")
    lines.append("3. Stacked ground super-cloud, then Statistical Outlier Removal (SOR).")
    lines.append("4. Harmonized DTM anchors via per-cell temporal percentile elevation.")
    lines.append("5. Optional slope-adaptive DEM merge: p10 (gentle) + p25 (steep).")
    lines.append("6. Interpolation methods: IDW-k6, TIN linear, Natural Neighbor linear.")
    lines.append("7. Post-processing variants: bilateral DTM and optional Gaussian CHM smoothing.")
    lines.append("8. CHM from point-wise HAG with configurable output clipping (supports negative near-ground values).")
    lines.append("")

    lines.append("## Core Equations")
    lines.append("- HAG at point i: `HAG_i = z_i - z_ground(x_i, y_i)`")
    lines.append("- CHM pixel p: `CHM[p] = max(HAG_i)` for points in pixel p")
    lines.append("- HAG constraint: keep only `HAG_min <= HAG_i <= HAG_max`, then clip output to [CHM_clip_min, HAG_max]")
    lines.append("- Temporal harmonization: `z_cell = percentile_10({z_t})` over stacked years")
    lines.append("- TIN/Natural-Neighbor local surface: `z = Ax + By + C` inside each triangle")
    lines.append("")

    lines.append("## Input Data")
    lines.append(f"- Tile ID: {payload['tile_id']}")
    lines.append(f"- Years: {payload['years']}")
    lines.append(f"- LAZ directory: {payload['paths']['laz_dir']}")
    lines.append(f"- Labels directory: {payload['paths']['labels_dir']}")
    lines.append("")

    lines.append("## SMRF Diagnostics")
    for y in payload["years"]:
        s = payload["smrf_year_stats"][str(y)]
        lines.append(
            f"- {y}: class2 ratio={s['ground_ratio_pct']:.2f}% | DEM-valid={s['dem_valid_pct']:.2f}% | class2 points={s['points_ground_class2']}"
        )
    lines.append("")

    lines.append("## Vertical Alignment and SOR")
    ref = payload["vertical_alignment"]["reference_year"]
    lines.append(f"- Reference year: {ref}")
    for y, item in payload["vertical_alignment"]["pairwise"].items():
        lines.append(
            f"- {y}: overlap_cells={item['overlap_cells']} | shift_to_ref={item['vertical_shift_to_reference_m']:.4f} m"
        )
    sor = payload["sor_stats"]
    lines.append(
        f"- SOR: input={sor.get('input_points')} working={sor.get('sor_working_points')} output={sor.get('output_points')} kept={sor.get('kept_pct'):.2f}%"
    )
    sadapt = payload.get("slope_adapt_stats")
    if sadapt:
        slope_thr = sadapt.get("slope_threshold_deg")
        if slope_thr is None:
            lines.append(
                f"- Slope-adaptive DEM: steep_cells={sadapt.get('steep_cells')} ({sadapt.get('steep_pct'):.2f}%)"
            )
        else:
            lines.append(
                f"- Slope-adaptive DEM: slope_thr={float(slope_thr):.2f} deg | steep_cells={sadapt.get('steep_cells')} ({sadapt.get('steep_pct'):.2f}%)"
            )
    lines.append("")

    lines.append("## Aggregate Evaluation")
    for m, ev in payload["evaluation_aggregate"].items():
        b = ev.get("best_youden_tile_max", {})
        lines.append(
            f"- {m}: AUC(tile_max)={ev.get('auc_tile_max')}, J(tile_max)={b.get('youden_j')} @thr={b.get('threshold')}, "
            f"CDW>=15cm={ev.get('cdw_detect_rate_15cm')}, NoCDW>=15cm={ev.get('no_false_high_rate_15cm')}"
        )
    lines.append("")

    lines.append("## Best Method")
    lines.append(f"**{payload['best_method']}**")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("- Bilateral variants should reduce micro-jitter while preserving terrain edges in DTM.")
    lines.append("- Natural-Neighbor/TIN methods are expected to avoid IDW pockmark artifacts in sparse-ground zones.")
    lines.append("- Last-return and class exclusion reduce canopy/building/water contamination in CHM." )
    lines.append("")

    lines.append("## Improvement Ideas")
    lines.append("1. Add strict PTD implementation (e.g., lidR ptd) and compare against SMRF.")
    lines.append("2. Add explicit stable-surface masks (roads/rock polygons) for stronger vertical datum alignment.")
    lines.append("3. Evaluate quantiles 5th/10th/15th for harmonization sensitivity.")
    lines.append("4. Introduce Kriging interpolation as an additional comparator.")
    lines.append("5. Run uncertainty maps (ensemble spread among methods) to guide manual QA.")
    lines.append("")

    lines.append("## Output Artifacts")
    lines.append(f"- JSON report: {payload['paths']['report_json']}")
    lines.append(f"- CSV summary: {payload['paths']['report_csv']}")
    lines.append(f"- This Markdown report: {payload['paths']['report_md']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="SOTA DTM/HAG experiment (SMRF + temporal p10 + TIN/NN)")
    ap.add_argument("--tile-id", default="436646")
    ap.add_argument("--years", default="2018,2020,2022,2024")
    ap.add_argument("--laz-dir", type=Path, default=Path("data/lamapuit/laz"))
    ap.add_argument("--labels-dir", type=Path, default=Path("output/onboarding_labels_v2_drop13"))
    ap.add_argument("--baseline-chm-dir", type=Path, default=Path("data/lamapuit/chm_max_hag_13_drop"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/dtm_hag_436646_sota_2026-04-14/results"),
    )

    ap.add_argument("--epsg", type=int, default=3301)
    ap.add_argument("--chunk-size", type=int, default=800_000)
    ap.add_argument("--align-resolution", type=float, default=2.0)
    ap.add_argument("--hag-min", type=float, default=0.0)
    ap.add_argument("--chm-clip-min", type=float, default=0.0)
    ap.add_argument("--dem-resolution", type=float, default=1.0)
    ap.add_argument("--hag-max", type=float, default=1.3)

    ap.add_argument("--smrf-scalar", type=float, default=1.2)
    ap.add_argument("--smrf-slope", type=float, default=0.2)
    ap.add_argument("--smrf-threshold", type=float, default=0.45)
    ap.add_argument("--smrf-window", type=float, default=16.0)
    ap.add_argument("--reuse-smrf", action="store_true")

    ap.add_argument("--ground-max-points-per-year", type=int, default=900_000)
    ap.add_argument("--sor-max-points", type=int, default=1_500_000)
    ap.add_argument("--sor-neighbors", type=int, default=16)
    ap.add_argument("--sor-std-mult", type=float, default=2.5)
    ap.add_argument("--temporal-quantile", type=float, default=10.0)
    ap.add_argument("--temporal-quantile-steep", type=float, default=25.0)
    ap.add_argument("--stable-slope-quantile", type=float, default=35.0)
    ap.add_argument("--enable-slope-adaptive", action="store_true")
    ap.add_argument("--slope-adapt-quantile", type=float, default=80.0)

    ap.add_argument("--idw-k", type=int, default=6)
    ap.add_argument("--idw-power", type=float, default=2.0)
    ap.add_argument("--bilateral-d", type=int, default=9)
    ap.add_argument("--bilateral-sigma-color", type=float, default=0.35)
    ap.add_argument("--bilateral-sigma-space", type=float, default=3.0)
    ap.add_argument("--gaussian-sigma", type=float, default=0.0)

    ap.add_argument("--return-mode", default="last", choices=["all", "last", "last2"])
    ap.add_argument("--exclude-classes", default="6,9")
    ap.add_argument("--seed", type=int, default=20260414)

    args = ap.parse_args()
    t0 = time.time()

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    exclude_classes = {
        int(v.strip()) for v in str(args.exclude_classes).split(",") if v.strip()
    }

    out_dir = args.out_dir
    scratch_dir = out_dir / "scratch"
    dtm_dir = out_dir / "dtm"
    chm_dir = out_dir / "chm"
    eval_dir = out_dir / "eval"
    for d in (out_dir, scratch_dir, dtm_dir, chm_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    year_inputs: List[YearInput] = []
    for y in years:
        laz = args.laz_dir / f"{args.tile_id}_{y}_madal.laz"
        labels = args.labels_dir / f"{args.tile_id}_{y}_madal_chm_max_hag_20cm_labels.csv"
        baseline = args.baseline_chm_dir / f"{args.tile_id}_{y}_madal_chm_max_hag_20cm.tif"
        if not laz.exists():
            raise FileNotFoundError(f"Missing LAZ: {laz}")
        if not labels.exists():
            raise FileNotFoundError(f"Missing labels CSV: {labels}")
        if not baseline.exists():
            raise FileNotFoundError(f"Missing baseline CHM: {baseline}")
        year_inputs.append(YearInput(year=y, laz_path=laz, labels_csv=labels, baseline_chm=baseline))

    # 1) SMRF classification per year.
    smrf_laz_by_year: Dict[int, Path] = {}
    for yi in year_inputs:
        out_laz = scratch_dir / f"{yi.laz_path.stem}_smrf_ground.laz"
        if args.reuse_smrf and out_laz.exists():
            print(f"reuse_smrf={out_laz}", flush=True)
        else:
            _pdal_smrf_reclassify(
                yi.laz_path,
                out_laz,
                scratch_dir,
                epsg=args.epsg,
                scalar=args.smrf_scalar,
                slope=args.smrf_slope,
                threshold=args.smrf_threshold,
                window=args.smrf_window,
            )
        smrf_laz_by_year[yi.year] = out_laz

    # 2) Common bounds for alignment and DEM grids.
    minx, miny, maxx, maxy = _global_bounds([yi.laz_path for yi in year_inputs])

    # 3) Vertical alignment diagnostics from coarse per-year DEMs.
    align_ox, _ = _snap_origin(minx, miny, args.align_resolution)
    align_w = int(math.ceil((maxx - align_ox) / args.align_resolution))
    align_h = int(math.ceil((maxy - (math.floor(miny / args.align_resolution) * args.align_resolution)) / args.align_resolution))
    align_oy = math.floor(miny / args.align_resolution) * args.align_resolution
    align_h = int(math.ceil((maxy - align_oy) / args.align_resolution))

    align_dems: Dict[int, np.ndarray] = {}
    smrf_year_stats: Dict[str, dict] = {}
    for yi in year_inputs:
        dem, stats = _build_year_ground_dem(
            smrf_laz=smrf_laz_by_year[yi.year],
            ox=align_ox,
            maxy=maxy,
            dem_resolution=args.align_resolution,
            dem_h=align_h,
            dem_w=align_w,
            chunk_size=args.chunk_size,
        )
        align_dems[yi.year] = dem
        smrf_year_stats[str(yi.year)] = stats
        _write_gtiff(
            dtm_dir / f"{yi.year}_smrf_ground_min_dem_{int(args.align_resolution * 100)}cm.tif",
            np.where(np.isfinite(dem), dem, NODATA).astype(np.float32),
            from_origin(align_ox, maxy, args.align_resolution, args.align_resolution),
            f"EPSG:{args.epsg}",
            NODATA,
            {"METHOD": "smrf_ground_min", "YEAR": yi.year, "RES": args.align_resolution},
        )

    vertical_offsets, vertical_diag = _estimate_vertical_offsets(
        align_dems,
        reference_year=min(years),
        stable_slope_quantile=args.stable_slope_quantile,
    )

    # 4) Build temporally aligned super-cloud and apply SOR.
    stacks = []
    stack_stats = {}
    for yi in year_inputs:
        pts = _collect_smrf_ground_points(
            smrf_laz=smrf_laz_by_year[yi.year],
            z_offset=vertical_offsets.get(yi.year, 0.0),
            chunk_size=args.chunk_size,
            max_points=args.ground_max_points_per_year,
            seed=args.seed + yi.year,
        )
        stacks.append(pts)
        stack_stats[str(yi.year)] = {
            "stacked_points": int(pts.shape[0]),
            "vertical_offset_m": float(vertical_offsets.get(yi.year, 0.0)),
        }

    super_cloud = np.vstack(stacks) if stacks else np.empty((0, 3), dtype=np.float64)
    sor_cloud, sor_stats = _sor_filter_points(
        super_cloud,
        neighbors=args.sor_neighbors,
        std_multiplier=args.sor_std_mult,
        max_points=args.sor_max_points,
        seed=args.seed + 99,
    )

    # 5) Harmonized DTM by temporal percentile.
    dem_ox, dem_oy = _snap_origin(minx, miny, args.dem_resolution)
    dem_w = int(math.ceil((maxx - dem_ox) / args.dem_resolution))
    dem_h = int(math.ceil((maxy - dem_oy) / args.dem_resolution))

    harmonized_dem, harmonized_stats = _temporal_percentile_dem(
        sor_cloud,
        ox=dem_ox,
        maxy=maxy,
        dem_resolution=args.dem_resolution,
        dem_h=dem_h,
        dem_w=dem_w,
        percentile=args.temporal_quantile,
    )

    _write_gtiff(
        dtm_dir / f"harmonized_p{int(args.temporal_quantile)}_dem_{int(args.dem_resolution * 100)}cm.tif",
        np.where(np.isfinite(harmonized_dem), harmonized_dem, NODATA).astype(np.float32),
        from_origin(dem_ox, maxy, args.dem_resolution, args.dem_resolution),
        f"EPSG:{args.epsg}",
        NODATA,
        {
            "METHOD": f"temporal_p{int(args.temporal_quantile)}",
            "SOR": "true",
            "DEM_RES": args.dem_resolution,
        },
    )

    harmonized_steep_stats = None
    slope_adapt_stats = None
    harmonized_dem_sadapt = None
    if args.enable_slope_adaptive:
        harmonized_dem_steep, harmonized_steep_stats = _temporal_percentile_dem(
            sor_cloud,
            ox=dem_ox,
            maxy=maxy,
            dem_resolution=args.dem_resolution,
            dem_h=dem_h,
            dem_w=dem_w,
            percentile=args.temporal_quantile_steep,
        )
        _write_gtiff(
            dtm_dir / f"harmonized_p{int(args.temporal_quantile_steep)}_dem_{int(args.dem_resolution * 100)}cm.tif",
            np.where(np.isfinite(harmonized_dem_steep), harmonized_dem_steep, NODATA).astype(np.float32),
            from_origin(dem_ox, maxy, args.dem_resolution, args.dem_resolution),
            f"EPSG:{args.epsg}",
            NODATA,
            {
                "METHOD": f"temporal_p{int(args.temporal_quantile_steep)}",
                "SOR": "true",
                "DEM_RES": args.dem_resolution,
            },
        )

        harmonized_dem_sadapt, slope_adapt_stats = _slope_adaptive_merge_dem(
            dem_base=harmonized_dem,
            dem_steep=harmonized_dem_steep,
            dem_resolution=args.dem_resolution,
            slope_quantile=args.slope_adapt_quantile,
        )
        _write_gtiff(
            dtm_dir / f"harmonized_slope_adapt_dem_{int(args.dem_resolution * 100)}cm.tif",
            np.where(np.isfinite(harmonized_dem_sadapt), harmonized_dem_sadapt, NODATA).astype(np.float32),
            from_origin(dem_ox, maxy, args.dem_resolution, args.dem_resolution),
            f"EPSG:{args.epsg}",
            NODATA,
            {
                "METHOD": "slope_adaptive_p10_p25",
                "BASE_Q": args.temporal_quantile,
                "STEEP_Q": args.temporal_quantile_steep,
                "STEEP_SLOPE_Q": args.slope_adapt_quantile,
            },
        )

    # 6) Interpolate DTM methods.
    anchors_xy, anchors_z = _dem_to_anchors(harmonized_dem, dem_ox, maxy, args.dem_resolution)

    idw = IDWInterpolator(anchors_xy, anchors_z, k=args.idw_k, power=args.idw_power)
    tin = TINInterpolator(anchors_xy, anchors_z)

    method_dtms: Dict[str, np.ndarray] = {}
    method_anchor_counts: Dict[str, int] = {}

    method_dtms[f"idw_k{args.idw_k}"] = _interpolate_grid_model(
        idw,
        ox=dem_ox,
        maxy=maxy,
        dem_resolution=args.dem_resolution,
        dem_h=dem_h,
        dem_w=dem_w,
    )
    method_anchor_counts[f"idw_k{args.idw_k}"] = int(anchors_xy.shape[0])

    method_dtms["tin_linear"] = _interpolate_grid_model(
        tin,
        ox=dem_ox,
        maxy=maxy,
        dem_resolution=args.dem_resolution,
        dem_h=dem_h,
        dem_w=dem_w,
    )
    method_anchor_counts["tin_linear"] = int(anchors_xy.shape[0])

    method_dtms["natural_neighbor_linear"] = _interpolate_natural_neighbor_grid(
        anchors_xy,
        anchors_z,
        ox=dem_ox,
        maxy=maxy,
        dem_resolution=args.dem_resolution,
        dem_h=dem_h,
        dem_w=dem_w,
    )
    method_anchor_counts["natural_neighbor_linear"] = int(anchors_xy.shape[0])

    if args.enable_slope_adaptive and harmonized_dem_sadapt is not None:
        sadapt_xy, sadapt_z = _dem_to_anchors(harmonized_dem_sadapt, dem_ox, maxy, args.dem_resolution)
        sadapt_tin = TINInterpolator(sadapt_xy, sadapt_z)

        method_dtms["tin_linear_sadapt"] = _interpolate_grid_model(
            sadapt_tin,
            ox=dem_ox,
            maxy=maxy,
            dem_resolution=args.dem_resolution,
            dem_h=dem_h,
            dem_w=dem_w,
        )
        method_anchor_counts["tin_linear_sadapt"] = int(sadapt_xy.shape[0])

        method_dtms["natural_neighbor_linear_sadapt"] = _interpolate_natural_neighbor_grid(
            sadapt_xy,
            sadapt_z,
            ox=dem_ox,
            maxy=maxy,
            dem_resolution=args.dem_resolution,
            dem_h=dem_h,
            dem_w=dem_w,
        )
        method_anchor_counts["natural_neighbor_linear_sadapt"] = int(sadapt_xy.shape[0])

    method_dtms["tin_linear_bilateral"] = _bilateral_filter_dem(
        method_dtms["tin_linear"],
        diameter=args.bilateral_d,
        sigma_color=args.bilateral_sigma_color,
        sigma_space=args.bilateral_sigma_space,
    )
    method_anchor_counts["tin_linear_bilateral"] = method_anchor_counts["tin_linear"]
    method_dtms["natural_neighbor_bilateral"] = _bilateral_filter_dem(
        method_dtms["natural_neighbor_linear"],
        diameter=args.bilateral_d,
        sigma_color=args.bilateral_sigma_color,
        sigma_space=args.bilateral_sigma_space,
    )
    method_anchor_counts["natural_neighbor_bilateral"] = method_anchor_counts["natural_neighbor_linear"]

    if args.enable_slope_adaptive and "tin_linear_sadapt" in method_dtms:
        method_dtms["tin_linear_sadapt_bilateral"] = _bilateral_filter_dem(
            method_dtms["tin_linear_sadapt"],
            diameter=args.bilateral_d,
            sigma_color=args.bilateral_sigma_color,
            sigma_space=args.bilateral_sigma_space,
        )
        method_anchor_counts["tin_linear_sadapt_bilateral"] = method_anchor_counts["tin_linear_sadapt"]

        method_dtms["natural_neighbor_sadapt_bilateral"] = _bilateral_filter_dem(
            method_dtms["natural_neighbor_linear_sadapt"],
            diameter=args.bilateral_d,
            sigma_color=args.bilateral_sigma_color,
            sigma_space=args.bilateral_sigma_space,
        )
        method_anchor_counts["natural_neighbor_sadapt_bilateral"] = method_anchor_counts[
            "natural_neighbor_linear_sadapt"
        ]

    dtm_transform = from_origin(dem_ox, maxy, args.dem_resolution, args.dem_resolution)
    dtm_crs = f"EPSG:{args.epsg}"

    for name, dtm in method_dtms.items():
        _write_gtiff(
            dtm_dir / f"harmonized_{name}_dem_{int(args.dem_resolution * 100)}cm.tif",
            np.where(np.isfinite(dtm), dtm, NODATA).astype(np.float32),
            dtm_transform,
            dtm_crs,
            NODATA,
            {
                "METHOD": name,
                "DEM_RES": args.dem_resolution,
                "ANCHORS": int(method_anchor_counts.get(name, anchors_xy.shape[0])),
            },
        )

    # 7) Generate CHMs at baseline grid resolution (0.2 m expected).
    chm_paths: Dict[str, Dict[int, Path]] = {}
    for yi in year_inputs:
        year_out = chm_dir / str(yi.year)
        year_out.mkdir(parents=True, exist_ok=True)
        out_map = _generate_year_chms(
            year_input=yi,
            method_dtms=method_dtms,
            dtm_transform=dtm_transform,
            dtm_crs=dtm_crs,
            out_dir=year_out,
            hag_min=args.hag_min,
            hag_max=args.hag_max,
            chm_clip_min=args.chm_clip_min,
            chunk_size=args.chunk_size,
            return_mode=args.return_mode,
            exclude_classes=exclude_classes,
            gaussian_sigma=args.gaussian_sigma,
            seed=args.seed,
        )
        for method_name, p in out_map.items():
            chm_paths.setdefault(method_name, {})[yi.year] = p

    baseline_key = "baseline_idw3_drop13"
    chm_paths[baseline_key] = {yi.year: yi.baseline_chm for yi in year_inputs}

    # 8) Evaluate methods using labels.
    eval_per_year: Dict[str, Dict[str, dict]] = {}
    eval_agg: Dict[str, dict] = {}

    for method_name, by_year in chm_paths.items():
        feats_all = []
        per_year = {}
        for yi in year_inputs:
            chm = by_year[yi.year]
            feats = _tile_features(chm, yi.labels_csv)
            feats_all.append(feats)
            per_year[str(yi.year)] = _eval_from_features(feats)
        eval_per_year[method_name] = per_year
        eval_agg[method_name] = _eval_from_features(pd.concat(feats_all, ignore_index=True))

    best_method = _choose_best_method(eval_agg)

    rows = []
    for method_name, ev in eval_agg.items():
        b = ev.get("best_youden_tile_max", {})
        rows.append(
            {
                "method": method_name,
                "auc_tile_max": ev.get("auc_tile_max"),
                "youden_tile_max": b.get("youden_j"),
                "thr_tile_max": b.get("threshold"),
                "cdw_detect_rate_15cm": ev.get("cdw_detect_rate_15cm"),
                "no_false_high_rate_15cm": ev.get("no_false_high_rate_15cm"),
                "cohens_d_tile_max": ev.get("cohens_d_tile_max"),
                "n_tiles": ev.get("n_tiles"),
            }
        )

    summary_csv = eval_dir / "method_summary.csv"
    pd.DataFrame(rows).sort_values(["youden_tile_max", "auc_tile_max"], ascending=False).to_csv(summary_csv, index=False)

    params_json = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }

    payload = {
        "tile_id": args.tile_id,
        "years": years,
        "elapsed_seconds": round(time.time() - t0, 2),
        "parameters": params_json,
        "paths": {
            "laz_dir": str(args.laz_dir),
            "labels_dir": str(args.labels_dir),
            "baseline_chm_dir": str(args.baseline_chm_dir),
            "out_dir": str(out_dir),
            "report_json": str(out_dir / "experiment_report.json"),
            "report_md": str(out_dir / "experiment_report.md"),
            "report_csv": str(summary_csv),
        },
        "smrf_year_stats": smrf_year_stats,
        "vertical_alignment": vertical_diag,
        "vertical_offsets_m": {str(k): float(v) for k, v in vertical_offsets.items()},
        "stack_stats": stack_stats,
        "sor_stats": sor_stats,
        "harmonized_stats": harmonized_stats,
        "harmonized_steep_stats": harmonized_steep_stats,
        "slope_adapt_stats": slope_adapt_stats,
        "anchors": int(anchors_xy.shape[0]),
        "method_anchor_counts": method_anchor_counts,
        "evaluation_per_year": eval_per_year,
        "evaluation_aggregate": eval_agg,
        "best_method": best_method,
        "chm_paths": {
            method: {str(y): str(p) for y, p in by_year.items()} for method, by_year in chm_paths.items()
        },
    }

    report_json = out_dir / "experiment_report.json"
    report_md = out_dir / "experiment_report.md"
    _write_json(report_json, payload)
    _write_markdown_report(report_md, payload)

    print(f"best_method={best_method}")
    print(f"report_json={report_json}")
    print(f"report_md={report_md}")
    print(f"summary_csv={summary_csv}")
    print(f"elapsed_seconds={payload['elapsed_seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

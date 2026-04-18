#!/usr/bin/env python3
"""Research experiment: CSF + multi-year harmonized DTM + TIN/TPS HAG for tile 436646.

This experiment implements and evaluates a workflow motivated by recent terrain-interpolation
recommendations:
1) Ground classification with PDAL CSF.
2) Multi-year harmonization by selecting the lowest *valid* ground cell value across years.
3) DTM interpolation alternatives (IDW-k12, TIN-linear, TPS).
4) HAG/CHM creation with strict drop mode: keep only 0 <= HAG <= hag_max.
5) Optional Gaussian smoothing of final CHM raster.

Outputs:
- CSF-reclassified LAZ files (scratch)
- Harmonized DTM and interpolated DTM rasters
- CHM rasters for each year and method
- JSON + CSV + Markdown report with metrics and recommendation
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import warnings
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
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.stats import rankdata

try:
    from scipy.interpolate import RBFInterpolator

    HAVE_RBF_INTERPOLATOR = True
except Exception:
    HAVE_RBF_INTERPOLATOR = False
    from scipy.interpolate import Rbf


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


def _snap_origin(minx: float, miny: float, resolution: float) -> Tuple[float, float]:
    ox = math.floor(minx / resolution) * resolution
    oy = math.floor(miny / resolution) * resolution
    return ox, oy


def _write_pdal_pipeline(pipeline: list, path: Path) -> None:
    path.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")


def _pdal_csf_reclassify(input_laz: Path, output_laz: Path, work_dir: Path, epsg: int = 3301) -> None:
    """Reclassify ground with CSF and write compressed LAZ.

    Strategy: reset classifications to 1, then let CSF assign ground (class 2).
    """
    output_laz.parent.mkdir(parents=True, exist_ok=True)
    pipe_path = work_dir / f"pdal_csf_{input_laz.stem}.json"

    pipeline = [
        {
            "type": "readers.las",
            "filename": str(input_laz),
            "override_srs": f"EPSG:{epsg}",
        },
        {"type": "filters.assign", "value": "Classification = 1"},
        {
            "type": "filters.csf",
            "resolution": 0.5,
            "rigidness": 3,
            "step": 0.65,
            "threshold": 0.5,
            "hdiff": 0.3,
        },
        {"type": "writers.las", "filename": str(output_laz), "compression": "true"},
    ]

    _write_pdal_pipeline(pipeline, pipe_path)
    _run(["pdal", "pipeline", str(pipe_path)])


def _laz_bounds(laz_path: Path) -> Tuple[float, float, float, float]:
    with laspy.open(str(laz_path)) as fh:
        hdr = fh.header
        minx = float(hdr.mins[0])
        miny = float(hdr.mins[1])
        maxx = float(hdr.maxs[0])
        maxy = float(hdr.maxs[1])
    return minx, miny, maxx, maxy


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
    csf_laz: Path,
    ox: float,
    maxy: float,
    dem_resolution: float,
    dem_h: int,
    dem_w: int,
    chunk_size: int,
) -> Tuple[np.ndarray, dict]:
    """Build per-year DEM using per-cell minimum z among CSF-ground points."""
    dem = np.full((dem_h, dem_w), np.nan, dtype=np.float32)
    n_points = 0
    n_ground = 0

    with laspy.open(str(csf_laz)) as fh:
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


def _harmonize_year_dems(
    dems_by_year: Dict[int, np.ndarray],
    mad_factor: float = 2.5,
    floor_m: float = 0.15,
) -> Tuple[np.ndarray, dict]:
    """Harmonize multi-year ground as lowest valid per cell.

    For each cell with values z_t over years t:
    - m = median(z_t)
    - MAD = median(|z_t - m|) * 1.4826
    - valid if z_t >= m - max(floor_m, mad_factor * MAD)
    - harmonized z = min(valid z_t), fallback min(all z_t)
    """
    years = sorted(dems_by_year.keys())
    stack = np.stack([dems_by_year[y] for y in years], axis=0).astype(np.float32)

    # Silence known all-NaN slice warnings; these cells are handled explicitly below.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(stack, axis=0)
            abs_dev = np.abs(stack - median[np.newaxis, :, :])
            mad = 1.4826 * np.nanmedian(abs_dev, axis=0)

    lower = median - np.maximum(floor_m, mad_factor * mad)
    finite = np.isfinite(stack)
    valid = finite & (stack >= lower[np.newaxis, :, :])

    with np.errstate(invalid="ignore"):
        filtered = np.where(valid, stack, np.nan)

    # Warning-safe nanmin reducers: cells with no finite values remain NaN.
    filtered_safe = np.where(np.isfinite(filtered), filtered, np.inf)
    harm = np.min(filtered_safe, axis=0)
    harm = np.where(np.isinf(harm), np.nan, harm)

    stack_safe = np.where(np.isfinite(stack), stack, np.inf)
    fallback = np.min(stack_safe, axis=0)
    fallback = np.where(np.isinf(fallback), np.nan, fallback)

    harm = np.where(np.isfinite(harm), harm, fallback).astype(np.float32)

    stats = {
        "years": years,
        "mad_factor": float(mad_factor),
        "floor_m": float(floor_m),
        "valid_cells": int(np.isfinite(harm).sum()),
        "valid_pct": float(100.0 * np.isfinite(harm).sum() / harm.size) if harm.size else 0.0,
        "median_mad": float(np.nanmedian(mad)) if np.isfinite(mad).any() else None,
        "p95_mad": float(np.nanpercentile(mad, 95)) if np.isfinite(mad).any() else None,
    }
    return harm, stats


def _dem_to_anchors(dem: np.ndarray, ox: float, maxy: float, dem_resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    rr, cc = np.where(np.isfinite(dem))
    if rr.size == 0:
        raise RuntimeError("Harmonized DEM has no finite cells")
    x = ox + (cc.astype(np.float64) + 0.5) * dem_resolution
    y = maxy - (rr.astype(np.float64) + 0.5) * dem_resolution
    z = dem[rr, cc].astype(np.float64)
    xy = np.column_stack((x, y))
    return xy, z


def _sample_points(xy: np.ndarray, z: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if xy.shape[0] <= max_points:
        return xy, z
    rng = np.random.default_rng(seed)
    idx = rng.choice(xy.shape[0], size=max_points, replace=False)
    return xy[idx], z[idx]


class IDWInterpolator:
    def __init__(self, xy: np.ndarray, z: np.ndarray, k: int = 12, power: float = 2.0, eps: float = 1e-8):
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


class TPSInterpolator:
    def __init__(self, xy: np.ndarray, z: np.ndarray, neighbors: int = 64, smoothing: float = 0.05):
        self.xy = xy.astype(np.float64)
        self.z = z.astype(np.float64)
        self.neighbors = int(neighbors)
        self.smoothing = float(smoothing)

        if HAVE_RBF_INTERPOLATOR:
            self.model = RBFInterpolator(
                self.xy,
                self.z,
                kernel="thin_plate_spline",
                neighbors=self.neighbors,
                smoothing=self.smoothing,
            )
            self.mode = "rbf_interpolator"
        else:
            # Fallback for older scipy versions; only safe for modest sample sizes.
            self.model = Rbf(self.xy[:, 0], self.xy[:, 1], self.z, function="thin_plate", smooth=self.smoothing)
            self.mode = "rbf_legacy"

    def __call__(self, xyq: np.ndarray) -> np.ndarray:
        if self.mode == "rbf_interpolator":
            return np.asarray(self.model(xyq), dtype=np.float64).reshape(-1)
        return np.asarray(self.model(xyq[:, 0], xyq[:, 1]), dtype=np.float64).reshape(-1)


def _interpolate_dtm_grid(
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


def _smooth_masked(
    arr: np.ndarray,
    nodata: float,
    sigma: float,
    min_clip: float | None = None,
    max_clip: float | None = None,
) -> np.ndarray:
    if sigma <= 0:
        out = arr.copy()
        if min_clip is not None or max_clip is not None:
            valid = out != nodata
            lo = min_clip if min_clip is not None else -np.inf
            hi = max_clip if max_clip is not None else np.inf
            out[valid] = np.clip(out[valid], lo, hi)
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

    if min_clip is not None or max_clip is not None:
        valid_out = out != nodata
        lo = min_clip if min_clip is not None else -np.inf
        hi = max_clip if max_clip is not None else np.inf
        out[valid_out] = np.clip(out[valid_out], lo, hi)

    return out


def _write_gtiff(
    path: Path,
    arr: np.ndarray,
    transform,
    crs,
    nodata: float,
    tags: dict,
) -> None:
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
    hag_max: float,
    chunk_size: int,
    chm_clip_min: float,
    hag_upper_mode: str,
    gaussian_sigma: float,
    point_sample_rate: float,
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

            if point_sample_rate < 1.0:
                keep_pts = rng.random(x.size) < point_sample_rate
                if not np.any(keep_pts):
                    continue
                x = x[keep_pts]
                y = y[keep_pts]
                z = z[keep_pts]

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
                # Keep points according to selected upper-bound policy.
                if hag_upper_mode == "clip":
                    keep = hag >= chm_clip_min
                else:
                    keep = (hag >= chm_clip_min) & (hag <= hag_max)
                if not np.any(keep):
                    continue

                valid_flat = flat[valid_dtm][keep]
                valid_hag = np.clip(hag[keep], chm_clip_min, hag_max).astype(np.float32)

                # Use np.maximum.at to aggregate max point HAG values in each pixel
                np.maximum.at(chm_arr.ravel(), valid_flat, valid_hag)

    outputs: Dict[str, Path] = {}
    for method_name, chm_arr in method_chms.items():
        raw_path = out_dir / f"{year_input.year}_{method_name}_raw_chm.tif"
        sm_path = out_dir / f"{year_input.year}_{method_name}_gauss_chm.tif"

        _write_gtiff(
            raw_path,
            chm_arr,
            spec["transform"],
            out_crs,
            nodata,
            {
                "SOURCE_LAZ": year_input.laz_path.name,
                "YEAR": year_input.year,
                "METHOD": method_name,
                "HAG_MAX": hag_max,
                "CHM_CLIP_MIN": chm_clip_min,
                "HAG_UPPER_MODE": hag_upper_mode,
                "FILTER_MODE": "drop_per_point",
                "POST_FILTER": "none",
                "POINT_SAMPLE_RATE": point_sample_rate,
            },
        )

        smoothed = _smooth_masked(
            chm_arr,
            nodata=nodata,
            sigma=gaussian_sigma,
            min_clip=chm_clip_min,
            max_clip=hag_max,
        )
        _write_gtiff(
            sm_path,
            smoothed,
            spec["transform"],
            out_crs,
            nodata,
            {
                "SOURCE_LAZ": year_input.laz_path.name,
                "YEAR": year_input.year,
                "METHOD": method_name,
                "HAG_MAX": hag_max,
                "CHM_CLIP_MIN": chm_clip_min,
                "HAG_UPPER_MODE": hag_upper_mode,
                "FILTER_MODE": "drop_per_point",
                "POST_FILTER": f"gaussian_sigma_{gaussian_sigma}",
                "POINT_SAMPLE_RATE": point_sample_rate,
            },
        )

        outputs[f"{method_name}_raw"] = raw_path
        outputs[f"{method_name}_gauss"] = sm_path

    return outputs


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
        j2 = float(r.get("best_youden_frac_above_15cm", {}).get("youden_j", 0.0))
        cdw = float(r.get("cdw_detect_rate_15cm", 0.0) or 0.0)
        no_fp = float(r.get("no_false_high_rate_15cm", 1.0) or 1.0)
        auc = float(r.get("auc_tile_max", 0.0) or 0.0)
        return 0.35 * j1 + 0.2 * j2 + 0.2 * cdw + 0.15 * (1.0 - no_fp) + 0.1 * auc

    return max(aggregate_eval.keys(), key=score)


def _sample_csf_ground_points(
    csf_laz: Path,
    max_points: int,
    chunk_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks: List[np.ndarray] = []
    per_chunk_cap = max(5000, max_points // 4)

    with laspy.open(str(csf_laz)) as fh:
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
            z = np.asarray(pts.z[keep], dtype=np.float64)
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


def _ground_fit_metrics(samples_xyz: np.ndarray, model) -> dict:
    if samples_xyz.shape[0] == 0:
        return {}
    xy = samples_xyz[:, :2]
    z = samples_xyz[:, 2]
    g = model(xy)
    resid = z - g
    absr = np.abs(resid)
    return {
        "n_samples": int(resid.size),
        "mean_residual": float(np.mean(resid)),
        "median_residual": float(np.median(resid)),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(absr)),
        "p95_abs_residual": float(np.percentile(absr, 95)),
    }


def _write_markdown_report(report_path: Path, payload: dict) -> None:
    lines: List[str] = []
    lines.append("# 436646 DTM/HAG Research Experiment")
    lines.append("")
    lines.append("## Goal")
    lines.append(
        "Evaluate a research-oriented workflow: CSF classification + multi-year harmonized ground + "
        "TIN/TPS interpolation + Gaussian-refined CHM/HAG."
    )
    lines.append("")

    lines.append("## Tested Workflow")
    lines.append("1. CSF reclassification for each year (ground class=2).")
    lines.append("2. Build yearly ground DEM from per-cell minimum class-2 z.")
    lines.append("3. Harmonize years by lowest valid cell value (MAD-guarded).")
    lines.append("4. Interpolate harmonized DTM with IDW-k12, TIN-linear, TPS.")
    lines.append("5. Compute HAG and CHM (drop mode: keep only 0 <= HAG <= 1.3 m).")
    lines.append("6. Apply Gaussian smoothing to final CHM and compare raw vs smoothed.")
    lines.append("")

    lines.append("## Core Formulas")
    lines.append("- HAG per point: `HAG = z - z_ground(x, y)`")
    lines.append("- Drop filter: keep points where `0 <= HAG <= HAG_max`.")
    lines.append("- CHM per pixel p: `CHM[p] = max(HAG_i)` over points in p.")
    lines.append("- IDW-k12: `z_ground = sum(w_i z_i) / sum(w_i)`, `w_i = 1/(d_i+eps)^p`, with `k=12`, `p=2`.")
    lines.append("- Harmonized ground validity: for yearly values z_t in one cell, valid if `z_t >= median(z_t) - max(floor, mad_factor*MAD)`.")
    lines.append("- Harmonized value: minimum of valid yearly values, fallback minimum of all yearly values.")
    lines.append("")

    lines.append("## Input")
    lines.append(f"- Tile: {payload['tile_id']}")
    lines.append(f"- Years: {payload['years']}")
    lines.append(f"- LAZ directory: {payload['paths']['laz_dir']}")
    lines.append(f"- Labels directory: {payload['paths']['labels_dir']}")
    lines.append("")

    lines.append("## CSF and Harmonization Diagnostics")
    for y in payload["years"]:
        yk = str(y)
        s = payload["csf_year_stats"][yk]
        lines.append(
            f"- {y}: class2 ratio={s['ground_ratio_pct']:.2f}% | DEM valid={s['dem_valid_pct']:.2f}% | "
            f"class2 points={s['points_ground_class2']}"
        )
    hs = payload["harmonization_stats"]
    lines.append(
        f"- Harmonized DEM valid={hs['valid_pct']:.2f}% | median MAD={hs['median_mad']} | p95 MAD={hs['p95_mad']}"
    )
    lines.append("")

    lines.append("## DTM Ground-Fit (CSF points)")
    for m, s in payload["dtm_ground_fit"].items():
        lines.append(
            f"- {m}: RMSE={s.get('rmse')} | MAE={s.get('mae')} | p95|res|={s.get('p95_abs_residual')} | n={s.get('n_samples')}"
        )
    lines.append("")

    lines.append("## Aggregate CHM/Label Evaluation")
    for m, ev in payload["evaluation_aggregate"].items():
        best1 = ev.get("best_youden_tile_max", {})
        best2 = ev.get("best_youden_frac_above_15cm", {})
        lines.append(
            f"- {m}: AUC(tile_max)={ev.get('auc_tile_max')}, "
            f"J(tile_max)={best1.get('youden_j')} @thr={best1.get('threshold')}, "
            f"J(frac>=15cm)={best2.get('youden_j')} @thr={best2.get('threshold')}, "
            f"CDW>=15cm={ev.get('cdw_detect_rate_15cm')}, NoCDW>=15cm={ev.get('no_false_high_rate_15cm')}"
        )
    lines.append("")

    lines.append("## Recommendation")
    lines.append(f"**Best method from aggregate score: {payload['best_method']}**")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- If TIN/TPS + Gaussian improves Youden/AUC while keeping NoCDW false-high low, it should replace baseline IDW3.")
    lines.append("- If smoothed versions consistently beat raw variants, keep Gaussian refinement in production.")
    lines.append("- If one method has better DTM residuals but worse CHM separability, prioritize CHM-label separability for CDW detection tasks.")
    lines.append("")
    lines.append("## Output Artifacts")
    lines.append(f"- JSON report: {payload['paths']['report_json']}")
    lines.append(f"- CSV summary: {payload['paths']['report_csv']}")
    lines.append(f"- This Markdown report: {payload['paths']['report_md']}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="CSF + harmonized DTM + TIN/TPS HAG experiment for 436646")
    ap.add_argument("--tile-id", default="436646")
    ap.add_argument("--years", default="2018,2020,2022,2024")
    ap.add_argument("--laz-dir", type=Path, default=Path("data/lamapuit/laz"))
    ap.add_argument("--labels-dir", type=Path, default=Path("output/onboarding_labels_v2_drop13"))
    ap.add_argument("--baseline-chm-dir", type=Path, default=Path("data/lamapuit/chm_max_hag_13_drop"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/dtm_hag_436646_research/results"))
    ap.add_argument("--dem-resolution", type=float, default=1.0)
    ap.add_argument("--hag-max", type=float, default=1.3)
    ap.add_argument(
        "--hag-upper-mode",
        choices=["drop", "clip"],
        default="drop",
        help="How to apply upper HAG bound: drop values above hag-max or clip them to hag-max.",
    )
    ap.add_argument("--chunk-size", type=int, default=800_000)
    ap.add_argument("--gaussian-sigma", type=float, default=1.0)
    ap.add_argument(
        "--chm-clip-min",
        type=float,
        default=0.0,
        help="Lower clamp for CHM HAG values before writing (can be negative).",
    )
    ap.add_argument("--seed", type=int, default=20260411)
    ap.add_argument("--mad-factor", type=float, default=2.5)
    ap.add_argument("--mad-floor", type=float, default=0.15)
    ap.add_argument("--idw-k", type=int, default=12)
    ap.add_argument("--idw-power", type=float, default=2.0)
    ap.add_argument("--idw-max-samples", type=int, default=220000)
    ap.add_argument("--tin-max-samples", type=int, default=120000)
    ap.add_argument("--tps-max-samples", type=int, default=35000)
    ap.add_argument("--tps-neighbors", type=int, default=64)
    ap.add_argument("--tps-smoothing", type=float, default=0.05)
    ap.add_argument("--epsg", type=int, default=3301)
    ap.add_argument(
        "--point-sample-rate",
        type=float,
        default=1.0,
        help="Fraction of points kept during HAG rasterization (0,1]. Use <1.0 for faster pilot runs.",
    )
    ap.add_argument(
        "--reuse-csf",
        action="store_true",
        help="Reuse existing CSF LAZ files from out_dir/scratch when present.",
    )
    ap.add_argument(
        "--only-raw-gauss",
        action="store_true",
        help="Produce only harmonized DEM raw+gauss CHMs and skip IDW/TIN/TPS outputs.",
    )
    args = ap.parse_args()

    t0 = time.time()
    if not (0.0 < args.point_sample_rate <= 1.0):
        raise ValueError("--point-sample-rate must be in (0,1].")
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

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

    # 1) CSF per year.
    csf_laz_by_year: Dict[int, Path] = {}
    for yi in year_inputs:
        out_laz = scratch_dir / f"{yi.laz_path.stem}_csf_ground.laz"
        if args.reuse_csf and out_laz.exists():
            print(f"reuse_csf={out_laz}", flush=True)
        else:
            _pdal_csf_reclassify(yi.laz_path, out_laz, scratch_dir, epsg=args.epsg)
        csf_laz_by_year[yi.year] = out_laz

    # 2) Build per-year class2 DEMs on common grid.
    minx, miny, maxx, maxy = _global_bounds([yi.laz_path for yi in year_inputs])
    ox, oy = _snap_origin(minx, miny, args.dem_resolution)
    dem_w = int(math.ceil((maxx - ox) / args.dem_resolution))
    dem_h = int(math.ceil((maxy - oy) / args.dem_resolution))

    dems_by_year: Dict[int, np.ndarray] = {}
    csf_year_stats: Dict[str, dict] = {}
    for yi in year_inputs:
        dem, stats = _build_year_ground_dem(
            csf_laz=csf_laz_by_year[yi.year],
            ox=ox,
            maxy=maxy,
            dem_resolution=args.dem_resolution,
            dem_h=dem_h,
            dem_w=dem_w,
            chunk_size=args.chunk_size,
        )
        dems_by_year[yi.year] = dem
        csf_year_stats[str(yi.year)] = stats
        _write_gtiff(
            dtm_dir / f"{yi.year}_csf_ground_min_dem_{int(args.dem_resolution*100)}cm.tif",
            np.where(np.isfinite(dem), dem, NODATA).astype(np.float32),
            from_origin(ox, maxy, args.dem_resolution, args.dem_resolution),
            f"EPSG:{args.epsg}",
            NODATA,
            {
                "METHOD": "csf_year_min_dem",
                "YEAR": yi.year,
                "DEM_RES": args.dem_resolution,
            },
        )

    # 3) Harmonize years: lowest valid per cell.
    harmonized_dem, harmonization_stats = _harmonize_year_dems(
        dems_by_year,
        mad_factor=args.mad_factor,
        floor_m=args.mad_floor,
    )
    _write_gtiff(
        dtm_dir / f"harmonized_ground_dem_{int(args.dem_resolution*100)}cm.tif",
        np.where(np.isfinite(harmonized_dem), harmonized_dem, NODATA).astype(np.float32),
        from_origin(ox, maxy, args.dem_resolution, args.dem_resolution),
        f"EPSG:{args.epsg}",
        NODATA,
        {
            "METHOD": "harmonized_lowest_valid_ground",
            "DEM_RES": args.dem_resolution,
            "MAD_FACTOR": args.mad_factor,
            "MAD_FLOOR": args.mad_floor,
            "YEARS": ",".join(str(y) for y in years),
        },
    )

    anchors_xy, anchors_z = _dem_to_anchors(harmonized_dem, ox=ox, maxy=maxy, dem_resolution=args.dem_resolution)

    dtm_stats: Dict[str, dict] = {}
    method_dtms: Dict[str, np.ndarray] = {"harmonized_dem": harmonized_dem}
    models: Dict[str, object] = {}
    tps_xy = np.empty((0, 2), dtype=np.float64)
    dtm_transform = from_origin(ox, maxy, args.dem_resolution, args.dem_resolution)
    dtm_crs = f"EPSG:{args.epsg}"
    p_raw = dtm_dir / f"harmonized_ground_dem_{int(args.dem_resolution*100)}cm.tif"
    try:
        dtm_stats["harmonized_dem"] = _raster_stats(p_raw)
    except Exception:
        dtm_stats["harmonized_dem"] = {}

    # 4) Build interpolators unless raw+gauss-only mode is requested.
    if args.only_raw_gauss:
        print("only_raw_gauss enabled - skipping IDW/TIN/TPS interpolation stage", flush=True)
    else:
        # Use all points for IDW/TIN for high fidelity, downsample only for TPS.
        tps_xy, tps_z = _sample_points(anchors_xy, anchors_z, args.tps_max_samples, args.seed + 3)

        models = {
            "idw_k3": IDWInterpolator(anchors_xy, anchors_z, k=3, power=args.idw_power),
            "idw_k6": IDWInterpolator(anchors_xy, anchors_z, k=6, power=args.idw_power),
            "tin_linear": TINInterpolator(anchors_xy, anchors_z),
            "tps": TPSInterpolator(tps_xy, tps_z, neighbors=args.tps_neighbors, smoothing=args.tps_smoothing),
        }

        # 5) Write interpolated DTM rasters for diagnostics.
        for name, model in models.items():
            dtm = _interpolate_dtm_grid(
                model,
                ox=ox,
                maxy=maxy,
                dem_resolution=args.dem_resolution,
                dem_h=dem_h,
                dem_w=dem_w,
                batch_size=200_000,
            )
            method_dtms[name] = dtm
            p = dtm_dir / f"harmonized_{name}_dem_{int(args.dem_resolution*100)}cm.tif"
            _write_gtiff(
                p,
                dtm,
                dtm_transform,
                dtm_crs,
                NODATA,
                {
                    "METHOD": f"harmonized_{name}",
                    "DEM_RES": args.dem_resolution,
                },
            )
            dtm_stats[name] = _raster_stats(p)

    # 6) Generate CHM rasters per year for each method (raw + gauss).
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
            hag_max=args.hag_max,
            chunk_size=args.chunk_size,
            chm_clip_min=args.chm_clip_min,
            hag_upper_mode=args.hag_upper_mode,
            gaussian_sigma=args.gaussian_sigma,
            point_sample_rate=args.point_sample_rate,
            seed=args.seed,
        )
        for mname, p in out_map.items():
            chm_paths.setdefault(mname, {})[yi.year] = p

    # Add baseline as a competing method.
    baseline_key = "baseline_idw3_drop13"
    chm_paths[baseline_key] = {yi.year: yi.baseline_chm for yi in year_inputs}

    # 7) DTM fit metrics using sampled CSF ground points.
    all_samples = []
    for yi in year_inputs:
        s = _sample_csf_ground_points(
            csf_laz_by_year[yi.year],
            max_points=60_000,
            chunk_size=args.chunk_size,
            seed=args.seed + yi.year,
        )
        if s.size:
            all_samples.append(s)
    sample_xyz = np.vstack(all_samples) if all_samples else np.empty((0, 3), dtype=np.float64)

    dtm_ground_fit: Dict[str, dict] = {}
    for name, model in models.items():
        dtm_ground_fit[name] = _ground_fit_metrics(sample_xyz, model)

    # 8) Evaluate each method with label CSVs (per-year + aggregate).
    eval_per_year: Dict[str, Dict[str, dict]] = {}
    eval_agg: Dict[str, dict] = {}
    raster_stats: Dict[str, Dict[str, dict]] = {}

    for method_name, by_year in chm_paths.items():
        feats_all = []
        per_year = {}
        ras_by_year = {}

        for yi in year_inputs:
            chm = by_year[yi.year]
            feats = _tile_features(chm, yi.labels_csv)
            feats_all.append(feats)
            per_year[str(yi.year)] = _eval_from_features(feats)
            ras_by_year[str(yi.year)] = _raster_stats(chm)

        eval_per_year[method_name] = per_year
        raster_stats[method_name] = ras_by_year
        eval_agg[method_name] = _eval_from_features(pd.concat(feats_all, ignore_index=True))

    best_method = _choose_best_method(eval_agg)

    # 9) Flat CSV summary for quick comparison.
    rows = []
    for method_name, ev in eval_agg.items():
        b1 = ev.get("best_youden_tile_max", {})
        b2 = ev.get("best_youden_frac_above_15cm", {})
        rows.append(
            {
                "method": method_name,
                "auc_tile_max": ev.get("auc_tile_max"),
                "auc_frac_above_15cm": ev.get("auc_frac_above_15cm"),
                "j_tile_max": b1.get("youden_j"),
                "thr_tile_max": b1.get("threshold"),
                "j_frac15": b2.get("youden_j"),
                "thr_frac15": b2.get("threshold"),
                "cdw_detect_rate_15cm": ev.get("cdw_detect_rate_15cm"),
                "no_false_high_rate_15cm": ev.get("no_false_high_rate_15cm"),
                "cohens_d_tile_max": ev.get("cohens_d_tile_max"),
            }
        )

    summary_csv = eval_dir / "method_summary.csv"
    pd.DataFrame(rows).sort_values("j_tile_max", ascending=False).to_csv(summary_csv, index=False)

    elapsed = time.time() - t0

    report_json = out_dir / "experiment_report.json"
    report_md = out_dir / "experiment_report.md"

    payload = {
        "tile_id": args.tile_id,
        "years": years,
        "parameters": {
            "dem_resolution": args.dem_resolution,
            "hag_max": args.hag_max,
            "gaussian_sigma": args.gaussian_sigma,
            "chunk_size": args.chunk_size,
            "idw_k": args.idw_k,
            "idw_power": args.idw_power,
            "mad_factor": args.mad_factor,
            "mad_floor": args.mad_floor,
            "tps_neighbors": args.tps_neighbors,
            "tps_smoothing": args.tps_smoothing,
            "anchors": int(anchors_xy.shape[0]),
            "tps_samples": int(tps_xy.shape[0]),
            "tps_impl": "RBFInterpolator" if HAVE_RBF_INTERPOLATOR else "Rbf_fallback",
            "point_sample_rate": args.point_sample_rate,
        },
        "paths": {
            "laz_dir": str(args.laz_dir),
            "labels_dir": str(args.labels_dir),
            "baseline_chm_dir": str(args.baseline_chm_dir),
            "out_dir": str(out_dir),
            "report_json": str(report_json),
            "report_csv": str(summary_csv),
            "report_md": str(report_md),
        },
        "csf_year_stats": csf_year_stats,
        "harmonization_stats": harmonization_stats,
        "dtm_stats": dtm_stats,
        "dtm_ground_fit": dtm_ground_fit,
        "raster_stats": raster_stats,
        "evaluation_per_year": eval_per_year,
        "evaluation_aggregate": eval_agg,
        "best_method": best_method,
        "elapsed_seconds": elapsed,
        "chm_paths": {
            method: {str(y): str(p) for y, p in by_year.items()} for method, by_year in chm_paths.items()
        },
    }

    _write_json(report_json, payload)
    _write_markdown_report(report_md, payload)

    print(f"best_method={best_method}")
    print(f"report_json={report_json}")
    print(f"report_md={report_md}")
    print(f"summary_csv={summary_csv}")
    print(f"elapsed_seconds={elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

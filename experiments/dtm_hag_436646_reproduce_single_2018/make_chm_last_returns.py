#!/usr/bin/env python3
"""Build CHM from a harmonized DEM using only last returns from a LAZ.

Creates raw and Gaussian-smoothed CHM rasters using the same grid as a
baseline CHM. This mirrors `run_experiment_copied.py`'s CHM aggregation but
filters points to last returns only.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
import laspy
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter

NODATA = -9999.0


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


def _smooth_masked(arr: np.ndarray, nodata: float, sigma: float, min_clip: float | None = None, max_clip: float | None = None) -> np.ndarray:
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Make CHM from harmonized DEM using last returns only")
    ap.add_argument("--tile-id", default="436646")
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--laz-dir", type=Path, default=Path("data/lamapuit/laz"))
    ap.add_argument("--baseline-chm", type=Path, required=False)
    ap.add_argument("--harmonized-dem", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/dtm_hag_436646_reproduce_single_2018/lastreturns_chm"))
    ap.add_argument("--chunk-size", type=int, default=800_000)
    ap.add_argument("--gaussian-sigma", type=float, default=0.3)
    ap.add_argument("--chm-clip-min", type=float, default=0.0)
    ap.add_argument("--hag-max", type=float, default=1.3)
    ap.add_argument("--hag-upper-mode", choices=["drop", "clip"], default="drop")
    ap.add_argument("--point-sample-rate", type=float, default=1.0)
    args = ap.parse_args()

    tile = args.tile_id
    y = args.year

    laz = args.laz_dir / f"{tile}_{y}_madal.laz"
    if not laz.exists():
        raise FileNotFoundError(f"Missing LAZ: {laz}")

    # baseline CHM for grid spec — if not provided, use an existing research baseline
    if args.baseline_chm:
        baseline_chm = args.baseline_chm
    else:
        baseline_chm = Path("data/lamapuit/chm_max_hag_13_drop") / f"{tile}_{y}_madal_chm_max_hag_20cm.tif"
    if not baseline_chm.exists():
        raise FileNotFoundError(f"Missing baseline CHM for grid spec: {baseline_chm}")

    spec = _read_baseline_grid_spec(baseline_chm)
    width = spec["width"]
    height = spec["height"]
    ox = spec["ox"]
    maxy = spec["maxy"]
    res = spec["res"]
    nodata = spec["nodata"]
    # read harmonized DEM and resample to CHM grid. Capture its CRS so we can
    # choose a sensible destination CRS if the baseline CHM lacks one.
    src_crs = None
    with rasterio.open(args.harmonized_dem) as src:
        src_arr = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs

    # Ensure we have a valid destination CRS for reproject; prefer baseline CHM CRS,
    # otherwise fall back to the harmonized DEM CRS or EPSG:3301.
    out_crs = spec["crs"] if spec["crs"] is not None else (src_crs if src_crs is not None else f"EPSG:3301")

    dtm_resampled = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dtm_resampled,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=spec["transform"],
        dst_crs=out_crs,
        resampling=Resampling.bilinear,
    )

    chm_arr = np.full((height, width), nodata, dtype=np.float32)

    rng = np.random.default_rng(20260411 + y)

    with laspy.open(str(laz)) as fh:
        for pts in fh.chunk_iterator(args.chunk_size):
            x = np.asarray(pts.x, dtype=np.float64)
            ycoords = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float32)
            if x.size == 0:
                continue

            # filter to last returns only
            try:
                rn = np.asarray(pts.return_number)
                nr = np.asarray(pts.number_of_returns)
                last_mask = rn == nr
            except Exception:
                last_mask = np.ones_like(x, dtype=bool)

            if args.point_sample_rate < 1.0:
                samp = rng.random(x.size) < args.point_sample_rate
                keep_pts = last_mask & samp
            else:
                keep_pts = last_mask

            if not np.any(keep_pts):
                continue

            x = x[keep_pts]
            ycoords = ycoords[keep_pts]
            z = z[keep_pts]

            col = ((x - ox) / res).astype(np.int32)
            row = ((maxy - ycoords) / res).astype(np.int32)
            valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
            if not np.any(valid):
                continue

            row = row[valid]
            col = col[valid]
            z = z[valid]

            flat = row * width + col
            dtm_z = dtm_resampled[row, col]
            valid_dtm = np.isfinite(dtm_z)
            if not np.any(valid_dtm):
                continue

            flat = flat[valid_dtm]
            hag = z[valid_dtm] - dtm_z[valid_dtm]

            if args.hag_upper_mode == "clip":
                keep = hag >= args.chm_clip_min
            else:
                keep = (hag >= args.chm_clip_min) & (hag <= args.hag_max)
            if not np.any(keep):
                continue

            valid_flat = flat[keep]
            valid_hag = np.clip(hag[keep], args.chm_clip_min, args.hag_max).astype(np.float32)

            np.maximum.at(chm_arr.ravel(), valid_flat, valid_hag)

    # write outputs
    year_out = args.out_dir / "chm" / str(args.year)
    raw_path = year_out / f"{args.year}_harmonized_dem_last_raw_chm.tif"
    gauss_path = year_out / f"{args.year}_harmonized_dem_last_gauss_chm.tif"

    _write_gtiff(
        raw_path,
        chm_arr,
        spec["transform"],
        out_crs,
        nodata,
        {
            "SOURCE_LAZ": laz.name,
            "YEAR": args.year,
            "METHOD": "harmonized_dem_last",
            "HAG_MAX": args.hag_max,
            "CHM_CLIP_MIN": args.chm_clip_min,
            "HAG_UPPER_MODE": args.hag_upper_mode,
            "FILTER_MODE": "last_return_only",
            "POST_FILTER": "none",
            "POINT_SAMPLE_RATE": args.point_sample_rate,
        },
    )

    smoothed = _smooth_masked(chm_arr, nodata=nodata, sigma=args.gaussian_sigma, min_clip=args.chm_clip_min, max_clip=args.hag_max)
    _write_gtiff(
        gauss_path,
        smoothed,
        spec["transform"],
        out_crs,
        nodata,
        {
            "SOURCE_LAZ": laz.name,
            "YEAR": args.year,
            "METHOD": "harmonized_dem_last",
            "HAG_MAX": args.hag_max,
            "CHM_CLIP_MIN": args.chm_clip_min,
            "HAG_UPPER_MODE": args.hag_upper_mode,
            "FILTER_MODE": "last_return_only",
            "POST_FILTER": f"gaussian_sigma_{args.gaussian_sigma}",
            "POINT_SAMPLE_RATE": args.point_sample_rate,
        },
    )

    print("wrote:", raw_path)
    print("wrote:", gauss_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

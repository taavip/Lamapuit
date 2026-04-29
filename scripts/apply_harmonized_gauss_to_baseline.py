#!/usr/bin/env python3
"""Resample baseline CHMs to a harmonized grid (default 0.8 m) and apply
masked Gaussian smoothing identical to the harmonized pipeline.

Saves results under `output/chm_variant_selection/chm_raw` and
`output/chm_variant_selection/chm_gauss`.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter


DEFAULT_NODATA = -9999.0


def _write_gtiff(path: Path, arr: np.ndarray, transform, crs, nodata: float, tags: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "crs": crs,
        "nodata": float(nodata),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


def _smooth_masked_cpu(
    arr: np.ndarray,
    nodata: float,
    sigma: float,
    min_clip: Optional[float] = None,
    max_clip: Optional[float] = None,
) -> np.ndarray:
    if sigma <= 0:
        out = arr.copy()
    else:
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
        valid = out != nodata
        lo = min_clip if min_clip is not None else -np.inf
        hi = max_clip if max_clip is not None else np.inf
        out[valid] = np.clip(out[valid], lo, hi)

    return out


def _snap_origin(minx: float, miny: float, resolution: float) -> (float, float):
    ox = math.floor(minx / resolution) * resolution
    oy = math.floor(miny / resolution) * resolution
    return ox, oy


def process_file(src_path: Path, out_root: Path, dem_resolution: float, gaussian_sigma: float, fallback_epsg: int = 3301, max_files: Optional[int] = None) -> None:
    with rasterio.open(src_path) as src:
        src_nodata = src.nodata if src.nodata is not None else DEFAULT_NODATA
        left, bottom, right, top = src.bounds
        ox = math.floor(left / dem_resolution) * dem_resolution
        # Snap top to a multiple of resolution that covers the tile (use ceil to avoid clipping)
        maxy = math.ceil(top / dem_resolution) * dem_resolution
        dst_width = int(math.ceil((right - ox) / dem_resolution))
        dst_height = int(math.ceil((maxy - bottom) / dem_resolution))
        dst_transform = from_origin(ox, maxy, dem_resolution, dem_resolution)

        dst_arr = np.full((dst_height, dst_width), src_nodata, dtype=np.float32)

        # If source CRS is missing, assume fallback EPSG (handled by caller via args)
        src_crs_val = src.crs if src.crs is not None else f"EPSG:{fallback_epsg}"
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src_crs_val,
            dst_transform=dst_transform,
            dst_crs=src_crs_val,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=src_nodata,
        )

        # Save raw resampled
        rel = src_path.name
        base = Path(rel).stem
        raw_dir = out_root / "chm_raw"
        gauss_dir = out_root / "chm_gauss"
        raw_out = raw_dir / f"{base}_raw_chm.tif"
        gauss_out = gauss_dir / f"{base}_gauss_chm.tif"

        common_tags = {
            "SOURCE": str(src_path),
            "POST_FILTER": "none",
        }

        out_crs = src_crs_val
        _write_gtiff(raw_out, dst_arr, dst_transform, out_crs, src_nodata, {**common_tags, "POST_FILTER": "none"})

        # Apply masked gaussian smoothing
        chm_gauss = _smooth_masked_cpu(dst_arr, src_nodata, gaussian_sigma, min_clip=None, max_clip=None)

        _write_gtiff(gauss_out, chm_gauss, dst_transform, out_crs, src_nodata, {**common_tags, "POST_FILTER": f"gaussian_sigma_{gaussian_sigma}"})

        print(f"Processed {src_path} -> {gauss_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=Path("data/chm_variants/baseline_chm_20cm"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/chm_variant_selection"))
    parser.add_argument("--dem-resolution", type=float, default=0.8)
    parser.add_argument("--gaussian-sigma", type=float, default=0.3)
    parser.add_argument("--fallback-epsg", type=int, default=3301, help="EPSG code to assume when source TIFF lacks CRS")
    parser.add_argument("--pattern", type=str, default="**/*.tif")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files processed (0 = all)")
    args = parser.parse_args()

    baseline_dir = args.baseline_dir
    out_root = args.out_dir
    dem_resolution = args.dem_resolution
    gaussian_sigma = args.gaussian_sigma

    if not baseline_dir.exists():
        raise SystemExit(f"Baseline directory not found: {baseline_dir}")

    files = list(baseline_dir.glob(args.pattern))
    if len(files) == 0:
        print(f"No files found in {baseline_dir} matching {args.pattern}")
        return

    limit = args.max_files if args.max_files and args.max_files > 0 else None
    if limit is not None:
        files = files[:limit]

    for p in files:
        try:
            process_file(p, out_root, dem_resolution, gaussian_sigma, args.fallback_epsg)
        except Exception as e:
            print(f"Failed {p}: {e}")


if __name__ == "__main__":
    main()

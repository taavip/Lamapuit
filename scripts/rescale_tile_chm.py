#!/usr/bin/env python3
"""
Rescale CHM raster to uint8 and tile for Label Studio.

Writes:
- output/tile_labels/{stem}_stats.json  (global stats + params)
- output/tile_labels/{stem}/images/*.png  (tiles 160x160)
- output/tile_labels/{stem}/tile_metadata.csv (per-tile metadata)

Usage: python scripts/rescale_tile_chm.py --input data/chm_max_hag/474659_2024_madal_chm_max_hag_20cm.tif
"""

from pathlib import Path
import argparse
import json
import csv
import math
import sys

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from tqdm import tqdm
from PIL import Image

try:
    import cv2

    _has_cv2 = True
except Exception:
    _has_cv2 = False


def compute_global_percentiles(src, sample_pct=1.0, max_samples=1_000_000):
    # Try reading whole band first
    h, w = src.height, src.width
    try:
        arr = src.read(1, masked=True)
        valid = arr.compressed()
        if valid.size == 0:
            raise ValueError("No valid pixels found")
        p2, p98 = np.nanpercentile(valid, [2, 98])
        stats = {
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "valid_count": int(valid.size),
            "total_count": int(h * w),
        }
        return p2, p98, stats
    except MemoryError:
        pass

    # Fallback: windowed sampling
    samples = []
    step = max(256, int(min(h, w) / 100))
    win = 512
    for r in range(0, h, step):
        for c in range(0, w, step):
            rw = min(win, h - r)
            cw = min(win, w - c)
            window = Window(r, c, rw, cw)
            block = src.read(1, window=window, masked=True)
            try:
                vals = block.compressed()
            except Exception:
                vals = block[~np.isnan(block)].ravel()
            if vals.size:
                samples.append(vals)
            if sum(s.size for s in samples) >= max_samples:
                break
        if sum(s.size for s in samples) >= max_samples:
            break
    if not samples:
        raise RuntimeError("Unable to sample raster for percentiles")
    all_samples = np.concatenate(samples)
    p2, p98 = np.nanpercentile(all_samples, [2, 98])
    stats = {
        "min": float(np.min(all_samples)),
        "max": float(np.max(all_samples)),
        "mean": float(np.mean(all_samples)),
        "std": float(np.std(all_samples)),
        "valid_count": int(all_samples.size),
        "total_count": int(h * w),
    }
    return p2, p98, stats


def rescale_to_uint8(arr, p2, p98, clip=True):
    # arr may be masked array or numeric
    is_masked = hasattr(arr, "mask")
    if is_masked:
        data = arr.filled(np.nan).astype("float32")
        mask = ~np.isnan(data)
    else:
        data = arr.astype("float32")
        mask = np.isfinite(data)

    denom = (p98 - p2) if (p98 - p2) != 0 else 1.0
    out = np.zeros(data.shape, dtype=np.uint8)
    if mask.any():
        norm = (data - p2) / denom
        if clip:
            norm = np.clip(norm, 0.0, 1.0)
        vals = (norm * 255.0).round().astype(np.uint8)
        out[mask] = vals[mask]
    return out


def apply_clahe_uint8(img_uint8, clipLimit=2.0, tileGridSize=(8, 8)):
    if not _has_cv2:
        raise RuntimeError("OpenCV is required for CLAHE but not available")
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img_uint8)


def tile_and_save(
    src_path, out_dir, tile_size=160, overlap=0.5, method="p2p98", clahe=False, sample_pct=1.0
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(src_path).stem
    images_dir = out_dir / stem / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    stats_json = out_dir / f"{stem}_stats.json"
    metadata_csv = out_dir / stem / "tile_metadata.csv"

    with rasterio.open(src_path) as src:
        mask_src = None
        # try to find a mask file alongside source: same stem + _mask.tif
        mask_path = Path(src_path).with_name(Path(src_path).stem + "_mask.tif")
        if mask_path.exists():
            try:
                mask_src = rasterio.open(mask_path)
                print(f"Using mask: {mask_path}")
            except Exception:
                mask_src = None
        p2, p98, stats = compute_global_percentiles(src, sample_pct=sample_pct)
        stats["method"] = method
        stats["p2"] = float(p2)
        stats["p98"] = float(p98)
        stats["clahe"] = bool(clahe)
        stats["dtype"] = str(src.dtypes[0])
        stats["width"] = src.width
        stats["height"] = src.height
        stats["crs"] = str(src.crs)
        stats["transform"] = src.transform.to_gdal()

        # Save global stats
        with open(stats_json, "w") as f:
            json.dump(stats, f, indent=2)

        stride = max(1, int(tile_size * (1.0 - overlap)))
        rows = list(range(0, src.height, stride))
        cols = list(range(0, src.width, stride))

        csv_fields = [
            "image_path",
            "row_off",
            "col_off",
            "width",
            "height",
            "minx",
            "miny",
            "maxx",
            "maxy",
            "tile_min",
            "tile_max",
            "tile_mean",
            "tile_std",
        ]
        with open(metadata_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writeheader()
            skipped_nodata = 0
            skipped_zero = 0
            written_tiles = 0

            for r in tqdm(rows, desc="Tiles rows"):
                for c in cols:
                    # rasterio Window is (col_off, row_off, width, height)
                    window = Window(c, r, tile_size, tile_size)
                    arr = src.read(1, window=window, boundless=True, fill_value=src.nodata)
                    # if mask present, skip tiles where mask indicates no data
                    if mask_src is not None:
                        mask_win = mask_src.read(1, window=window, boundless=True, fill_value=0)
                        if not mask_win.any():
                            # skip nodata-only tile
                            continue
                    # arr is ndarray; convert masked if nodata present
                    if src.nodata is not None:
                        arr = np.ma.masked_equal(arr, src.nodata)

                    # skip tiles where all values are nodata
                    if np.ma.isMaskedArray(arr) and np.ma.count(arr) == 0:
                        skipped_nodata += 1
                        continue

                    tile_min = (
                        float(np.min(arr)) if np.ma.is_masked(arr) is False else float(arr.min())
                    )
                    tile_max = (
                        float(np.max(arr)) if np.ma.is_masked(arr) is False else float(arr.max())
                    )
                    tile_mean = float(np.mean(arr) if np.ma.is_masked(arr) is False else arr.mean())
                    tile_std = float(np.std(arr) if np.ma.is_masked(arr) is False else arr.std())

                    uint8 = rescale_to_uint8(arr, p2, p98)
                    if clahe:
                        if not _has_cv2:
                            raise RuntimeError("CLAHE requested but OpenCV not available")
                        uint8 = apply_clahe_uint8(uint8)

                    # skip all-zero tiles (black PNGs)
                    if not uint8.any():
                        skipped_zero += 1
                        continue

                    # Save PNG
                    img_path = images_dir / f"tile_{r:06d}_{c:06d}.png"
                    Image.fromarray(uint8).convert("L").save(img_path)
                    written_tiles += 1

                    # compute geobox bounds for the window
                    try:
                        minx, miny, maxx, maxy = window_bounds(window, src.transform)
                    except Exception:
                        minx = miny = maxx = maxy = None

                    writer.writerow(
                        {
                            "image_path": str(img_path.relative_to(out_dir)),
                            "row_off": int(r),
                            "col_off": int(c),
                            "width": int(tile_size),
                            "height": int(tile_size),
                            "minx": minx,
                            "miny": miny,
                            "maxx": maxx,
                            "maxy": maxy,
                            "tile_min": tile_min,
                            "tile_max": tile_max,
                            "tile_mean": tile_mean,
                            "tile_std": tile_std,
                        }
                    )

    print(f"Wrote stats to {stats_json}")
    print(f"Wrote tile metadata to {metadata_csv}")
    print(f"Saved tiles to {images_dir}")
    print(f"Tile summary: written={written_tiles}, skipped_nodata={skipped_nodata}, skipped_zero={skipped_zero}")


def main():
    parser = argparse.ArgumentParser(description="Rescale CHM and tile for Label Studio")
    parser.add_argument("--input", "-i", required=True, help="Input CHM TIFF")
    parser.add_argument(
        "--out-dir", "-o", default="output/tile_labels", help="Output base directory"
    )
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--method", choices=["p2p98", "minmax"], default="p2p98")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE after rescaling")
    parser.add_argument(
        "--sample-pct",
        type=float,
        default=1.0,
        help="Sampling percent used for percentile calc (1.0=all)",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input file not found: {inp}")
        sys.exit(1)

    tile_and_save(
        str(inp),
        args.out_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        method=args.method,
        clahe=args.clahe,
        sample_pct=args.sample_pct,
    )


if __name__ == "__main__":
    main()

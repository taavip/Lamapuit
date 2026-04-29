#!/usr/bin/env python3
"""
Build 4-band composite rasters from harmonized CHM sources with explicit mask channels.

All inputs are warped to 0.2 m resolution (finest grid from baseline_chm_20cm).

Output bands:
  1. Gaussian-smoothed CHM (0.2 m)
  2. Raw CHM (0.2 m)
  3. Baseline CHM 20cm (0.2 m reference)
  4. Composite mask (1=valid in ALL sources, 0=any nodata)

NoData value: -9999
Mask channel allows self-attention mechanisms to explicitly "see" which pixels
have valid measurements across all three sources.
"""

import os
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from tqdm import tqdm


NODATA_VAL = -9999.0
NODATA_THRESHOLD = -9998.0  # Pixels with value <= this are treated as nodata


def safe_list_tifs(path: str) -> list:
    """Safely list .tif files in a directory."""
    try:
        return sorted([f for f in os.listdir(path) if f.lower().endswith('.tif')])
    except Exception:
        return []


def extract_tile_key(filename: str) -> Tuple[str, str]:
    """Extract (id, year) from filename using pattern {id}_{year}*."""
    pattern = re.compile(r"(?P<id>\d{6})[_-](?P<year>\d{4})")
    match = pattern.search(filename)
    if match:
        return (match.group('id'), match.group('year'))
    return None


def find_matching_triples(gauss_dir: str, raw_dir: str, base_dir: str) -> list:
    """
    Find matching tile IDs across three CHM sources.
    Returns list of (id, year, gauss_path, raw_path, base_path).
    """
    gauss_files = safe_list_tifs(gauss_dir)
    raw_files = safe_list_tifs(raw_dir)
    base_files = safe_list_tifs(base_dir)

    def map_files(files):
        m = {}
        for f in files:
            key = extract_tile_key(f)
            if key:
                m.setdefault(key, []).append(f)
        return m

    gmap = map_files(gauss_files)
    rmap = map_files(raw_files)
    bmap = map_files(base_files)

    keys = sorted(set(gmap.keys()) & set(rmap.keys()) & set(bmap.keys()))

    matches = []
    for key in keys:
        tile_id, year = key
        ga_file = gmap[key][0]
        ra_file = rmap[key][0]
        ba_file = bmap[key][0]
        matches.append(
            (
                tile_id,
                year,
                os.path.join(gauss_dir, ga_file),
                os.path.join(raw_dir, ra_file),
                os.path.join(base_dir, ba_file),
            )
        )

    return matches


def read_and_warp(
    src_path: str,
    target_bounds: rasterio.coords.BoundingBox,
    target_res: Tuple[float, float],
    target_crs,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Read and warp a raster to target grid.

    Args:
        src_path: Input raster path
        target_bounds: Target bounding box
        target_res: Target pixel resolution (width, height)
        target_crs: Target coordinate reference system
        height: Target height in pixels
        width: Target width in pixels

    Returns:
        Warped raster data as numpy array (single band)
    """
    with rasterio.open(src_path) as src:
        with WarpedVRT(
            src,
            crs=target_crs,
            resampling=Resampling.bilinear,
            left=target_bounds.left,
            bottom=target_bounds.bottom,
            right=target_bounds.right,
            top=target_bounds.top,
            width=width,
            height=height,
        ) as vrt:
            data = vrt.read(1).astype(np.float32)
            # Normalize nodata values to standard value
            data[data <= NODATA_THRESHOLD] = NODATA_VAL
            return data


def create_composite_with_masks(
    gauss_path: str,
    raw_path: str,
    base_path: str,
    output_path: str,
) -> bool:
    """
    Create 4-band composite (gauss, raw, base, mask) at 0.2m resolution.

    Warps all inputs to match baseline_chm_20cm grid (0.2m resolution).
    Creates composite mask: 1 where all inputs are valid, 0 where any is nodata.

    Returns True on success, False on failure.
    """
    try:
        # Use baseline as reference (0.2 m resolution)
        with rasterio.open(base_path) as base_src:
            ref_bounds = base_src.bounds
            ref_crs = base_src.crs
            ref_transform = base_src.transform
            height = base_src.height
            width = base_src.width
            px_width = abs(ref_transform.a)
            px_height = abs(ref_transform.e)
            target_res = (px_width, px_height)

        # Read baseline (already at target resolution)
        with rasterio.open(base_path) as src:
            base_data = src.read(1).astype(np.float32)
            base_data[base_data <= NODATA_THRESHOLD] = NODATA_VAL

        # Warp gauss and raw to baseline grid (0.2 m)
        gauss_data = read_and_warp(
            gauss_path, ref_bounds, target_res, ref_crs, height, width
        )
        raw_data = read_and_warp(
            raw_path, ref_bounds, target_res, ref_crs, height, width
        )

        # Create conservative mask using ONLY Raw + Baseline (true measurements)
        # Gaussian is smoothed/interpolated, so we exclude it from validity check
        # A pixel is valid (mask=1) only if BOTH Raw AND Baseline have valid data
        mask = np.ones((height, width), dtype=np.float32)
        mask[raw_data <= NODATA_THRESHOLD] = 0
        mask[base_data <= NODATA_THRESHOLD] = 0
        # Note: Gaussian band is included for model input but not for mask validation

        # Stack into 4-band composite
        composite = np.stack([gauss_data, raw_data, base_data, mask], axis=0)

        # Write output with tiling and compression
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=4,
            dtype=rasterio.float32,
            crs=ref_crs,
            transform=ref_transform,
            nodata=NODATA_VAL,
            TILED='YES',
            COMPRESS='DEFLATE',
            BLOCKXSIZE=256,
            BLOCKYSIZE=256,
            BIGTIFF='IF_SAFER',
        ) as dst:
            dst.write(composite)

        return True

    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    gauss_dir = "data/chm_variants/harmonized_0p8m_chm_gauss"
    raw_dir = "data/chm_variants/harmonized_0p8m_chm_raw"
    base_dir = "data/chm_variants/baseline_chm_20cm"
    # Fallback to actual path if symlink doesn't work
    if not os.path.isdir(base_dir):
        base_dir = "data/lamapuit/chm_max_hag_13_drop"

    out_dir = "data/chm_variants/composite_3band_with_masks"
    max_tiles = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Validate input directories
    for d, name in [(gauss_dir, "gauss"), (raw_dir, "raw"), (base_dir, "baseline")]:
        if not os.path.isdir(d):
            print(f"Error: {name} directory not found: {d}")
            return

    print("Scanning for matching tiles across three sources...")
    matches = find_matching_triples(gauss_dir, raw_dir, base_dir)
    print(f"Found {len(matches)} matching triples.\n")

    if not matches:
        print("No matching triples. Exiting.")
        return

    print(f"Output resolution: 0.2 m (from baseline_chm_20cm)")
    print(f"Output directory: {out_dir}/\n")

    processed = 0
    failed = 0

    for tile_id, year, gauss_path, raw_path, base_path in tqdm(
        matches, desc="Processing tiles", unit="tile"
    ):
        if max_tiles > 0 and processed >= max_tiles:
            break

        out_path = os.path.join(out_dir, f"{tile_id}_{year}_4band.tif")

        if os.path.exists(out_path):
            continue

        success = create_composite_with_masks(gauss_path, raw_path, base_path, out_path)

        if success:
            processed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done. Processed {processed} composites → {out_dir}/")
    if failed > 0:
        print(f"Failed: {failed}")
    print(f"{'='*60}")
    print("\nBand descriptions:")
    print("  Band 1: Gaussian-smoothed CHM (0.2 m)")
    print("  Band 2: Raw CHM (0.2 m)")
    print("  Band 3: Baseline CHM (0.2 m)")
    print("  Band 4: Composite mask (1=valid in all, 0=any nodata)")


if __name__ == "__main__":
    main()

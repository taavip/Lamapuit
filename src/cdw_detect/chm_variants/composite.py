"""
4-Band Composite CHM Generator

Creates composite rasters with:
  Band 1: Gaussian-smoothed CHM (0.2m)
  Band 2: Raw CHM (0.2m)
  Band 3: Baseline CHM (0.2m)
  Band 4: Composite Mask (1=valid in Raw+Base, 0=any missing)

Mask Strategy (Conservative):
  - Only includes pixels where BOTH Raw and Baseline have valid data
  - Excludes Gaussian-only interpolations (synthetic data)
  - Gaussian band available for model features, but not mask validation
  - Results in ~22% valid pixels (high confidence)
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from tqdm import tqdm


NODATA_VAL = -9999.0
NODATA_THRESHOLD = -9998.0


class CompositeGenerator:
    """Generate 4-band composite with conservative masking."""

    def __init__(self, gauss_dir: str, raw_dir: str, base_dir: str, output_dir: str):
        """
        Initialize composite generator.

        Args:
            gauss_dir: Gaussian-smoothed CHM directory
            raw_dir: Raw CHM directory
            base_dir: Baseline CHM directory
            output_dir: Output directory for composites
        """
        self.gauss_dir = Path(gauss_dir)
        self.raw_dir = Path(raw_dir)
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_list_tifs(path: Path) -> list:
        """Safely list .tif files in directory."""
        try:
            return sorted([f for f in path.iterdir() if f.suffix.lower() == '.tif'])
        except Exception:
            return []

    def find_matching_triples(self) -> list:
        """Find tiles matching across all three CHM sources."""
        import re

        pattern = re.compile(r"(?P<id>\d{6})[_-](?P<year>\d{4})")

        def extract_key(filename):
            match = pattern.search(filename.name)
            return (match.group('id'), match.group('year')) if match else None

        def map_files(files):
            m = {}
            for f in files:
                key = extract_key(f)
                if key:
                    m.setdefault(key, []).append(f)
            return m

        gauss_files = self.safe_list_tifs(self.gauss_dir)
        raw_files = self.safe_list_tifs(self.raw_dir)
        base_files = self.safe_list_tifs(self.base_dir)

        gmap = map_files(gauss_files)
        rmap = map_files(raw_files)
        bmap = map_files(base_files)

        keys = sorted(set(gmap.keys()) & set(rmap.keys()) & set(bmap.keys()))

        matches = []
        for key in keys:
            tile_id, year = key
            matches.append((tile_id, year, gmap[key][0], rmap[key][0], bmap[key][0]))

        return matches

    @staticmethod
    def read_and_warp(
        src_path: Path,
        target_bounds,
        target_res: Tuple[float, float],
        target_crs,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Warp raster to target grid."""
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
                data[data <= NODATA_THRESHOLD] = NODATA_VAL
                return data

    def create_composite(self, gauss_path: Path, raw_path: Path, base_path: Path, output_path: Path) -> bool:
        """
        Create 4-band composite with conservative mask.

        Mask = 1 only where BOTH Raw and Baseline have valid data.
        Gaussian excluded from mask validation (interpolations not trusted).
        """
        try:
            # Use Baseline as reference (0.2m resolution)
            with rasterio.open(base_path) as base_src:
                ref_bounds = base_src.bounds
                ref_crs = base_src.crs
                ref_transform = base_src.transform
                height = base_src.height
                width = base_src.width
                px_width = abs(ref_transform.a)
                px_height = abs(ref_transform.e)
                target_res = (px_width, px_height)

            # Read baseline
            with rasterio.open(base_path) as src:
                base_data = src.read(1).astype(np.float32)
                base_data[base_data <= NODATA_THRESHOLD] = NODATA_VAL

            # Warp gauss and raw to baseline grid
            gauss_data = self.read_and_warp(
                gauss_path, ref_bounds, target_res, ref_crs, height, width
            )
            raw_data = self.read_and_warp(
                raw_path, ref_bounds, target_res, ref_crs, height, width
            )

            # Conservative mask: 1 only where Raw AND Baseline both valid
            mask = np.ones((height, width), dtype=np.float32)
            mask[raw_data <= NODATA_THRESHOLD] = 0
            mask[base_data <= NODATA_THRESHOLD] = 0

            # Stack 4 bands
            composite = np.stack([gauss_data, raw_data, base_data, mask], axis=0)

            # Write output
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
            print(f"Error creating composite: {e}")
            return False

    def generate(self, max_tiles: int = 0) -> int:
        """Generate all 4-band composites. Returns count of processed tiles."""
        matches = self.find_matching_triples()
        if not matches:
            print("No matching triples found")
            return 0

        processed = 0
        for tile_id, year, gauss_path, raw_path, base_path in tqdm(
            matches, desc="Generating 4-band composites", unit="tile"
        ):
            if max_tiles > 0 and processed >= max_tiles:
                break

            output_path = self.output_dir / f"{tile_id}_{year}_4band.tif"
            if output_path.exists():
                continue

            if self.create_composite(gauss_path, raw_path, base_path, output_path):
                processed += 1

        return processed

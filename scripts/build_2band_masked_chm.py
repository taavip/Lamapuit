#!/usr/bin/env python3
"""
Build 2-band masked CHM dataset from harmonized raw CHM rasters.

Output bands:
  1. Raw CHM values (0 to ~1.3 m)
  2. Binary mask where 1=valid data, 0=nodata

This format is optimal for attention-based models, as the mask channel allows
the self-attention mechanism to explicitly "see" which pixels contain valid
measurements vs. nodata regions. Even if the value in Band 1 is 0 (valid
measurement), the presence of 1 in Band 2 tells the model that 0 is real data,
not a missing value.

NoData value: -9999 (same as input rasters)
"""

import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm


NODATA_VAL = -9999.0
NODATA_THRESHOLD = -9998.0  # Pixels with value < this are treated as nodata


def process_raster(input_path: str, output_path: str) -> bool:
    """
    Read raw CHM, create mask channel, and write 2-band output.
    Returns True on success.
    """
    try:
        with rasterio.open(input_path) as src:
            # Read the raw CHM
            data = src.read(1)
            nodata = src.nodata if src.nodata is not None else NODATA_VAL

            # Create binary mask: 1 = valid, 0 = nodata
            mask = np.ones_like(data, dtype=np.float32)
            mask[data <= NODATA_THRESHOLD] = 0

            # Normalize nodata values in data band to standard value
            data_out = data.astype(np.float32)
            data_out[data_out <= NODATA_THRESHOLD] = NODATA_VAL

            # Stack into 2-band array
            composite = np.stack([data_out, mask], axis=0)

            # Write output
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=src.height,
                width=src.width,
                count=2,
                dtype=composite.dtype,
                crs=src.crs,
                transform=src.transform,
                nodata=NODATA_VAL,
                TILED='YES',
                COMPRESS='DEFLATE',
                BIGTIFF='IF_SAFER',
            ) as dst:
                dst.write(composite)

        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    input_dir = "data/chm_variants/harmonized_0p8m_chm_raw"
    output_dir = "data/chm_variants/harmonized_0p8m_chm_raw_2band_masked"
    max_tiles = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # List input files
    input_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith('.tif')]
    )
    print(f"Found {len(input_files)} raw CHM rasters in {input_dir}/")

    if not input_files:
        print("No input files. Exiting.")
        return

    processed = 0
    for filename in tqdm(input_files, desc="Processing rasters"):
        if max_tiles > 0 and processed >= max_tiles:
            break

        input_path = os.path.join(input_dir, filename)
        # Replace .tif with _2band.tif
        output_filename = filename.replace('.tif', '_2band.tif')
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            continue

        success = process_raster(input_path, output_path)
        if success:
            processed += 1

    print(f"\nDone. Processed {processed} rasters → {output_dir}/")
    print(
        "2-band format: Band 1 = raw CHM, Band 2 = mask (1=valid, 0=nodata)"
    )


if __name__ == "__main__":
    main()

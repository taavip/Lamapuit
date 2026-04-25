"""
2-Band Masked CHM Generator

Creates 2-band rasters with:
  Band 1: Raw CHM values (0 to 1.3m)
  Band 2: Binary mask (1=valid, 0=nodata)

Mask Channel Benefits:
  - Explicit validity signal for attention mechanisms
  - Distinguishes real 0m (bare ground) from -9999 (nodata)
  - Model can learn which regions are trustworthy
  - Conservative: only marks valid where data exists
"""

import os
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


NODATA_VAL = -9999.0
NODATA_THRESHOLD = -9998.0


class MaskedCHMGenerator:
    """Generate 2-band masked CHM (Raw + Mask)."""

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize masked CHM generator.

        Args:
            input_dir: Input raw CHM directory
            output_dir: Output directory for 2-band masked CHMs
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_raster(self, input_path: Path, output_path: Path) -> bool:
        """
        Read raw CHM, create mask channel, write 2-band output.

        Mask = 1 where CHM is valid, 0 where CHM is nodata.
        """
        try:
            with rasterio.open(input_path) as src:
                # Read raw CHM
                data = src.read(1)
                nodata = src.nodata if src.nodata is not None else NODATA_VAL

                # Create binary mask: 1=valid, 0=nodata
                mask = np.ones_like(data, dtype=np.float32)
                mask[data <= NODATA_THRESHOLD] = 0

                # Normalize nodata in data band
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
                    BLOCKXSIZE=256,
                    BLOCKYSIZE=256,
                    BIGTIFF='IF_SAFER',
                ) as dst:
                    dst.write(composite)

                return True

        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")
            return False

    def generate(self, max_tiles: int = 0) -> int:
        """Generate all 2-band masked CHMs. Returns count of processed tiles."""
        input_files = sorted(
            [f for f in self.input_dir.iterdir() if f.suffix.lower() == '.tif']
        )

        if not input_files:
            print(f"No TIF files found in {self.input_dir}")
            return 0

        processed = 0
        for input_path in tqdm(input_files, desc="Generating 2-band masked CHMs", unit="tile"):
            if max_tiles > 0 and processed >= max_tiles:
                break

            output_filename = input_path.name.replace('.tif', '_2band.tif')
            output_path = self.output_dir / output_filename

            if output_path.exists():
                continue

            if self.process_raster(input_path, output_path):
                processed += 1

        return processed

#!/usr/bin/env python3
"""Generate CHM variants from existing baseline CHM file."""

import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine

def load_chm(path):
    """Load CHM raster."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
    return data, profile, transform

def save_chm(path, data, profile, transform):
    """Save CHM raster."""
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)

def create_mask(chm):
    """Create validity mask (where CHM > 0)."""
    return (chm > 0).astype(np.uint8) * 255

def gaussian_kernel_1d(sigma, size=None):
    """Create 1D Gaussian kernel."""
    if size is None:
        size = int(6 * sigma + 1)
    x = np.arange(-size // 2 + 1., size // 2 + 1.)
    gauss = np.exp(-x ** 2 / (2 * sigma ** 2))
    return gauss / gauss.sum()

def apply_gaussian_smooth(chm, kernel_size=0.8, resolution=0.2):
    """Apply Gaussian smoothing using separable convolution."""
    sigma_pixels = kernel_size / resolution
    kernel = gaussian_kernel_1d(sigma_pixels)

    # Apply 1D convolution in each dimension
    result = chm.copy()
    for axis in [0, 1]:
        result = np.apply_along_axis(
            lambda arr: np.convolve(arr, kernel, mode='same'),
            axis,
            result
        )
    return result

def main():
    input_chm = Path("/home/tpipar/project/Lamapuit/chm_max_hag/406455_2021_tava_chm_max_hag_20cm.tif")
    output_base = Path("/home/tpipar/project/Lamapuit/source/406455_2021_tava/chm_variants")

    if not input_chm.exists():
        print(f"Error: Input CHM not found: {input_chm}")
        sys.exit(1)

    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline CHM: {input_chm}")
    baseline_chm, profile, transform = load_chm(input_chm)

    # 1. Baseline (copy)
    print("1. Creating baseline variant...")
    baseline_dir = output_base / "baseline_chm_0p2m"
    baseline_dir.mkdir(exist_ok=True)
    save_chm(baseline_dir / "406455_2021_tava_chm.tif", baseline_chm, profile, transform)

    # 2. Raw (same as baseline for our case)
    print("2. Creating harmonized raw variant...")
    raw_dir = output_base / "harmonized_raw_0p2m"
    raw_dir.mkdir(exist_ok=True)
    save_chm(raw_dir / "406455_2021_tava_chm_raw.tif", baseline_chm, profile, transform)

    # 3. Gaussian smoothed
    print("3. Creating Gaussian smoothed variant...")
    gauss_chm = apply_gaussian_smooth(baseline_chm, kernel_size=0.8, resolution=0.2)
    gauss_dir = output_base / "harmonized_gauss_kernel0p8m_0p2m"
    gauss_dir.mkdir(exist_ok=True)
    save_chm(gauss_dir / "406455_2021_tava_chm_gauss.tif", gauss_chm, profile, transform)

    # 4. Composite 4-band
    print("4. Creating 4-band composite variant...")
    mask = create_mask(baseline_chm)
    composite_dir = output_base / "composite_4band_raw_base_mask"
    composite_dir.mkdir(exist_ok=True)

    # Save as 4-band GeoTIFF
    profile_4band = profile.copy()
    profile_4band.update(count=4, dtype=rasterio.float32)
    with rasterio.open(composite_dir / "406455_2021_tava_composite.tif", 'w', **profile_4band) as dst:
        dst.write(gauss_chm.astype(np.float32), 1)  # Gaussian
        dst.write(baseline_chm.astype(np.float32), 2)  # Raw
        dst.write(baseline_chm.astype(np.float32), 3)  # Baseline
        dst.write(mask.astype(np.float32), 4)  # Mask

    # 5. 2-band masked raw
    print("5. Creating 2-band masked raw variant...")
    masked_dir = output_base / "masked_raw_2band_0p2m"
    masked_dir.mkdir(exist_ok=True)

    profile_2band = profile.copy()
    profile_2band.update(count=2, dtype=rasterio.float32)
    with rasterio.open(masked_dir / "406455_2021_tava_masked_raw.tif", 'w', **profile_2band) as dst:
        dst.write(baseline_chm.astype(np.float32), 1)  # Raw CHM
        dst.write(mask.astype(np.float32), 2)  # Mask

    print("\n✓ All variants generated successfully!")
    print(f"\nOutput location: {output_base}")
    for variant in sorted(output_base.iterdir()):
        if variant.is_dir():
            file_count = len(list(variant.glob("*.tif")))
            print(f"  {variant.name}: {file_count} TIF file(s)")

if __name__ == "__main__":
    main()

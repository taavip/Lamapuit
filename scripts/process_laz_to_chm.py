"""
LAZ to CHM Raster Processing

Convert LiDAR LAZ files to CHM (Canopy Height Model) GeoTIFF rasters.
Computes Height Above Ground (HAG) and creates 0.2m resolution CHM suitable for CDW detection.

Usage:
    python scripts/process_laz_to_chm.py --input path/to/file.laz --output path/to/chm.tif
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def compute_hag_raster(laz_path: str, output_tif: str, resolution: float = 0.2, 
                       hag_max: float = 1.5, nodata: float = -9999.0):
    """
    Convert LAZ file to CHM raster.
    
    Args:
        laz_path: Path to input LAZ file
        output_tif: Path to output GeoTIFF
        resolution: Raster cell size in meters (default: 0.2)
        hag_max: Maximum HAG value to include (default: 1.5m for CDW)
        nodata: NoData value for empty cells
    
    Returns:
        Path to created GeoTIFF
    """
    try:
        import laspy
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.enums import Resampling
        from scipy.spatial import cKDTree
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Install with: pip install laspy[lazrs] rasterio scipy")
        sys.exit(1)
    
    print(f"Reading LAZ file: {laz_path}")
    las = laspy.read(laz_path)
    
    # Extract coordinates
    x = np.asarray(las.x, dtype=float)
    y = np.asarray(las.y, dtype=float)
    z = np.asarray(las.z, dtype=float)
    
    print(f"  Points: {len(x):,}")
    print(f"  Bounds: X={x.min():.2f} to {x.max():.2f}, Y={y.min():.2f} to {y.max():.2f}")
    
    # Get classification (ground = 2)
    try:
        classification = np.asarray(las.classification)
        ground_mask = classification == 2
        n_ground = np.count_nonzero(ground_mask)
        print(f"  Ground points: {n_ground:,}")
        
        if n_ground < 3:
            print("Error: Not enough ground points for HAG computation")
            sys.exit(1)
    except Exception:
        print("Warning: No classification found - using all points as ground")
        ground_mask = np.ones(len(x), dtype=bool)
    
    # Compute grid parameters
    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    
    print(f"  Raster size: {width} x {height} pixels ({resolution}m resolution)")
    
    # Convert points to pixel coordinates
    col = ((x - minx) / resolution).astype(int)
    row = ((maxy - y) / resolution).astype(int)
    
    # Clip to raster bounds
    valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    row = row[valid]
    col = col[valid]
    x = x[valid]
    y = y[valid]
    z = z[valid]
    ground_mask = ground_mask[valid]
    
    # Compute HAG using ground interpolation
    print("Computing Height Above Ground...")
    ground_pts = np.vstack([x[ground_mask], y[ground_mask]]).T
    ground_z = z[ground_mask]
    
    # Build KDTree for fast nearest neighbor search
    tree = cKDTree(ground_pts)
    
    # Query 3 nearest ground points for each point
    pts = np.vstack([x, y]).T
    try:
        dists, idx = tree.query(pts, k=3, workers=-1)
    except TypeError:
        dists, idx = tree.query(pts, k=3)
    
    # Inverse distance weighted interpolation
    weights = 1.0 / (dists + 1e-8)
    ground_z_interp = (weights * ground_z[idx]).sum(axis=1) / weights.sum(axis=1)
    
    # Compute HAG
    hag = z - ground_z_interp
    hag = np.clip(hag, 0.0, hag_max)
    
    # Rasterize using max HAG per pixel
    print("Rasterizing...")
    raster = np.full((height, width), nodata, dtype=np.float32)
    
    # Vectorized max reduction
    flat_idx = row * width + col
    np.maximum.at(raster.ravel(), flat_idx, hag.astype(np.float32))
    
    # Create GeoTIFF
    print(f"Writing GeoTIFF: {output_tif}")
    transform = from_origin(minx, maxy, resolution, resolution)
    
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'float32',
        'transform': transform,
        'nodata': nodata,
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'compress': 'lzw',
    }
    
    # Extract CRS if available
    try:
        crs = las.header.parse_crs()
        if crs is not None:
            profile['crs'] = rasterio.crs.CRS.from_wkt(crs.to_wkt())
    except Exception:
        print("Warning: Could not extract CRS from LAZ file")
    
    output_path = Path(output_tif)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(raster, 1)
        
        # Add overviews for faster display
        try:
            dst.build_overviews([2, 4, 8, 16], resampling=Resampling.nearest)
            dst.update_tags(ns='rio_overview', resampling='nearest')
        except Exception:
            pass
    
    # Compute statistics
    valid_pixels = raster[raster != nodata]
    if len(valid_pixels) > 0:
        print(f"\nRaster statistics:")
        print(f"  Valid pixels: {len(valid_pixels):,} ({100*len(valid_pixels)/(height*width):.1f}%)")
        print(f"  HAG range: {valid_pixels.min():.3f} - {valid_pixels.max():.3f} m")
        print(f"  Mean HAG: {valid_pixels.mean():.3f} m")
    else:
        print("Warning: No valid pixels in output raster")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert LAZ file to CHM GeoTIFF raster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/process_laz_to_chm.py --input points.laz --output chm.tif
  
  # Custom resolution and height limit
  python scripts/process_laz_to_chm.py --input points.laz --output chm.tif --resolution 0.1 --hag-max 2.0

Requirements:
  - LAZ file must have ground classification (class 2) from SMRF or similar algorithm
  - If no classification exists, all points will be used as ground (may produce poor results)
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input LAZ file path')
    parser.add_argument('--output', '-o', required=True, help='Output GeoTIFF path')
    parser.add_argument('--resolution', '-r', type=float, default=0.2, 
                       help='Raster cell size in meters (default: 0.2)')
    parser.add_argument('--hag-max', type=float, default=1.5,
                       help='Maximum HAG value in meters (default: 1.5 for CDW)')
    parser.add_argument('--nodata', type=float, default=-9999.0,
                       help='NoData value (default: -9999.0)')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print("=" * 70)
    print("LAZ TO CHM RASTER CONVERSION")
    print("=" * 70)
    
    compute_hag_raster(
        laz_path=args.input,
        output_tif=args.output,
        resolution=args.resolution,
        hag_max=args.hag_max,
        nodata=args.nodata,
    )
    
    print("\n✓ Processing complete!")


if __name__ == '__main__':
    main()

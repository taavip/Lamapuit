"""Create small sample data for GitHub repository examples.

This script extracts a small tile from the large CHM raster and copies
the small label GeoPackage for inclusion in the repository.

Usage:
    python create_sample_data.py
"""

import os
import shutil
from pathlib import Path
import rasterio
from rasterio.windows import Window
import geopandas as gpd


def create_sample_chm(
    input_raster: str = "merged041225.tif",
    output_raster: str = "examples/data/sample_chm_tile.tif",
    tile_size: int = 500,  # pixels
    start_x: int = 1000,
    start_y: int = 1000
):
    """Extract small tile from large CHM raster.
    
    Args:
        input_raster: Path to large input CHM
        output_raster: Path to output sample tile
        tile_size: Size of tile in pixels (default 500x500)
        start_x: Starting X pixel coordinate
        start_y: Starting Y pixel coordinate
    """
    print(f"Creating sample CHM tile from {input_raster}...")
    
    with rasterio.open(input_raster) as src:
        # Create window
        window = Window(start_x, start_y, tile_size, tile_size)
        
        # Read data
        data = src.read(window=window)
        
        # Update profile for output
        profile = src.profile.copy()
        profile.update({
            'height': tile_size,
            'width': tile_size,
            'transform': rasterio.windows.transform(window, src.transform)
        })
        
        # Write output
        os.makedirs(os.path.dirname(output_raster), exist_ok=True)
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(data)
        
        # Get file size
        size_mb = os.path.getsize(output_raster) / (1024 * 1024)
        print(f"✓ Created {output_raster} ({size_mb:.2f} MB)")
        print(f"  Resolution: {src.res[0]}m")
        print(f"  Tile size: {tile_size}x{tile_size} pixels")
        print(f"  Area: {tile_size * src.res[0]:.0f}x{tile_size * src.res[1]:.0f} meters")


def copy_sample_labels(
    input_gpkg: str = "lamapuit.gpkg",
    output_gpkg: str = "examples/data/lamapuit_labels.gpkg"
):
    """Copy small label GeoPackage to examples.
    
    Args:
        input_gpkg: Path to input labels
        output_gpkg: Path to output labels
    """
    print(f"\nCopying sample labels from {input_gpkg}...")
    
    # Read labels
    gdf = gpd.read_file(input_gpkg)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_gpkg), exist_ok=True)
    
    # Copy file
    shutil.copy2(input_gpkg, output_gpkg)
    
    size_mb = os.path.getsize(output_gpkg) / (1024 * 1024)
    print(f"✓ Copied {output_gpkg} ({size_mb:.2f} MB)")
    print(f"  Features: {len(gdf)}")
    print(f"  CRS: {gdf.crs}")


def create_data_readme():
    """Create README for examples/data directory."""
    
    readme_content = """# Example Data

This folder contains small sample datasets for testing the CDW detection pipeline.

## Files Included

### `lamapuit_labels.gpkg` (~0.14 MB)
Sample CDW training labels with line geometries representing fallen logs.
- **Features**: 50 manually digitized CDW instances
- **CRS**: EPSG:3301 (Estonian Coordinate System)
- **Format**: GeoPackage vector layer

### `sample_chm_tile.tif` (~5-10 MB)
Small excerpt from Canopy Height Model (CHM) raster.
- **Resolution**: 0.2m per pixel
- **Size**: 500x500 pixels (100x100 meters)
- **Format**: GeoTIFF with elevation in meters above ground
- **Source**: LiDAR-derived maximum height

## Quick Test

```bash
# Prepare training data from samples
python scripts/prepare_data.py \\
  --raster examples/data/sample_chm_tile.tif \\
  --labels examples/data/lamapuit_labels.gpkg \\
  --output ./test_dataset \\
  --buffer-width 0.5

# Train on sample (will be very limited due to small area)
python scripts/train_model.py \\
  --config configs/default.yaml \\
  --data ./test_dataset/data.yaml \\
  --epochs 10
```

**Note**: This sample is too small for real training - download full dataset for production use.

## Full Dataset

The complete dataset is available on **Zenodo**:

📦 **DOI**: https://doi.org/10.5281/zenodo.XXXXXXX *(Update after upload)*

**Contents**:
- `merged041225.tif` (280 MB) - Complete CHM raster covering full study area
- `training_boxes_lamapuit.gpkg` (0.37 MB) - All training labels
- Additional CHM tiles and metadata

**Download**:
```bash
# Using wget
wget https://zenodo.org/record/XXXXXXX/files/merged041225.tif

# Using curl
curl -O https://zenodo.org/record/XXXXXXX/files/merged041225.tif
```

## Pre-trained Model

Download trained model weights from GitHub Releases:

```bash
wget https://github.com/taavip/cdw-detect/releases/download/v1.0.0/cdw_detect_v1.0.0.pt
```

## Data Specifications

### CHM Raster Specifications
- **Method**: Maximum height above ground (HAG) from LiDAR point cloud
- **Ground Classification**: SMRF algorithm (class 2)
- **Spatial Resolution**: 0.2m per pixel
- **Vertical Accuracy**: ±0.15m
- **Coverage**: Lamapuit forest area, Estonia

### Label Specifications
- **Digitization**: Manual on-screen digitization from CHM
- **Minimum CDW Size**: 2m length, 0.3m diameter
- **Geometry**: LineString centerlines along fallen logs
- **Buffer**: 0.5m applied during training to create 1m width masks

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{lamapuit_cdw_2025,
  author       = {Taavi Pipar},
  title        = {CDW Detection Dataset - Lamapuit Forest LiDAR},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

## License

This dataset is licensed under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

You are free to:
- Share and redistribute
- Adapt and build upon

Under the following terms:
- Attribution required
"""
    
    output_path = "examples/data/README.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✓ Created {output_path}")


def main():
    """Create all sample data files."""
    
    print("=" * 70)
    print("Creating Sample Data for GitHub Repository")
    print("=" * 70)
    
    # Check if input files exist
    if not os.path.exists("merged041225.tif"):
        print("❌ Error: merged041225.tif not found!")
        print("   Place the CHM raster in the current directory first.")
        return
    
    if not os.path.exists("lamapuit.gpkg"):
        print("❌ Error: lamapuit.gpkg not found!")
        print("   Place the label GeoPackage in the current directory first.")
        return
    
    try:
        # Create sample CHM tile
        create_sample_chm(
            input_raster="merged041225.tif",
            output_raster="examples/data/sample_chm_tile.tif",
            tile_size=500,  # 100x100m at 0.2m resolution
            start_x=1000,
            start_y=1000
        )
        
        # Copy sample labels
        copy_sample_labels(
            input_gpkg="lamapuit.gpkg",
            output_gpkg="examples/data/lamapuit_labels.gpkg"
        )
        
        # Create README
        create_data_readme()
        
        print("\n" + "=" * 70)
        print("✓ SUCCESS - Sample data created!")
        print("=" * 70)
        print("\nFiles created in examples/data/:")
        print("  - sample_chm_tile.tif (~5-10 MB)")
        print("  - lamapuit_labels.gpkg (~0.14 MB)")
        print("  - README.md (documentation)")
        print("\nTotal size: ~10-15 MB (safe for Git commit)")
        print("\nNext steps:")
        print("  1. Review files in examples/data/")
        print("  2. Update examples/data/README.md with actual Zenodo DOI")
        print("  3. Add to Git: git add examples/")
        print("  4. Commit: git commit -m 'Add sample data for testing'")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

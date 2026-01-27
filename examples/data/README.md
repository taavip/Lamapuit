# Example Data

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
python scripts/prepare_data.py \
  --raster examples/data/sample_chm_tile.tif \
  --labels examples/data/lamapuit_labels.gpkg \
  --output ./test_dataset \
  --buffer-width 0.5

# Train on sample (will be very limited due to small area)
python scripts/train_model.py \
  --config configs/default.yaml \
  --data ./test_dataset/data.yaml \
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

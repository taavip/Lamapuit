# LAZ Classifier Module

This module trains an open-source point classifier from LAZ files using:
- Geometry: `x,y,z`
- LiDAR radiometry and returns: `intensity`, `return_number`, `number_of_returns`
- Spectral channels when available: `red`, `green`, `blue`, `nir`
- Acquisition fields when available: `scan_angle`, `point_source_id`, `gps_time`
- Optional local-structure descriptors from neighborhoods (linearity, planarity, verticality, etc.)

## Why this design

- For dense multispectral ALS, recent Finland benchmark evidence shows point-transformer methods are strongest.
- For smaller training sets or sparser data, Random Forest remains robust and practical.
- This module provides a production-friendly RF baseline using all available fields and is easy to extend toward deep models later.

## Quick start

Train (example using LAS `classification` as label):

```bash
python -m cdw_detect.laz_classifier.cli train \
  --laz data/lamapuit/laz/436646_2018_madal.laz \
  --label-dim classification \
  --exclude-labels 0,7,18 \
  --max-points 200000 \
  --out-dir runs/laz_classifier_436646_2018 \
  --use-neighborhood-features \
  --knn 16 --radius-m 1.0
```

Predict class distribution on a new LAZ sample:

```bash
python -m cdw_detect.laz_classifier.cli predict \
  --laz data/lamapuit/laz/436646_2018_madal.laz \
  --model runs/laz_classifier_436646_2018/rf_laz_classifier.joblib \
  --max-points 200000 \
  --use-neighborhood-features \
  --knn 16 --radius-m 1.0
```

## Outputs

Training produces:
- `rf_laz_classifier.joblib` (model)
- `metrics.json` (report, confusion matrix, F1)
- `feature_importances.csv`
- `training_metadata.json`

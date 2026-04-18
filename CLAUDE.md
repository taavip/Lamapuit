# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**Lamapuit** — Master's thesis on detecting Coarse Woody Debris (CWD) from low-density airborne LiDAR (ALS) in Estonian forests.

**Core challenge**: Estonian ALS is 1–4 pts/m², too sparse for reliable CWD detection. The strategy is to produce high-quality CHMs and labeled masks from denser reference data, train on those, and generalize to low-density inputs.

**Active direction**: LAZ point-feature classification (`laz_classifier/`) and tile-level classification (`scripts/model_search_v3/`).

**Key bottleneck**: labeled training data. A custom browser-based brush segmentation tool is being built to annotate CWD vs background masks directly on CHM rasters — this is the next priority.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate cwd-detect    # env name differs from package name cdw-detect
pip install -e .
```

## Commands

### Tests
```bash
pytest                          # all tests with coverage (≥70% required)
pytest -m unit                  # unit tests only
pytest -m "not slow"            # skip slow tests
pytest tests/test_prepare.py    # single test file
pytest -x                       # stop on first failure
```

### Linting / formatting
```bash
ruff check src/ tests/
black src/ tests/
```

Line length: 100 characters (configured in pyproject.toml).

### Core pipeline
```bash
# 0. LAZ → CHM (harmonized pipeline preferred)
python experiments/laz_to_chm_harmonized_0p8m/build_dataset.py

# Or legacy:
python scripts/process_laz_to_chm.py --input points.laz --output chm.tif --resolution 0.2

# 1. Prepare training tiles
python scripts/prepare_data.py --chm chm.tif --labels lamapuit.gpkg --output data/dataset

# 2. Train
python scripts/train_model.py --data data/dataset/dataset.yaml

# 3. Detect
python scripts/run_detection.py --chm chm.tif --model runs/.../best.pt --output detections.gpkg
```

### CLI entry points (after `pip install -e .`)
```bash
cdw-detect           # src/cdw_detect/cli.py
cdw-laz-classifier   # src/cdw_detect/laz_classifier/cli.py
```

## Architecture

```
src/cdw_detect/
  __init__.py               # exports YOLODataPreparer, CDWDetector
  prepare.py                # YOLODataPreparer: CHM + line vectors → tiled dataset
  detect.py                 # CDWDetector: sliding-window inference → georef'd GeoPackage
  train.py                  # Training wrapper
  dataset_tiles.py          # Tile/window helpers
  laz_classifier/           # Sub-package: Random Forest classifier on LAZ point features
    features.py             # LiDAR feature extraction
    rf.py                   # sklearn RF training/prediction
    cli.py                  # cdw-laz-classifier entry point
  wms_utils.py              # WMS tile fetching helpers

scripts/
  model_search_v3/          # Latest model selection experiments (tile classifier)
  process_laz_to_chm.py     # LAZ → CHM (clip or drop HAG filter, two modes)
  prepare_data.py
  train_model.py
  finetune_model.py
  run_detection.py
  cleanup_memory.py

experiments/laz_to_chm_harmonized_0p8m/
  build_dataset.py          # Harmonized DEM + raw/Gaussian-smoothed CHM at 0.8 m
```

## CHM Pipelines

Two CHM pipelines — prefer the harmonized one for new work:

- **Legacy** (`scripts/process_laz_to_chm.py`) → `chm_max_hag_13_drop/` at 0.2 m; supports clip vs drop mode for HAG filtering
- **Harmonized** (`experiments/laz_to_chm_harmonized_0p8m/build_dataset.py`) → `laz_to_chm_harmonized_0p8m/` at 0.8 m; produces harmonized DEMs, raw CHM, and Gaussian-smoothed CHM

## Data Conventions

- CHM rasters: GeoTIFF, 0.2–0.8 m resolution, HAG 0–1.5 m, nodata = 0 or −9999
- Labels: `lamapuit.gpkg` in repo root — LineString geometries of manually annotated CWD centerlines
- Tiled dataset output: `data/` subdirectories (gitignored)
- Model weights: gitignored; canonical checkpoints documented in `models/MODEL_REGISTRY.md`

## What to Commit vs Ignore

**Commit**: `src/`, `scripts/`, `LaTeX/`, `labeler/`, `configs/`, `docs/`, `tests/`, `lamapuit.gpkg`, summary CSVs/YAMLs at repo root, `Dockerfile*`, `docker-compose.*.yml`, `environment.yml`, `pyproject.toml`

**Do NOT commit**: `*.laz`, `*.las`, `*.tif` (except tiny test fixtures in `tests/fixtures/`), `output/`, `runs/`, `experiments/` outputs, `data/`, large model weights, `*_tmp/`, `htmlcov/`, `.pytest_cache/`

## Planned: Brush Segmentation Labeling Tool

A custom web tool for annotating binary masks (CWD vs background) on CHM rasters:

- **Backend**: FastAPI — serves CHM tiles as PNGs, saves masks, runs model inference for pre-annotation
- **Frontend**: Leaflet.js + canvas brush (adjustable size, undo/redo)
- **Layers**: raw CHM, Gaussian-smoothed CHM, WMS orthophoto, model confidence heatmap, neighboring tile context, multi-year switcher, same SLD symbology as QGIS
- **Mask format**: single-channel PNG (0 = background, 1 = CWD, 255 = ignore)

## Content to Remove (off-topic)

- `labelstudio_sam_demo/` — unrelated to thesis
- `UrbanCar_LiDAR_Dataset_Report.md` — separate urban car LiDAR project
- `export_coco-instance_Pealik_pen_coin_v*.json` — unrelated annotation exports
- `configs/label_studio/car_polygon_label_config.xml`
- Session-note `.md` files at repo root — distill key findings into thesis or `docs/`, then delete
- `*_tmp/` directories (kpconv_tmp, myria3d_tmp, openpcseg_tmp, randla_tmp)

## Testing Notes

- `tests/conftest.py` — shared fixtures (synthetic rasters, tmp dirs)
- `tests/fixtures/` — small sample rasters and vectors for unit tests
- Coverage measured on `src/cdw_detect` only; 70% minimum enforced
- Mark slow/integration tests with `@pytest.mark.slow` / `@pytest.mark.integration`

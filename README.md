# Lamapuit — CWD Detection from Airborne LiDAR

**Master's thesis project** — Detecting Coarse Woody Debris (CWD) in Estonian forests from low-density airborne LiDAR (ALS) using machine learning.

## Research Challenge

Estonian ALS data are low-density (typically 1–4 pts/m²), making direct CWD detection unreliable. The approach:

1. Generate high-quality labeled CHMs from denser reference data
2. Train segmentation models on manually curated CWD masks
3. Simulate low-density conditions to improve generalization across scan densities

The **key bottleneck** is labeled training data — a custom brush segmentation tool is being built to accelerate mask annotation directly on CHM rasters.

## Repository Contents

| Path | Contents |
|---|---|
| `LaTeX/` | Thesis source (main write-up) |
| `src/cdw_detect/` | Core Python package: data preparation, training, detection, LAZ classifier |
| `scripts/` | Pipeline and experiment scripts |
| `scripts/model_search_v3/` | Latest model selection experiments (CWD / not-CWD tile classification) |
| `labeler/` | Manual tile review and label curation utilities |
| `configs/` | YAML/XML configuration files |
| `docs/` | Stable documentation and experiment runbooks |
| `examples/` | Small sample data for quick-start testing |

## CHM Generation Pipelines

Two pipelines produce Canopy Height Models from LAZ:

- **Legacy pipeline** (`scripts/process_laz_to_chm.py`) — produces CHMs in `chm_max_hag_13_drop/`, clips or drops points above HAG threshold
- **Harmonized pipeline** (`experiments/laz_to_chm_harmonized_0p8m/`) — produces harmonized DEMs + both raw and Gaussian-smoothed CHMs at 0.8 m resolution; this is the current best pipeline

## Installation

```bash
conda env create -f environment.yml
conda activate cwd-detect
pip install -e .
```

## Workflow

### 1. Convert LAZ → CHM

```bash
python scripts/process_laz_to_chm.py \
  --input points.laz \
  --output chm.tif \
  --resolution 0.8 \
  --drop-above-hag-max
```

Requirements: LAZ must have ground classification (class 2) from SMRF or similar.

### 2. Prepare Training Data

```python
from cdw_detect import YOLODataPreparer

preparer = YOLODataPreparer(
    output_dir="data/dataset",
    buffer_width=0.5,
)
stats = preparer.prepare(
    chm_path="path/to/chm.tif",
    labels_path="lamapuit.gpkg",
)
```

### 3. Train and Detect

```bash
python scripts/train_model.py --data data/dataset/dataset.yaml --epochs 50
python scripts/run_detection.py --chm path/to/chm.tif --model runs/.../best.pt --output detections.gpkg
```

## Key Scripts

| Script | Purpose |
|---|---|
| `scripts/process_laz_to_chm.py` | LAZ → CHM GeoTIFF (clip or drop HAG filter) |
| `scripts/prepare_data.py` | CHM + line labels → training dataset |
| `scripts/train_model.py` | Train segmentation model |
| `scripts/finetune_model.py` | Fine-tune from existing checkpoint |
| `scripts/run_detection.py` | Run CWD detection on CHM rasters |
| `scripts/cleanup_memory.py` | Clear GPU/CPU memory, kill lingering processes |
| `scripts/model_search_v3/` | Model selection experiments (tile classifier) |

## Data

### Input CHM
- GeoTIFF, 0.2–0.8 m resolution
- Height Above Ground (HAG) values, clipped to 0–1.5 m
- Nodata: 0 or −9999

### Training Labels (`lamapuit.gpkg`)
- LineString geometries representing CWD centerlines
- Same CRS as the CHM raster
- Committed to this repository

Large outputs (`output/`, `runs/`, `experiments/`, `data/`) are not tracked in Git — see `models/MODEL_REGISTRY.md` for the list of trained checkpoints.

## Project Structure

```
Lamapuit/
├── src/cdw_detect/            # Core package
│   ├── prepare.py             # Training data preparation
│   ├── train.py               # Model training wrapper
│   ├── detect.py              # Sliding-window inference → GeoPackage
│   ├── laz_classifier/        # Random Forest classifier on LAZ point features
│   └── wms_utils.py           # WMS tile fetching
├── scripts/                   # Pipeline and experiment scripts
│   └── model_search_v3/       # Latest model search
├── LaTeX/                     # Thesis source
├── labeler/                   # Tile review and label curation
├── configs/                   # YAML and XML configs
├── docs/                      # Documentation and runbooks
├── examples/                  # Small sample CHM tile + labels
├── tests/                     # pytest suite
├── environment.yml            # Conda environment
├── pyproject.toml             # Package metadata
└── lamapuit.gpkg              # CWD line labels
```

## Experiment Tracking

- `experiments_results.yaml` — consolidated results
- `experiments_progress.yaml` — in-progress tracking
- `massive_multirun_all_runs.csv` / `massive_multirun_statistics.csv` — multi-run summaries
- `docs/` — per-experiment runbooks and analysis notes

## CLI Entry Points

After `pip install -e .`:

```bash
cdw-detect           # main detection CLI
cdw-laz-classifier   # LAZ point-feature RF classifier
```

## Troubleshooting

### Memory errors

```bash
python scripts/cleanup_memory.py
python scripts/train_model.py --data dataset.yaml --batch 2 --imgsz 512
```

## Citation

```bibtex
@software{lamapuit_2026,
  title  = {Lamapuit: Coarse Woody Debris Detection from Airborne LiDAR},
  author = {Taavi Pipar},
  year   = {2026},
  url    = {https://github.com/taavip/cdw-detect}
}
```

## License

MIT License — see LICENSE file for details.

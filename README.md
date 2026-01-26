# CDW-Detect

**Coarse Woody Debris Detection from LiDAR using Deep Learning**

Detects fallen logs (coarse woody debris) in LiDAR-derived Canopy Height Model (CHM) rasters using YOLO11 instance segmentation.

## Features

- 🎯 YOLO11n-seg instance segmentation for CDW detection
- 📊 Training data preparation from vector line annotations
- 🗺️ Georeferenced GeoPackage output
- 🔄 Nodata-robust augmentation for LiDAR edge artifacts

## Installation

```bash
# Clone repository
git clone https://github.com/taavip/cdw-detect.git
cd cdw-detect

# Create conda environment
conda env create -f environment.yml
conda activate cdw-detect

# Install package
pip install -e .
```

## Quick Start

### 0. (Optional) Convert LAZ to CHM Raster

If you have raw LiDAR LAZ files, convert them to CHM first:

```bash
python scripts/process_laz_to_chm.py --input points.laz --output chm.tif --resolution 0.2
```

**Requirements**: LAZ file must have ground classification (class 2) from SMRF or similar algorithm.

### 1. Prepare Training Data

```python
from cdw_detect import YOLODataPreparer

preparer = YOLODataPreparer(
    output_dir='data/yolo_dataset',
    buffer_width=0.5,  # 1m total width for CDW lines
)

stats = preparer.prepare(
    chm_path='path/to/chm.tif',
    labels_path='path/to/labels.gpkg',  # LineString geometries
)
print(f"Created {stats['total']} tiles ({stats['with_cdw']} with CDW)")
```

### 2. Train Model

```python
from cdw_detect.train import train

best_model = train(
    dataset_yaml='data/yolo_dataset/dataset.yaml',
    epochs=50,
    device='cpu',  # or '0' for GPU
)
```

### 3. Run Detection

```python
from cdw_detect import CDWDetector

detector = CDWDetector(
    model_path='runs/cdw_detect/train/weights/best.pt',
    confidence=0.15,
)

detections = detector.detect_to_vector(
    raster_path='path/to/new_chm.tif',
    output_path='detections.gpkg',
)
print(f"Found {len(detections)} CDW features")
```

## Data Requirements

### Input CHM
- GeoTIFF format
- 0.2m resolution recommended
- Height Above Ground (HAG) values 0-1.5m
- Can be created from LAZ using `scripts/process_laz_to_chm.py`

### Training CHM Files (Included via Git LFS)
- **Location**: `visuals/chm_max_hag/`
- **~25 files** of 10-20 MB each (~350 MB total)
- Ready-to-use for creating training datasets
- See [docs/TRAINING_DATA_SETUP.md](docs/TRAINING_DATA_SETUP.md) for details

### Training Labels
- GeoPackage with LineString geometries
- Same CRS as CHM raster
- Lines representing log centerlines

## Scripts

| Script | Purpose |
|--------|---------|
| `process_laz_to_chm.py` | Convert LAZ LiDAR files to CHM GeoTIFF rasters |
| `prepare_data.py` | Create YOLO training dataset from CHM + labels |
| `train_model.py` | Train YOLO11n-seg model |
| `run_detection.py` | Run CDW detection on new CHM rasters |
| `cleanup_memory.py` | Clear memory and kill lingering processes |

## Troubleshooting

### Memory Errors During Training

If you see `RuntimeError: not enough memory` during training on CPU:

1. **Run cleanup script first:**
   ```bash
   python scripts/cleanup_memory.py
   ```

2. **Reduce batch size:**
   ```bash
   python scripts/train_model.py --data dataset.yaml --batch 2  # or even --batch 1
   ```

3. **Use smaller image size:**
   ```bash
   python scripts/train_model.py --data dataset.yaml --imgsz 512
   ```

4. **The package automatically disables AMP on CPU** (saves memory)

### Lingering Processes

If detection fails with import errors after training, kill old Python processes:
```bash
python scripts/cleanup_memory.py
# Answer 'y' to kill processes
```

## Project Structure

```
cdw-detect/
├── src/cdw_detect/        # Main package
│   ├── prepare.py         # Training data preparation
│   ├── train.py           # Model training
│   └── detect.py          # Inference
├── scripts/               # CLI tools
│   ├── process_laz_to_chm.py
│   ├── prepare_data.py
│   ├── train_model.py
│   └── run_detection.py
├── docs/                  # Documentation
│   ├── presentations/     # Project presentations
│   └── references/        # Scientific references
├── examples/              # Usage examples
├── configs/               # Configuration files
├── environment.yml        # Conda dependencies
└── pyproject.toml         # Package metadata
```

## 📚 Documentation

- **Presentations**: See [docs/presentations/cdw_detection_overview.pptx](docs/presentations/cdw_detection_overview.pptx) for project overview
- **References**: [Joyce et al. 2019 - CDW LiDAR Detection](docs/references/Joyce_et_al_2019_CDW_LiDAR.pdf)
- **Guides**:
  - [MEMORY_FIXES.md](MEMORY_FIXES.md) - Memory optimization troubleshooting
  - [LAZ_PROCESSING_INTEGRATION.md](LAZ_PROCESSING_INTEGRATION.md) - LAZ to CHM workflow
- **Examples**: See [examples/](examples/) folder for quick start tutorials

## 📦 Model Weights

Pre-trained model is not included in repository due to file size (11.4MB).

**Download from GitHub Releases**:
```bash
# Using wget
wget https://github.com/taavip/cdw-detect/releases/download/v1.0.0/cdw_detect_v1.0.0.pt -O models/best.pt

# Using curl
curl -LO https://github.com/taavip/cdw-detect/releases/download/v1.0.0/cdw_detect_v1.0.0.pt
```

Or **train your own model**:
```bash
python scripts/train_model.py --config configs/default.yaml --data ./yolo_dataset/data.yaml --epochs 50
```

## Citation

If you use this code, please cite:

```bibtex
@software{cdw_detect_2026,
  title = {CDW-Detect: Coarse Woody Debris Detection from LiDAR},
  author = {Taavi Pipar},
  year = {2026},
  url = {https://github.com/taavip/cdw-detect}
}
```

**Reference methodology**:
> Joyce, M. J., Erb, J. D., & Sampson, B. A. (2019). Detection of coarse woody debris using airborne light detection and ranging (LiDAR). *Forest Science*, 65(6), 711-724.

## License

MIT License - see LICENSE file for details.

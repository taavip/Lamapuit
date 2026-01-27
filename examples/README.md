# Examples

Quick start examples for the CDW Detection package.

## Directory Contents

This folder can contain:
- **Small sample datasets** (<1MB) for testing the pipeline
- **Example configuration files** for different scenarios
- **Tutorial notebooks** demonstrating the workflow

## Quick Start Example

### 1. Process LAZ to CHM
```bash
python scripts/process_laz_to_chm.py \
  --input /path/to/your/lidar.laz \
  --output /path/to/output_chm.tif \
  --resolution 0.2 \
  --method max
```

### 2. Prepare Training Data
```bash
python scripts/prepare_data.py \
  --raster /path/to/chm.tif \
  --labels /path/to/labels.gpkg \
  --output ./yolo_dataset \
  --buffer-width 0.5 \
  --tile-size 640
```

### 3. Train Model
```bash
python scripts/train_model.py \
  --config configs/default.yaml \
  --data ./yolo_dataset/data.yaml \
  --epochs 50
```

### 4. Run Detection
```bash
python scripts/run_detection.py \
  --raster /path/to/chm.tif \
  --model ./runs/cdw_detect/*/weights/best.pt \
  --output ./detections.gpkg \
  --conf-threshold 0.25
```

## Programmatic Usage

### Data Preparation
```python
from src.cdw_detect import YOLODataPreparer

preparer = YOLODataPreparer(
    raster_path='chm.tif',
    labels_path='labels.gpkg',
    output_dir='./yolo_dataset',
    buffer_width=0.5
)
preparer.prepare()
```

### Detection
```python
from src.cdw_detect import CDWDetector

detector = CDWDetector(
    model_path='best.pt',
    conf_threshold=0.25
)

results = detector.detect_to_vector(
    raster_path='chm.tif',
    output_path='detections.gpkg'
)
print(f"Detected {len(results)} CDW features")
```

## Sample Data

Due to file size constraints, sample data is not included in this repository.

To create your own test dataset:
1. Obtain LiDAR data (.laz/.las) covering a forested area
2. Process to CHM using `process_laz_to_chm.py`
3. Manually label some CDW features in QGIS
4. Use `prepare_data.py` to create YOLO dataset

Alternatively, download sample data from [link to data repository].

## Configuration Examples

See `configs/default.yaml` for a complete configuration template.

### Low Memory Training
```yaml
# config_low_memory.yaml
epochs: 30
batch: 2
imgsz: 320
patience: 10
```

### High Precision Detection
```yaml
# config_high_precision.yaml
conf_threshold: 0.35  # Higher confidence
iou_threshold: 0.3    # Stricter NMS
overlap: 0.3          # More overlap for edge cases
```

## Troubleshooting

See [MEMORY_FIXES.md](../MEMORY_FIXES.md) for memory optimization tips.

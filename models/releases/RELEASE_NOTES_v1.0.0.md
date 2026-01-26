# CDW Detection v1.0.0

**Release Date**: January 25, 2026  
**Model**: YOLO11n-seg for Coarse Woody Debris detection from LiDAR CHM

---

## 🎯 Model Details

- **Architecture**: YOLO11n-seg (Ultralytics)
- **Parameters**: 2.8M
- **Input Size**: 640x640 pixels
- **Task**: Instance segmentation
- **Training Device**: CPU-optimized

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Box mAP50 | 11.35% |
| Mask mAP50 | 8.89% |
| Training Epochs | 50 |
| Training Time | 8-10 hours (CPU) |
| Detection Speed | 2-3 sec/tile |

---

## 🗂️ Dataset

- **Training Images**: 448
- **Validation Images**: 112
- **Augmentation**: 30% nodata patches for edge robustness
- **Buffer Width**: 0.5m (creates 1m total CDW width)
- **Resolution**: 0.2m per pixel
- **Source**: Lamapuit forest, Estonia (LiDAR-derived CHM)

---

## ⚙️ Training Configuration

```yaml
epochs: 50
batch_size: 4
optimizer: SGD
learning_rate: 0.01
learning_rate_final: 0.0001
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
patience: 15
amp: false  # Disabled for CPU
cache: disk
```

---

## 📦 Download

### Model Weights
```bash
# Using wget
wget https://github.com/taavip/cdw-detect/releases/download/v1.0.0/cdw_detect_v1.0.0.pt

# Using curl
curl -LO https://github.com/taavip/cdw-detect/releases/download/v1.0.0/cdw_detect_v1.0.0.pt
```

**File**: `cdw_detect_v1.0.0.pt` (5.72 MB)  
**SHA256**: `[Generate after upload]`

### Metadata
```bash
wget https://github.com/taavip/cdw-detect/releases/download/v1.0.0/cdw_detect_v1.0.0_info.yaml
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/taavip/cdw-detect.git
cd cdw-detect
conda env create -f environment.yml
conda activate cdw-detect
pip install -e .
```

### Download Model
```bash
wget https://github.com/taavip/cdw-detect/releases/download/v1.0.0/cdw_detect_v1.0.0.pt
```

### Run Detection
```bash
python scripts/run_detection.py \
  --model cdw_detect_v1.0.0.pt \
  --raster your_chm.tif \
  --output detections.gpkg \
  --conf-threshold 0.25
```

### Programmatic Usage
```python
from src.cdw_detect import CDWDetector

detector = CDWDetector(
    model_path='cdw_detect_v1.0.0.pt',
    conf_threshold=0.25
)

results = detector.detect_to_vector(
    raster_path='your_chm.tif',
    output_path='detections.gpkg'
)

print(f"Detected {len(results)} CDW features")
```

---

## 🎯 Use Cases

- ✅ Forest inventory - automated CDW mapping
- ✅ LiDAR analysis - 0.2m resolution CHM rasters
- ✅ Research baseline - reproducible CDW detection
- ✅ Production deployment - CPU-optimized inference

---

## ⚠️ Known Limitations

1. **Moderate Precision**: Detection performance is modest (mAP50~11%), suitable for initial screening but may require manual review
2. **Nodata Sensitivity**: Model can be affected by raster edge artifacts (mitigated with 30% nodata augmentation)
3. **CPU-Only Training**: Trained without GPU acceleration (limits model complexity)
4. **Small Objects**: May miss very small or partially obscured CDW (<2m length)

---

## 📋 Requirements

### Software
- Python 3.8+
- PyTorch 2.0+
- Ultralytics 8.4.0+
- GDAL 3.0+
- GeoPandas 0.10+

### Hardware (Minimum)
- **RAM**: 8 GB
- **Storage**: 1 GB (model + dependencies)
- **CPU**: Multi-core recommended
- **GPU**: Optional (CPU inference supported)

### Input Data
- LiDAR-derived Canopy Height Model (CHM)
- GeoTIFF format
- Resolution: 0.1-0.5m (optimized for 0.2m)
- Ground-classified point cloud (SMRF or similar)

---

## 🔄 Changes from Previous Versions

This is the initial release (v1.0.0).

---

## 📚 References

This model is based on methodology from:

> Joyce, M. J., Erb, J. D., & Sampson, B. A. (2019). Detection of coarse woody debris using airborne light detection and ranging (LiDAR). *Forest Science*, 65(6), 711-724.

---

## 📜 License

This model is released under the **MIT License**.

You are free to:
- Use commercially
- Modify and distribute
- Use privately

See [LICENSE](../LICENSE) for full terms.

---

## 🙏 Citation

If you use this model in your research, please cite:

```bibtex
@software{cdw_detect_v1_2026,
  title = {CDW-Detect: Coarse Woody Debris Detection Model v1.0.0},
  author = {Taavi Pipar},
  year = {2026},
  month = {1},
  version = {1.0.0},
  url = {https://github.com/taavip/cdw-detect/releases/tag/v1.0.0},
  note = {YOLO11n-seg model for LiDAR-based CDW detection}
}
```

---

## 🐛 Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/taavip/cdw-detect/issues)
- **Documentation**: [README.md](../README.md)
- **Discussions**: [GitHub Discussions](https://github.com/taavip/cdw-detect/discussions)

---

## 🔗 Related Resources

- **Source Code**: [GitHub Repository](https://github.com/taavip/cdw-detect)
- **Documentation**: [Full Documentation](https://github.com/taavip/cdw-detect#readme)
- **Training Data**: [Zenodo Dataset](https://doi.org/10.5281/zenodo.XXXXXXX)
- **Model Registry**: [All Versions](../MODEL_REGISTRY.md)

---

**Thank you for using CDW-Detect!** 🌲

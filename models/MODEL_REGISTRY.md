# CDW Detection Model Registry

This file tracks all released versions of the CDW detection model.

## Model Versions

| Version | Date | mAP50(B) | mAP50(M) | Epochs | Size | Notes | Download |
|---------|------|----------|----------|--------|------|-------|----------|
| v1.0.0 | 2026-01-25 | 11.35% | 8.89% | 50 | 5.72 MB | Initial release | [GitHub Release](https://github.com/user/cdw-detect/releases/v1.0.0) |

## Version Details

### v1.0.0 (Current)
**Release Date**: 2026-01-25  
**Training Run**: `cdw_lamapuit_robust`

**Architecture**:
- Model: YOLO11n-seg
- Parameters: 2.8M
- Input Size: 640x640
- Device: CPU-optimized

**Dataset**:
- Images: 448 (train), 112 (val)
- Augmentation: 30% nodata patches
- Buffer Width: 0.5m (1m total CDW width)
- Resolution: 0.2m per pixel

**Training Configuration**:
- Epochs: 50
- Batch Size: 4
- Optimizer: SGD
- Learning Rate: 0.01
- AMP: Disabled (CPU)
- Cache: Disk

**Performance Metrics**:
- Box mAP50: 11.35%
- Mask mAP50: 8.89%
- Training Time: ~8-10 hours (CPU)
- Detection Speed: ~2-3 sec/tile

**Files**:
- Model: `cdw_detect_v1.0.0.pt` (5.72 MB)
- Metadata: `cdw_detect_v1.0.0_info.yaml`
- Training Logs: `runs/cdw_detect/cdw_lamapuit_robust/`

**Known Limitations**:
- Moderate precision on small/obscured CDW
- Sensitive to nodata edges (mitigated by augmentation)
- CPU-only training (no GPU optimization)

**Use Cases**:
- Forest inventory CDW detection
- LiDAR-derived CHM analysis (0.2m resolution)
- Research and development baseline

---

## How to Use

### Download Model
```bash
# From GitHub Release
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

### Load Programmatically
```python
from src.cdw_detect import CDWDetector

detector = CDWDetector(
    model_path='cdw_detect_v1.0.0.pt',
    conf_threshold=0.25
)

results = detector.detect_to_vector(
    raster_path='chm.tif',
    output_path='detections.gpkg'
)
```

---

## Release Process

1. **Train/Fine-tune Model**
   ```bash
   python scripts/train_model.py --config configs/default.yaml
   ```

2. **Copy to Releases**
   ```bash
   cp runs/cdw_detect/[run_name]/weights/best.pt models/releases/cdw_detect_vX.X.X.pt
   ```

3. **Create Version Info**
   ```bash
   # Create version_info.yaml with metrics
   ```

4. **Update Registry**
   - Add entry to table above
   - Document performance and changes

5. **Git Tag**
   ```bash
   git tag -a vX.X.X -m "Release vX.X.X: Description"
   git push origin vX.X.X
   ```

6. **GitHub Release**
   ```bash
   gh release create vX.X.X models/releases/cdw_detect_vX.X.X.pt \
     models/releases/cdw_detect_vX.X.X_info.yaml \
     --title "CDW Detection vX.X.X" \
     --notes-file models/releases/RELEASE_NOTES_vX.X.X.md
   ```

---

## Citation

If you use these models in your research, please cite:

```bibtex
@software{cdw_detect_2026,
  title = {CDW-Detect: Coarse Woody Debris Detection from LiDAR},
  author = {Taavi Pipar},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/taavip/cdw-detect},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

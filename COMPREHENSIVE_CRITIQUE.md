# Comprehensive Project Critique & Improvement Roadmap
**Date**: January 28, 2026  
**Reviewer**: Complete System Analysis  
**Project**: CDW-Detect (Coarse Woody Debris Detection using YOLO)

---

## 📊 Executive Summary

### Overall Grade: B (70/100)
- **Code Quality**: B+ (Well-structured package, good separation of concerns)
- **Documentation**: B+ (Comprehensive but some inconsistencies)
- **Testing**: D (No automated tests, no CI/CD)
- **ML Pipeline**: C+ (Works but severe data limitations)
- **Production Readiness**: B- (Functional but needs hardening)

### Critical Finding
**🚨 PRIMARY BOTTLENECK: Dataset size (~33 training samples) makes all ML improvements futile**
- Current performance: mAP50 = 0.11 ± 0.08 (79% CV)
- 77% overfitting across 9 independent runs
- No amount of model tuning, regularization, or architecture changes can overcome this

---

## 🎯 Critical Issues (Must Fix)

### 1. **DATASET SIZE IS THE SHOW-STOPPER** 🚨
**Status**: ❌ Critical
**Impact**: Makes all model development efforts ineffective

**Current State**:
- Train: 33 samples
- Val: 14 samples
- Result: 79% coefficient of variation (unstable)
- Result: 77% mean overfitting (severe memorization)

**Evidence from 9-run experiment**:
```
mAP50:      0.1067 ± 0.0848 (79% CV)
Overfitting: 76.83% ± 26.57%
6/9 runs stopped at epochs 7-24 (zero predictions)
```

**Impact**:
- Models cannot learn robust features
- Random initialization dominates results
- Impossible to distinguish good/bad architectural choices
- Cannot deploy with confidence

**Solution** (HIGHEST PRIORITY):
1. **Immediate**: Expand to 500+ training samples
2. **Short-term**: Target 1000+ samples for production
3. **Method**: Semi-automated labeling pipeline (see Section 8.1)

---

### 2. **No Automated Testing** ❌
**Status**: Critical for production
**Impact**: High - Cannot verify correctness, regression bugs likely

**Missing**:
- No `tests/` directory
- No unit tests
- No integration tests
- No CI/CD pipeline

**Risk Examples**:
```python
# src/cdw_detect/prepare.py - No input validation
def prepare(self, chm_path: str, labels_path: str) -> dict:
    labels_gdf = gpd.read_file(labels_path)  # What if file doesn't exist?
    # What if CRS conversion fails?
    # What if geometries are invalid?
```

**Required Tests**:
1. **Unit Tests**:
   - Data loading/validation
   - Coordinate transformations
   - NMS algorithm correctness
   - Edge cases (empty images, nodata, overlaps)

2. **Integration Tests**:
   - Full pipeline end-to-end
   - Small dataset training
   - Detection on synthetic data

3. **Regression Tests**:
   - Model performance benchmarks
   - Output format consistency

**Implementation**:
```bash
tests/
├── test_prepare.py
├── test_train.py
├── test_detect.py
├── test_integration.py
├── fixtures/
│   ├── sample_chm.tif
│   └── sample_labels.gpkg
└── conftest.py
```

---

### 3. **No CI/CD Pipeline** ❌
**Status**: Important for collaboration
**Impact**: Medium - Manual testing burden, delayed feedback

**Missing**: `.github/workflows/`

**Required Workflows**:
```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: pytest tests/
      - name: Lint
        run: ruff check src/

# .github/workflows/package.yml
name: Build Package
on: release
  types: [published]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build
        run: python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

---

### 4. **Error Handling Gaps** ⚠️
**Status**: Important
**Impact**: Medium - Poor user experience, hard to debug

**Issues**:

```python
# src/cdw_detect/prepare.py
def prepare(self, chm_path: str, labels_path: str) -> dict:
    # No file existence check
    # No CRS validation
    # No geometry validity check
    
# src/cdw_detect/detect.py
def detect(self, chm_path: str, output_path: str = None):
    # No model path validation
    # No output directory creation
    # Silent failures possible
```

**Required Improvements**:
```python
def prepare(self, chm_path: str, labels_path: str) -> dict:
    """Prepare training data with robust error handling."""
    # Validate inputs
    if not Path(chm_path).exists():
        raise FileNotFoundError(f"CHM raster not found: {chm_path}")
    
    if not Path(labels_path).exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    
    try:
        labels_gdf = gpd.read_file(labels_path)
    except Exception as e:
        raise ValueError(f"Failed to read labels: {e}")
    
    if len(labels_gdf) == 0:
        raise ValueError("Labels GeoPackage is empty")
    
    # Validate geometries
    invalid = ~labels_gdf.geometry.is_valid
    if invalid.any():
        n_invalid = invalid.sum()
        raise ValueError(f"Found {n_invalid} invalid geometries in labels")
    
    # Continue with processing...
```

---

### 5. **Documentation Inconsistencies** ⚠️
**Status**: Important
**Impact**: Medium - Confuses users, reduces adoption

**Issues**:

A. **environment.yml missing dependencies**:
```yaml
# Current environment.yml
dependencies:
  - python=3.13
  - pdal, laspy  # For LAZ
  - numpy, pandas
  # ❌ Missing: geopandas, rasterio, opencv

# pyproject.toml requires:
"geopandas>=0.14",   # MISSING
"rasterio>=1.3",     # MISSING  
"opencv-python>=4.8", # MISSING
"ultralytics>=8.0",  # MISSING
```

**Fix**: Synchronize environment.yml with pyproject.toml

B. **README examples don't match API**:
```python
# README.md shows (WRONG):
detector.detect_to_vector(raster_path='...', output_path='...')

# Actual API (CORRECT):
detector.detect(chm_path='...', output_path='...')
```

C. **Placeholder values not replaced**:
- `YOUR_USERNAME` → should be `taavip`
- `Your Name` → should be actual name
- Zenodo DOI: `XXXXXXX` → pending

---

## 🔧 Important Improvements (Priority 2)

### 6. **Code Structure & Organization**

**Current State**: Good foundation but can improve

**Issues**:
1. **58 legacy files at project root** (clutters workspace)
2. **Duplicate functionality** across multiple scripts
3. **No logging framework** (print statements everywhere)

**Refactoring Recommendations**:

A. **Consolidate Training Scripts**:
```
scripts/
├── train_model.py              # Keep (simple API)
├── train_multirun.py           # Keep (variability analysis)
├── train_conservative.py       # ❌ REMOVE (superseded)
├── train_enhanced.py           # ❌ REMOVE (superseded)
├── train_ultimate.py           # ❌ REMOVE (superseded)
├── train_experiment.py         # ❌ REMOVE (superseded)
├── train_with_suggestions.py   # ❌ REMOVE (superseded)
└── experiment_runner.py        # ❌ REMOVE (superseded)
```

B. **Add Logging Framework**:
```python
# src/cdw_detect/utils/logging.py
import logging
import sys

def setup_logger(name: str, level=logging.INFO):
    """Setup consistent logging across modules."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Usage in modules:
from cdw_detect.utils.logging import setup_logger
logger = setup_logger(__name__)

logger.info("Starting detection...")
logger.warning("Low confidence detections")
logger.error("Failed to read raster")
```

C. **Add Configuration Management**:
```python
# src/cdw_detect/config.py
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

@dataclass
class PrepareConfig:
    """Configuration for data preparation."""
    tile_size: int = 640
    buffer_width: float = 0.5
    overlap: float = 0.2
    min_log_pixels: int = 50
    val_split: float = 0.2
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            return cls(**yaml.safe_load(f))
    
    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)

@dataclass
class TrainConfig:
    """Configuration for training."""
    model: str = 'yolo11n-seg.pt'
    epochs: int = 50
    batch: int = 4
    patience: int = 15
    # ... more params

@dataclass
class DetectConfig:
    """Configuration for detection."""
    tile_size: int = 640
    stride: int = 480
    confidence: float = 0.15
    # ... more params
```

---

### 7. **Performance & Resource Management**

**Issues**:
1. **No GPU memory monitoring**
2. **Sliding window could be more efficient**
3. **No batch processing for multiple files**

**Improvements**:

A. **GPU Memory Monitoring**:
```python
# src/cdw_detect/utils/gpu.py
import torch

def get_gpu_memory_info():
    """Get GPU memory usage information."""
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - reserved
    }

def log_gpu_memory(logger):
    """Log GPU memory usage."""
    info = get_gpu_memory_info()
    if info:
        logger.info(f"GPU Memory: {info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB")
```

B. **Batch Detection Script**:
```python
# scripts/batch_detect.py
"""
Process multiple CHM files in batch.

Usage:
    python scripts/batch_detect.py \\
        --input-dir chm_tiles/ \\
        --output-dir detections/ \\
        --model best.pt \\
        --pattern "*.tif"
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from cdw_detect import CDWDetector
from tqdm import tqdm

def process_single(args):
    chm_path, output_dir, model_path, confidence = args
    detector = CDWDetector(model_path=model_path, confidence=confidence)
    output_path = output_dir / f"{chm_path.stem}_detections.gpkg"
    return detector.detect(str(chm_path), str(output_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--pattern', default='*.tif')
    parser.add_argument('--confidence', type=float, default=0.15)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(input_dir.glob(args.pattern))
    print(f"Found {len(files)} CHM files")
    
    if args.workers == 1:
        # Serial processing
        for chm_path in tqdm(files):
            process_single((chm_path, output_dir, args.model, args.confidence))
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            tasks = [(f, output_dir, args.model, args.confidence) for f in files]
            list(tqdm(executor.map(process_single, tasks), total=len(files)))
    
    print(f"✓ Processed {len(files)} files")

if __name__ == '__main__':
    main()
```

---

## 🚀 Strategic Improvements (Priority 3)

### 8. **Data Pipeline & Labeling**

**8.1 Semi-Automated Labeling Pipeline**

**Problem**: Manual labeling is slow (current: 33 samples after months)

**Solution**: Active learning + human-in-the-loop
```python
# scripts/active_learning_pipeline.py
"""
Semi-automated labeling pipeline using active learning.

Workflow:
1. Train initial model on 33 labeled samples
2. Run detection on unlabeled CHM tiles
3. Sort by uncertainty (low confidence or overlapping detections)
4. Human reviews and corrects top-N uncertain cases
5. Retrain model, repeat

Can label 100-200 samples/day vs 5-10 manually.
"""

from cdw_detect import CDWDetector, YOLODataPreparer
import geopandas as gpd
from pathlib import Path

def run_active_learning_round(
    model_path: str,
    unlabeled_chms: list,
    output_dir: Path,
    top_n: int = 100
):
    """Run one round of active learning."""
    detector = CDWDetector(model_path=model_path, confidence=0.05)  # Low threshold
    
    uncertain_samples = []
    
    for chm in unlabeled_chms:
        detections = detector.detect(chm)
        
        # Calculate uncertainty metrics
        for idx, row in detections.iterrows():
            uncertainty = 1 - row['confidence']  # Simple uncertainty
            overlaps = count_overlaps(detections, row)  # Complex cases
            
            uncertain_samples.append({
                'chm': chm,
                'detection_id': idx,
                'geometry': row['geometry'],
                'confidence': row['confidence'],
                'uncertainty': uncertainty,
                'overlaps': overlaps,
                'score': uncertainty + 0.3 * overlaps  # Combined score
            })
    
    # Sort by uncertainty score
    uncertain_samples.sort(key=lambda x: x['score'], reverse=True)
    
    # Export top-N for human review
    review_gdf = gpd.GeoDataFrame(uncertain_samples[:top_n])
    review_gdf.to_file(output_dir / f"review_round_{round_num}.gpkg")
    
    print(f"Exported {top_n} uncertain samples for review")
    print(f"Human: Please review and correct in QGIS")
    print(f"Save corrected labels to: {output_dir / 'corrected.gpkg'}")
    
    return review_gdf

def count_overlaps(detections_gdf, detection_row):
    """Count how many other detections overlap with this one."""
    overlaps = detections_gdf[
        detections_gdf.geometry.intersects(detection_row.geometry)
    ]
    return len(overlaps) - 1  # Exclude self
```

**Expected Speedup**: 10-20x faster than pure manual labeling

**8.2 Data Augmentation for Small Datasets**

```python
# src/cdw_detect/augmentation.py
"""
Heavy augmentation for extremely small datasets.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_heavy_augmentation():
    """
    Augmentation pipeline for datasets with <100 samples.
    Creates synthetic diversity to prevent overfitting.
    """
    return A.Compose([
        # Geometric
        A.Rotate(limit=180, p=1.0),  # All rotations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.3,
            rotate_limit=180,
            p=0.8
        ),
        A.ElasticTransform(
            alpha=120,
            sigma=6,
            alpha_affine=6,
            p=0.3
        ),
        
        # Photometric
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.8
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        
        # Dropout augmentations (simulate missing data)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            p=0.5
        ),
        
        ToTensorV2(),
    ])
```

---

### 9. **Model Architecture Experiments** (AFTER getting more data)

**Current**: YOLOv11n-seg (2.8M parameters)

**Experiments to Try** (once dataset > 500 samples):

A. **Ensemble Methods**:
```python
# src/cdw_detect/ensemble.py
class EnsembleDetector:
    """
    Ensemble of multiple models for robust detection.
    Reduces variance, improves generalization.
    """
    def __init__(self, model_paths: list):
        self.models = [CDWDetector(path) for path in model_paths]
    
    def detect(self, chm_path: str):
        # Run all models
        all_detections = []
        for model in self.models:
            dets = model.detect(chm_path)
            all_detections.append(dets)
        
        # Voting/consensus fusion
        return self._consensus_fusion(all_detections)
    
    def _consensus_fusion(self, detections_list):
        """
        Fusion strategy: keep detection if 2+ models agree.
        """
        # Implement spatial clustering and voting
        pass
```

B. **Two-Stage Detection** (Region Proposal + Refinement):
```python
# Stage 1: Fast YOLO detection (low confidence)
# Stage 2: Refinement network on ROIs
class TwoStageDetector:
    """
    Stage 1: Fast detection with YOLOv11n-seg
    Stage 2: Refine detections with larger model or specialized network
    """
    def __init__(self, proposal_model, refinement_model):
        self.proposal = proposal_model
        self.refinement = refinement_model
    
    def detect(self, chm_path):
        # Stage 1: Get proposals
        proposals = self.proposal.detect(chm_path, confidence=0.05)
        
        # Stage 2: Refine each proposal
        refined = []
        for proposal in proposals:
            roi = self._extract_roi(chm_path, proposal)
            refined_det = self.refinement.detect(roi, confidence=0.25)
            refined.append(refined_det)
        
        return refined
```

C. **Attention Mechanisms** (for elongated objects):
```python
# Custom YOLO head with attention for CDW's elongated shape
# Focuses on aspect ratio and orientation

class OrientedDetectionHead:
    """
    Modified YOLO head that outputs:
    - Bounding box
    - Rotation angle
    - Aspect ratio confidence
    
    Better for elongated objects like fallen logs.
    """
    pass
```

---

### 10. **Deployment & Productionization**

**10.1 Model Serving API**

```python
# api/app.py
"""
FastAPI server for CDW detection as a service.

Usage:
    uvicorn api.app:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /detect - Upload CHM, get detections
    GET /status - Check server health
    GET /models - List available models
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import tempfile
from pathlib import Path
from cdw_detect import CDWDetector

app = FastAPI(title="CDW Detection API", version="1.0.0")

# Load model at startup
detector = None

@app.on_event("startup")
async def load_model():
    global detector
    model_path = "models/releases/cdw_detect_v1.0.0.pt"
    detector = CDWDetector(model_path=model_path, device='cuda')

@app.get("/status")
async def health_check():
    """Check API health."""
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "version": "1.0.0"
    }

@app.post("/detect")
async def detect_cdw(
    file: UploadFile = File(...),
    confidence: float = 0.15
):
    """
    Detect CDW in uploaded CHM raster.
    
    Returns:
        GeoJSON with detection polygons
    """
    if not file.filename.endswith('.tif'):
        raise HTTPException(400, "Only GeoTIFF files supported")
    
    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_chm:
        tmp_chm.write(await file.read())
        chm_path = tmp_chm.name
    
    # Run detection
    output_path = chm_path.replace('.tif', '_detections.gpkg')
    detections = detector.detect(chm_path, output_path)
    
    # Convert to GeoJSON
    geojson = detections.to_json()
    
    # Cleanup
    Path(chm_path).unlink()
    
    return {"detections": geojson, "count": len(detections)}

@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "id": "v1.0.0",
                "name": "CDW Detect v1.0.0",
                "metrics": {
                    "mAP50": 0.11,
                    "description": "Initial release, limited training data"
                }
            }
        ]
    }
```

**Docker Deployment**:
```dockerfile
# Dockerfile.api
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/releases/ models/releases/
COPY api/ api/

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Deploy
docker build -f Dockerfile.api -t cdw-detect-api .
docker run -d -p 8000:8000 --gpus all cdw-detect-api

# Test
curl -X POST -F "file=@test_chm.tif" http://localhost:8000/detect
```

**10.2 QGIS Plugin** (for end-users)

```python
# qgis_plugin/cdw_detect_plugin.py
"""
QGIS plugin for CDW detection.

Provides GUI for:
- Loading CHM rasters
- Running detection
- Visualizing results
- Exporting to GeoPackage
"""

from qgis.PyQt.QtWidgets import QAction, QFileDialog, QProgressDialog
from qgis.core import QgsVectorLayer, QgsProject

class CDWDetectPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.detector = None
    
    def initGui(self):
        """Initialize GUI elements."""
        self.action = QAction("Detect CDW", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
    
    def run(self):
        """Run CDW detection."""
        # Get CHM layer
        chm_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select CHM Raster",
            "",
            "GeoTIFF (*.tif)"
        )
        
        if not chm_path:
            return
        
        # Get model
        model_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Model",
            "",
            "YOLO Model (*.pt)"
        )
        
        if not model_path:
            return
        
        # Run detection with progress
        progress = QProgressDialog("Detecting CDW...", "Cancel", 0, 100)
        progress.show()
        
        from cdw_detect import CDWDetector
        detector = CDWDetector(model_path=model_path)
        output_path = chm_path.replace('.tif', '_detections.gpkg')
        
        detections = detector.detect(chm_path, output_path)
        
        # Load results
        layer = QgsVectorLayer(output_path, "CDW Detections", "ogr")
        QgsProject.instance().addMapLayer(layer)
        
        progress.close()
        self.iface.messageBar().pushSuccess(
            "CDW Detection",
            f"Found {len(detections)} CDW features"
        )
```

---

## 📈 Metrics & Monitoring

### 11. **Model Performance Tracking**

```python
# src/cdw_detect/metrics.py
"""
Comprehensive metrics for CDW detection evaluation.
"""

import numpy as np
import geopandas as gpd
from shapely.ops import unary_union

def calculate_detection_metrics(
    predictions: gpd.GeoDataFrame,
    ground_truth: gpd.GeoDataFrame,
    iou_threshold: float = 0.5
):
    """
    Calculate precision, recall, F1, mAP for CDW detection.
    
    Returns:
        dict with metrics
    """
    # Match predictions to ground truth
    matches = match_detections(predictions, ground_truth, iou_threshold)
    
    tp = matches['true_positives']
    fp = matches['false_positives']
    fn = matches['false_negatives']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }

def match_detections(preds, gt, iou_thresh):
    """Match predictions to ground truth using Hungarian algorithm."""
    # ... implementation
    pass
```

**Benchmark Suite**:
```python
# scripts/benchmark_model.py
"""
Benchmark model on standardized test set.

Tracks:
- Detection accuracy
- Inference speed
- Memory usage
- Model size
"""

def benchmark_model(model_path: str, test_data_dir: str):
    """Run full benchmark suite."""
    results = {
        'model': model_path,
        'timestamp': datetime.now(),
        'metrics': {},
        'performance': {},
        'resources': {}
    }
    
    # Accuracy metrics
    detector = CDWDetector(model_path=model_path)
    for test_chm in test_data_dir.glob('*.tif'):
        gt_path = test_chm.with_suffix('.gpkg')
        preds = detector.detect(test_chm)
        gt = gpd.read_file(gt_path)
        
        metrics = calculate_detection_metrics(preds, gt)
        results['metrics'][test_chm.name] = metrics
    
    # Speed benchmark
    import time
    start = time.time()
    for _ in range(10):
        _ = detector.detect(test_chm)
    elapsed = time.time() - start
    results['performance']['avg_inference_time'] = elapsed / 10
    
    # Memory usage
    import psutil
    process = psutil.Process()
    results['resources']['memory_mb'] = process.memory_info().rss / 1e6
    results['resources']['model_size_mb'] = Path(model_path).stat().st_size / 1e6
    
    return results
```

---

## 🛡️ Quality Assurance

### 12. **Input Validation & Sanity Checks**

```python
# src/cdw_detect/validation.py
"""
Input validation and sanity checks.
"""

import rasterio
import geopandas as gpd
from pathlib import Path

class InputValidator:
    """Validate inputs before processing."""
    
    @staticmethod
    def validate_chm_raster(chm_path: str) -> dict:
        """
        Validate CHM raster for CDW detection.
        
        Returns:
            dict with validation results and warnings
        """
        results = {'valid': True, 'warnings': [], 'errors': []}
        
        # Check file exists
        if not Path(chm_path).exists():
            results['valid'] = False
            results['errors'].append(f"File not found: {chm_path}")
            return results
        
        try:
            with rasterio.open(chm_path) as src:
                # Check CRS
                if src.crs is None:
                    results['warnings'].append("No CRS defined")
                
                # Check resolution
                res = src.res[0]
                if res > 0.5:
                    results['warnings'].append(
                        f"Resolution {res}m is coarse, recommended: 0.1-0.3m"
                    )
                
                # Check data range
                sample = src.read(1, window=Window(0, 0, 100, 100))
                if sample.max() > 5.0:
                    results['warnings'].append(
                        "Height values >5m detected, expected 0-1.5m for CDW"
                    )
                
                # Check nodata
                nodata_frac = np.isnan(sample).mean()
                if nodata_frac > 0.5:
                    results['warnings'].append(
                        f"{nodata_frac*100:.1f}% nodata pixels"
                    )
                
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Failed to read raster: {e}")
        
        return results
    
    @staticmethod
    def validate_labels(labels_path: str) -> dict:
        """Validate label GeoPackage."""
        results = {'valid': True, 'warnings': [], 'errors': []}
        
        try:
            gdf = gpd.read_file(labels_path)
            
            # Check geometry types
            if not all(gdf.geometry.type.isin(['LineString', 'Polygon'])):
                results['errors'].append(
                    "Only LineString or Polygon geometries supported"
                )
                results['valid'] = False
            
            # Check invalid geometries
            invalid = ~gdf.geometry.is_valid
            if invalid.any():
                n = invalid.sum()
                results['errors'].append(f"{n} invalid geometries found")
                results['valid'] = False
            
            # Check empty geometries
            empty = gdf.geometry.is_empty
            if empty.any():
                n = empty.sum()
                results['warnings'].append(f"{n} empty geometries (will skip)")
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Failed to read labels: {e}")
        
        return results
```

---

## 📚 Documentation Improvements

### 13. **API Documentation**

```python
# Use Sphinx for auto-generated API docs

# docs/conf.py
project = 'CDW-Detect'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google/NumPy docstrings
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
]

# Generate docs:
# cd docs
# sphinx-apidoc -o source ../src/cdw_detect
# make html
```

**Add comprehensive docstrings**:
```python
def detect(
    self,
    chm_path: str,
    output_path: str = None,
    confidence: float = None,
    min_area_m2: float = None
) -> gpd.GeoDataFrame:
    """
    Run CDW detection on a CHM raster using sliding window approach.
    
    The method:
    1. Slides a window across the raster (with overlap)
    2. Runs YOLO segmentation on each tile
    3. Converts detections to georeferenced polygons
    4. Applies Non-Maximum Suppression (NMS) to remove duplicates
    5. Filters by area and confidence
    
    Args:
        chm_path: Path to Canopy Height Model (CHM) GeoTIFF raster.
            Should contain height above ground values (0-1.5m for CDW).
            Resolution: 0.1-0.3m recommended.
        output_path: Optional path to save detections as GeoPackage.
            If None, results are returned but not saved.
        confidence: Override default confidence threshold (0.0-1.0).
            Lower = more detections but more false positives.
            Default: 0.15
        min_area_m2: Override minimum detection area in square meters.
            Smaller CDW pieces may be noise.
            Default: 0.5
    
    Returns:
        GeoDataFrame with CDW detection polygons containing:
        - geometry: Polygon geometry in CHM's CRS
        - confidence: Detection confidence (0-1)
        - area_m2: Area in square meters
        - tile_id: Source tile index (for debugging)
        - detection_id: Unique detection ID
    
    Raises:
        FileNotFoundError: If chm_path doesn't exist
        RuntimeError: If model fails to load or predict
        ValueError: If CHM raster is invalid (no data, wrong format, etc.)
    
    Examples:
        >>> detector = CDWDetector(model_path='best.pt')
        >>> detections = detector.detect('forest_chm.tif')
        >>> print(f"Found {len(detections)} CDW features")
        Found 42 CDW features
        
        >>> # Save to file
        >>> detections = detector.detect(
        ...     'forest_chm.tif',
        ...     output_path='cdw_detections.gpkg',
        ...     confidence=0.25
        ... )
        
        >>> # High-confidence only
        >>> high_conf = detections[detections['confidence'] > 0.5]
    
    Notes:
        - Uses sliding window with overlap to handle edge effects
        - NMS removes duplicate detections in overlap regions
        - Performance: ~5-10 seconds per 1000x1000 pixel tile (GPU)
        - Memory: Loads one tile at a time (low memory footprint)
    
    See Also:
        - CDWDetector.__init__: Configure detection parameters
        - _apply_nms: Non-maximum suppression algorithm details
    """
    pass
```

---

## 🎯 Recommended Priority Order

### Phase 1: Data Foundation (CRITICAL - 1-2 months)
1. **Expand dataset to 500+ samples** using active learning pipeline
2. **Add input validation** to all modules
3. **Write unit tests** for core functions
4. **Fix documentation inconsistencies**

### Phase 2: Production Hardening (1 month)
5. **Add CI/CD pipeline** (GitHub Actions)
6. **Implement logging framework**
7. **Add error handling** throughout
8. **Create benchmark suite**
9. **Write integration tests**

### Phase 3: Advanced Features (2 months)
10. **Experiment with ensemble methods** (if data > 500)
11. **Build API server** for deployment
12. **Create QGIS plugin** for users
13. **Implement batch processing**
14. **Add monitoring/metrics tracking**

### Phase 4: Research & Optimization (Ongoing)
15. **Try advanced architectures** (attention, two-stage)
16. **Optimize inference speed**
17. **Experiment with augmentation**
18. **Publish results**, write paper

---

## 📊 Success Metrics

### Short-term (3 months):
- ✅ Dataset size: 500+ samples
- ✅ Test coverage: >70%
- ✅ mAP50: >0.30 (triple current)
- ✅ CV: <15% (5x improvement)
- ✅ Overfitting: <30% (2.5x better)

### Long-term (6 months):
- ✅ Dataset: 1000+ samples
- ✅ Test coverage: >90%
- ✅ mAP50: >0.50
- ✅ CV: <10%
- ✅ Deployable API + QGIS plugin
- ✅ Published methodology

---

## 🎓 Conclusion

**Current Strengths**:
- Clean, modular codebase
- Good documentation foundation
- Working ML pipeline

**Critical Gap**:
- Dataset size is THE bottleneck (33 samples → need 500+)

**Path Forward**:
1. **Immediate**: Expand dataset using active learning (10-20x speedup)
2. **Short-term**: Add tests, CI/CD, production hardening
3. **Medium-term**: Deploy as API/plugin, advanced ML experiments
4. **Long-term**: Publish, scale to production

**Bottom Line**: This is a B-grade research project that can become an A-grade production system with 3-6 months of focused work, primarily on **data collection**.

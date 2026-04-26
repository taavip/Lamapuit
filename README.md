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

## Critical Resources (2026-04-26)

### 📊 Training Labels
**Primary File:** `data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv` (580,136 rows, updated 2026-04-26)

- **Size:** 580,136 labeled locations across 23 map sheets and 8 years (2018–2024)
- **Content:** All labels with train/val/test splits + retrained ensemble probabilities
- **Splits (stratified):** 
  - Train: 67,290 (11.6%) — high quality, 93% mean CWD probability
  - Val: 13,850 (2.4%) — high quality, 95% mean CWD probability
  - Test: 56,521 (9.7%) — high quality, 96% mean CWD probability
  - Buffer/ineligible: 442,475 (76.3%) — properly excluded uncertain zones
- **Class distribution (eligible):** 71.8% CWD, 28.2% NO_CWD
- **Spatial isolation:** 51.2 m gap (exceeds 50 m CWD autocorrelation threshold per Gu et al. 2024)
- **Methodology:** Spatial-temporal splits prevent leakage across years and locations
- **Probabilities:** Ensemble soft-voting from 4-model ensemble (CNN×3 + EfficientNet)
- **Reference:** See `SPLIT_ASSIGNMENT_REPORT.md` for methodology, `OPTION_B_SPATIAL_SPLITS_COMPARISON.md` for ensemble details

**Legacy Versions:**
- `labels_canonical_with_splits.csv` — Original split assignment (2026-04-23)
- `labels_canonical_with_splits_recalculated.csv` — Intermediate version (2026-04-23)

### 🏔️ CHM Variants

**Location:** `data/chm_variants/` — multiple variants for comparison

| Variant | Path | Channels | Resolution | Coverage | Status | Notes |
|---------|------|----------|------------|----------|--------|-------|
| **Baseline** | `baseline_chm_20cm/` | 1 | 0.2 m | ~119 tiles | ✅ Full | Raw single-band CHM |
| **Composite 3-band** | `composite_3band/` | 2 | 0.2 m | ~119 tiles | ✅ Full | Raw + Gaussian + difference |
| **Composite 4-band (masked)** | `composite_3band_with_masks/` | 4 | 0.2 m | 2 test tiles | ⚠️ Partial | Conservative mask strategy |
| **2-band Masked** | `harmonized_0p8m_chm_raw_2band_masked/` | 2 | 0.8 m | 2 test tiles | ⚠️ Partial | Raw + mask channel |
| **Harmonized Raw** | `harmonized_0p8m_chm_raw/` | 1 | 0.8 m | ✓ Available | ✅ Ready | Harmonized DEM + raw CHM |
| **Harmonized Gaussian** | `harmonized_0p8m_chm_gaussian/` | 1 | 0.8 m | ✓ Available | ✅ Ready | Harmonized DEM + smoothed CHM |

**Evaluation Status (2026-04-26):** Comprehensive benchmark in progress — testing 3-4 variants × 3 architectures (ConvNeXt, EfficientNet, ResNet) with 3-fold CV. Expected completion: 2026-04-26 ~05:00 UTC. See `CHM_VARIANT_EVALUATION_PLAN.md` for methodology.

**Generation Script:** `src/cdw_detect/chm_variants/` — Modular Python API
```python
from src.cdw_detect.chm_variants import CHMVariantGenerator

gen = CHMVariantGenerator(
    laz_dir="/data/laz",
    output_dir="data/chm_variants"
)
results = gen.generate(variants=['baseline', 'raw', 'gaussian', 'composite', 'masked-raw'])
```

### 🧠 Trained Models

**Location:** `output/tile_labels_spatial_splits/` — Final ensemble (2026-04-25)

| Model | File | Architecture | Seed | Train Data | Test Performance |
|-------|------|--------------|------|-----------|------------------|
| Model 1 | `cnn_seed42_spatial.pt` | CNN-Deep-Attn | 42 | 67,290 labels | AUC 0.9884 |
| Model 2 | `cnn_seed43_spatial.pt` | CNN-Deep-Attn | 43 | 67,290 labels | F1 0.9819 @ 0.4 |
| Model 3 | `cnn_seed44_spatial.pt` | CNN-Deep-Attn | 44 | 67,290 labels | (ensemble metric) |
| Model 4 | `effnet_b2_spatial.pt` | EfficientNet-B2 | — | 67,290 labels | (ensemble metric) |

**Ensemble Performance (Test Set: 56,521 labels)**
- **AUC-ROC:** 0.9884
- **F1 Score:** 0.9819 @ threshold=0.4
- **Class Separation:** 5.55× (CWD mean / NO_CWD mean)
- **Confidence:** 89.5% mean probability on CWD samples, 16.1% on NO_CWD

**Training Details:** 3-year temporal window (2018–2024), spatial-temporal stratification, 67.3K training labels (3.4× increase from Option A), ensemble soft-voting. See `OPTION_B_SPATIAL_SPLITS_COMPARISON.md` for full comparison.

**Legacy Models:** Old model runs in `output/tile_labels_*` (archive before thesis submission). Canonical references documented in `models/MODEL_REGISTRY.md`.

### 📖 Related Documentation
- **Methodology:** `SPLIT_ASSIGNMENT_REPORT.md` (6,000+ words) — spatial-temporal split strategy with Chebyshev buffers
- **Results:** `FINAL_DELIVERABLES_SUMMARY.md` — label statistics, model validation, performance metrics
- **Inference Guide:** `RECALCULATE_MODEL_PROBS_README.md` — CNN inference & probability recalculation
- **Comparison:** `OPTION_B_SPATIAL_SPLITS_COMPARISON.md` — Original vs. retrained ensemble
- **Timeline & Inventory:** `PROJECT_TIMELINE_AND_EXPERIMENTS.md` — Chronological overview of all experiments
- **Cleanup Plan:** `CLEANUP_CHECKLIST.md` — What to keep vs. archive/delete

## CHM Generation Pipelines

Two pipelines produce Canopy Height Models from LAZ:

- **Legacy pipeline** (`scripts/process_laz_to_chm.py`) — produces CHMs in `chm_max_hag_13_drop/`, clips or drops points above HAG threshold
- **Harmonized pipeline** (`experiments/laz_to_chm_harmonized_0p8m/`) — produces harmonized DEMs + both raw and Gaussian-smoothed CHMs at 0.8 m resolution; this is the current best pipeline
- **Variant module** (`src/cdw_detect/chm_variants/`) — Programmatic API for generating multiple CHM variants (baseline, raw, Gaussian, composite, masked) for comparison

## Project Status (2026-04-26)

| Component | Status | Details |
|-----------|--------|---------|
| **Pipeline & Core Code** | ✅ Complete | Modular architecture, tested, production-ready |
| **Ground Truth Labels** | ✅ Complete | 580K labels standardized across CHM variants |
| **Spatial Split Methodology** | ✅ Complete | Stride-aware, year-safe, prevents 1.4M label leakage |
| **Model Training** | ✅ Complete | 4-model ensemble trained on 67.3K labels |
| **Model Performance** | ✅ Complete | AUC 0.9884, F1 0.9819 on test set (56.5K labels) |
| **CHM Variant Evaluation** | 🔄 In Progress | Benchmark running; due ~05:00 UTC 2026-04-26 |
| **Thesis Documentation** | 🚀 In Progress | Methodology sections complete; results integration ongoing |
| **Project Cleanup** | ⚠️ Pending | 100+ experimental scripts to archive/delete (see `CLEANUP_CHECKLIST.md`) |

**Key Achievement:** Spatial-temporal split strategy prevents leakage while maximizing training data (67.3K labels, up 3.4× from initial 19.8K).

## Installation

```bash
conda env create -f environment.yml
conda activate cwd-detect
pip install -e .
```

## Quick Start

### Use Pre-trained Ensemble Model
```python
import torch
from src.cdw_detect import CDWDetector

# Load ensemble (4 models)
models = [
    torch.load('output/tile_labels_spatial_splits/cnn_seed42_spatial.pt'),
    torch.load('output/tile_labels_spatial_splits/cnn_seed43_spatial.pt'),
    torch.load('output/tile_labels_spatial_splits/cnn_seed44_spatial.pt'),
    torch.load('output/tile_labels_spatial_splits/effnet_b2_spatial.pt'),
]

detector = CDWDetector(models=models, ensemble=True)
detections_gpkg = detector.detect(chm_path='data/chm_variants/baseline_chm_20cm/tile_0.tif')
detections_gpkg.to_file('detections.gpkg')
```

### Access Training Labels with Splits & Probabilities
```python
import pandas as pd

# Load 580K labels with splits and retrained ensemble probabilities
labels = pd.read_csv('data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv')

# Filter by split
train_labels = labels[labels['split'] == 'train']  # 67,290 labels
val_labels = labels[labels['split'] == 'val']      # 13,850 labels
test_labels = labels[labels['split'] == 'test']    # 56,521 labels

# Access ensemble probabilities
train_probs = train_labels['ensemble_prob_cwd']  # Retrained ensemble predictions
print(f"Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")
print(f"CWD distribution: {(train_labels['class'] == 'CWD').mean():.1%}")
print(f"Mean CWD probability (train): {train_probs[train_labels['class'] == 'CWD'].mean():.1%}")
```

### Evaluate CHM Variants
```bash
# Benchmark 3-4 CHM variants with 3-fold CV
python scripts/chm_variant_benchmark_quick.py \
  --chm-dir data/chm_variants \
  --labels data/chm_variants/labels_canonical_with_splits.csv \
  --output results.json
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

### Core Pipeline (Production)
| Script | Purpose |
|---|---|
| `scripts/process_laz_to_chm.py` | LAZ → CHM GeoTIFF (clip or drop HAG filter) |
| `scripts/prepare_data.py` | CHM + line labels → training dataset |
| `scripts/train_model.py` | Train segmentation model |
| `scripts/finetune_model.py` | Fine-tune from existing checkpoint |
| `scripts/run_detection.py` | Run CWD detection on CHM rasters |
| `scripts/cleanup_memory.py` | Clear GPU/CPU memory, kill lingering processes |

### Spatial Split & Labeling (Methodology-Critical)
| Script | Purpose |
|---|---|
| `scripts/assign_label_splits.py` | Assign spatial-temporal train/val/test splits (prevents leakage) |
| `scripts/split_utils.py` | Utilities for spatial split validation |
| `scripts/standardize_labels_for_chm_variants.py` | Standardize labels across CHM variants |
| `scripts/check_split_leakage.py` | Validate zero spatial-temporal leakage |
| `scripts/retrain_ensemble_spatial_splits.py` | Train final ensemble on stratified splits |
| `scripts/recalculate_model_probs.py` | Ensemble inference & probability calculation |

### CHM Variant Evaluation (Current 2026-04-26)
| Script | Purpose |
|---|---|
| `scripts/chm_variant_benchmark_quick.py` | Fast variant comparison (3-fold CV, 6-hour limit) |
| `scripts/chm_variant_selection_improved.py` | Comprehensive variant testing |
| `scripts/chm_ablation_train.py` | Ablation study: raw vs. smoothed vs. variants |
| `scripts/chm_ablation_analyze.py` | Analyze ablation results |

### Experiment Archives (Reference Only)
| Script | Purpose | Status |
|---|---|---|
| `scripts/model_search_v3/` | Model hyperparameter search | ✅ Completed; keep final config |
| `scripts/model_search*.py` | Earlier search iterations | ⚠️ Archive outputs; keep final results |
| `scripts/analyze_experiments.py` | Experiment analysis utilities | ⚠️ Reference only |

## Data & Files Guide

### Ground Truth Labels
**Primary File:** `lamapuit.gpkg` (committed to repo)
- LineString geometries representing CWD centerlines
- 23 map sheets, 8 years of observations
- CRS: same as CHM rasters

**Processed Labels (Canonical for Training):** `data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv` (580,136 rows, 2026-04-26)
- 580K labeled locations with train/val/test splits
- Includes retrained ensemble probabilities (4-model soft-voting)
- Split assignment: 67.3K train, 13.9K val, 56.5K test, 442.5K buffer
- Ready for model training and evaluation

### Input CHM Rasters
**Location:** `data/chm_variants/` (generated data, not tracked in Git)

Format specifications:
- GeoTIFF, 0.2–0.8 m resolution (variant-dependent)
- Height Above Ground (HAG) values, 0–1.5 m
- Nodata: 0 or −9999
- Single or multi-band (see variant table in Critical Resources section)

### Trained Models
**Location:** `output/tile_labels_spatial_splits/` (generated data, not tracked in Git)

- Final ensemble: 4 PyTorch models (.pt files)
- See Critical Resources section for details
- Old runs: `output/tile_labels_*` (archive before submission)
- Canonical references: `models/MODEL_REGISTRY.md` (when published)

### Experiment Outputs
**Location:** `experiments/`, `output/` (generated data, not tracked in Git)

Large outputs and experiment results are **not tracked in Git** per `.gitignore`:
- `output/` — model checkpoints and inference results
- `experiments/` — experimental runs and ablations
- `data/` — generated training datasets
- `runs/` — training logs and outputs

See `PROJECT_TIMELINE_AND_EXPERIMENTS.md` for inventory and cleanup plan.

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

## Project Documentation & Inventory

### Critical References for Thesis Writers
- **`PROJECT_TIMELINE_AND_EXPERIMENTS.md`** — Chronological overview of all work (Jan 2026–Apr 2026)
  - 5 major phases: foundation, RF classifier, spatial splits, CHM variants, cleanup
  - Per-experiment goal/methodology/results
  - Thesis relevance ratings (✅ CRITICAL, ⚠️ REFERENCE, ❌ ARCHIVE)
  - Clear guidance on what to report

- **`CLEANUP_CHECKLIST.md`** — What to keep vs. archive/delete
  - 40 thesis-critical items (keep)
  - 20+ items to archive if cited
  - 100+ experimental scripts to delete
  - Cleanup steps and success criteria

### Methodology Documentation (for Thesis)
- `SPLIT_ASSIGNMENT_REPORT.md` — Full spatial-temporal split strategy (6,000+ words)
- `OPTION_B_SPATIAL_SPLITS_COMPARISON.md` — Original vs. retrained ensemble comparison
- `FINAL_DELIVERABLES_SUMMARY.md` — Label statistics, model validation, performance metrics
- `CNN_INFERENCE_RESULTS.md` — Ensemble inference details and test metrics
- `CHM_VARIANT_EVALUATION_PLAN.md` — Variant selection methodology (in progress)

### Experiment Tracking
- `experiments_results.yaml` — consolidated results
- `experiments_progress.yaml` — in-progress tracking
- `massive_multirun_all_runs.csv` / `massive_multirun_statistics.csv` — multi-run summaries
- `docs/` — per-experiment runbooks and analysis notes

### Legacy Documentation (Archive After Review)
- `*_SUMMARY.md` files (100+) — old session notes and results
- `TRAINING_SESSION_SUMMARY.md` — past session notes
- `UPDATE_SUMMARY.md` — old status updates
- See `CLEANUP_CHECKLIST.md` for full list

## CLI Entry Points

After `pip install -e .`:

```bash
cdw-detect           # main detection CLI
cdw-laz-classifier   # LAZ point-feature RF classifier
```

## Key Metrics & Results Summary

**Final Model Performance (Test Set: 56,521 labels)**
```
Ensemble (4 models, soft-voting)
├── AUC-ROC:           0.9884
├── F1 Score:          0.9819 @ threshold=0.4
├── Precision:         0.9724
├── Recall:            0.9915
├── Class Separation:  5.55× (CWD vs. NO_CWD)
└── Confidence:        89.5% mean on CWD, 16.1% on NO_CWD
```

**Training Data (Spatial-Temporal Splits)**
```
Total Labels:         580,136
├── Train:            67,290 (11.6%)  — high quality, 93% mean CWD prob
├── Val:              13,850 (2.4%)   — high quality, 95% mean CWD prob
├── Test:             56,521 (9.7%)   — high quality, 96% mean CWD prob
└── Buffer/Ineligible: 442,475 (76.3%) — excluded from training

Class Distribution (Eligible: 142,465)
├── CWD:              102,290 (71.8%)
└── NO_CWD:            40,175 (28.2%)

Geographic & Temporal Coverage
├── Map Sheets:       23 (national coverage)
├── Years:            8 (2018–2024)
├── Spatial Isolation: 51.2 m buffer gap (meets 50 m CWD autocorr threshold)
└── Year Consistency: Maintained across observations
```

**CHM Processing Summary**
```
Input Resolution:     0.2–0.8 m (variant-dependent)
Height Range:         0–1.5 m (Height Above Ground)
Variants Evaluated:   6 (baseline, raw, Gaussian, composite 3/4-band, masked)
Benchmark Status:     IN PROGRESS (started 2026-04-26 02:30 UTC)
Expected Completion:  ~05:00 UTC 2026-04-26
```

## File Paths Quick Reference

```
📁 Critical Data & Models
├── 📄 data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv (580K + probs)
├── 📄 lamapuit.gpkg                                            (ground truth lines)
├── 📁 data/chm_variants/baseline_chm_20cm/                    (full variant)
├── 📁 data/chm_variants/composite_3band/                      (full variant)
├── 📁 output/tile_labels_spatial_splits/
│   ├── cnn_seed42_spatial.pt                                  (model 1)
│   ├── cnn_seed43_spatial.pt                                  (model 2)
│   ├── cnn_seed44_spatial.pt                                  (model 3)
│   └── effnet_b2_spatial.pt                                   (model 4)
└── 📄 SPLIT_ASSIGNMENT_REPORT.md                              (methodology)

📁 Methodology & Results
├── 📄 SPLIT_ASSIGNMENT_REPORT.md                              (6,000+ words)
├── 📄 OPTION_B_SPATIAL_SPLITS_COMPARISON.md
├── 📄 FINAL_DELIVERABLES_SUMMARY.md
├── 📄 CNN_INFERENCE_RESULTS.md
├── 📄 CHM_VARIANT_EVALUATION_PLAN.md
└── 📄 PROJECT_TIMELINE_AND_EXPERIMENTS.md                     (this overview)

📁 Core Code (Thesis-Ready)
├── src/cdw_detect/prepare.py         (labeling→dataset)
├── src/cdw_detect/detect.py          (inference)
├── src/cdw_detect/train.py           (training)
├── src/cdw_detect/chm_variants/      (variant generation)
├── scripts/assign_label_splits.py    (spatial split logic)
├── scripts/retrain_ensemble_spatial_splits.py
├── scripts/recalculate_model_probs.py
└── scripts/chm_variant_benchmark_quick.py

📁 Configuration & Environment
├── environment.yml                    (Conda environment)
├── pyproject.toml                     (Package metadata)
├── Dockerfile                         (Reproducibility)
├── docker-compose.benchmark.yml       (CHM variant testing)
└── docker-compose.labeler.yml        (Inference container)
```

## Troubleshooting

### Memory errors

```bash
python scripts/cleanup_memory.py
python scripts/train_model.py --data dataset.yaml --batch 2 --imgsz 512
```

### Check project status

```bash
# View critical locations
cat README.md | grep -A 30 "Critical Resources"

# See what was done and what to report
cat PROJECT_TIMELINE_AND_EXPERIMENTS.md | head -100

# Understand cleanup needs
cat CLEANUP_CHECKLIST.md | grep -E "^##|✅|❌"
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

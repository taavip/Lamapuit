# Lamapuit Project: Chronological Timeline & Experiment Inventory

**Document Created:** 2026-04-26  
**Purpose:** Comprehensive overview of all experiments, scripts, and outputs to determine what to report in thesis and what to archive/delete

---

## 🎯 Executive Summary

This document maps all major work phases, experiments, and their status (thesis-relevant vs. archive/delete). The project spans from January 2026 (pipeline setup) to April 2026 (final benchmarking).

---

## 📅 PHASE 1: Foundation & Setup (Jan 27 – Apr 18, 2026)

### 1.1 Pipeline Architecture Setup
**Dates:** 2026-01-27 to 2026-01-28  
**Commit:** `0aa0975` (feat: Add scripts for CDW detection pipeline)

| Item | Details |
|------|---------|
| **Goal** | Build core detection pipeline from LAZ → CHM → labels → detection |
| **Methodology** | Structure: prepare.py, detect.py, train.py as core modules |
| **Scripts Created** | `src/cdw_detect/prepare.py`, `detect.py`, `train.py` |
| **Status** | ✅ **THESIS-CRITICAL** — Core pipeline for all experiments |
| **Output** | Modular pipeline enabling reproducible experiments |
| **To Report** | Yes — Architecture section of methodology |
| **Archive?** | NO |

### 1.2 Label Dataset Acquisition
**Dates:** 2026-01-28  
**Commit:** `1244d58` (Add lamapuit.gpkg CDW labels file)

| Item | Details |
|------|---------|
| **Goal** | Establish ground truth for CWD detection (line geometries) |
| **Methodology** | Manual annotation of CWD centerlines in QGIS |
| **Data Created** | `lamapuit.gpkg` (LineString geometries) |
| **Schema** | Lines representing CWD center positions |
| **Status** | ✅ **THESIS-CRITICAL** — Foundation for all training |
| **Coverage** | 23 map sheets, 8 years (2018–2024) |
| **To Report** | Yes — Data section |
| **Archive?** | NO — Live database |

### 1.3 Code Refactoring & Cleanup
**Dates:** 2026-01-28  
**Commit:** `b75db9d` (Refactor code structure)

| Item | Details |
|------|---------|
| **Goal** | Improve code quality and maintainability |
| **Actions** | Code style, validation, dependency cleanup |
| **Status** | ✅ Housekeeping (foundational) |
| **To Report** | No |
| **Archive?** | NO — Ongoing basis |

---

## 📅 PHASE 2: Random Forest Classifier (Apr 18, 2026)

### 2.1 LAZ Classifier Subpackage
**Date:** 2026-04-18  
**Commit:** `0967c3f` (feat: add laz_classifier subpackage with Random Forest classifier)

| Item | Details |
|------|---------|
| **Goal** | Point-level classification of LAZ files (alternative to CHM) |
| **Methodology** | Random Forest on point features (height, intensity, planarity, etc.) |
| **Scripts** | `src/cdw_detect/laz_classifier/` |
|  | — `features.py` (LiDAR feature extraction) |
|  | — `rf.py` (sklearn Random Forest) |
|  | — `cli.py` (cdw-laz-classifier entry point) |
| **Status** | ✅ Implemented; **LIMITED THESIS VALUE** (not main approach) |
| **Performance** | Benchmark not reported |
| **To Report** | Optional — Mention as explored alternative |
| **Archive?** | Keep (implemented) but don't expand |

### 2.2 Model Search V3 (Hyperparameter Search)
**Date:** 2026-04-18  
**Commit:** `63b257a` (feat: add model search v3 script)

| Item | Details |
|------|---------|
| **Goal** | Optimize hyperparameters for tile-level classification |
| **Methodology** | Grid search over model architectures, LR, batch size, etc. |
| **Script** | `scripts/model_search_v3/` (latest hyperparameter search) |
| **Status** | ✅ **THESIS-RELEVANT** — Best practices for tuning |
| **Focus** | YOLO and CNN variants (see FEEDBACK below) |
| **To Report** | Yes — Hyperparameter tuning section |
| **Archive?** | NO — Document final configs used |
| **Note** | ⚠️ YOLO experiments failed per project memory; remove YOLO refs from docs |

### 2.3 Docker & Infrastructure
**Date:** 2026-04-18  
**Commit:** `b49cff1` (chore: add Docker infrastructure)

| Item | Details |
|------|---------|
| **Goal** | Reproducible training environment |
| **Files** | `Dockerfile`, `docker-compose.*.yml` |
| **Status** | ✅ **THESIS-RELEVANT** — Reproducibility |
| **To Report** | Yes — In Appendix (reproducibility) |
| **Archive?** | NO — Keep for code release |

### 2.4 LaTeX Thesis Source
**Date:** 2026-04-18  
**Commit:** `34eb269` (docs: add LaTeX thesis source)

| Item | Details |
|------|---------|
| **Goal** | Main thesis document |
| **Location** | `LaTeX/Lamapuidu_tuvastamine/` |
| **Status** | 🚀 **PRIMARY OUTPUT** — Your actual thesis |
| **Chapters** | Metoodika (Methodology), Tulemused (Results), etc. |
| **To Report** | Yes — This IS your thesis |
| **Archive?** | NO — Keep and finalize |

---

## 📅 PHASE 3: Spatial Splits & Data Standardization (Apr 21–25, 2026)

### 3.1 CHM Ablation Experiment
**Dates:** 2026-04-21 to 2026-04-23  
**Commit:** Related PRs on Apr 21

| Item | Details |
|------|---------|
| **Goal** | Test different CHM preprocessing (raw vs. smoothed vs. HAG filters) |
| **Methodology** | Train same model on baseline, raw, Gaussian variants |
| **Scripts** | `scripts/chm_ablation_train.py`, `chm_ablation_analyze.py`, `chm_ablation_final_report.py` |
| **Documentation** | `scripts/CHM_ABLATION_EXPERIMENT.md` |
| **Result** | Ablation completed; findings documented |
| **Status** | ✅ **THESIS-RELEVANT** — Ablation study validates CHM choice |
| **To Report** | Yes — Ablation section in methodology/results |
| **Archive?** | NO — Results needed for thesis |

### 3.2 Label Splits Assignment (Spatial-Temporal)
**Dates:** 2026-04-22 to 2026-04-23  
**Commit:** `e137431` (feat: standardize onboarding_labels_v2_drop13)

| Item | Details |
|------|---------|
| **Goal** | Create train/val/test splits preventing spatial-temporal leakage |
| **Methodology** | Stride-aware spatial blocking + year-safe seeding |
|  | — Chebyshev distance in stride coords |
|  | — Buffer gap: 51.2m (exceeds 50m CWD autocorr) |
| **Scripts** | `scripts/assign_label_splits.py`, `scripts/split_utils.py` |
|  | `scripts/standardize_labels_for_chm_variants.py` |
| **Documentation** | `SPLIT_ASSIGNMENT_REPORT.md` (6,000+ words, academic-grade) |
| **Data** | `data/chm_variants/labels_canonical_with_splits.csv` (580,136 rows) |
| **Results** | ✅ All 580K labels assigned |
|  | ✅ 67.3K train, 13.9K val, 56.5K test |
|  | ✅ Year consistency maintained |
|  | ✅ Zero leakage (spatial isolation verified) |
| **Status** | 🚀 **THESIS-CRITICAL** — Core contribution |
| **To Report** | **YES** — Full methodology chapter |
|  | — Explain spatial-temporal strategy |
|  | — Document split sizes and class balance |
|  | — Cite Valavi 2019, Gu et al. 2024 |
| **Archive?** | NO — Primary data product |
| **Academic Value** | Suitable for publication (blockCV comparison, stride-aware spatial CV) |

### 3.3 CNN Inference & Probability Recalculation
**Dates:** 2026-04-23 to 2026-04-24  
**Commit:** Related to final deliverables

| Item | Details |
|------|---------|
| **Goal** | Validate model probabilities on test set |
| **Methodology** | CNN ensemble inference with soft-voting |
|  | — Models: CNN-Deep-Attn (3 seeds), EfficientNet-B2 |
|  | — Input normalization: p2-p98 + CLAHE |
| **Scripts** | `scripts/recalculate_model_probs.py` |
|  | `scripts/recalculate_model_probs_ensemble.py` |
|  | `scripts/recalculate_model_probs_tta_ensemble.py` |
| **Documentation** | `CNN_INFERENCE_RESULTS.md` |
|  | `RECALCULATE_MODEL_PROBS_README.md` |
| **Results** | ✅ Test set AUC: 0.9884 |
|  | ✅ F1: 0.9819 @ threshold=0.4 |
|  | ✅ Class separation: 5.55× |
|  | ✅ Zero probability changes needed (production-ready) |
| **Status** | ✅ **THESIS-RELEVANT** — Final model validation |
| **To Report** | Yes — Performance metrics & evaluation |
| **Archive?** | NO — Core results |

### 3.4 Ensemble Training on Spatial Splits (Option B)
**Dates:** 2026-04-25  
**Completion:** 2026-04-25T22:25:48  
**Commit:** Related to spatial split implementation

| Item | Details |
|------|---------|
| **Goal** | Retrain ensemble with proper spatial-temporal splits |
| **Methodology** | Same as Option A but with train/val/test stratification |
| **Training Setup** | 4-model ensemble (3× CNN + 1× EfficientNet) |
| **Data Increase** | 19.8K → 67.3K training labels (3.4× more) |
| **Scripts** | `scripts/retrain_ensemble_spatial_splits.py` |
|  | `scripts/postprocess_spatial_split_retraining.py` |
| **Models Created** | `output/tile_labels_spatial_splits/` |
|  | — `cnn_seed42_spatial.pt` |
|  | — `cnn_seed43_spatial.pt` |
|  | — `cnn_seed44_spatial.pt` |
|  | — `effnet_b2_spatial.pt` |
| **Results** | Test set: AUC 0.9884, F1 0.9819 |
| **Documentation** | `OPTION_B_SPATIAL_SPLITS_SUMMARY.md` |
|  | `OPTION_B_SPATIAL_SPLITS_COMPARISON.md` |
|  | `OPTION_B_SPATIAL_SPLITS_RETRAINING.md` |
| **Status** | ✅ **THESIS-CRITICAL** — Final trained models |
| **To Report** | **YES** — Training setup, results, comparison |
| **Archive?** | NO — These are final production models |

---

## 📅 PHASE 4: CHM Variant Evaluation (Apr 22–26, 2026)

### 4.1 CHM Variant Module Creation
**Dates:** 2026-04-22 to 2026-04-26  
**Commits:** 
- `5e4605d` (refactor: move CHM variant generation to src/ module)
- `a0a6a79` (feat: add improved CHM variant selection scripts)
- `7e408a6` (docs: add CHM variant evaluation plan)

| Item | Details |
|------|---------|
| **Goal** | Compare different CHM preprocessing approaches |
|  | — Baseline (0.2m raw) |
|  | — Raw CHM (gap-filled) |
|  | — Gaussian smoothed |
|  | — Composite 3-band (raw + Gaussian + diff) |
|  | — Composite 4-band with masks |
|  | — 2-band masked CHM |
| **Methodology** | Generate CHM variants and benchmark with 3-fold CV |
| **Module** | `src/cdw_detect/chm_variants/` |
|  | — `generator.py` (317 lines) |
|  | — `composite.py` (207 lines) |
|  | — `masked.py` (115 lines) |
| **Scripts** | `scripts/chm_variant_benchmark_quick.py` (400 lines) |
|  | `scripts/chm_variant_selection_improved.py` (400 lines) |
|  | `scripts/chm_variant_selection.py` (existing) |
|  | `scripts/chm_variant_selection_analyze.py` |
| **Models Tested** | ConvNeXt, EfficientNet, ResNet (3 architectures) |
| **Configuration** | Time-optimized: 3-fold CV, 15 epochs, 200-tile sample |
| **Estimated Runtime** | 2–3 hours (6-hour limit) |
| **Documentation** | `CHM_VARIANT_EVALUATION_PLAN.md` (comprehensive) |
|  | `BENCHMARK_SETUP_SUMMARY.md` |
| **Status** | 🔄 **IN PROGRESS** (benchmark started 2026-04-26 02:30) |
|  | Expected completion: ~05:00 UTC |
| **To Report** | **YES (pending results)** — Will inform final CHM choice |
| **Archive?** | NO — Results determine methodology chapter |
| **Key Question** | Does complex 4-band fusion beat simple baselines? |

### 4.2 Docker Benchmark Environment
**Date:** 2026-04-26  
**File:** `docker-compose.benchmark.yml`

| Item | Details |
|------|---------|
| **Goal** | GPU-accelerated CHM variant testing in container |
| **Setup** | Builds `lamapuit:gpu-benchmark` image |
|  | Mounts workspace, uses CUDA if available |
| **Status** | ✅ Ready, in use |
| **To Report** | Optional (infrastructure) |
| **Archive?** | NO — Keep for reproducibility |

---

## 📅 PHASE 5: Cleanup & Documentation Updates (Apr 25–26, 2026)

### 5.1 .gitignore Refinement
**Date:** 2026-04-25  
**Commit:** `678aacb` (chore: add explicit .gitignore rules)

| Item | Details |
|------|---------|
| **Goal** | Prevent committing large data/output files |
| **Status** | ✅ Housekeeping |
| **Archive?** | NO |

### 5.2 LaTeX & Thesis Updates
**Dates:** 2026-04-25 to 2026-04-26  
**Commits:**
- `a26f69f`, `b4d4144` (docs: expand thesis content)
- `ffdd6db` (docs: update LaTeX from Overleaf)
- `6710cd3`, `678aacb` (citation updates)

| Item | Details |
|------|---------|
| **Goal** | Integrate findings into thesis document |
| **Content Added** | APA citations, extended literature review |
| **Status** | 🚀 **THESIS IN PROGRESS** |
| **To Report** | Yes — This is your thesis |
| **Archive?** | NO — Actively developed |

---

## 📊 Summary: Scripts Inventory

### Core Pipeline (KEEP - THESIS-CRITICAL)
```
src/cdw_detect/
├── prepare.py              ✅ KEEP — Label→tiles conversion
├── detect.py               ✅ KEEP — Inference pipeline
├── train.py                ✅ KEEP — Training wrapper
├── laz_classifier/         ✅ KEEP — RF alternative (explored)
└── chm_variants/           ✅ KEEP — Variant generation module

scripts/
├── prepare_data.py         ✅ KEEP — CLI wrapper for prepare
├── train_model.py          ✅ KEEP — CLI wrapper for train
├── run_detection.py        ✅ KEEP — CLI wrapper for detect
├── process_laz_to_chm.py   ✅ KEEP — LAZ→CHM conversion
└── chm_variant_selection*.py  ✅ KEEP — Variant benchmarking
```

### Spatial Split Methodology (KEEP - THESIS-CRITICAL)
```
scripts/
├── assign_label_splits.py              ✅ KEEP — Split assignment
├── split_utils.py                      ✅ KEEP — Utilities
├── standardize_labels_*.py             ✅ KEEP — Label standardization
├── check_split_leakage.py              ✅ KEEP — Validation
├── remediate_split_leakage.py          ✅ KEEP — Fixes
└── test_evaluate_spatial_splits.py     ✅ KEEP — Tests
```

### Model Training & Inference (KEEP - THESIS-CRITICAL)
```
scripts/
├── retrain_ensemble_spatial_splits.py  ✅ KEEP — Final training
├── recalculate_model_probs*.py         ✅ KEEP — Validation/inference
├── fine_tune_cnn.py                    ✅ KEEP — Fine-tuning
├── train_ensemble.py                   ✅ KEEP — Ensemble training
└── train_tile_classifier.py            ✅ KEEP — Tile classification
```

### Experimental/Exploratory (ARCHIVE or DELETE)
```
scripts/
├── analyze_*.py            ⚠️  Keep only if actively cited in thesis
├── model_search*.py        ⚠️  Keep summary; archive runs
├── run_massive_multirun.py ⚠️  Archive (intermediate experiment)
├── label_tiles.py          ⚠️  Legacy GUI; not part of final pipeline
├── label_all_rasters*.py   ⚠️  Legacy annotation tools
├── train_*_manual_masks.py ⚠️  Archive (explored but not used)
├── rank_tiles_for_manual_masks.py ⚠️ Archive
├── generate_*_visualizations.py ⚠️ Keep if used in thesis
└── [100+ other utility/debug scripts] ⚠️ Clean up
```

### What to DELETE
```
❌ DELETE:
├── labelstudio_sam_demo/               (unrelated SAM project)
├── labeler/                            (old annotation tool)
├── manual_masks_lineaware_browser/     (GUI prototype)
├── UrbanCar_LiDAR_Dataset_Report.md   (separate project)
├── prototypes/                         (abandoned prototypes)
├── *_tmp/                              (temporary directories)
├── kpconv_tmp/, myria3d_tmp/, etc.     (failed approaches)
├── export_coco-instance*.json          (unrelated)
├── configs/label_studio/car_*.xml      (unrelated)
└── Session-note .md files at root      (distill to thesis, delete)
```

### Data & Outputs (SELECTIVE KEEP)
```
data/
├── chm_variants/           ✅ KEEP — Working data for variant selection
└── generated products      ⚠️ Keep canonical versions; archive intermediate

output/
├── tile_labels_spatial_splits/ ✅ KEEP — Final trained models
├── chm_variant_benchmark/  ⚠️ KEEP until results analyzed; then archive
└── [experiment runs]       ⚠️ Archive old runs; keep latest

experiments/
├── chm_ablation_436646/    ⚠️ KEEP if cited; otherwise archive
├── dtm_hag_436646_*/       ⚠️ Archive (exploration)
└── randla_chm_class2_*     ⚠️ Archive (exploration)
```

---

## 📋 Documentation & Thesis Materials

### Required for Thesis ✅

| File | Purpose | Status |
|------|---------|--------|
| `SPLIT_ASSIGNMENT_REPORT.md` | Spatial-temporal split methodology (6,000+ words) | ✅ Ready |
| `OPTION_B_SPATIAL_SPLITS_COMPARISON.md` | Training comparison (Option A vs B) | ✅ Ready |
| `CHM_VARIANT_EVALUATION_PLAN.md` | Variant selection methodology | ✅ Ready |
| `FINAL_DELIVERABLES_SUMMARY.md` | Label statistics & model validation | ✅ Ready |
| `CNN_INFERENCE_RESULTS.md` | Model performance metrics | ✅ Ready |
| `LaTeX/Lamapuidu_tuvastamine/` | Main thesis document | 🚀 In progress |

### Exploratory (Optional / Archive) ⚠️

| File | Purpose | Action |
|------|---------|--------|
| `OPTION_B_METHODOLOGY_DRAFT.tex` | Early methodology draft | Archive or merge to main |
| `COMPREHENSIVE_METRICS_SUMMARY.md` | Detailed metrics | Archive (covered above) |
| `MANUAL_MASK_EXPERIMENT_REPORT.md` | Manual annotation experiments | Archive (not used) |
| `MASK_COMPARISON.md` | Mask strategy variants | Archive (covered by variant eval) |
| `UPDATE_SUMMARY.md` | Session notes | Archive (distill key findings) |
| Various `*_SUMMARY.md` files | Old summaries | Archive |
| `TRAINING_SESSION_SUMMARY.md` | Past session notes | Archive |

---

## 🎯 Recommendations for Thesis Writing

### What to Report ✅

1. **Problem & Background**
   - Estonian ALS sparsity (1–4 pts/m²)
   - CWD importance in forest management
   - Literature review (Gu et al. 2024, Valavi 2019, etc.)

2. **Methodology** 
   - CHM preprocessing & variant evaluation (pending 4-26 results)
   - Spatial-temporal split strategy (→ `SPLIT_ASSIGNMENT_REPORT.md`)
   - Label standardization (580K labels, 142K eligible)
   - Model architecture (CNN-Deep-Attn, EfficientNet ensemble)
   - Training/validation setup (Option B with spatial splits)

3. **Results**
   - Ablation study (CHM variants impact)
   - Model performance (AUC 0.9884, F1 0.9819)
   - Class separation analysis
   - Comparison: Option A (original splits) vs. Option B (spatial splits)

4. **Discussion**
   - Why spatial splits matter (leakage prevention)
   - Generalization to low-density ALS (future work)
   - Model confidence calibration
   - Uncertainty & buffer zones

5. **Reproducibility**
   - Docker environment
   - Python package (`pip install -e .`)
   - Dataset location & structure

### What to Archive/Delete ⚠️

1. **Abandoned Exploratory Work**
   - YOLO experiments (failed)
   - Label Studio experiments (superseded)
   - Manual mask annotation GUIs (prototypes)
   - UrbanCar project files
   - `*_tmp` directories
   - Old session notes (distill findings first)

2. **Intermediate Experiment Outputs**
   - `experiments/` subdirectories (save summaries, delete large files)
   - Old model runs (keep latest ensemble only)
   - Intermediate datasets (keep canonical versions)

3. **Code Cleanup**
   - Remove dead code branches
   - Archive experimental scripts (80+ utility scripts)
   - Document which hyperparameter search configs were used (final)

---

## 📈 Key Metrics to Report

| Metric | Value | Source |
|--------|-------|--------|
| **Training Labels** | 67,290 | `OPTION_B_SPATIAL_SPLITS_SUMMARY.md` |
| **Test Labels** | 56,521 | `SPLIT_ASSIGNMENT_REPORT.md` |
| **CWD Class Distribution** | 71.8% (eligible labels) | `FINAL_DELIVERABLES_SUMMARY.md` |
| **Test Set AUC** | 0.9884 | `CNN_INFERENCE_RESULTS.md` |
| **Test Set F1** | 0.9819 | `OPTION_B_SPATIAL_SPLITS_SUMMARY.md` |
| **Buffer Gap (Spatial)** | 51.2m | `SPLIT_ASSIGNMENT_REPORT.md` |
| **Class Separation Ratio** | 5.55× | `FINAL_DELIVERABLES_SUMMARY.md` |
| **Geographic Coverage** | 23 map sheets | `SPLIT_ASSIGNMENT_REPORT.md` |
| **Temporal Coverage** | 8 years (2018–2024) | `FINAL_DELIVERABLES_SUMMARY.md` |

---

## ⚙️ Next Steps

### Immediate (Apr 26–27)
- [ ] Monitor CHM variant benchmark completion (~05:00 UTC)
- [ ] Analyze variant results → update `CHM_VARIANT_EVALUATION_PLAN.md`
- [ ] Decide final CHM variant for thesis

### Manual Labeling & Mask Refinement (Optional - For Additional Training Data)

If you need to create additional labeled training data beyond the 67.3K already available:

**Batch label all CHM rasters (keyboard-based, 128×128 chunks):**
```bash
python scripts/label_all_rasters.py \
  --chm-dir data/chm_max_hag \
  --output output/tile_labels
```
⏱️ **Time:** ~2 hours for 100 rasters  
📊 **Output:** CSV files with chunk-level CDW/NO_CDW labels  
📖 **Guide:** See `LABELING_TOOLS_GUIDE.md`

**Pixel-level mask refinement with interactive brush:**
```bash
python scripts/brush_mask_labeler.py \
  --tile-csv queue.csv \
  --output output/refined_masks
```
⏱️ **Time:** ~2-5 minutes per tile for detailed refinement  
🎨 **Controls:** B (brush), E (eraser), C (clear), S (save), N/P (navigate)  
📦 **Output:** Binary masks (NPY) + confidence maps  
📖 **Guide:** See `LABELING_TOOLS_GUIDE.md`

**Alternative: AI-assisted semi-automated labeling (60% faster):**
```bash
python scripts/label_all_rasters_segmentation.py \
  --chm-dir data/chm_max_hag \
  --model-dir output/tile_labels_spatial_splits \
  --output output/segmentation_labels
```

### Short-term (Apr 27–May 5)
- [ ] Write full methodology chapter (spatial splits + variant choice)
- [ ] Generate ablation study figures for results section
- [ ] Create final model architecture diagram
- [ ] Document all hyperparameters used
- [ ] (Optional) Create additional training data using labeling tools above

### Medium-term (May 5–15)
- [ ] Archive exploratory scripts (80+ utility files)
- [ ] Delete unrelated data (UrbanCar, SAM demos, old experiments)
- [ ] Consolidate documentation (merge scattered `*SUMMARY.md` files)
- [ ] Create final code release version

### Before Submission
- [ ] Verify all figures/tables reference correct data
- [ ] Ensure reproducibility: `pip install -e . && pytest`
- [ ] Update README with final results
- [ ] Add `models/MODEL_REGISTRY.md` (canonical checkpoint references)

---

## 📂 File Organization Summary

```
KEEP (Thesis-Critical)
├── LaTeX/Lamapuidu_tuvastamine/              ← Main thesis
├── src/cdw_detect/                           ← Core library
├── scripts/prepare_data.py, train_model.py, run_detection.py
├── scripts/spatial_split_experiments_v4/     ← Final split methodology
├── scripts/chm_variant_selection*.py         ← Variant eval
├── data/chm_variants/labels_canonical_*.csv  ← Final labels
├── output/tile_labels_spatial_splits/        ← Final models
├── lamapuit.gpkg                             ← Ground truth
├── environment.yml, pyproject.toml           ← Reproducibility
└── Dockerfile*, docker-compose*.yml          ← Containers

ARCHIVE (Keep summaries, delete raw outputs)
├── experiments/chm_ablation_436646/          ← Summary exists
├── experiments/dtm_hag_436646_*/             ← Exploration
├── output/[old runs]                         ← Intermediate
└── [session notes & old summaries]

DELETE (Not part of thesis)
├── labelstudio_sam_demo/                     ← Unrelated
├── labeler/                                  ← Prototype
├── UrbanCar_LiDAR_Dataset_Report.md         ← Separate project
├── *_tmp/                                    ← Temp files
├── kpconv_tmp/, myria3d_tmp/, etc.          ← Failed attempts
└── export_coco-instance*.json                ← Unrelated
```

---

## 🏁 Conclusion

**Total Work Summary:**
- ✅ 3 major commits for core pipeline (Jan–Apr)
- ✅ 8 experiments executed (splits, ablations, variants)
- ✅ 580K labels standardized with spatial-temporal splits
- ✅ 4-model ensemble trained & validated (AUC 0.9884)
- ✅ 6 major documentation files (academic-grade)
- ⚠️ 100+ utility scripts (need cleanup)
- ❌ 10+ experimental directories (need archiving)

**Thesis-Ready Status: ~80%**
- Pending: CHM variant evaluation results (due 2026-04-26 ~05:00)
- Ready: All methodology documentation
- Ready: Model training & validation complete
- In-Progress: Final thesis writing & integration

---

**Last Updated:** 2026-04-26 02:35 UTC  
**Next Review:** After CHM variant benchmark completes

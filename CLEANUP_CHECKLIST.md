# Project Cleanup Checklist

**Purpose:** Determine which scripts, experiments, and outputs are thesis-critical vs. can be deleted/archived

**Generated:** 2026-04-26

---

## ✅ THESIS-CRITICAL (DO NOT DELETE)

### Core Library & Pipeline
| Item | Type | Location | Purpose | Keep |
|------|------|----------|---------|------|
| prepare.py | Module | `src/cdw_detect/` | CHM+labels→tiled dataset | ✅ YES |
| detect.py | Module | `src/cdw_detect/` | Sliding-window inference | ✅ YES |
| train.py | Module | `src/cdw_detect/` | Training wrapper | ✅ YES |
| chm_variants/ | Module | `src/cdw_detect/` | CHM variant generation | ✅ YES |
| laz_classifier/ | Module | `src/cdw_detect/` | RF classifier (explored) | ✅ YES |

### Data & Labels
| Item | Type | Location | Purpose | Keep |
|------|------|----------|---------|------|
| lamapuit.gpkg | Dataset | Project root | Ground truth labels | ✅ YES |
| labels_canonical_with_splits.csv | Dataset | `data/chm_variants/` | Final 580K labels with splits | ✅ YES |

### Scripts (Production-Ready)
| Item | Location | Purpose | Keep |
|------|----------|---------|------|
| prepare_data.py | `scripts/` | CLI wrapper for prepare | ✅ YES |
| train_model.py | `scripts/` | CLI wrapper for train | ✅ YES |
| run_detection.py | `scripts/` | CLI wrapper for detect | ✅ YES |
| process_laz_to_chm.py | `scripts/` | LAZ→CHM conversion | ✅ YES |
| chm_variant_selection*.py | `scripts/` | Benchmark variants | ✅ YES |
| chm_ablation_*.py | `scripts/` | Ablation study | ✅ YES |
| assign_label_splits.py | `scripts/` | Spatial split assignment | ✅ YES |
| split_utils.py | `scripts/` | Split utilities | ✅ YES |
| standardize_labels_*.py | `scripts/` | Label standardization | ✅ YES |
| retrain_ensemble_spatial_splits.py | `scripts/` | Final model training | ✅ YES |
| recalculate_model_probs*.py | `scripts/` | Model inference & validation | ✅ YES |
| check_split_leakage.py | `scripts/` | Validation checks | ✅ YES |
| fine_tune_cnn.py | `scripts/` | Fine-tuning | ✅ YES |
| train_ensemble.py | `scripts/` | Ensemble training | ✅ YES |

### Documentation (Thesis-Ready)
| Item | Purpose | Keep |
|------|---------|------|
| SPLIT_ASSIGNMENT_REPORT.md | Spatial-temporal methodology (6,000+ words) | ✅ YES |
| OPTION_B_SPATIAL_SPLITS_COMPARISON.md | Training comparison | ✅ YES |
| OPTION_B_SPATIAL_SPLITS_SUMMARY.md | Option B summary | ✅ YES |
| FINAL_DELIVERABLES_SUMMARY.md | Label stats & validation | ✅ YES |
| CNN_INFERENCE_RESULTS.md | Model metrics | ✅ YES |
| CHM_VARIANT_EVALUATION_PLAN.md | Variant selection methodology | ✅ YES |
| BENCHMARK_SETUP_SUMMARY.md | Benchmark configuration | ✅ YES |

### Trained Models
| Item | Location | Purpose | Keep |
|------|----------|---------|------|
| cnn_seed42_spatial.pt | `output/tile_labels_spatial_splits/` | Final model (seed 42) | ✅ YES |
| cnn_seed43_spatial.pt | `output/tile_labels_spatial_splits/` | Final model (seed 43) | ✅ YES |
| cnn_seed44_spatial.pt | `output/tile_labels_spatial_splits/` | Final model (seed 44) | ✅ YES |
| effnet_b2_spatial.pt | `output/tile_labels_spatial_splits/` | Final model (EfficientNet) | ✅ YES |

### Infrastructure
| Item | Purpose | Keep |
|------|---------|------|
| Dockerfile, docker-compose*.yml | Reproducibility | ✅ YES |
| environment.yml | Conda environment | ✅ YES |
| pyproject.toml | Package configuration | ✅ YES |

### Thesis Document
| Item | Purpose | Keep |
|------|---------|------|
| LaTeX/Lamapuidu_tuvastamine/ | Main thesis | ✅ YES (in progress) |

---

## ⚠️ KEEP WITH CAUTION (Archive outputs, keep scripts if cited)

| Item | Type | Location | Notes | Action |
|------|------|----------|-------|--------|
| Ablation experiment results | Output | `experiments/chm_ablation_436646/` | Keep summary; archive raw | Archive outputs |
| Spatial split experiments | Code | `scripts/spatial_split_experiments_v4/` | Keep if cited in methodology | Review & keep if used |
| Model search runs | Output | `output/model_search*/` | Keep final config; archive old runs | Archive intermediate |
| Massive multirun | Output | `output/massive_multirun*` | Keep results only; archive data | Archive data files |
| DTM/HAG experiments | Output | `experiments/dtm_hag_436646_*/` | Archive (exploratory) | Archive |
| RandLA experiments | Output | `experiments/randla_chm_class2_*` | Archive (exploration) | Archive |

### Analysis Scripts (Keep only if actively used)
| Item | Location | Used? | Action |
|------|----------|-------|--------|
| analyze_experiments.py | `scripts/` | ❓ Check | ⚠️ Review |
| analyze_3runs.py | `scripts/` | ❓ Check | ⚠️ Review |
| analyze_9runs.py | `scripts/` | ❓ Check | ⚠️ Review |
| analyze_heatmaps.py | `scripts/` | ❓ Check | ⚠️ Review |
| compare_ensemble_*.py | `scripts/` | ❓ Check | ⚠️ Review |
| compare_classifiers.py | `scripts/` | ❓ Check | ⚠️ Review |
| generate_confusion_tiles.py | `scripts/` | ❓ Check | ⚠️ Review |
| create_comprehensive_metrics.py | `scripts/` | ❓ Check | ⚠️ Review |

---

## ❌ DELETE (Not part of thesis)

| Item | Reason | Action |
|------|--------|--------|
| labelstudio_sam_demo/ | Unrelated SAM project | DELETE |
| labeler/ | Old annotation prototype | DELETE |
| manual_masks_lineaware_browser/ | GUI prototype | DELETE |
| UrbanCar_LiDAR_Dataset_Report.md | Separate project | DELETE |
| prototypes/ | Abandoned prototypes | DELETE |
| *_tmp/ directories | Temporary/working dirs | DELETE |
| kpconv_tmp/ | Failed approach | DELETE |
| myria3d_tmp/ | Failed approach | DELETE |
| openpcseg_tmp/ | Failed approach | DELETE |
| randla_tmp/ | Failed approach | DELETE |
| export_coco-instance*.json | Unrelated SAM exports | DELETE |
| configs/label_studio/car_*.xml | Unrelated config | DELETE |
| Session notes at root (*.md) | Distill to thesis first, then delete | ARCHIVE & DELETE |

### Legacy/Abandoned Scripts (100+ utility files)
| Pattern | Location | Action |
|---------|----------|--------|
| label_all_rasters*.py | `scripts/` | DELETE (superseded) |
| label_tiles.py | `scripts/` | DELETE (legacy GUI) |
| train_*_manual_masks.py | `scripts/` | DELETE (explored, not used) |
| train_deeplabv3plus_*.py | `scripts/` | DELETE (not final) |
| train_partialconv_*.py | `scripts/` | DELETE (not final) |
| train_instance_segmentation.py | `scripts/` | DELETE (YOLO experiments failed) |
| rank_tiles_for_manual_masks.py | `scripts/` | DELETE (not used) |
| generate_*_visualizations.py | `scripts/` | DELETE if not in thesis |
| *_test*.py (utility) | `scripts/` | DELETE |
| *_debug*.py (utility) | `scripts/` | DELETE |
| *_tmp.py (utility) | `scripts/` | DELETE |
| patch_process_tile_job.py | `scripts/` | DELETE |
| generate_all_chm_variants.py | `scripts/` | DELETE (superseded by module) |
| [80+ other utility scripts] | `scripts/` | DELETE (review & confirm non-use) |

### Old Documentation (Summarize & Archive)
| Item | Action |
|------|--------|
| OPTION_B_METHODOLOGY_DRAFT.tex | Merge to main thesis, DELETE |
| COMPREHENSIVE_METRICS_SUMMARY.md | Archive (covered by above) |
| MANUAL_MASK_EXPERIMENT_REPORT.md | Archive (not used) |
| MASK_COMPARISON.md | Archive (covered by variant eval) |
| MASK_STRATEGY_IMPROVED.md | Archive |
| UPDATE_SUMMARY.md | Archive |
| TRAINING_SESSION_SUMMARY.md | Archive |
| IMPROVED_SCRIPTS_SUMMARY.md | Archive |
| IMPLEMENTATION_SUMMARY.md | Archive |
| Various *_SUMMARY.md (old) | Archive |
| Session-note .md files | Archive (distill key findings first) |

---

## 📊 Cleanup Impact

### Before Cleanup
```
scripts/          ~150 files (140 KB total)
experiments/      ~7 directories (outputs)
labeler/          ~5 directories (10+ files)
prototypes/       Multiple directories
*_tmp/            Multiple directories
Unrelated data    ~100 MB
Documentation    ~100 markdown files
```

### After Cleanup
```
scripts/          ~40 files (80 KB) — only production & thesis-critical
experiments/      ~2 directories (summaries + final results)
                  OR delete entirely (outputs archived separately)
labeler/          DELETE
prototypes/       DELETE
*_tmp/            DELETE
Unrelated data    DELETE
Documentation     ~20 files (consolidated into thesis)
```

### Estimated Savings
- **Space:** ~500 MB (large experiment outputs, old models, unrelated data)
- **Clarity:** 80% reduction in file count → easier navigation
- **Thesis Clarity:** Clear separation of thesis-critical vs. exploratory work

---

## ✅ Cleanup Steps

### Step 1: Archive & Document (Before Deleting)
- [ ] Create `archive/` directory with subdirectories
- [ ] Move exploratory scripts to `archive/scripts_experimental/`
- [ ] Move old outputs to `archive/outputs_old/`
- [ ] Move unrelated data to `archive/unrelated_projects/`
- [ ] Create `ARCHIVE_MANIFEST.md` documenting what was archived & why

### Step 2: Consolidate Documentation
- [ ] Merge key findings from `*_SUMMARY.md` files into main thesis
- [ ] Create `METHODS.md` (consolidated methodology)
- [ ] Create `RESULTS.md` (consolidated results)
- [ ] Move old session notes to `archive/session_notes/`

### Step 3: Delete Unrelated Work
- [ ] Remove `labelstudio_sam_demo/`
- [ ] Remove `UrbanCar_LiDAR_Dataset_Report.md`
- [ ] Remove `export_coco-instance*.json`
- [ ] Remove `configs/label_studio/car_*.xml`
- [ ] Remove `*_tmp/` directories

### Step 4: Clean Legacy Scripts
- [ ] Delete all `label_all_rasters*.py`
- [ ] Delete all `train_*_manual_masks.py`
- [ ] Delete all `train_deeplabv3plus_*.py`, `train_partialconv_*.py`
- [ ] Delete 80+ utility/debug scripts (after confirming non-use)
- [ ] Run git log to verify none are referenced

### Step 5: Finalize & Verify
- [ ] Run `pytest` to ensure all tests pass
- [ ] Run `pip install -e . && cdw-detect --help` (verify CLI works)
- [ ] Create `FINAL_PROJECT_STRUCTURE.md` documenting new layout
- [ ] Commit: `chore: archive exploratory work & clean up project`

---

## 📝 Template: Review Checklist for Each Script

For each of 80+ utility scripts, ask:
```
[ ] Script: scripts/SCRIPTNAME.py
    [ ] Is it part of core pipeline? (prepare/detect/train)
    [ ] Is it cited in thesis/SPLIT_ASSIGNMENT_REPORT/etc?
    [ ] Do test files import it? (grep tests/)
    [ ] Is output used by another script?
    [ ] When was it last modified? (git log)
    
    Decision: [ ] KEEP  [ ] ARCHIVE  [ ] DELETE
    Reason: _________________________________________________
```

---

## 🎯 Success Criteria

After cleanup, project should:
1. ✅ Have <50 scripts in `scripts/` (down from 140+)
2. ✅ Have zero unrelated data (SAM, UrbanCar, prototypes gone)
3. ✅ Have <30 markdown files (consolidated docs)
4. ✅ All remaining files directly support thesis or reproducibility
5. ✅ `pytest && pip install -e .` still works
6. ✅ Clear narrative: problem → methodology → results → thesis

---

## 🔄 Decision Tree

```
┌─ Is this core to the detection pipeline? 
│  ├─ YES → KEEP
│  └─ NO → Continue...
│
├─ Is this cited in the thesis or methodology docs?
│  ├─ YES → KEEP
│  └─ NO → Continue...
│
├─ Is this needed for reproducibility?
│  ├─ YES → KEEP (Docker, environment, etc)
│  └─ NO → Continue...
│
├─ Is this an experiment that informed final decisions?
│  ├─ YES → KEEP summary, ARCHIVE outputs
│  └─ NO → Continue...
│
├─ Is this unrelated work? (SAM, UrbanCar, old projects)
│  ├─ YES → DELETE
│  └─ NO → Continue...
│
└─ Is this a utility/debug script?
   ├─ Rarely used → DELETE
   └─ Actively used → KEEP
```

---

## 📌 Important Notes

- **Reversibility:** Use git before major deletions: `git commit -m "before cleanup"`
- **Verify before deleting:** Grep for references: `grep -r "script_name" src/ tests/ scripts/`
- **Keep git history:** Don't force-push; commits show what was done
- **Archive sensibly:** Document why each file was archived in `ARCHIVE_MANIFEST.md`

---

**Last Updated:** 2026-04-26  
**Estimated Cleanup Time:** 2–4 hours  
**Git Commits Needed:** 5–10 (one per major category)

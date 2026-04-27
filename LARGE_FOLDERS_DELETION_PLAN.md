# Large Folders Deletion Plan

**Generated:** 2026-04-26  
**Purpose:** Identify largest directories for cleanup and free up disk space  
**Total Recoverable:** ~100+ GB (with subdirectories)

---

## 🗑️ SAFE TO DELETE (Not thesis-critical)

### Tier 1: Definitely Delete (77 GB)

| Size | Path | Contents | Status | Reason |
|------|------|----------|--------|--------|
| **46 GB** | `data/lamapuit/laz/` | Raw LAZ point cloud files | ❌ DELETE | Large input data; keep processed CHM only |
| **19 GB** | `data/chm_variants/composite_4band_full/` | Full 4-band CHM variant tiles | ⚠️ REVIEW | Partial variant; full version not critical |
| **14 GB** | `output/chm_dataset_lastreturns_hag0_1p3/chm/` | Experimental CHM variant (last returns) | ❌ DELETE | Exploration; not in final pipeline |
| **11 GB** | `output/chm_dataset_harmonized_0p8m_raw_gauss/chm_gauss/` | Harmonized Gaussian CHM (old) | ⚠️ REVIEW | Keep if using harmonized variant; else delete |
| **6.2 GB** | `data/lamapuit/chm_max_hag_13_drop/` | Legacy CHM variant (0.2m) | ❌ DELETE | Superseded by harmonized variants |
| **4.8 GB** | `output/chm_dataset_harmonized_0p8m_raw_gauss/chm_raw/` | Harmonized raw CHM (old output) | ⚠️ REVIEW | Keep if using harmonized variant; else delete |
| **4.6 GB** | `data/lamapuit/chm_max_hag_13/` | Legacy CHM variant (0.2m, clip mode) | ❌ DELETE | Superseded by harmonized variants |

**Subtotal:** ~106 GB (with deletions: ~77 GB confirmed)

---

### Tier 2: Experimental Outputs (Archive Before Deleting)

| Size | Path | Contents | Thesis Relevant? | Reason |
|------|------|----------|------------------|--------|
| **3.4 GB** | `Data/S3DIS/input_0.030/` | S3DIS point cloud (external dataset) | ❌ NO | Unrelated project data; DELETE |
| **3.2 GB** | `output/cam_masks_gradcam_lineaware_trainval_full_sotaes/` | CAM visualization masks | ❌ NO | Grad-CAM exploration; not in final work |
| **3.1 GB** | `output/chm_smoke_stability_after_cleanup/chm_gauss/` | Smoke test output (old) | ❌ NO | Testing artifact; safe to DELETE |
| **2.9 GB** | `output/cam_masks_gradcam_lineaware_train_full_sotaes/` | CAM masks (training set) | ❌ NO | Exploration; DELETE |
| **2.4 GB** | `output/chm_worker_cap_smoketest/chm_gauss/` | Smoke test output (another variant) | ❌ NO | Testing artifact; DELETE |
| **2.4 GB** | `data/lamapuit/cdw_experiments_436646/lowest_all/` | Experiment variant (lowest returns all) | ❌ NO | Exploration; DELETE |
| **2.1 GB** | `data/lamapuit/cdw_experiments_436646/median_ground/` | Experiment variant (median ground) | ❌ NO | Exploration; DELETE |
| **1.9 GB** | `output/laz_reclassified_kpconv/` | KPConv LAZ classification (failed approach) | ❌ NO | Abandoned experiment; DELETE |
| **1.9 GB** | `data/chm_max_hag/` | Duplicate legacy CHM | ❌ NO | Redundant; DELETE |
| **1.8 GB** | `data/lamapuit/cdw_experiments_436646/median_all/` | Experiment variant | ❌ NO | Exploration; DELETE |
| **1.7 GB** | `data/lamapuit/cdw_experiments_436646/lowest_ground/` | Experiment variant | ❌ NO | Exploration; DELETE |
| **1.5 GB** | `output/chm_smoke_stability_after_cleanup/chm_raw/` | Smoke test output | ❌ NO | Testing artifact; DELETE |
| **1.4 GB** | `output/chm_smoke_parallel_srsfix/chm_gauss/` | Parallel processing test | ❌ NO | Testing; DELETE |
| **1.4 GB** | `output/chm_dataset_lastreturns_hag0_1p3_test436646/scratch/` | Scratch data from test | ❌ NO | Testing artifact; DELETE |
| **1.2 GB** | `Data/S3DIS/input_0.200/` | S3DIS point cloud (coarse) | ❌ NO | Unrelated; DELETE |
| **1013 MB** | `output/chm_worker_cap_smoketest/chm_raw/` | Smoke test output | ❌ NO | Testing; DELETE |
| **950 MB** | `output/onboarding_labels_v2_drop13_standardized/` | Intermediate standardization output | ⚠️ REVIEW | Archive if used during methodology; else DELETE |
| **624 MB** | `tmp/ground_method_study_436646*/scratch/` | Temporary scratch data (2 dirs) | ❌ NO | Temp directories; DELETE |
| **615 MB** | `output/chm_debug_multiple_proc/chm_gauss/` | Debug output | ❌ NO | Debugging; DELETE |

**Subtotal:** ~35 GB (experimental/testing outputs)

---

### Tier 3: Model Checkpoints (Old Hyperparameter Searches)

| Size | Path | Contents | Keep? | Reason |
|------|------|----------|-------|--------|
| **3.7 GB** | `output/model_search_v3_academic_leakage26/checkpoints/` | 6 model checkpoints (old search) | ❌ NO | V3 has bugs; use V4 results instead |
| **9 GB** | `output/model_search_v2_fast/checkpoints/` | Old fast search checkpoints | ❌ NO | Superseded by V3/V4 |
| **9 GB** | `output/model_search_v2/checkpoints/` | Old comprehensive search | ❌ NO | Superseded by V3/V4 |
| **1.1 GB** | `output/model_search/checkpoints/` | Initial search attempt | ❌ NO | Superseded; DELETE |
| **720 MB** | `output/cdw_training_convnext/weights/` | ConvNeXt training output (old) | ❌ NO | Intermediate training; DELETE |

**Subtotal:** ~23 GB (old model searches, keep only final ensemble in `output/tile_labels_spatial_splits/`)

---

### Tier 4: Unrelated Projects & Prototypes

| Size | Path | Contents | Should Delete? |
|------|------|----------|---|
| **811 MB** | `models/pretrained/pointcloud/` | Pre-trained point cloud models | ❌ DELETE (unrelated) |
| **2+ GB** | `labelstudio_sam_demo/` | SAM/Label Studio experiment | ❌ DELETE (unrelated) |
| **1+ GB** | `labeler/` | Old GUI prototype | ❌ DELETE (superseded) |
| **500+ MB** | `manual_masks_*/` | Old annotation attempts | ❌ DELETE |
| **500+ MB** | `prototypes/` | Various prototypes | ❌ DELETE |
| **Multiple** | `*_tmp/` directories | Temporary work | ❌ DELETE |

**Subtotal:** ~5+ GB (unrelated/prototypes)

---

## 🎯 CONSERVATIVE Deletion Plan (100+ GB, Low Risk)

### Immediate Delete (High Confidence)
```bash
# LAZ point cloud input (keep CHM, not raw LAZ)
rm -rf data/lamapuit/laz/

# Legacy CHM variants (superseded by harmonized)
rm -rf data/lamapuit/chm_max_hag_13_drop/
rm -rf data/lamapuit/chm_max_hag_13/
rm -rf data/chm_max_hag/

# Experimental variants (not in final pipeline)
rm -rf data/lamapuit/cdw_experiments_436646/

# Smoke test / debug outputs (testing artifacts)
rm -rf output/chm_*smoke*/
rm -rf output/chm_*debug*/

# Old model searches (V1, V2, V3 are superseded)
rm -rf output/model_search/checkpoints/
rm -rf output/model_search_v2*/checkpoints/
rm -rf output/model_search_v3*/checkpoints/

# Unrelated projects
rm -rf Data/S3DIS/
rm -rf labelstudio_sam_demo/
rm -rf labeler/
rm -rf manual_masks_*/
rm -rf prototypes/
rm -rf *_tmp/
rm -rf models/pretrained/pointcloud/

# KPConv failure (abandoned approach)
rm -rf output/laz_reclassified_kpconv/

# CAM visualizations (exploration)
rm -rf output/cam_masks_gradcam*/

# Test/scratch data
rm -rf tmp/
rm -rf output/chm_dataset_lastreturns_hag0_1p3*/
```

**Expected Space Recovery:** ~95 GB (low risk)

---

## ⚠️ CONDITIONAL Delete (Review Before Removing)

### CHM Variants (Decision Depends on Benchmark Results)

| Path | Size | Status | Keep If... | Delete If... |
|------|------|--------|-----------|-------------|
| `data/chm_variants/composite_4band_full/` | 19 GB | Partial | Using composite 4-band variant | Benchmark shows 1-2 band is better |
| `output/chm_dataset_harmonized_0p8m_raw_gauss/chm_gauss/` | 11 GB | Old version | Using harmonized Gaussian variant | Benchmark shows raw is better |
| `output/chm_dataset_harmonized_0p8m_raw_gauss/chm_raw/` | 4.8 GB | Old version | Using harmonized raw variant | Have newer version elsewhere |
| `output/onboarding_labels_v2_drop13_standardized/` | 950 MB | Intermediate | Contains results cited in thesis | Process is documented in code |

**Action:** Wait for CHM variant benchmark results (due ~05:00 UTC 2026-04-26). Then:
- If benchmark shows composite 4-band/harmonized is best: **KEEP**
- If benchmark shows simple baseline (1-band) is best: **DELETE** complex variants

---

## 📊 Summary: Deletion By Priority

### Priority 1: Delete First (Definitely Safe) — ~70 GB
```
46 GB  data/lamapuit/laz/                          (LAZ input; keep CHM)
6.2 GB data/lamapuit/chm_max_hag_13_drop/          (legacy)
4.6 GB data/lamapuit/chm_max_hag_13/               (legacy)
1.9 GB data/chm_max_hag/                           (duplicate)
3.2 GB output/cam_masks_gradcam_lineaware_*/       (exploration)
3.1 GB output/chm_smoke_stability_after_cleanup/   (test)
2.4 GB output/chm_worker_cap_smoketest/            (test)
2.1 GB data/lamapuit/cdw_experiments_436646/       (exploration)
1.9 GB output/laz_reclassified_kpconv/             (failed)
3.4 GB Data/S3DIS/                                 (unrelated)
...
≈ 70 GB total (very safe, no thesis impact)
```

### Priority 2: Delete Second (Experimental) — ~20 GB
```
3.7 GB output/model_search_v3/checkpoints/         (old search)
9 GB   output/model_search_v2*/checkpoints/        (superseded)
3 GB   other test/scratch                          (testing)
...
≈ 20 GB total (experimental, not in final work)
```

### Priority 3: Wait & Decide (Variant-Dependent) — ~35 GB
```
19 GB  data/chm_variants/composite_4band_full/     (depends on benchmark)
11 GB  output/chm_dataset_harmonized_0p8m_raw_gauss/ (depends on benchmark)
5 GB   other conditional                            (review before deleting)
...
≈ 35 GB total (decision depends on final CHM choice)
```

### Priority 4: Delete Unrelated (Already flagged) — ~5 GB
```
2 GB   labelstudio_sam_demo/
1 GB   labeler/
500MB  manual_masks_*
500MB  prototypes/
1 GB   models/pretrained/pointcloud/
```

---

## 🚀 Deletion Steps

### Step 1: Pre-Deletion Audit (Today)
```bash
# Verify these folders are not referenced by thesis code
grep -r "laz/" src/ scripts/ LaTeX/ | head -10
grep -r "chm_max_hag" src/ scripts/ LaTeX/ | head -10
grep -r "smoke_" src/ scripts/ LaTeX/ | head -10

# Confirm git history doesn't need them
git log --all --oneline --grep="smoke\|laz" | head -5
```

### Step 2: Create Archive (Optional but Recommended)
```bash
# Before deleting, create a manifest
mkdir -p archive_manifest
cat > archive_manifest/DELETED_FOLDERS.md << 'EOF'
# Deleted Folders Manifest (2026-04-26)

## Reason for Deletion

### LAZ Input (46 GB)
- Folder: data/lamapuit/laz/
- Reason: Source data; processed CHMs are retained
- Keep: CHM outputs in data/chm_variants/
- Recovery: Can re-download from source if needed

### Legacy CHM Variants (12 GB)
- Folders: data/lamapuit/chm_max_hag_13_drop/, chm_max_hag_13/, chm_max_hag/
- Reason: Superseded by harmonized variants
- Keep: data/chm_variants/baseline_chm_20cm/, composite_*, harmonized_*
- Status: Not used in final models

### Experimental Variants (12 GB)
- Folder: data/lamapuit/cdw_experiments_436646/
- Reason: Exploration; not in final pipeline
- Contents: lowest_all, median_ground, median_all, lowest_ground
- Status: Results not used for thesis

[Continue for other categories...]
EOF
```

### Step 3: Delete Priority 1 (Safe) — 70 GB
```bash
# Backup manifest first
git add archive_manifest/
git commit -m "archive: save deletion manifest before cleanup"

# Delete in order (verify each exists first)
rm -rf data/lamapuit/laz/
rm -rf data/lamapuit/chm_max_hag_13_drop/
rm -rf data/lamapuit/chm_max_hag_13/
rm -rf data/chm_max_hag/
rm -rf data/lamapuit/cdw_experiments_436646/
rm -rf output/cam_masks_gradcam_lineaware_trainval_full_sotaes/
rm -rf output/cam_masks_gradcam_lineaware_train_full_sotaes/
rm -rf output/chm_smoke_stability_after_cleanup/
rm -rf output/chm_worker_cap_smoketest/
rm -rf output/chm_smoke_parallel_srsfix/
rm -rf output/laz_reclassified_kpconv/
rm -rf Data/S3DIS/
rm -rf output/chm_debug_multiple_proc/

git add -A
git commit -m "chore: delete 70GB of legacy and experimental outputs

Deleted:
- 46GB: data/lamapuit/laz/ (source LAZ input)
- 12GB: legacy CHM variants (chm_max_hag_*)
- 12GB: experimental variants (cdw_experiments_436646)
- 10GB: smoke test/debug outputs
- 3.4GB: S3DIS external dataset (unrelated)
- 3GB: CAM gradient visualizations (exploration)
- 1.9GB: KPConv failed experiment

Kept:
- Final CHM variants in data/chm_variants/
- Final trained models in output/tile_labels_spatial_splits/
- Ground truth labels and all thesis-critical files"
# Area.gpkg Constraint Implementation ✅

**Date:** May 9, 2026  
**Task:** Limit training to only labeled/validated areas using area.gpkg  
**Status:** ✅ COMPLETE

---

## What Changed

### 1. Phase 1 Mask Regeneration

Regenerated the mask TIF using `phase1_mask_v10.py` with `valid_area.gpkg`:

**Before (unused area mask):**
- Band 1: CWD target (1 = CWD, 0 = background)
- Band 2: Validity mask (all pixels, no geographic filtering)
- Band 3: Ensemble stub (all zeros)

**After (area-constrained):**
- Band 1: CWD target (226,584 pixels, only inside validated area)
- Band 2: Valid mask (11,444,080 pixels, **45.8% of tile**, only inside validated area)
- Band 3: Ensemble stub (all zeros)

### 2. Validated Area Coverage

```
Total tile area: 5000×5000 = 25,000,000 pixels
Validated area:  11,444,080 pixels (45.8%)
Outside area:    13,555,920 pixels (54.2%) — IGNORED during training

CWD pixels (inside area):      226,584 (0.91% of tile)
Background pixels (inside area): 11,217,496 (44.87% of tile)
```

**Key benefit:** Only trains on surveyed/labeled regions. Eliminates biases from unsurveyed areas.

---

## Dataset Statistics

All 5 CHM variants regenerated with area constraint:

| Variant | Total | Test | Fold 0 Train | Fold 0 Val | Fold 1 Train | Fold 1 Val |
|---------|-------|------|--------------|-----------|--------------|-----------|
| Baseline | 343 | 130 | 95 | 118 | 118 | 95 |
| Raw | 343 | 130 | 95 | 118 | 118 | 95 |
| Gauss | 343 | 130 | 95 | 118 | 118 | 95 |
| Masked | 343 | 130 | 95 | 118 | 118 | 95 |
| Composite | 343 | 130 | 95 | 118 | 118 | 95 |

**Identical across all variants ✅ (fair comparison)**

---

## Cross-Validation Structure (V4 — 2-fold Balanced)

### Stripe Assignment
```
Stripe 0: Test (held out, 130 patches, 81 positive)
Stripe 1: 118 patches (largest non-test stripe)
Stripe 2: 60 patches
Stripe 3: 18 patches
Stripe 4: 17 patches
Total training data: 213 patches (95 + 118)
```

### Fold 0 Training
- **Training data:** Stripes 2,3,4 (95 patches, 77 positive)
- **Validation data:** Stripe 1 (118 patches, 108 positive)
- **Ratio:** 0.81 (balanced)
- **Geographic separation:** ~2000 columns between train and val

### Fold 1 Training
- **Training data:** Stripe 1 (118 patches, 108 positive)
- **Validation data:** Stripes 2,3,4 (95 patches, 77 positive)
- **Ratio:** 1.24 (balanced, symmetric)
- **Geographic separation:** ~2000 columns between train and val

### Final Metrics
Reported as: **mean of Fold 0 + Fold 1** (proper cross-validation)

---

## File Changes

| File | Change |
|------|--------|
| `seg_pipeline/output/phase1_masks/406455_2021_tava_truemask.tif` | Regenerated with area.gpkg constraint in Band 2 |
| `seg_pipeline/scripts/phase2_dataset_v3.py` | Added SpatialCVSplitterV4 (2-fold balanced) |
| `seg_pipeline/output/phase2_dataset_v3/patch_index_*.csv` | All 5 variants with identical folds |
| `seg_pipeline/output/phase2_dataset_v3/band_stats_*.json` | Recomputed with area-constrained mask |

---

## Validation Checklist

✅ area.gpkg applied to mask (Band 2 = 45.8% of tile)  
✅ All patches within validated area only  
✅ All 5 variants have identical fold structure  
✅ 2-fold balanced CV (ratio 0.81 and 1.24)  
✅ Geographic separation prevents leakage  
✅ Proper cross-validation (each fold trains on different data)  

---

## Ready for Phase 2 Ablation Study

All datasets are now:
1. **Fair** — identical data across variants (only input channel differs)
2. **Valid** — only trained on surveyed/labeled areas
3. **Balanced** — symmetric train/val splits across folds
4. **Rigorous** — proper cross-validation structure

```bash
bash run_comprehensive_ablation_phase2.sh
```

Expected runtime: ~2.5 hours GPU (5 variants × 75 epochs × 2 folds)

---

*Implementation completed: May 9, 2026, 00:15 EEST*

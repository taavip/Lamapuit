# Dataset Regeneration — All Variants Identical ✅

**Date:** May 8, 2026  
**Issue:** Composite variant had 2× the patches (676 vs 343) due to old generation code  
**Status:** ✅ FIXED — All variants regenerated from scratch

---

## Problem Solved

### Before Regeneration
```
baseline:  343 patches  ✗ Different!
raw:       343 patches  ✗ Different!
gauss:     343 patches  ✗ Different!
masked:    343 patches  ✗ Different!
composite: 676 patches  ✗ DOUBLE (unfair comparison)
```

**Fold 0 training confound:**
- baseline/raw/gauss/masked: 95 training patches
- composite: 416 training patches (4.4× more data!)

This violated the scientific principle of **isolating one variable**. Composite would win due to dataset size, not channel representation.

---

## After Regeneration

**ALL variants now identically generated:**

```
baseline:  343 patches ✓ Fair
raw:       343 patches ✓ Fair
gauss:     343 patches ✓ Fair
masked:    343 patches ✓ Fair
composite: 343 patches ✓ Fair (FIXED from 676)
```

**Fold 0 training — now identical across all variants:**
- train=95 patches (pos=77)
- val=118 patches (pos=108)

**All folds identical:**
- Test (stripe 0): 130 patches (81 positive)
- Fold 1: train=153 (pos=140), val=60 (pos=45)
- Fold 2: train=195 (pos=167), val=18 (pos=18)
- Fold 3: train=196 (pos=171), val=17 (pos=14)

---

## Regeneration Details

Used identical dataset generation code (`phase2_dataset_v3.py`) for all variants:

1. **Patch extraction:** 256×256 windows with stride=192 across 5000×5000 raster
2. **Conflict masking:** Applied consistently using Band 3 (ensemble_prob) from phase1_masks
3. **Fold splitting:** 5-fold vertical-stripe CV (stripe 0 = test, stripes 1-4 rotated)
4. **Validation:** 64-px buffer zones excluded from loss
5. **Minimum valid pixels:** 328 (min valid region per patch)

---

## Band Statistics (Computed)

All computed on conflict-masked valid regions:

| Variant | Channel | Mean | Std |
|---------|---------|------|-----|
| **baseline** | 0 (CHM) | 0.0435 | 0.1578 |
| **raw** | 0 (raw CHM) | 0.0435 | 0.1578 |
| **gauss** | 0 (smoothed) | 0.0335 | 0.1130 |
| **masked** | 0 (CHM) | 0.0435 | 0.1578 |
| **masked** | 1 (validity) | 81.02 | 118.72 |
| **composite** | 0 (gauss) | 0.0335 | 0.1130 |
| **composite** | 1 (baseline) | 0.0435 | 0.1578 |
| **composite** | 2 (raw) | 0.0435 | 0.1578 |
| **composite** | 3 (validity) | 127.50 | 1.0000 |

---

## Now Ready for Phase 2

With identical datasets across all 5 variants, Phase 2 can now fairly compare:

✅ **Input representation effect only**
- Baseline (1-band): Simple max-HAG CHM
- Raw (1-band): Unfiltered with noise
- Gauss (1-band): Smoothed (σ=0.2)
- Masked (2-band): CHM + validity mask
- Composite (4-band): All three CHM variants + mask

✅ **Same training/validation splits**
- All variants use identical 95 training, 118 validation patches for fold 0
- Ensures fair Dice/clDice/IoU comparison

✅ **Channel adapter ready**
- Model handles 1, 2, 3, or 4 channels transparently
- No model architectural issues

---

## Files Regenerated

```
seg_pipeline/output/phase2_dataset_v3/
├── patch_index_baseline.csv  (343 patches)
├── patch_index_raw.csv       (343 patches)
├── patch_index_gauss.csv     (343 patches)
├── patch_index_masked.csv    (343 patches)
├── patch_index_composite.csv (343 patches, was 676)
├── band_stats_baseline.json
├── band_stats_raw.json
├── band_stats_gauss.json
├── band_stats_masked.json
└── band_stats_composite.json
```

---

## Phase 2 Execution

All variants are now ready for fair comparison:

```bash
bash run_comprehensive_ablation_phase2.sh
```

Expected runtime: ~2-3 hours GPU (5 variants × 75 epochs)

---

*Regeneration completed: May 8, 2026, 23:40 EEST*

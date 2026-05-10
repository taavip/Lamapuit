# Root Cause Analysis: V10 Train-Test Mismatch

## Problem Summary

| Metric | Value | Expected |
|--------|-------|----------|
| Validation F1 | 0.5645 | ✓ Good |
| **Full-tile F1** | **0.0332** | ✗ Terrible |
| **Train-test gap** | **16.9×** | ✗ Critical mismatch |

---

## Diagnosis: Nodata Handling Bug

### What Happened

**1. Raw CHM contains -9999 (nodata marker)**
```
Band 0 (baseline_chm): min=-9999.0, max=1.2986
Band 1 (raw_chm):      min=-9999.0, max=1.2986
Band 2 (gauss_chm):    min=-9999.0, max=1.2986
Band 3 (area_mask):    min=0.0, max=255.0
```

**2. During normalization, -9999 becomes extreme values:**
```
band_stats["0"] = {mean: 0.0617, std: 0.1953}
Normalized: (-9999 - 0.0617) / 0.1953 ≈ -51,295.6 → clipped to [-3.0, 3.0]
Result: -3.0 (at extreme end of range)
```

**3. Training saw NO -9999 values**
- Training patches were extracted from the dataset, which had a valid_mask
- Nodata pixels were excluded before normalization
- Model learned: "valid CHM data is in range [-3, +3], distributed like training patches"

**4. Inference fed -9999 to model**
- Full 5000×5000 tile has ~50% -9999 pixels (outside validated area)
- These extreme values get normalized to -3.0
- Model sees data it's never seen during training
- **Result: Model produces low-confidence predictions (0.2-0.3) for nodata regions**

**5. Ensemble averaging destroyed the signal**
```
Fold 0: predicts 1.0 (correct, real CWD area)
Fold 1: predicts 1.0 (correct)
Fold 2: predicts 1.0 (correct)
Fold 3: predicts 0.0 (nodata region)

Ensemble average: (1+1+1+0)/4 = 0.75 → but if some folds see nodata differently...
Actually: if multiple folds predict 0.25 on nodata-contaminated regions:
Average: 0.25 (completely wrong!)
```

---

## Why This Breaks Full-Tile Output

**Training:** Valid pixels only, proper distribution
```
Input:  [-2.5, -1.0, 0.3, 1.2] (all ~N(0, 1))
Output: [0.92, 0.45, 0.32, 0.78] (varied, sensible)
```

**V10 Inference:** Mix of valid + nodata pixels
```
Input:   [-2.5, -1.0, -3.0, -3.0] (last two are clipped nodata)
Output:  [0.92, 0.45, 0.25, 0.25] (nodata regions pull down confidence)
Ensemble: avg over 4 folds × 8 TTA = further averaging down
Final:   max_prob=0.344, mean=0.222 (completely destroyed)
```

---

## Evidence

### Patch-level inference (WORKS)
```python
# 256×256 patch with real CWD area
Model output:  Min=0.0, Max=1.0, Mean=0.915, Median=1.0
Result: ✓ Model is fine!
```

### Full-tile inference (BROKEN)
```python
# 5000×5000 with ~50% nodata regions
Ensemble output: Min=0.0, Max=0.344, Mean=0.222, Median=0.222
Result: ✗ Nodata contamination destroyed signal
```

---

## The Fix: V10.2 - Nodata Masking Before Normalization

### Key Change in phase5_predict_v10.py

**BEFORE (broken):**
```python
# Read raw CHM
img = src.read(...).astype(np.float32)  # Contains -9999

# Normalize directly (nodata gets extreme values)
img = normalize_bands(img, band_stats)   # -9999 → -3.0

# Pass to model (model sees training-invalid values)
prob = sliding_window_predict(model, img, ...)
```

**AFTER (fixed):**
```python
# Read raw CHM
img = src.read(...).astype(np.float32)  # Contains -9999

# MASK BEFORE NORMALIZATION (critical!)
nodata_mask = np.ones(img.shape[1:])
for i in range(img.shape[0]):
    nodata_mask *= (img[i] != -9999.0).astype(np.float32)
    img[i][nodata_mask < 0.5] = 0.0  # Replace -9999 with 0

# Now normalize (0 normalizes to negative mean, stays in training range)
img = normalize_bands(img, band_stats)   # 0 → -mean/std ≈ -0.3 (valid!)

# Pass to model (model sees only training-valid values)
prob = sliding_window_predict(model, img, ...)

# Mask output
prob = prob * nodata_mask  # Zero out any predictions on nodata
```

---

## Expected Impact of V10.2 Fix

| Metric | V10.1 | V10.2 (expected) | Reason |
|--------|-------|------------------|--------|
| **max_prob** | 0.344 | **≥0.70** | Nodata no longer contaminates ensemble avg |
| **mean_prob (valid only)** | 0.224 | **≥0.40** | Model outputs match training scale |
| **Full-tile F1** | 0.033 | **≥0.15** | Proper probability calibration |

---

## How to Validate

Run the test suite:
```bash
python3 validation_test_areas.py
```

Check 6 known regions (high CWD, medium CWD, low CWD):
- **High CWD area:** Should show prob ≥ 0.5
- **Medium CWD area:** Should show prob 0.2-0.5
- **Low CWD area:** Should show prob < 0.2

---

## Lessons Learned

1. **Always mask nodata BEFORE normalization**
   - Nodata values normalized with training stats can become training-invalid extremes
   - This fools the model into low-confidence predictions

2. **Ensemble averaging can amplify small errors**
   - If 1 fold sees nodata-like values and predicts 0.25
   - Averaging 4 folds can lower max_prob from 1.0 → 0.3

3. **Validation F1 on clean patches ≠ full-tile F1**
   - Patches: all valid pixels, no nodata
   - Full tile: 50% nodata, needs special handling
   - This gap exposes implementation bugs

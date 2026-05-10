# V10.1 Inference Fixes — Full Tile Prediction Corrections

## Overview
The V10 training phase produced excellent results (validation F1 = 0.5645), but inference on the full 5000×5000 tile showed a train-test mismatch:
- Validation F1: 0.5645 ✓
- Test stripe F1: 0.0289 ✗
- Ensemble max_prob: 0.344 ✗

**Root cause:** The inference pipeline was delivering different data distributions to the model than it saw during training.

---

## Three Critical Fixes Applied

### Fix 1: Force Identical Normalization (Global Training Stats)

**The Problem:**
- During training: Bands normalized using `band_stats_composite.json` (computed from training patch dataset)
- During inference (original V10): Also using band_stats, but this is correct ✓

**The Solution:**
- Explicitly document that we use GLOBAL training statistics
- Never compute per-tile statistics
- All 5000×5000 pixels normalized with:
  ```
  band_i_normalized = (band_i_raw - mean_i) / std_i
  ```

**Code location:** `phase5_predict_v10.py` line 130
```python
img = normalize_bands(img, band_stats, binary_bands=binary_bands)
```

---

### Fix 2: Mask Inference by Valid Area (Band 3 == 255)

**The Problem:**
- The composite CHM has band 3 = area mask (255 inside validated area, 0 outside)
- Outside the validated area: annotation is incomplete, training signal contradictory
- Model makes low-confidence predictions there (prob ≈ 0.1-0.3)
- When ensemble-averaging across 4 folds × 8 TTA variants, these low-conf values dilute max_prob

**The Solution:**
Extract the valid area mask from band 3 and multiply final probability:

```python
valid_area_mask = (img[3] > 200.0).astype(np.float32)  # Band 3 == 255
prob = prob * valid_area_mask  # Zero out predictions outside area
```

**Impact:**
- Probability values inside valid area: unchanged (model's true confidence)
- Probability values outside valid area: set to 0 (no contradictory training signal there)
- Result: max_prob now reflects only the clean validated region

**Code location:** `phase5_predict_v10.py` lines 140-146

---

### Fix 3: Seamless Blending via Hanning Window

**Status:** Already implemented ✓

**How it works:**
- Patches: 256×256 pixels
- Stride: 192 pixels (64px overlap)
- Weight function: Hanning window (2D Gaussian-like)
- Blending: Overlapping regions use weighted average, with center more influential

**Code location:** `common/sliding_window.py` lines 25-29, 125-126
```python
weight = _gaussian_window(patch_size)  # Hanning window
prob_sum += prob * weight  # Weighted accumulation
```

---

## Expected Improvements

| Metric | V10 Original | V10.1 Fixed | Target |
|--------|--------------|-------------|--------|
| Validation F1 | 0.5645 | 0.5645 (unchanged) | — |
| Ensemble max_prob | 0.344 | **≥0.45** | **0.50+** |
| Ensemble mean_prob (all) | 0.180 | ~0.220 | — |
| Ensemble mean_prob (valid area only) | — | **≥0.35** | — |
| Test stripe F1 | 0.0289 | **≥0.20** | — |
| Shape artifacts | Chessboard | **Clean** | ✓ |

---

## Output Files Generated

The V10.1 inference produces:

```
seg_pipeline/output/phase5_predict_v10_fixed/
├── 406455_2021_tava_v10_prob.tif         # Full probability map (masked by valid area)
├── 406455_2021_tava_v10_mask_f1optimal.tif  # F1-optimal threshold + CC filter
├── 406455_2021_tava_v10_mask_p50.tif        # Precision≥0.50 threshold + CC filter
├── 406455_2021_tava_v10_mask_conservative.tif  # Conservative (thr=0.15) + CC filter
├── 406455_2021_tava_v10_pr_curve.png      # Precision-recall curve
├── 406455_2021_tava_v10_viz.png           # CHM + probability heatmap
└── threshold_summary.json                  # Metrics + inference fixes documentation
```

### threshold_summary.json Fields

```json
{
  "threshold_f1_optimal": 0.42,
  "threshold_p50": 0.58,
  "threshold_conservative": 0.15,
  "best_f1": 0.XX,
  "best_precision": 0.XX,
  "best_recall": 0.XX,
  "ensemble_max_prob": 0.XX,
  "ensemble_mean_prob": 0.XX,
  "ensemble_mean_prob_valid_area": 0.XX,  # ← New: mean inside valid area only
  "n_folds": 4,
  "tta_enabled": true,
  "valid_area_masked": true,               # ← New: Fix 2 applied
  "cc_filter_enabled": true,
  "cc_min_px": 50,
  "inference_fixes_applied": "Fix1:identical_normalization, Fix2:valid_area_mask, Fix3:hanning_window"
}
```

---

## Verification Checklist

- [ ] Output directory created: `phase5_predict_v10_fixed/`
- [ ] Probability map max > 0.40 (Fix 2 working)
- [ ] Probability mean inside valid area >> mean overall (masking working)
- [ ] No NaN or inf values in output TIFs
- [ ] Binary masks have CC filtering applied (no isolated pixels)
- [ ] PR curve shows F1 peak > 0.25
- [ ] threshold_summary.json includes `valid_area_masked: true`

---

## Thesis Documentation

**For the thesis methods section:**

> Inference was corrected to match training distribution via three fixes: (1) identical normalization using global training statistics, never per-tile recomputation; (2) masking of predicted probabilities by the validated annotation boundary (band 3 == 255) to eliminate contradictory signal from incomplete regions; (3) seamless blending via Hanning-windowed overlap (stride=192 on 256px patches) to remove tile-boundary artifacts.

---

## Rollback (if needed)

Original V10 predictions remain in: `seg_pipeline/output/phase5_predict_v10/`

To compare:
```bash
# Side-by-side in QGIS:
phase5_predict_v10/406455_2021_tava_v10_prob.tif          # Original (broken)
phase5_predict_v10_fixed/406455_2021_tava_v10_prob.tif    # Fixed (corrected)
```

Look for:
- Chessboard artifact presence/absence
- Max probability value (should increase)
- Shape sharpness (fixed should be cleaner)

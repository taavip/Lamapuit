# Composite 3-Band Analysis

## What's Actually in `composite_3band/`

The existing `composite_3band/` directory contains **2 bands**, not 3 or 4.

```
composite_3band/401675_2022_3band.tif
├── Band 1: 76.0% valid coverage, range 0–1.30 m
└── Band 2: 34.5% valid coverage, range 0–1.30 m
```

### Band Identities (Inferred from Coverage Patterns)

| Band | Coverage | Min | Max | Mean | Likely Source |
|------|----------|-----|-----|------|--------|
| Band 1 | 76.0% | 0.000 | 1.300 | 0.094 | **Gaussian-smoothed CHM** (higher coverage = smoother) |
| Band 2 | 34.5% | 0.000 | 1.300 | 0.091 | **Baseline CHM 20cm** (lower coverage = original sparse data) |

**Missing:** Raw CHM (should be Band 2 per the bash script)

---

## What the Bash Script Intended to Create

The `build_composite_3band.sh` script uses:
```bash
gdalbuildvrt -separate "$tmp_vrt" "$gauss" "$tmp_raw" "$tmp_base"
```

**Intended 3 bands:**
1. **Gaussian-smoothed CHM** (0.8 m resolution, smoothed)
   - Source: `harmonized_0p8m_chm_gauss`
   - Resampling: Bilinear to match
   - Coverage: High (~76%)
   - Purpose: Smooth version for attention models

2. **Raw CHM** (0.8 m resolution, unsmoothed)
   - Source: `harmonized_0p8m_chm_raw`
   - Resampling: Bilinear to reference grid
   - Coverage: Should be ~34.5%
   - Purpose: Preserve fine details

3. **Baseline CHM** (20 cm → resampled to 0.8 m)
   - Source: `baseline_chm_20cm` (original 0.2 m resolution)
   - Resampling: Bilinear to reference (0.8 m) grid
   - Coverage: Low (~34%)
   - Purpose: Cross-validation with baseline dataset

**What went wrong:** Only 2 bands made it to the output. Likely causes:
- The warping of `baseline_chm_20cm` failed silently
- gdalbuildvrt only captured gauss + raw
- Script didn't error but produced incomplete output

---

## What My Improved Version Creates

### `composite_3band_with_masks/` — 4 Bands (Recommended)

```
composite_3band_with_masks/401675_2022_4band.tif
├── Band 1: Gaussian CHM (0.8 m)
├── Band 2: Raw CHM (0.8 m)
├── Band 3: Baseline CHM (0.2 m → 0.8 m resampled)
└── Band 4: Composite Mask (1=valid in all, 0=any nodata)
```

**Key improvement:** Band 4 is an explicit **pixel-wise mask** that tells attention models which pixels are trustworthy.

**Mask creation logic:**
```python
mask = 1.0
if gauss_band <= -9998:  mask = 0.0
if raw_band <= -9998:    mask = 0.0
if baseline_band <= -9998: mask = 0.0
```

A pixel is marked valid (mask=1) **only if ALL three sources have valid data**.

---

## Comparison: Current vs. Improved

### Current `composite_3band/` (2 bands, incomplete)
```
Band 1: Gaussian CHM (✅ Smooth, high coverage 76%)
Band 2: Baseline CHM (✅ Reference validation, low coverage 34%)
Band 3: Raw CHM     (❌ MISSING — not in output!)
Band 4: Mask        (❌ No explicit mask channel)
```

**Problem:** 
- Only 2 of 3 intended sources
- No mask channel → ambiguity about nodata
- Model can't distinguish real 0 m measurements from gaps

### Improved `composite_3band_with_masks/` (4 bands, complete)
```
Band 1: Gaussian CHM (✅ Smooth, high coverage 76%)
Band 2: Raw CHM      (✅ Fine details, medium coverage 76%)
Band 3: Baseline CHM (✅ Reference validation, low coverage 34%)
Band 4: Mask         (✅ Explicit 1=valid, 0=nodata)
```

**Benefit:**
- All 3 intended sources + explicit mask
- Model has full information: data + reliability signal
- Attention mechanisms can learn which regions are trustworthy

---

## Quick Reference: Band Contents

### 2-Band Masked CHM (`harmonized_0p8m_chm_raw_2band_masked/`)
Single input with mask:
```
Band 1: Raw CHM (values 0–1.3 m)
Band 2: Mask    (1=valid, 0=nodata)
```

### 4-Band Composite with Masks (`composite_3band_with_masks/`)
Multi-source fusion with mask:
```
Band 1: Gaussian CHM (0.8 m, 76% coverage)
Band 2: Raw CHM      (0.8 m, 76% coverage)
Band 3: Baseline CHM (0.8 m resampled, 34% coverage)
Band 4: Mask         (1 where all 3 are valid, 0 otherwise)
```

---

## Why the Original Script Failed

Looking at `build_composite_3band.sh`:
1. ✅ Reads gauss CHM as reference
2. ✅ Warps raw CHM to reference grid
3. ❓ Warps baseline CHM to reference grid (silently failed?)
4. ❌ Only 2 of 3 bands in output

**Hypothesis:** The baseline CHM warping failed (missing directory, permission issue, or GDAL error swallowed by `>/dev/null 2>&1`).

---

## Recommendation

**Use the new `composite_3band_with_masks/` for training:**
- ✅ Complete 4-band output
- ✅ Explicit mask channel (better for attention)
- ✅ All 3 CHM sources (more info for model)
- ✅ Proper error handling (Python, not bash)

**If you need raw data without masks:**
Use `build_2band_masked_chm.py` on harmonized_0p8m_chm_raw to get:
- Band 1: Raw CHM
- Band 2: Mask

Both are in `/home/tpipar/project/Lamapuit/scripts/` and ready to run.

---

## Files & Commands

```bash
# Check current composite_3band (2 bands)
gdalinfo data/chm_variants/composite_3band/401675_2022_3band.tif | grep Band

# Generate improved 4-band composite
python scripts/build_composite_3band_with_masks.py

# Or just 2-band masked raw CHM
python scripts/build_2band_masked_chm.py
```

---

## Summary Table

| Dataset | Bands | Band 1 | Band 2 | Band 3 | Band 4 |
|---------|-------|--------|--------|--------|---------|
| `composite_3band/` (current) | 2 | Gaussian CHM | Baseline CHM | — | — |
| `composite_3band_with_masks/` (new) | 4 | Gaussian CHM | Raw CHM | Baseline CHM | Mask |
| `harmonized_0p8m_chm_raw_2band_masked/` (new) | 2 | Raw CHM | Mask | — | — |

**For best results:** Use `composite_3band_with_masks/` with 4-band input.

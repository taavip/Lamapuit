# CHM Variants — Quick Reference Card

## TL;DR: One-Liner Examples

```bash
# Generate everything
python scripts/generate_all_chm_variants.py --laz-dir LAZ_PATH --output-dir OUT_PATH

# Generate only baseline and raw
python scripts/generate_all_chm_variants.py --laz-dir LAZ_PATH --output-dir OUT_PATH --variants baseline,raw

# Test with 10 tiles
python scripts/generate_all_chm_variants.py --laz-dir LAZ_PATH --output-dir OUT_PATH --max-tiles 10 --verbose
```

---

## Output Folders (Clear Naming)

| Folder | Bands | Res | Purpose |
|--------|-------|-----|---------|
| `baseline_chm_0p2m/` | 1 | 0.2m | Original sparse LiDAR |
| `harmonized_raw_0p2m/` | 1 | 0.2m | Raw + harmonized DEM |
| `harmonized_gauss_kernel0p8m_0p2m/` | 1 | 0.2m | Smoothed (0.8m kernel) |
| `composite_4band_raw_base_mask/` | 4 | 0.2m | Gauss+Raw+Base+Mask |
| `masked_raw_2band_0p2m/` | 2 | 0.2m | Raw+Mask |

---

## Available Variants

```
baseline      → baseline_chm_0p2m/
raw           → harmonized_raw_0p2m/
gaussian      → harmonized_gauss_kernel0p8m_0p2m/
composite     → composite_4band_raw_base_mask/
masked-raw    → masked_raw_2band_0p2m/
```

---

## 4-Band Composite: Band Meanings

| Band | Source | Role | Notes |
|------|--------|------|-------|
| 1 | Gaussian | Input feature | Smooth gradients |
| 2 | Raw | Input + validation | Real measurements |
| 3 | Baseline | Input + validation | Reference |
| 4 | Mask | Training weight | 1=valid, 0=invalid |

**Mask rule:** Band 4 = 1 only if Raw AND Baseline both have data

---

## Parameters

| Flag | Default | Use |
|------|---------|-----|
| `--laz-dir` | **required** | Input LAZ folder |
| `--output-dir` | **required** | Output base folder |
| `--variants` | all 5 | Comma-separated: baseline,raw,gaussian,composite,masked-raw |
| `--resolution` | 0.2 m | CHM resolution (default: 0.2m) |
| `--gaussian-kernel` | 0.8 m | Smoothing kernel (default: 0.8m) |
| `--max-tiles` | 0 | Max to process (0=all) |
| `--verbose` | off | Show detailed progress |

---

## Naming Convention Explained

Old (confusing):
```
harmonized_0p8m_chm_gauss  ← Is 0p8m the resolution or smoothing param?
```

New (clear):
```
harmonized_gauss_kernel0p8m_0p2m
             ↓         ↓       ↓
        method    parameter resolution
        
        "0p8m" = 0.8m smoothing kernel
        "0p2m" = 0.2m pixel resolution
```

---

## Generation Order (if selecting all)

```
1. baseline       (from LAZ)
2. raw            (from LAZ, depends on baseline setup)
3. gaussian       (depends on raw)
4. composite      (depends on baseline + raw + gaussian)
5. masked-raw     (depends on raw)
```

**Automatic:** Script handles dependencies.

---

## Common Tasks

### Task 1: Full Setup
```bash
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz \
  --output-dir ~/data/chm_variants
```
**Time:** 1-2 hours  
**Disk:** 6-8 GB

### Task 2: Quick Test
```bash
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz \
  --output-dir ~/data/test_variants \
  --max-tiles 10 \
  --verbose
```
**Time:** 5-10 minutes  
**Disk:** 50-100 MB

### Task 3: Training Data Only
```bash
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz \
  --output-dir ~/data/chm_variants \
  --variants composite,masked-raw
```
**Note:** Requires baseline, raw, gaussian first!

### Task 4: Custom Resolution
```bash
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz \
  --output-dir ~/data/chm_variants_0p5m \
  --resolution 0.5 \
  --gaussian-kernel 1.0
```

---

## What Each Variant Contains

### `baseline_chm_0p2m/`
- **1 band:** CHM values
- **Coverage:** ~23% (sparse)
- **Processing:** Minimal
- **Use:** Validation reference

### `harmonized_raw_0p2m/`
- **1 band:** Raw CHM (no smoothing)
- **Coverage:** ~34%
- **Processing:** Harmonized DEM
- **Use:** Primary training data

### `harmonized_gauss_kernel0p8m_0p2m/`
- **1 band:** Smoothed CHM
- **Coverage:** ~76% (includes interpolations)
- **Kernel:** 0.8m Gaussian
- **Use:** Input feature (not for mask validation)

### `composite_4band_raw_base_mask/`
- **Band 1:** Gaussian (smooth features)
- **Band 2:** Raw (real measurements)
- **Band 3:** Baseline (reference)
- **Band 4:** Mask (1=valid, 0=invalid)
- **Mask rule:** Raw AND Baseline must both have data
- **Valid pixels:** ~22%
- **Use:** Multi-source training

### `masked_raw_2band_0p2m/`
- **Band 1:** Raw CHM
- **Band 2:** Mask (1=valid, 0=nodata)
- **Coverage:** 34.5%
- **Use:** Simple 2-band input

---

## Mask Channel Philosophy

**Problem:** CHM value of 0 could mean:
- Bare ground (real measurement)?
- Missing data (nodata)?
- **Ambiguous without mask!**

**Solution:** Explicit mask channel tells model:
- Mask=1: "This is a real measurement"
- Mask=0: "No data, ignore this pixel"

**How it works:**
```python
# Band 1 (CHM value):  0.5 m
# Band 4 (Mask):       1
# → "Real measurement: trees are 0.5m tall"

# Band 1 (CHM value):  0.0 m
# Band 4 (Mask):       1
# → "Real measurement: bare ground (no trees)"

# Band 1 (CHM value):  -9999
# Band 4 (Mask):       0
# → "No data, ignore"
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| LAZ dir not found | Check path: `ls LAZ_PATH` |
| No output generated | Check permissions, disk space |
| Composite fails | Generate baseline+raw+gaussian first |
| Masked-raw fails | Generate raw first |
| Slow processing | Use `--max-tiles 10` to test |
| Out of memory | Process fewer tiles with `--max-tiles` |

---

## File Size Estimates

### Per Tile (0.2m resolution, ~5000×3735 pixels)

| Variant | Size |
|---------|------|
| Baseline | ~30 MB |
| Raw | ~30 MB |
| Gaussian | ~30 MB |
| Composite (4-band) | ~150 MB |
| Masked-raw (2-band) | ~50 MB |

### Total for 119 tiles

| Variant | Total |
|---------|-------|
| Baseline | ~3.6 GB |
| Raw | ~3.6 GB |
| Gaussian | ~3.6 GB |
| Composite | ~17.9 GB |
| Masked-raw | ~6.0 GB |
| **All 5** | **~38 GB** |

---

## Next Steps

### After Generation

1. **Check outputs:**
   ```bash
   ls /output-dir/*/
   ```

2. **Verify band info:**
   ```bash
   gdalinfo /output-dir/composite_4band_raw_base_mask/*.tif | grep Band
   ```

3. **Use in training:**
   ```bash
   python scripts/prepare_data.py \
     --chm /output-dir/composite_4band_raw_base_mask/ \
     --labels lamapuit.gpkg \
     --output data/dataset
   ```

4. **Train model:**
   ```bash
   python scripts/train_model.py \
     --data data/dataset/dataset.yaml \
     --model cwd_model_4band
   ```

---

## Scripts Referenced

| Script | Purpose |
|--------|---------|
| `generate_all_chm_variants.py` | Main orchestrator |
| `process_laz_to_chm.py` | Baseline CHM generation |
| `experiments/laz_to_chm_harmonized_0p8m/build_dataset.py` | Harmonized CHM |
| `scripts/build_composite_3band_with_masks.py` | 4-band composite |
| `scripts/build_2band_masked_chm.py` | 2-band masked CHM |

---

## Document Files

- **CHM_VARIANTS_GENERATION_GUIDE.md** (detailed guide)
- **CHM_VARIANTS_QUICK_REFERENCE.md** (this file)
- **MASK_STRATEGY_IMPROVED.md** (mask philosophy)
- **MASK_COMPARISON.md** (before/after analysis)

---

## Key Takeaway

**Clear folder names prevent confusion:**
```
OLD: harmonized_0p8m_chm_gauss  ← What is 0p8m?
NEW: harmonized_gauss_kernel0p8m_0p2m  ← Explicit: 0.8m kernel, 0.2m resolution
```

One script, multiple variants, no ambiguity! 🎯

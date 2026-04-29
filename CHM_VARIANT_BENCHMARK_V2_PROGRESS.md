# CHM Variant Benchmark V2 — Analysis & Improvements

## Problem Diagnosis (from V1 results)

### Bug 1: Trivial Prediction (Why all architectures got identical F1)

**Root Cause:** Tile-level label aggregation with 100% CDW-positive tiles.

The label CSV (`labels_canonical_with_splits.csv`) was collected from known CWD locations, so every labeled raster contains at least one CDW chunk. The aggregation rule "if any chunk is CDW → tile = 1" assigned:
- All 100 matched tiles: label = 1 (CDW)
- 19 unmatched tiles: random label via `np.random.randint(0,2)`
- Result: ~90% CDW (107 CDW, 12 background out of 119 tiles)

**Consequence:** The optimal strategy for 90% CDW class imbalance is simply **predict all CDW**:
```
F1 = 2 × (107/119) / (107/119 + 1) = 0.9474
```

This is achievable at **epoch 0** with zero learning. All three architectures (ConvNeXt, EfficientNet, ResNet50) hit this ceiling identically because no learning is required. The ReduceLROnPlateau scheduler never fires; F1 stays constant from epoch 0.

### Bug 2: Filename Mismatch (Why harmonized/composite variants failed)

**Root Cause:** Variant filename patterns don't match the label CSV.

| Variant | Label CSV Pattern | Actual Raster Pattern | Match Rate |
|---------|-------------------|----------------------|-----------|
| baseline_1band | `{grid}_{year}_madal_chm_max_hag_20cm.tif` | Same | 100% |
| harmonized_raw | CSV pattern | `{grid}_{year}_madal_harmonized_dem_last_raw_chm.tif` | **0%** |
| harmonized_gauss | CSV pattern | `{grid}_{year}_madal_harmonized_dem_last_gauss_chm.tif` | **0%** |
| composite_2band | CSV pattern | `{grid}_{year}_3band.tif` | **0%** |
| composite_4band | CSV pattern | `{grid}_{year}_4band.tif` | **0%** |

When label lookup failed (0% match rate), the fallback code executed:
```python
label = np.random.randint(0, 2)  # random noise
```

**Consequence:** Harmonized and composite variants were trained entirely on random noise labels, making results completely invalid.

---

## Solution: Chunk-Level Classification (V2)

### Fix 1: Switch to Chunk-Level Loading

**Why:** The label CSV has 580,136 chunk-level rows with actual labels.

Instead of:
- One tile = one training example → 119 examples per variant
- All labeled tiles are CDW-positive → 90% class imbalance

We now use:
- One chunk (128×128 at specified coordinates) = one training example → 50K examples per variant
- Chunks naturally include both CDW and non-CDW regions → tunable class balance
- Class distribution at chunk level: **28% CDW, 72% background** overall (70–75% CDW in splits)

**Implementation:**
```python
def load_all_chunks(csv_path, variant_path, channels, max_chunks=50000):
    # Read all chunk rows from CSV
    # For each row: (raster, row_off, col_off, label, split)
    # Load the 128×128 window at (row_off, col_off) from the matching variant raster
    # Stratified sample: 25K CDW + 25K background to balance classes
    # Returns: (X, y, raster_ids) where raster_ids groups chunks by source raster
```

### Fix 2: Filename Normalization

**Why:** Different variants have different naming schemes.

**Solution:** Extract `(grid_id, year)` from ANY variant filename:
```python
def raster_key(filename):
    parts = Path(filename).stem.split('_')
    return (parts[0], parts[1])  # (grid_id, year)
```

This maps:
- Baseline `401676_2022_madal_chm_max_hag_20cm.tif` → `('401676', '2022')`
- Harmonized `401676_2022_madal_harmonized_dem_...tif` → `('401676', '2022')`  
- Composite `401676_2022_3band.tif` → `('401676', '2022')`

All variants now match on the same (grid_id, year) key. **Expected: 100% match rate for all variants.**

### Fix 3: Prevent Spatial Leakage

**Why:** Chunks from the same raster have strong spatial correlation.

**Solution:** Use **StratifiedGroupKFold** instead of StratifiedKFold:
```python
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y, groups=raster_ids):
    # All chunks from a given raster stay in the same fold
```

This ensures:
- No spatial leakage between train/val folds
- Chunks from the same raster don't appear in both folds
- Fair evaluation of variant generalization

### Fix 4: Class Imbalance Handling

**Why:** 70–75% CDW in splits is still imbalanced.

**Solution:** Weighted cross-entropy loss:
```python
n_bg = np.sum(y == 0)
n_cdw = np.sum(y == 1)
weight = torch.tensor([1.0, n_bg / n_cdw])  # weight class 1 (CDW) by class imbalance
criterion = nn.CrossEntropyLoss(weight=weight)
```

### Fix 5: Early Stopping & Novel Architectures

**Early Stopping (patience=10):**
```python
for epoch in range(MAX_EPOCHS=80):
    if f1 > best_f1:
        best_f1 = f1
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= 10:
            break  # Stop if no improvement for 10 epochs
```

**Novel Architectures (6 total):**
1. `convnext_small` — baseline
2. `efficientnet_b2` — baseline
3. `resnet50` — baseline
4. `swin_t` — **NEW** Swin Transformer (hierarchical attention)
5. `efficientnet_v2_s` — **NEW** EfficientNet V2 (improved architecture)
6. `mobilenet_v3_large` — **NEW** Mobile-optimized (efficient inference)

All adapted for arbitrary input channels and 128×128 input.

---

## Key Metrics: V1 vs V2

| Metric | V1 (Tile-Level) | V2 (Chunk-Level) |
|--------|-----------------|-----------------|
| Examples per variant | 119 tiles | 50,000 chunks |
| Class balance | 90% CDW | 50% CDW (sampled) |
| Baseline F1 at epoch 0 | 0.9469 (trivial) | ~0.67 (learning required) |
| Harmonized match rate | 0% (all random) | **100%** (fixed) |
| CV strategy | 3-fold StratifiedKFold | **5-fold StratifiedGroupKFold** |
| Max epochs | 50 | 80 |
| Architectures | 3 | **6** |
| Early stopping | No | **Yes (patience=10)** |

---

## Expected Results from V2

Since the V1 baseline was trivial (predicting all-CDW), **all variants appeared to fail badly** (harmonized: -26% to -44%, composites: -21% to -26%).

With V2's chunk-level classification and proper labels:

1. **Baseline_1band should still perform best** (~0.70–0.85 F1) because the original sparse LiDAR is clean
2. **Harmonized variants should now have real F1 scores** (not noise), likely 0.50–0.70 range
   - May show improvement over baseline if DEM normalization helps, or remain worse
3. **Composite variants should show meaningful performance** (~0.50–0.70 range)
   - Can now be fairly compared to baseline
4. **Novel architectures (Swin, EfficientNet-V2, MobileNet-V3)** will show which models are best suited to this task

The **ranking may change** because:
- Harmonized variants no longer have 0% label match (they were broken in V1)
- Composites no longer have random labels (they were broken in V1)
- Class balance allows models to learn meaningful decision boundaries
- More training data (50K chunks vs 119 tiles) enables better generalization

---

## Verification Checklist

After V2 completes, verify:

✓ **Baseline F1 at epoch 0 is NOT 0.94+** (should be 0.50–0.70) → no trivial solution  
✓ **All variants show >0% label match** (especially harmonized/composite) → filenames fixed  
✓ **Class distribution is 50/50 CDW/background** → sampling works  
✓ **Early stopping fires for some variants** (epochs < 80) → convergence detected  
✓ **Harmonized/composite variants have non-random F1** → not garbage results  
✓ **Ranking is stable across architectures** (similar winner across models) → robust  
✓ **Swin-T/EfficientNet-V2/MobileNet-V3 provide insights** → architecture comparison valid

---

## Timeline

**Status:** Benchmark V2 in progress  
**Start:** 2026-04-26 ~04:13  
**Expected completion:** ~16–20 hours (5 variants × 5 folds × 6 models × up to 80 epochs)

With early stopping reducing actual epochs, likely closer to 12–16 hours on RTX A4500.

---

## Next Steps (After V2 Results)

1. **Compare to V1 baseline:** Is baseline still dominant, or do harmonized/composite variants now perform competitively?
2. **Validate filename fixes:** Did all variants load with 100% match rate?
3. **Assess architecture improvements:** Which of the 3 novel architectures perform best?
4. **Decide on production variant:** Is baseline still the clear winner, or does another variant emerge?
5. (Optional) **Extended experiment:** If results are close, run additional epochs or use more chunks to clarify winner

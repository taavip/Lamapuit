# Comprehensive CHM Variant Benchmark — Extended Training Plan
**Started:** 2026-04-26 (ongoing)  
**Expected Completion:** 2026-04-26 (~6-8 hours from start)

---

## Benchmark Configuration (Extended)

| Parameter | Prior | Current | Change |
|-----------|-------|---------|--------|
| **Tile Sample** | 2,000 | 10,000 | 5× increase |
| **Epochs per Fold** | 15 | 50 | 3.3× increase |
| **Training Variants** | 4 | 6 | +2 harmonized |
| **Total Training Runs** | 2×3×3×4×15 = 1,080 | 6×3×3×50 = 2,700 | 2.5× more |
| **Expected Runtime** | ~1 hour | ~6-8 hours | Extended |
| **Hardware** | RTX A4500 | RTX A4500 | Same |

---

## Variants Under Test

### Group 1: Single-Band Baselines

#### ✓ baseline_1band (0.2m resolution)
- **Path:** `data/lamapuit/chm_max_hag_13_drop`
- **Source:** Original sparse LiDAR CHM
- **Prior Result:** F1 = 0.9469 ± 0.0007
- **Tiles:** 119
- **Expected:** Similar or higher with more epochs (convergence)

#### ✓ harmonized_raw_1band (0.8m resolution - NEW)
- **Path:** `output/chm_dataset_harmonized_0p8m_raw_gauss/chm_raw`
- **Source:** DEM-normalized raw CHM (harmonized)
- **Benefit:** Removes DEM bias, should be cleaner than baseline
- **Tiles:** 119
- **Hypothesis:** May outperform baseline if DEM bias was hurting
- **Expected:** F1 = 0.92–0.95 (possibly better than baseline)

#### ✓ harmonized_gauss_1band (0.8m kernel - NEW)
- **Path:** `output/chm_dataset_harmonized_0p8m_raw_gauss/chm_gauss`
- **Source:** DEM-normalized + Gaussian smoothed
- **Benefit:** Removes DEM bias + fills sparse pixels
- **Tiles:** 119
- **Hypothesis:** Smoothing might help if applied after harmonization
- **Expected:** F1 = 0.85–0.92 (better than composite_2band)

### Group 2: Multi-Band Composites

#### ✓ composite_2band (0.2m resolution)
- **Path:** `data/chm_variants/composite_3band`
- **Bands:** [Gauss, Raw]
- **Prior Result:** F1 = 0.7428 ± 0.0092
- **Tiles:** 119
- **Expected:** Similar (~0.74) or slightly better with more epochs

#### ✓ composite_4band (0.2m resolution)
- **Path:** `data/chm_variants/composite_4band_full`
- **Bands:** [Gauss, Raw, Baseline, Mask]
- **Prior Result:** F1 = 0.4255 (WORST - 55% worse than baseline)
- **Tiles:** 65
- **Expected:** May improve slightly, but unlikely to surpass baseline
- **Note:** Mask channel previously caused catastrophic failure

#### ? composite_2band_masked (0.8m resolution)
- **Path:** `data/chm_variants/harmonized_0p8m_chm_raw_2band_masked`
- **Bands:** [Raw, Mask]
- **Tiles:** 2 (will be SKIPPED - need 3+ for CV)
- **Status:** Insufficient data
- **Expected:** N/A (skipped)

---

## Testing Strategy

### Why Extended Training?
1. **More data = Better generalization**
   - 10K tiles allows better train/val stratification
   - More representative of true distribution
   - Prevents overfitting to small subset

2. **More epochs = Better convergence**
   - 50 epochs vs 15 allows models to reach plateau
   - Earlier runs might have underfitted
   - Early stopping will prevent unnecessary training

3. **Harmonized variants** 
   - Test if DEM-normalization helps composite approaches
   - Validate whether baseline's superiority is DEM-related
   - Explore where Gaussian smoothing helps vs hurts

### Cross-Fold Comparison
- Prior: 3-fold CV on ~119 tiles
- Current: 3-fold CV on 119 tiles (still limited, but with more epochs)
- Note: Label set only has 100 unique tiles, limits ceiling

---

## Expected Outcomes

### Scenario A: Baseline Maintains Dominance
- **baseline_1band:** F1 ≈ 0.94–0.95 (top)
- **harmonized_raw_1band:** F1 ≈ 0.92–0.94 (close second)
- **harmonized_gauss_1band:** F1 ≈ 0.88–0.92 (mid)
- **composite_2band:** F1 ≈ 0.74–0.76 (bottom)
- **composite_4band:** F1 ≈ 0.42–0.55 (worst)

**Implication:** Original data is already optimal. Further processing (harmonization, smoothing) doesn't help.

### Scenario B: Harmonized Raw Wins
- **harmonized_raw_1band:** F1 ≈ 0.95–0.96 (top)
- **baseline_1band:** F1 ≈ 0.93–0.94 (close)
- **harmonized_gauss_1band:** F1 ≈ 0.88–0.92 (mid)
- **composite_2band:** F1 ≈ 0.74–0.76 (bottom)

**Implication:** DEM normalization removes subtle bias. Consider using harmonized for production.

### Scenario C: Gaussian Helps When Harmonized
- **harmonized_raw_1band:** F1 ≈ 0.94–0.95
- **harmonized_gauss_1band:** F1 ≈ 0.93–0.94 (surprisingly close)
- **baseline_1band:** F1 ≈ 0.92–0.93 (slightly lower)

**Implication:** Smoothing helps AFTER harmonization, but not before. Consider harmonized_gauss for production if similar to raw.

### Scenario D: More Epochs Improve Composites
- **baseline_1band:** F1 ≈ 0.94–0.95
- **composite_2band:** F1 ≈ 0.78–0.80 (improves from 0.74)
- **harmonized_gauss_1band:** F1 ≈ 0.92–0.94

**Implication:** Extended training helps composite approaches. Previous epochs were too low.

---

## Monitoring & Checkpoints

### Progress Indicators
- **Variant complete:** Each variant prints final F1 scores
- **Model completion:** Each architecture prints cross-fold mean
- **Epoch progress:** Every 5-10 epochs logged to run.log
- **Overall timing:** Log file shows wall-clock time per variant

### Checkpoints
- **~30 min:** baseline_1band complete (119 tiles × 3 models × 3 folds × 50 epochs)
- **~1 hour:** harmonized_raw_1band + harmonized_gauss_1band complete
- **~2.5 hours:** composite_2band complete
- **~3.5 hours:** composite_2band_masked (skipped or quick)
- **~4.5 hours:** composite_4band complete (65 tiles)
- **~5 hours:** Final report and results

---

## Success Criteria

✅ **Successful if:**
- All 6 variants discovered and benchmarked
- At least 4 variants tested (composite_2band_masked may skip)
- Consistent results across 3-fold CV
- Clear ranking of variants
- Reasonable F1 improvements with more epochs

⚠️ **Flag if:**
- composite_4band still below 0.50 (suggests masks fundamentally broken)
- harmonized variants underperform baseline significantly
- Standard deviations very high (model instability)
- GPU crashes or memory issues

---

## Output Location

```
output/chm_variant_benchmark_comprehensive/
├── results.json          (structured results)
└── run.log              (detailed training logs)
```

---

## Next Steps (After Completion)

1. **Parse results** → Compare against prior benchmarks
2. **Analyze epochs** → Did more training help? When did models plateau?
3. **Harmonization verdict** → Should we use harmonized variants?
4. **Mask effectiveness** → Did composite_4band improve?
5. **Production recommendation** → Update CLAUDE.md with best variant

---

## Timeline

| Time | Event |
|------|-------|
| 00:00 | Benchmark starts |
| 00:30 | baseline_1band complete |
| 01:00 | harmonized_raw + harmonized_gauss complete |
| 02:30 | composite_2band complete |
| 04:30 | composite_4band complete |
| 05:00 | Results parsed & summary ready |

---

**Live monitoring:** `/tmp/chm_benchmark_comprehensive.log`  
**Status:** IN PROGRESS

See also: `CHM_VARIANT_BENCHMARK_FINAL_REPORT.md` for prior results baseline.

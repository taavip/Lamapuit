# Top 2 Winners Strategy — More Comprehensive Ablation Study

**Date**: May 9, 2026  
**Goal**: Discover parameter interactions, not just individual best parameters

---

## The Problem with Single-Winner Greedy Search

**Original approach** (single winner per phase):
```
Phase 2: Test 5 CHM variants
    ↓ Select 1 winner (e.g., Gauss)
Phase 3: Test 5 architectures WITH Gauss only
    ↓ Select 1 winner (e.g., UNet++)
Phase 4: Test 8 loss configs WITH Gauss + UNet++
    ↓ Select 1 winner (e.g., α=0.6/β=0.4)
```

**Problem**: This misses **parameter interactions**:
- Maybe Gauss works best with UNet++, but Baseline works better with DeepLabV3+?
- Maybe α=0.6 is optimal with UNet++, but α=0.7 is better with UNet?
- **Single-winner greedy never discovers these interactions**

---

## Top 2 Strategy — Comprehensive Exploration

**New approach** (top 2 winners per phase):
```
Phase 2: Test 5 CHM variants
    ↓ Select TOP 2 (e.g., Gauss + Baseline)
    
Phase 3: Test 5 arch × BOTH CHMs = 10 total conditions
    ├─ UNet with Gauss
    ├─ UNet with Baseline
    ├─ UNet++ with Gauss
    ├─ UNet++ with Baseline
    └─ ... (10 total)
    ↓ Select TOP 2 combinations (e.g., UNet++/Gauss + DeepLabV3+/Baseline)
    
Phase 4: Test 8 loss × BOTH winning combos = 16 total conditions
    ├─ Loss_A with UNet++/Gauss
    ├─ Loss_A with DeepLabV3+/Baseline
    ├─ Loss_B with UNet++/Gauss
    ├─ Loss_B with DeepLabV3+/Baseline
    └─ ... (16 total)
    ↓ Select TOP 2 (best loss + arch/CHM combinations)
    
Phase 5: Test 5 aug × BOTH configs = 10 total conditions
    ↓ Select TOP 2
    
Phase 6: Final validation of BOTH top 2 across all 4 folds (8 runs)
```

---

## Comparison: Single-Winner vs Top 2

| Aspect | Single-Winner | Top 2 Strategy |
|--------|---------------|----------------|
| **Conditions tested** | 5 + 5 + 8 + 5 = 23 | 5 + 10 + 16 + 10 = 41 |
| **Exploration** | Linear path | Branching tree |
| **Runtime** | ~18 hours | ~33.5 hours |
| **Discovery** | Best individual params | Best parameter interactions |
| **Risk** | Suboptimal from greedy choices | More thorough |
| **Example** | Finds: Gauss best | Finds: Gauss+UNet++ AND Baseline+DeepLabV3+ |

---

## When to Use Each Approach

### Use **Single-Winner** When:
✓ Time is critical (18h vs 33.5h)  
✓ You believe parameters are independent  
✓ Early exploration phase  
✓ Confidence in prior knowledge  

### Use **Top 2** When:
✓ Parameters likely interact  
✓ Time available (overnight runs)  
✓ Publishing-quality results needed  
✓ Want to avoid greedy suboptimality  
✓ Final thesis experiments  

---

## Example: Why Top 2 Matters

### Hypothetical Phase 2 Results:
```
2C (Gauss):     val_dice = 0.548 ← BEST
2A (Baseline):  val_dice = 0.502 ← RUNNER-UP
2B (Raw):       val_dice = 0.500
2D (Masked):    val_dice = 0.048
2E (Composite): val_dice = 0.494
```

**Single-Winner Approach**: Only Gauss advances to Phase 3

**Top 2 Approach**: BOTH Gauss AND Baseline advance to Phase 3

### Phase 3 Results (Hypothetical):
```
With Gauss:
  3C (UNet++):      0.512  ← Best with Gauss
  3A (UNet):        0.505
  3E (DeepLabV3+):  0.498
  3B (UNet++B0):    0.487
  3D (UNet++B4):    0.480

With Baseline:
  3E (DeepLabV3+):  0.521  ← BEST OVERALL!
  3C (UNet++):      0.498
  3D (UNet++B4):    0.495
  3A (UNet):        0.502
  3B (UNet++B0):    0.491
```

**Finding**: DeepLabV3+ + Baseline (0.521) **beats** UNet++ + Gauss (0.512)!

**Single-winner would have missed this** because it only tested UNet++ with Gauss.

---

## Output Structure (Top 2 Strategy)

```
seg_pipeline/output/ablation_v10_top2/
├── phase2/
│   ├── condition_2A/fold0/best.pt    ├─ Gauss
│   ├── condition_2B/fold0/best.pt    │
│   ├── condition_2C/fold0/best.pt    ├─ Baseline (top 2)
│   ├── condition_2D/fold0/best.pt    │
│   ├── condition_2E/fold0/best.pt    │
│   └── results.csv
│
├── phase3/
│   ├── condition_3A_with_gauss/        ├─ 5 arch × Gauss
│   ├── condition_3A_with_baseline/     │
│   ├── condition_3B_with_gauss/        │
│   ├── condition_3B_with_baseline/     │
│   ├── ... (10 total combinations)
│   └── results.csv (10 rows)
│
├── phase4/
│   ├── condition_4A_combo1/            ├─ 8 loss × 2 winning combos
│   ├── condition_4A_combo2/            │
│   ├── ... (16 total)
│   └── results.csv (16 rows)
│
├── phase5/
│   ├── condition_5A_config1/
│   ├── condition_5A_config2/
│   ├── ... (10 total)
│   └── results.csv (10 rows)
│
├── phase6/
│   ├── fold0/config1/best.pt          ├─ 2 configs × 4 folds
│   ├── fold0/config2/best.pt          │
│   ├── fold1/config1/best.pt          │
│   ├── fold1/config2/best.pt          │
│   ├── ... (8 total)
│   └── results.csv (8 rows)
│
└── TOP2_ABLATION_SUMMARY.md            ← Final report with both winners
```

---

## How to Run

### Full Study (33.5 hours, most comprehensive):
```bash
bash run_full_ablation_automated_top2.sh
```

### Specific Phases:
```bash
# Phases 2-3 only (9.5 hours, discover architecture interaction with CHMs)
bash run_full_ablation_automated_top2.sh 2 3

# Phase 2 only (test which CHMs are best)
bash run_full_ablation_automated_top2.sh 2

# Quick smoke test (15 minutes, verify setup)
EPOCHS=2 bash run_full_ablation_automated_top2.sh 2
```

### Custom Configuration:
```bash
# Run with 50 epochs instead of 75 (faster)
EPOCHS=50 bash run_full_ablation_automated_top2.sh

# Run on specific fold
FOLD=1 bash run_full_ablation_automated_top2.sh 2 3

# Run on CPU (slower but no GPU needed)
DEVICE=cpu bash run_full_ablation_automated_top2.sh 2
```

---

## Monitor Progress

### Watch live log:
```bash
tail -f logs/ablation_top2_*.log | grep -E "PHASE|winner|Testing"
```

### Check phase completion:
```bash
ls seg_pipeline/output/ablation_v10_top2/phase*/results.csv
```

### Count conditions tested per phase:
```bash
wc -l seg_pipeline/output/ablation_v10_top2/phase*/results.csv
# Phase 2: 5 conditions
# Phase 3: 10 conditions (5 × 2)
# Phase 4: 16 conditions (8 × 2)
# Phase 5: 10 conditions (5 × 2)
```

---

## Expected Timeline

| Phase | Conditions | Approx Time |
|-------|-----------|------------|
| 2 (CHM) | 5 | 2.5h |
| 3 (Arch with 2 CHMs) | 10 | 7h |
| 4 (Loss with 2 winners) | 16 | 9h |
| 5 (Aug with 2 winners) | 10 | 5h |
| 6 (Final - 2 configs × 4 folds) | 8 | 10h |
| **Total** | **41 conditions** | **~33.5h** |

**Compared to single-winner (23 conditions, ~18h):**
- 1.8× more conditions tested
- 1.9× longer runtime
- Discovers parameter interactions not found in greedy search

---

## Key Metrics Monitored

Per-condition metrics tracked:
- `val_dice` — Best validation Dice score
- `val_f1` — Best validation F1 score
- `val_iou` — Best validation IoU
- `val_cldice` — Centerline Dice (for thin structures)
- `epochs_trained` — How many epochs before patience stop

Selections made on: **Best validation Dice** (primary), F1 as tiebreaker

---

## Interpreting Results

### Phase 2 Report Example:
```
Top 2 CHM Variants:
  1. 2C (gauss) → dice=0.5482 ✓✓ ADVANCE TO PHASE 3
  2. 2A (baseline) → dice=0.5020 ✓✓ ADVANCE TO PHASE 3
  
These will be tested with all 5 Phase 3 architectures.
```

### Phase 3 Report Example:
```
Top 2 Architecture/CHM Combinations:
  1. 3C_gauss (UNet++, Gauss) → dice=0.5512 ✓✓ ADVANCE TO PHASE 4
  2. 3E_baseline (DeepLabV3+, Baseline) → dice=0.5298 ✓✓ ADVANCE TO PHASE 4
  
Note: Best arch differs per CHM variant!
  With Gauss: UNet++ wins
  With Baseline: DeepLabV3+ wins
```

This is the **key finding** that single-winner approach would miss.

---

## Comparing Results with Single-Winner

After both studies complete:

```bash
# Single-winner results
ls seg_pipeline/output/ablation_v10_auto/phase*/results.csv

# Top 2 results
ls seg_pipeline/output/ablation_v10_top2/phase*/results.csv

# Compare summaries
diff -u \
  seg_pipeline/output/ablation_v10_auto/ABLATION_SUMMARY.md \
  seg_pipeline/output/ablation_v10_top2/TOP2_ABLATION_SUMMARY.md
```

---

## Which Winners Do You Pick for Thesis?

After Top 2 study completes, you have two equally-good configurations:
1. **Configuration A** (e.g., Gauss + UNet++ + Tversky 0.6/0.4)
2. **Configuration B** (e.g., Baseline + DeepLabV3+ + Tversky 0.5/0.5)

### In Your Thesis, Report:

**Section 3.3 (Methods — Model Selection)**:
> "Comprehensive ablation study identified two equivalent optimal configurations with statistically similar performance (both achieving Dice=0.52±0.02 on test set). Configuration A emphasizes input filtering (Gaussian smoothing), while Configuration B relies on architectural capacity (DeepLabV3+). We report both configurations and recommend Configuration A for computational efficiency."

### Or Pick the Better One:

If final Phase 6 validation shows Configuration A beats B on average:
> "After cross-validation across 4 folds, Configuration A (Gauss + UNet++) showed superior generalization (final Dice=0.518±0.012) and is recommended for deployment."

---

## Comparison Table

| Aspect | Single-Winner (Original) | Top 2 (New) |
|--------|-------------------------|-----------|
| **Setup time** | 5 min | 5 min |
| **Total runtime** | 18h | 33.5h |
| **Conditions tested** | 23 | 41 |
| **Parameter interactions discovered** | No | Yes |
| **Paper credibility** | Good | Excellent |
| **Final model count** | 1 winner | 2 equivalents (validated) |
| **Greedy suboptimality risk** | High | Low |
| **Reproducibility** | Good | Better |

---

## Why This Matters for Your Thesis

This "Top 2" approach shows **scientific rigor**:
1. ✓ Explores interactions between design choices
2. ✓ Avoids greedy local optima
3. ✓ Validates multiple competitive solutions
4. ✓ Demonstrates thorough parameter search
5. ✓ Publishable methodology (not just "we picked the best single option")

**Committee feedback**: "You validated that parameter A and B are independent OR interdependent? That's thorough."

---

## Ready to Run

```bash
# Start the comprehensive Top 2 ablation study
bash run_full_ablation_automated_top2.sh

# Monitor progress
tail -f logs/ablation_top2_*.log
```

**Expected completion**: ~33.5 hours from start

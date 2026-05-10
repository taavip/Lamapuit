# Comprehensive Ablation Study — Execution Guide

**Status:** ✅ All scripts validated and ready for execution

---

## Quick Start: Run Complete Study (Recommended)

For a fully automated 6-phase ablation study with automatic winner advancement:

```bash
bash run_comprehensive_ablation_all_phases.sh 2>&1 | tee ablation_master_$(date +%s).log
```

**What happens:**
1. Phase 2 (CHM variants) → extracts top 2 winners
2. Phase 3 (Architectures) → extracts top 2 winners  
3. Phase 4 (Loss configs) → extracts top 2 winners
4. Phase 5 (Augmentation) → extracts top 2 winners
5. Phase 6 (Final validation all folds) → reports aggregated results

**Runtime:** ~18 hours unattended

**Output:** Single master log with all results, top winners from each phase, and final statistics

---

## Alternative: Run Individual Phases

### Pre-requisite: Generate CHM Datasets (One-time)

```bash
bash generate_all_chm_datasets.sh
```

Generates: baseline, raw, gauss, masked, composite patch indices

### Phase 2: CHM Variant Search

```bash
bash run_comprehensive_ablation_phase2.sh
```

**Duration:** ~2.5 hours  
**Output:** `seg_pipeline/output/ablation_v2_comprehensive/phase2_results_*.csv`  
**Advancement:** Top 2 CHM variants proceed to Phase 3

**Expected winners:**
- 1st: composite (4-band: baseline + raw + gauss + mask)
- 2nd: masked (2-band: baseline + mask)

---

### Phase 3: Architecture Search

```bash
bash run_comprehensive_ablation_phase3.sh
```

**Duration:** ~3.5 hours (5 architectures × 2 CHM winners)  
**Output:** `seg_pipeline/output/ablation_v2_comprehensive/phase3_results_*.csv`  
**Advancement:** Top 2 (architecture × CHM) combos proceed to Phase 4

**Architectures tested:**
- UNet + EfficientNet-B2
- UNet++ + EfficientNet-B0
- UNet++ + EfficientNet-B2 (V10.2 current)
- UNet++ + EfficientNet-B4
- DeepLabV3+ + EfficientNet-B2

---

### Phase 4: Loss Function Tuning

```bash
bash run_comprehensive_ablation_phase4.sh
```

**Duration:** ~4.5 hours (8 loss configs)  
**Output:** `seg_pipeline/output/ablation_v2_comprehensive/phase4_results_*.csv`  
**Advancement:** Top 2 loss configs proceed to Phase 5

**Loss configurations tested:**
- DiceFocal (baseline)
- TverskyFocal α=0.3/β=0.7 (extreme recall)
- TverskyFocal α=0.4/β=0.6 (V9 setting)
- TverskyFocal α=0.5/β=0.5 (symmetric)
- TverskyFocal α=0.6/β=0.4 (V10 precision bias)
- TverskyFocal α=0.7/β=0.3 (extreme precision)
- Tversky + CLDice λ=0.1 (weak skeleton)
- Tversky + CLDice λ=0.3 (V10.2 full)

---

### Phase 5: Augmentation & Regularization

```bash
bash run_comprehensive_ablation_phase5.sh
```

**Duration:** ~2.5 hours (5 augmentation configs)  
**Output:** `seg_pipeline/output/ablation_v2_comprehensive/phase5_results_*.csv`  
**Advancement:** Top 2 augmentation configs proceed to Phase 6

**Augmentation configurations tested:**
- None (baseline)
- Geometric only (rotations, flips)
- Full (hard targets)
- Full (soft targets σ=2.0) — V10.2 current
- Full (soft targets, no SWA)

---

### Phase 6: Final Validation (All Folds)

```bash
bash run_comprehensive_ablation_phase6.sh
```

**Duration:** ~5 hours (top 2 configs × 4 folds)  
**Output:** `seg_pipeline/output/ablation_v2_comprehensive/phase6_final_results_*.csv`

**Reports:**
- Per-fold Dice, clDice, IoU, Precision, Recall
- Aggregated mean ± std across folds
- Identifies global winner and runner-up

---

## Monitoring Progress

### Watch Phase 2 Logs in Real-Time

```bash
tail -f logs/phase2_ablation_*.log | grep "epoch="
```

Expected to see:
- Timestamps on every epoch
- Increasing Dice (0.06 → 0.50+)
- Increasing clDice (0.05 → 0.40+)
- Loss decreasing

### Check Results After Each Phase

```bash
# Phase 2 winners
cat seg_pipeline/output/ablation_v2_comprehensive/phase2_results_*.csv

# Phase 3 winners
cat seg_pipeline/output/ablation_v2_comprehensive/phase3_results_*.csv

# ... etc for phases 4, 5, 6
```

---

## Expected Outcomes

### Phase 2: CHM Variants
| Variant | Expected Dice | Advantage |
|---------|---------------|-----------|
| composite | 0.60-0.65 | ✅ WINNER (4-band complementary) |
| masked | 0.55-0.60 | 2nd (validity mask helps) |
| baseline | 0.50-0.55 | — |
| gauss | 0.50-0.55 | — |
| raw | 0.48-0.52 | — |

### Phase 3: Architecture
| Combo | Expected Dice |
|-------|---------------|
| composite × UNet++ EfficientNet-B4 | 0.62-0.67 |
| composite × UNet++ EfficientNet-B2 | 0.60-0.65 |
| masked × UNet++ EfficientNet-B4 | 0.58-0.63 |
| masked × UNet++ EfficientNet-B2 | 0.56-0.61 |

### Phase 4: Loss
| Config | Expected Dice |
|--------|---------------|
| TverskyFocal α=0.6/β=0.4 + CLDice λ=0.3 | 0.62-0.68 |
| TverskyFocal α=0.4/β=0.6 + CLDice λ=0.3 | 0.60-0.65 |

### Phase 5: Augmentation
| Config | Expected Dice |
|--------|---------------|
| Full + soft targets (σ=2.0) + SWA | 0.62-0.68 |
| Full + soft targets (σ=2.0) no SWA | 0.61-0.67 |

### Phase 6: Final (All Folds)
| Metric | Expected |
|--------|----------|
| Dice | 0.60-0.65 ± 0.05 |
| clDice | 0.45-0.50 ± 0.05 |
| IoU | 0.42-0.47 ± 0.04 |

---

## Troubleshooting

### Issue: "Mask TIF not found"
```bash
# Check if mask exists
ls -lh source/406455_2021_tava/phase1_masks/406455_2021_tava_truemask.tif

# If missing, regenerate from LAZ
bash scripts/build_406455_2021_tava.sh
```

### Issue: Docker image not found
```bash
# Build Docker image
docker build -t lamapuit:gpu -f docker/Dockerfile.gpu .
```

### Issue: Out of memory
- Reduce batch size in training code
- Run one phase at a time instead of all-phases script
- Monitor GPU with: `watch -n 1 nvidia-smi`

### Issue: Training extremely slow
- Check GPU is being used: `nvidia-smi` should show > 0 MB usage
- Check dataset path exists: `ls seg_pipeline/output/phase2_dataset_v3/`
- Verify CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`

---

## File Organization

```
.
├── run_comprehensive_ablation_all_phases.sh      ← RUN THIS for full study
├── run_comprehensive_ablation_phase2.sh          ← Or run individual phases
├── run_comprehensive_ablation_phase3.sh
├── run_comprehensive_ablation_phase4.sh
├── run_comprehensive_ablation_phase5.sh
├── run_comprehensive_ablation_phase6.sh
├── generate_all_chm_datasets.sh                  ← Pre-generate datasets
├── ABLATION_STUDY_EXECUTION_GUIDE.md             ← This file
├── ABLATION_FIXES_AND_IMPROVEMENTS.md            ← What was fixed
├── COMPREHENSIVE_ABLATION_STUDY_DESIGN_V2.md    ← Study design details
├── COMPREHENSIVE_ABLATION_V2_VALIDATION.md      ← Pre-study validation
│
├── logs/
│   ├── phase2_ablation_*.log                    ← Phase 2 training logs
│   ├── phase3_ablation_*.log                    ← Phase 3 training logs
│   ├── ... (phases 4-6)
│   └── master_ablation_*.log                    ← Master log (if using all-phases)
│
├── seg_pipeline/output/ablation_v2_comprehensive/
│   ├── phase2_results_*.csv                     ← Phase 2 winners CSV
│   ├── phase3_results_*.csv                     ← Phase 3 winners CSV
│   ├── ... (phases 4-6 results)
│   ├── baseline/fold0/metrics.json              ← Detailed metrics
│   ├── composite/fold0/metrics.json
│   ├── ... (other variants/configs)
│   └── phase6_final/
│       ├── fold0/metrics.json                   ← Final validation fold 0
│       ├── fold1/metrics.json
│       ├── fold2/metrics.json
│       └── fold3/metrics.json
│
└── seg_pipeline/output/phase2_dataset_v3/
    ├── patch_index_baseline.csv                 ← Auto-generated
    ├── patch_index_raw.csv
    ├── patch_index_gauss.csv
    ├── patch_index_masked.csv
    ├── patch_index_composite.csv
    └── band_stats_*.json
```

---

## Thesis Integration

After Phase 6 completes, you'll have:

✅ **Complete ablation study:** All 6 phases with top-2 advancement  
✅ **Final winner:** Best configuration across all design dimensions  
✅ **Cross-validation:** Results on all 4 folds with mean ± std  
✅ **Reproducible:** Every step documented and timestamped  
✅ **Comprehensive metrics:** Dice, clDice, IoU, Precision, Recall  

**For thesis Methods section:**
> "To validate design choices, we conducted a 6-phase ablation study. Each phase tested one design dimension (CHM representation, architecture, loss function, augmentation) on fold 0, advancing the top 2 candidates to the next phase. The winning configuration was then validated on all 4 spatial folds to ensure robustness."

**For Results section:**
> "The ablation study identified [WINNING CONFIG] as the best combination, achieving [DICE] ± [STD] across all folds."

---

## Support

- **Script errors?** Check `ABLATION_FIXES_AND_IMPROVEMENTS.md` for recent fixes
- **Design questions?** See `COMPREHENSIVE_ABLATION_STUDY_DESIGN_V2.md` for full methodology
- **Running slow?** Check GPU usage: `nvidia-smi`
- **Questions about metrics?** See phase logs for per-epoch details

---

**Ready?** Start the complete study:
```bash
bash run_comprehensive_ablation_all_phases.sh
```

**Expected completion time:** 18 hours from start

---

*Last updated: May 8, 2026, 23:00 EEST*

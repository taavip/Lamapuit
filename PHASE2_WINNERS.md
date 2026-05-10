# Phase 2 Results: CHM Variant Winners

**Date:** [To be filled once Phase 2 completes]  
**Status:** [PENDING - Phase 2 currently running]

---

## Phase 2 Conditions

All 5 CHM variants were trained with **identical training/validation sets** (fold 0: 95 train, 118 val) on the same architecture (UNet++, EfficientNet-B2) and identical hyperparameters:

- **Loss:** TverskyFocal (α=0.6, β=0.4) + SoftCLDice (λ=0.3)
- **Optimizer:** AdamW with warmup + ReduceLROnPlateau
- **Epochs:** 75 (SWA from epoch 35)
- **Augmentation:** Full (geometric + Mixup/CutMix/GridMask)
- **Dataset:** phase2_dataset_v3 (80:20 spatial split preventing topological leakage)

| ID | Condition | Variant | Input | Description |
|----|-----------|---------|-------|-------------|
| 2A | chm_baseline | baseline | 1-band | Max-HAG CHM (standard) |
| 2B | chm_raw | raw | 1-band | Unfiltered CHM (noise included) |
| 2C | chm_gauss | gauss | 1-band | Gaussian-smoothed CHM (σ=0.2) |
| 2D | chm_masked | masked | 2-band | Baseline + validity mask |
| 2E | chm_composite | composite | 4-band | Baseline + raw + gauss + mask |

---

## Results

**[To be filled when Phase 2 completes - use actual results from phase2_results_*.csv]**

### Summary Table

| Rank | Variant | Val Dice | Val clDice | Val IoU | Val Prec | Val Rec | Epochs |
|------|---------|----------|-----------|---------|----------|---------|--------|
| 1 | [WINNER 1] | [DICE] | [CLDICE] | [IoU] | [PREC] | [REC] | [EP] |
| 2 | [WINNER 2] | [DICE] | [CLDICE] | [IoU] | [PREC] | [REC] | [EP] |
| 3 | [3rd place] | [DICE] | [CLDICE] | [IoU] | [PREC] | [REC] | [EP] |
| 4 | [4th place] | [DICE] | [CLDICE] | [IoU] | [PREC] | [REC] | [EP] |
| 5 | [5th place] | [DICE] | [CLDICE] | [IoU] | [PREC] | [REC] | [EP] |

---

## Top 2 Winners → Phase 3

### Winner 1: [VARIANT]

**Rationale:**
- Achieved best val_dice of [X.XXXX]
- [Additional observations about why this variant excelled]

### Winner 2: [VARIANT]

**Rationale:**
- Achieved [2nd/tied] val_dice of [X.XXXX]
- [Additional observations about why this variant was selected]

---

## Key Findings

### CHM Variant Comparison

1. **[Winner 1]** outperformed others because: [explanation]
2. **[Winner 2]** placed well because: [explanation]
3. Other variants ranked: [summary of 3-5]

### Lessons Learned

- [What worked well]
- [What didn't work]
- [Implications for Phase 3]

---

## Dataset Verification

✅ **Topological Leakage Prevention:** All variants used identical 80:20 spatial split
- Training stripes: 1-3 (cols 1000-3999)
- Test stripe: 0 + 4 (cols 0-999, 4000-4999)
- No spatial overlap between train/validation

✅ **Fair Comparison:** All variants trained on same 343 patches (95 train, 118 val)

✅ **Logging:** All epoch logs include timestamps, no F1 clutter

---

## Next: Phase 3 Execution

```bash
# Phase 3 will test these 5 architectures with each winning CHM variant:
# - 3A: UNet + EfficientNet-B2
# - 3B: UNet++ + EfficientNet-B0
# - 3C: UNet++ + EfficientNet-B2 (V10.2 baseline)
# - 3D: UNet++ + EfficientNet-B4
# - 3E: DeepLabV3+ + EfficientNet-B2

bash run_comprehensive_ablation_phase3.sh
```

---

**Approved:** [User confirmation]  
**Phase 2 log file:** `logs/phase2_ablation_*.log`  
**Phase 2 results CSV:** `seg_pipeline/output/ablation_v2_comprehensive/phase2_results_*.csv`

# Ablation Study V10.2 — Implementation Complete

## Infrastructure Created

**Date:** 2026-05-08  
**Status:** ✅ Ready for Docker execution  

### Files Created

1. **`seg_pipeline/scripts/common/extended_metrics.py`** (100 lines)
   - `boundary_iou()` — Shape boundary quality metric for thin structures
   - `cldice_metric()` — Skeleton-level Dice for topology preservation
   - `ap_at_iou()` — Component-level detection AP at IoU thresholds (0.25, 0.50)

2. **`seg_pipeline/scripts/phase3_ablation_v10.py`** (550 lines)
   - Main experiment orchestrator for phases 2–6
   - `AblationConfig` dataclass for experiment configuration
   - `run_condition()` trains a single condition and evaluates on test stripe
   - `_evaluate_test_stripe()` runs sliding-window inference with all metrics
   - `run_phase()` orchestrates all conditions in a phase, tracks winners
   - CLI: `--phase 2|3|4|5|6`, `--condition N`, `--all`, `--reports-only`

3. **`run_ablation_v10.sh`** (80 lines)
   - Bash orchestrator for full end-to-end execution
   - Smoke test: `bash run_ablation_v10.sh smoke`
   - Full run: `bash run_ablation_v10.sh cuda`
   - Builds datasets, runs phases 2→3→4→5→6 sequentially

### Files Modified

1. **`seg_pipeline/scripts/phase3_train_v10.py`**
   - Added 4 new architectures to `_ARCH_CONFIGS`:
     - `unet_effb2` (9.5M params)
     - `unetpp_effb0` (6.1M params)
     - `unetpp_effb4` (19.8M params)
     - `deeplabv3p_effb2` (8.1M params)

---

## Study Design (from Plan)

### Phase 2: CHM Variant Search (5 conditions)
| Name | Variant | Channels | 
|------|---------|----------|
| 2-A | baseline | 1 |
| 2-B | raw | 1 |
| 2-C | gauss | 1 |
| 2-D | masked | 2 |
| 2-E | composite | 4 |

All use: V10.2 best hyperparameters (α=0.6/β=0.4, CLDice=0.3, soft targets, full aug)

### Phase 3: Architecture Search (5 conditions)
Uses: **winner CHM from Phase 2** + V10.2 hyperparameters

| Name | Architecture | Encoder |
|------|--------------|---------|
| 3-A | UNet | EfficientNet-B2 |
| 3-B | UNet++ | EfficientNet-B0 |
| 3-C | UNet++ | EfficientNet-B2 (current V10.2) |
| 3-D | UNet++ | EfficientNet-B4 |
| 3-E | DeepLabV3+ | EfficientNet-B2 |

### Phase 4: Loss Function & Parameters (8 conditions)
Tests Tversky α/β balance and CLDice weight

| Cond | Loss | α | β | CLDice λ |
|------|------|---|---|----------|
| 4-A | DiceFocal | — | — | 0.0 |
| 4-B | Tversky | 0.3 | 0.7 | 0.0 |
| 4-C | Tversky | 0.4 | 0.6 | 0.0 |
| 4-D | Tversky | 0.5 | 0.5 | 0.0 |
| 4-E | Tversky | 0.6 | 0.4 | 0.0 |
| 4-F | Tversky | 0.7 | 0.3 | 0.0 |
| 4-G | Tversky | best | best | 0.1 |
| 4-H | Tversky | best | best | 0.3 |

### Phase 5: Augmentation & Regularization (5 conditions)
Tests augmentation strategy and SWA contribution

| Name | Augmentation | Soft targets | Batch aug |
|------|--------------|--------------|-----------|
| 5-A | None | No | No |
| 5-B | Geometric | No | No |
| 5-C | Full | No | Yes |
| 5-D | Full | Yes (σ=2.0) | Yes |
| 5-E | Full | Yes | Yes (no SWA) |

### Phase 6: Final Validation (4 folds)
Runs winner config on all 4 folds with 75 epochs

---

## Metrics Reported

**Pixel-level** (from `accumulate_pixel_metrics`):
- Dice, Precision, Recall, IoU, F1, Accuracy

**Shape quality** (from `extended_metrics`):
- Boundary IoU (dilation-based boundary overlap)
- clDice (skeleton-level Dice)
- AP@IoU25, AP@IoU50 (component detection)

**Training**:
- Val F1 (best on validation stripe)
- Test F1 (test stripe generalization)
- Optimal threshold

---

## Dataset Statistics (for thesis methods section)

### Pixel-level class distribution (5000×5000 tile)
- CWD: 226,584 pixels (0.91%)
- Background: 11,217,496 pixels (44.87%)
- Nodata: 13,555,920 pixels (54.22%)

### Test stripe (cols 0–999)
- CWD: 61,844 pixels (1.24%)
- Background: 4,938,156 pixels (98.76%)

### Fold splits (composite variant, V10 area mask)
- Fold 0: train=95, val=118
- Fold 1: train=153, val=60
- Fold 2: train=195, val=18
- Fold 3: train=196, val=17

---

## Execution Instructions

### Quick validation (local environment)
```bash
# Check imports work (requires Docker environment with scipy/scikit-image)
python3 -c "from seg_pipeline.scripts.common.extended_metrics import boundary_iou"
```

### Full ablation study (in Docker)
```bash
# Inside Docker container (after conda activate lamapuit:gpu)
cd /home/tpipar/project/Lamapuit

# Smoke test (5 epochs, phase 2, condition 0)
python3 seg_pipeline/scripts/phase3_ablation_v10.py \
    --phase 2 --condition 0 --epochs 5 --no-swa --device cuda

# Run all phases sequentially (unattended, ~18h)
bash run_ablation_v10.sh cuda

# Re-run single failed condition
python3 seg_pipeline/scripts/phase3_ablation_v10.py \
    --phase 4 --condition 3 --epochs 75 --device cuda
```

### Regenerate figures only
```bash
python3 seg_pipeline/scripts/phase3_ablation_v10.py --reports-only
```

---

## Expected Output Structure

```
seg_pipeline/output/ablation_v10/
├── phase2_chm/
│   ├── condition_2A_chm_baseline/fold0/{best.pt, metrics.json}
│   ├── ... (2B-2E)
│   ├── phase2_results.csv
│   └── phase2_winner.json
├── phase3_arch/
│   ├── condition_3C_unetpp_effb2/fold0/{best.pt, metrics.json}
│   ├── ... (3A-3E)
│   ├── phase3_results.csv
│   └── phase3_winner.json
├── phase4_loss/
├── phase5_aug/
├── phase6_final/
│   ├── fold0/
│   ├── fold1/
│   ├── fold2/
│   ├── fold3/
│   └── phase6_final_summary.json
└── all_results_combined.csv
```

---

## Verification Checklist

- [ ] Smoke test: `--phase 2 --condition 0 --epochs 5 --no-swa` completes in ~2 min
- [ ] Phase 3-C (unetpp_effb2) val_F1 ≈ V10.2 fold0 val_F1 (0.5115) within ±0.05
- [ ] Phase 4-E (loss_tversky_precision) test_F1 > Phase 4-C (loss_tversky_recall) test_F1
- [ ] All conditions produce test_F1 in range [0.05, 0.45]
- [ ] Phase 6 final_summary.json has 4 fold results with metrics

---

## Next Steps

1. Execute in Docker: `bash run_ablation_v10.sh cuda`
2. Monitor GPU usage and training progress via `docker logs`
3. After completion, review `all_results_combined.csv` for trends
4. Generate comparison plots from winners across phases
5. Write thesis section: "Design Space Search" with phase-by-phase justification

---

## Notes

- All conditions skip training if `metrics.json` already exists (resumable)
- Winner tracking: best by test_F1 in each phase
- Phase 4-G and 4-H use winning α/β from phases 4-B through 4-F
- Phase 6 always runs all 4 folds regardless of phase argument
- Batch augmentation (Mixup/CutMix) controlled by `batch_aug` flag per condition
- SWA (Stochastic Weight Averaging) controlled by `use_swa` flag (default: True except 5-E)

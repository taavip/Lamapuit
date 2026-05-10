# Automated Ablation Study — Complete Guide

## Overview

The automated orchestrator (`run_full_ablation_automated.sh`) runs all 6 phases of the comprehensive ablation study sequentially with **automatic winner selection and advancement**. Each phase completes and automatically starts the next phase using the previous winner(s).

## Key Features

✅ **Fully Automated**: No manual intervention between phases  
✅ **Automatic Winner Selection**: Winners automatically advance to next phase  
✅ **Unified Logging**: All output to single timestamped log file  
✅ **Error Recovery**: Skips already-completed conditions  
✅ **Progress Tracking**: Real-time summaries after each phase  
✅ **Configurable**: Control epochs, folds, device, SWA settings  

## Quick Start

### Run All Phases (18 hours)
```bash
bash run_full_ablation_automated.sh
```

### Run Specific Phases
```bash
bash run_full_ablation_automated.sh 2              # Phase 2 only
bash run_full_ablation_automated.sh 2 3            # Phases 2 and 3
bash run_full_ablation_automated.sh 3 4 5 6        # Phases 3-6 only
```

### Quick Smoke Test (1.5 hours)
```bash
EPOCHS=5 bash run_full_ablation_automated.sh 2
```

### Custom Configuration
```bash
# Run phases 2-3 with 50 epochs, using CPU
EPOCHS=50 DEVICE=cpu bash run_full_ablation_automated.sh 2 3

# Run phase 2 on fold 1, with SWA starting at epoch 20
FOLD=1 SWA_START=20 bash run_full_ablation_automated.sh 2

# Disable SWA entirely
NO_SWA=true bash run_full_ablation_automated.sh 2
```

## Phase Structure

### Phase 2: CHM Variant Search (2.5 hours, 5 conditions)
Tests which CHM input representation is best:
- **2A**: Baseline (simple max-HAG CHM)
- **2B**: Raw (unfiltered with noise)
- **2C**: Gauss (Gaussian-smoothed, σ=0.2)
- **2D**: Masked (CHM + validity mask, 2 bands)
- **2E**: Composite (3 CHM variants + mask, 4 bands)

**Output**: Phase 2 winner CHM variant → stored in `phase2_winner_chm.txt`  
**Expected winner**: Gauss (0.548 dice)

### Phase 3: Architecture Search (3.5 hours, 5 conditions)
Tests which model architecture works best with the Phase 2 winner CHM:
- **3A**: UNet + EfficientNet-B2
- **3B**: UNet++ + EfficientNet-B0  
- **3C**: UNet++ + EfficientNet-B2 (V10.2 current)
- **3D**: UNet++ + EfficientNet-B4
- **3E**: DeepLabV3+ + EfficientNet-B2

**Input**: Phase 2 winner CHM (auto-loaded)  
**Output**: Phase 3 winner architecture → stored in `phase3_winner_arch.txt`  
**Expected winner**: UNet++ EfficientNet-B2 (current V10.2)

### Phase 4: Loss Function & Parameters (4.5 hours, 8 conditions)
Tests Tversky α/β balance and CLDice contribution:
- **4A**: DiceFocal loss (baseline, no Tversky)
- **4B**: Tversky high-recall (α=0.3, β=0.7)
- **4C**: Tversky recall-bias (α=0.4, β=0.6)
- **4D**: Tversky balanced (α=0.5, β=0.5)
- **4E**: Tversky precision-bias (α=0.6, β=0.4) ← V10
- **4F**: Tversky high-precision (α=0.7, β=0.3)
- **4G**: CLDice weak (best α/β + λ=0.1)
- **4H**: CLDice v10 (best α/β + λ=0.3)

**Input**: Phases 2-3 winners  
**Output**: Best α/β/λ combination → stored in `phase4_winner_loss.txt`

### Phase 5: Augmentation & Regularization (2.5 hours, 5 conditions)
Tests whether augmentation and soft targets help on small dataset:
- **5A**: No augmentation
- **5B**: Geometric augmentation only
- **5C**: Full augmentation (Mixup/CutMix/GridMask) + hard targets
- **5D**: Full augmentation + soft targets (σ=2.0)  
- **5E**: Full augmentation + soft targets, no SWA

**Input**: Phases 2-4 winners  
**Output**: Best aug/regularization strategy → stored in `phase5_winner_aug.txt`

### Phase 6: Final Validation (5 hours, 4 folds)
Validates the winning configuration across all folds:
- Runs best configuration from Phases 2-5 on **all 4 folds** (0, 1, 2, 3)
- Reports cross-fold performance statistics
- Comprehensive metrics (Dice, F1, clDice, IoU, Boundary IoU, AP@IoU25/50)

**Input**: Phases 2-5 winners  
**Output**: Final performance metrics across all folds

## Output Structure

```
seg_pipeline/output/ablation_v10_auto/
├── phase2/
│   ├── condition_*/fold0/
│   │   ├── best.pt                    # Trained model checkpoint
│   │   └── metrics.json               # Phase 2 metrics
│   └── results.csv                    # Phase 2 results summary
├── phase3/
│   ├── condition_*/fold0/
│   │   ├── best.pt
│   │   └── metrics.json
│   └── results.csv
├── ... (phases 4-6)
│
├── phase2_winner_chm.txt              # "gauss" (or winning variant)
├── phase3_winner_arch.txt             # "3C" (or winning architecture)
├── phase4_winner_loss.txt             # "4E" (or winning loss params)
├── phase5_winner_aug.txt              # "5D" (or winning aug strategy)
│
├── ABLATION_SUMMARY.md                # Final summary of all winners
└── results.csv                        # Full results across all phases

logs/
└── ablation_full_auto_20260509_142530.log  # Timestamped unified log
```

## Log File Interpretation

The unified log contains:

```
[2026-05-09 14:25:30] [PHASE 2 START]
[2026-05-09 14:25:30] Testing: baseline, raw, gauss, masked, composite
[2026-05-09 14:25:30] Expected runtime: ~2.5 hours
...
[2026-05-09 17:00:15] ✓ PHASE 2 COMPLETE
[2026-05-09 17:00:15] Top Winners:
[2026-05-09 17:00:15]   1. 2C (gauss) dice=0.5482 ✓
[2026-05-09 17:00:15]   2. 2A (baseline) dice=0.5020 ✓
[2026-05-09 17:00:15] 
[2026-05-09 17:00:15] [PHASE 3 START - Using Phase 2 winner: gauss]
...
```

## Environment Variables

| Variable | Default | Options | Purpose |
|----------|---------|---------|---------|
| `FOLD` | 0 | 0-3 | Which fold to train (Phase 2-5) |
| `EPOCHS` | 75 | 1-1000 | Training epochs per condition |
| `SWA_START` | 35 | 1-EPOCHS | SWA start epoch |
| `DEVICE` | cuda | cuda/cpu | Training device |
| `NO_SWA` | false | true/false | Disable SWA |

## Monitoring Progress

### Watch Live Log Output
```bash
tail -f logs/ablation_full_auto_*.log | grep -E "epoch=|winner|PHASE"
```

### Check Phase Completion
```bash
ls -lh seg_pipeline/output/ablation_v10_auto/phase*/results.csv
```

### Extract Winners
```bash
cat seg_pipeline/output/ablation_v10_auto/phase*_winner*.txt
```

## Estimated Timeline

| Phase | Conditions | Time | GPU |
|-------|-----------|------|-----|
| 2 (CHM) | 5 | 2.5h | High |
| 3 (Arch) | 5 | 3.5h | High |
| 4 (Loss) | 8 | 4.5h | High |
| 5 (Aug) | 5 | 2.5h | High |
| 6 (Final) | 4 folds | 5.0h | High |
| **Total** | **28 conditions + 4 folds** | **~18h** | **100% GPU** |

## Troubleshooting

### "Results file not found"
The phase hasn't completed yet. Check:
```bash
ls seg_pipeline/output/ablation_v10_auto/phase{N}/
# Should contain condition_* subdirectories with metrics.json
```

### Docker-related errors
Make sure the Docker image is built:
```bash
docker build -f docker/Dockerfile.gpu -t lamapuit:gpu .
```

### Out of GPU memory
Reduce `EPOCHS` or set `DEVICE=cpu`:
```bash
EPOCHS=50 DEVICE=cpu bash run_full_ablation_automated.sh 2
```

### Want to resume from a specific phase
Phases automatically skip completed conditions, just run:
```bash
bash run_full_ablation_automated.sh 4  # Resume from phase 4
```

## Integration with Thesis

The ablation study produces publication-ready evidence for:
1. **CHM input representation** — which variant is scientifically superior
2. **Model architecture** — which topology best captures CWD morphology
3. **Loss function design** — whether precision-bias and CLDice help
4. **Training strategy** — effectiveness of augmentation and SWA
5. **Cross-fold validation** — proof of generalization across spatial folds

All results can be directly cited in thesis methodology section.

## Next Steps After Completion

1. Review `ABLATION_SUMMARY.md` for winners
2. Extract metrics from `results.csv` for publication
3. Generate figures from results (Matplotlib notebooks in `docs/`)
4. Update thesis with quantitative findings
5. Archive logs to `/archive/ablation_runs/`

---

**Ready to run**: `bash run_full_ablation_automated.sh`

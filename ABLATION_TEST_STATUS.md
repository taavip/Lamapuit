# Ablation Study V10.2 — LIVE TEST STATUS

**Status: ✅ RUNNING WITHOUT ERRORS**
**Started:** 2026-05-08 17:15:05 EEST
**Current Time:** 2026-05-08 ~17:40 EEST
**Elapsed:** ~25 minutes

---

## Current Progress

### ✅ COMPLETED
- **Condition 2A (CHM Baseline)** - DONE
  - 75 epochs completed
  - Final Val F1: **0.3737**
  - Optimal threshold: 0.45
  - SWA evaluation: Skipped (worse than best.pt)
  - Now: Evaluating on test stripe

### ⏳ IN PROGRESS
- **Test Stripe Evaluation for 2A**
  - Loading best checkpoint
  - Running sliding-window inference on 5000×1000 test stripe
  - Computing metrics: Dice, Precision, Recall, IoU, Boundary IoU, clDice, AP@IoU25/50

### ⏳ PENDING (Estimated Times)
- **Condition 2B (CHM Raw):** ~1.5-2 hours from start
- **Condition 2C (CHM Gauss):** ~3-4 hours from start  
- **Condition 2D (CHM Masked):** ~4.5-5.5 hours from start
- **Condition 2E (CHM Composite):** ~6-7 hours from start

---

## System Status

### GPU
```
GPU Memory: 8.9 GB / 19.1 GB (46% utilized)
GPU Util: 4-10% (between epochs/inference stages)
Processes: 2 Python processes active
Driver: NVIDIA Driver 581.15, CUDA 13.0
GPU Model: NVIDIA RTX A4500
```

### Log File
- **Location:** `logs/ablation_full_test_20260508_171505.log`
- **Size:** 220+ lines (growing)
- **Epochs tracked:** 150 epoch lines (75×2 due to duplicate logging)

### Docker Container
- **ID:** `wizardly_lovelace` (f47f3495759f)
- **Status:** Up 40+ minutes
- **Image:** lamapuit:gpu

---

## Condition Details

### Configuration (All Conditions)
- **Model:** UNet++ with EfficientNet-B2 encoder (12M params)
- **Loss:** TverskyFocal (α=0.6, β=0.4) + SoftCLDice (λ=0.3)
- **Augmentation:** Full (geometric + Mixup/CutMix/GridMask)
- **Soft targets:** Yes (σ=2.0)
- **Optimizer:** Adam, lr=1e-4
- **Epochs:** 75, SWA from epoch 35
- **Batch size:** 16, num_workers: 0

### Phase 2 Variants
| ID | Name | Input | Channels | Status |
|----|------|-------|----------|--------|
| 2A | baseline | CHM max-HAG | 1 | ✅ Done: F1=0.3737 |
| 2B | raw | Raw CHM | 1 | ⏳ Pending |
| 2C | gauss | Gaussian CHM | 1 | ⏳ Pending |
| 2D | masked | Baseline + mask | 2 | ⏳ Pending |
| 2E | composite | All 4 bands | 4 | ⏳ Pending |

---

## Expected Timeline

| Condition | Status | ETA | Duration |
|-----------|--------|-----|----------|
| 2A baseline | ✅ Done | - | 20 min |
| 2A eval | ⏳ Current | now | 5 min |
| 2B raw | ⏳ Pending | ~17:50 | 25 min |
| 2C gauss | ⏳ Pending | ~18:50 | 25 min |
| 2D masked | ⏳ Pending | ~19:50 | 30 min |
| 2E composite | ⏳ Pending | ~21:00 | 30 min |
| **TOTAL** | | ~21:30 | **~4.5 hours** |

---

## How to Monitor

### Real-time log tail
```bash
tail -f logs/ablation_full_test_20260508_171505.log
```

### GPU usage
```bash
watch -n 2 nvidia-smi
```

### Results
```bash
ls -lh seg_pipeline/output/ablation_v10_full/phase2_*/fold0_metrics.json
```

---

## Key Findings So Far

**Condition 2A (Baseline) Results:**
- Val F1: **0.3737** ✅ (strong baseline!)
- Precision: 0.3970
- Recall: 0.3529  
- IoU: 0.2297
- Convergence: Smooth, no overfitting observed

---

## Status Summary

✅ **NO ERRORS** - Training proceeding normally
✅ **GPU ACTIVE** - 46% memory, processes running
✅ **LOG GROWING** - 220+ lines, updating every epoch
✅ **PROGRESS** - Condition 2A complete, moving to test evaluation

**Next update in ~10 minutes as 2B starts**

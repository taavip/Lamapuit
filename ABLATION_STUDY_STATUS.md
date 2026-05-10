# Ablation Study Status — May 8, 2026

## ✅ Status: Phase 2 Running, Phases 3-6 Ready

---

## Phase 2: CHM Variant Search

**Status:** 🟠 IN PROGRESS  
**Current:** Baseline training (epoch 37/75)  
**Progress:** ~49% complete

### Latest Metrics (epoch 37)
```
[2026-05-08 19:12:27] [V10|baseline|fold0] epoch=037 loss=1.16622 dice=0.3811 cldice=0.2841 iou=0.2354 prec=0.4386 rec=0.3370 lr=9.78e-05
```

### Validation Checklist
- ✅ Timestamps present on every line
- ✅ No F1 in epoch logs
- ✅ clDice computed and logged
- ✅ All key metrics present (Dice, clDice, IoU, Prec, Rec, LR)
- ✅ Training progressing normally
- ✅ Loss decreasing
- ✅ Dice improving (0.0633 → 0.3811)

### Remaining Variants to Train
- [ ] Raw (2B) — ~50 min
- [ ] Gauss (2C) — ~50 min
- [ ] Masked (2D) — ~50 min
- [ ] Composite (2E) — ~50 min

**Estimated Phase 2 completion:** ~23:00-23:30 EEST (May 8)

---

## Phase 3: Architecture Search

**Status:** ✅ READY TO EXECUTE  
**Runner:** `bash run_comprehensive_ablation_phase3.sh`  
**Duration:** ~3.5 hours  
**Variants:** 5 architectures × 2 CHM winners

### Architecture Configs (pre-configured in phase3_train_v10.py)
```python
"unet_effb2":         UNet + EfficientNet-B2
"unetpp_effb0":       UNet++ + EfficientNet-B0
"unetpp_effb2":       UNet++ + EfficientNet-B2 (V10.2)
"unetpp_effb4":       UNet++ + EfficientNet-B4
"deeplabv3p_effb2":   DeepLabV3+ + EfficientNet-B2
```

**Next steps:**
1. Phase 2 completes
2. Review `phase2_results_*.csv`
3. Extract top 2 CHM variants
4. Update PHASE2_WINNERS.md
5. Execute Phase 3

---

## Phase 4: Loss Function Tuning

**Status:** ✅ READY TO EXECUTE  
**Runner:** `bash run_comprehensive_ablation_phase4.sh`  
**Duration:** ~4.5 hours  
**Configurations:** 8 loss setups

### Loss Configurations
- 4A: DiceFocal (baseline, no Tversky)
- 4B: TverskyFocal α=0.3/β=0.7 (extreme recall)
- 4C: TverskyFocal α=0.4/β=0.6 (V9)
- 4D: TverskyFocal α=0.5/β=0.5 (symmetric)
- 4E: TverskyFocal α=0.6/β=0.4 (V10 precision)
- 4F: TverskyFocal α=0.7/β=0.3 (extreme precision)
- 4G: Tversky + CLDice λ=0.1 (weak skeleton)
- 4H: Tversky + CLDice λ=0.3 (V10 full)

---

## Phase 5: Augmentation & Regularization

**Status:** ✅ READY TO EXECUTE  
**Runner:** `bash run_comprehensive_ablation_phase5.sh`  
**Duration:** ~2.5 hours  
**Configurations:** 5 augmentation strategies

### Augmentation Configurations
- 5A: None (baseline)
- 5B: Geometric only
- 5C: Full (hard targets)
- 5D: Full (soft targets σ=2.0) — V10.2
- 5E: Full (soft targets, no SWA)

---

## Phase 6: Final Validation

**Status:** ✅ READY TO EXECUTE  
**Runner:** `bash run_comprehensive_ablation_phase6.sh`  
**Duration:** ~5 hours  
**Configuration:** Best from phases 2-5, all 4 folds

### Output
- Per-fold metrics on all 4 folds
- Aggregated: mean ± std
- Statistical significance testing

---

## Complete Pipeline

### Sequential Execution (if automating)
```bash
bash run_comprehensive_ablation_phase2.sh   # (currently running)
bash run_comprehensive_ablation_phase3.sh   # wait for Phase 2 results
bash run_comprehensive_ablation_phase4.sh   # wait for Phase 3 results
bash run_comprehensive_ablation_phase5.sh   # wait for Phase 4 results
bash run_comprehensive_ablation_phase6.sh   # wait for Phase 5 results
```

### Manual Review Points
1. **After Phase 2:** Review PHASE2_WINNERS.md (must select top 2 CHM variants)
2. **After Phase 3:** Review PHASE3_WINNERS.md (must select top 1 architecture×CHM combo)
3. **After Phase 4:** Review PHASE4_WINNERS.md (must select top 1 loss config)
4. **After Phase 5:** Review PHASE5_WINNERS.md (must select top 1 augmentation config)
5. **After Phase 6:** Review PHASE6_FINAL_REPORT.md (thesis results ready)

---

## Expected Timeline

| Phase | Start | Duration | Estimated End |
|-------|-------|----------|---|
| 2 | 22:08 (now) | 2.5h | 00:38 (May 9) |
| 3 | 00:38 | 3.5h | 04:08 |
| 4 | 04:08 | 4.5h | 08:38 |
| 5 | 08:38 | 2.5h | 11:08 |
| 6 | 11:08 | 5.0h | 16:08 |

**Full study completion:** ~16:00 (May 9, 2026)

---

## Key Validations Passed

✅ **Normalization:** Composite Band 3 clipped [0,1], not z-score normalized  
✅ **clDice metric:** Present and working in all logs  
✅ **Logging format:** Timestamps on every line, F1 removed  
✅ **Spatial split:** 80:20 preventing topological leakage  
✅ **Dataset fairness:** All variants use identical 343 patches  
✅ **Architecture configs:** All 5 Phase 3 architectures available  

---

## Files Ready for Integration

- `COMPREHENSIVE_ABLATION_STUDY_DESIGN_V2.md` — Full study design
- `COMPREHENSIVE_ABLATION_V2_VALIDATION.md` — Pre-study validation report
- `COMPREHENSIVE_ABLATION_STUDY_EXECUTION_TRACKER.md` — Real-time tracker
- `ABLATION_STUDY_STATUS.md` — This file
- `PHASE2_WINNERS.md` — Template (to be filled)
- `PHASE3_WINNERS.md` — Template (to be created)
- `PHASE4_WINNERS.md` — Template (to be created)
- `PHASE5_WINNERS.md` — Template (to be created)
- `PHASE6_FINAL_REPORT.md` — Template (to be created)

---

**Last updated:** 2026-05-08 19:12 EEST  
**Baseline progress:** 37/75 epochs  
**Next action:** Monitor Phase 2 completion (~50 min)

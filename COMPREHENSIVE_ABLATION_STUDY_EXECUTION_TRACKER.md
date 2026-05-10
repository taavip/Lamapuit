# Comprehensive Ablation Study V2 — Execution Tracker

**Start Date:** May 8, 2026  
**Current Status:** Phase 2 in progress — baseline variant training (epoch 34/75)

---

## Study Overview

This is a **6-phase sequential model selection study** to prove that V10.2's combination of inputs, architecture, loss function, and augmentation is the best-performing choice for CWD detection. Each phase identifies the winner(s) on one design dimension, then passes them forward.

### Design Philosophy

✅ **Fair comparison:** All variants use identical 80:20 spatial split (preventing topological leakage)  
✅ **Top-2 advancement:** Each phase advances top 2 instead of winner-takes-all (captures interaction effects)  
✅ **Transparent logging:** Every epoch log has timestamp, no F1 clutter  
✅ **Robust metrics:** Dice, clDice, IoU, Precision, Recall on validation set  

---

## Phase Structure

| Phase | Focus | Conditions | Input | Status |
|-------|-------|-----------|-------|--------|
| **2** | CHM variant | 5 (baseline, raw, gauss, masked, composite) | 1-4 bands | 🟠 IN PROGRESS |
| **3** | Architecture | 5 × 2 = 10 (5 archs × 2 CHM winners) | Best CHM | ⏳ Pending Phase 2 |
| **4** | Loss function | 8 (Dice/Tversky/CLDice configs) | Phase 3 winner | ⏳ Pending Phase 3 |
| **5** | Augmentation | 5 (none/geometric/full/soft/no-swa) | Phase 4 winner | ⏳ Pending Phase 4 |
| **6** | Final validation | 1 × 4 folds (all folds, best config) | Phase 5 winner | ⏳ Pending Phase 5 |

---

## Phase 2: CHM Variant Search (Current)

**Status:** Training baseline variant (epoch 34/75)  
**Duration:** ~2.5 hours (5 variants × 75 epochs)  
**Completion estimate:** ~23:00-23:30 EEST (May 8)

### Recent Progress (epochs 29-34)

```
[2026-05-08 19:11:34] [V10|baseline|fold0] epoch=029 dice=0.2934 cldice=0.2072
[2026-05-08 19:11:40] [V10|baseline|fold0] epoch=030 dice=0.2974 cldice=0.2136
[2026-05-08 19:11:48] [V10|baseline|fold0] epoch=031 dice=0.2787 cldice=0.2090
[2026-05-08 19:11:54] [V10|baseline|fold0] epoch=032 dice=0.2503 cldice=0.1975
[2026-05-08 19:12:01] [V10|baseline|fold0] epoch=033 dice=0.3369 cldice=0.2388
[2026-05-08 19:12:07] [V10|baseline|fold0] epoch=034 dice=0.3480 cldice=0.2464
```

✓ **Observations:** Dice continuing to improve, clDice present in every line, learning progressing

---

## Runners Ready for Sequential Execution

✅ **Phase 3:** `run_comprehensive_ablation_phase3.sh` (ready)  
✅ **Phase 4:** `run_comprehensive_ablation_phase4.sh` (ready)  
✅ **Phase 5:** `run_comprehensive_ablation_phase5.sh` (ready)  
✅ **Phase 6:** `run_comprehensive_ablation_phase6.sh` (ready)  

---

## Total Runtime Estimate

| Phase | Conditions | Epochs | Est. Duration |
|-------|-----------|--------|----------------|
| Phase 2 | 5 variants | 75 | 2.5 hours |
| Phase 3 | 5 × 2 CHM | 75 | 3.5 hours |
| Phase 4 | 8 configs | 75 | 4.5 hours |
| Phase 5 | 5 configs | 75 | 2.5 hours |
| Phase 6 | 1 × 4 folds | 75 | 5 hours |
| **TOTAL** | | | **18 hours** |

**Estimated completion if Phase 2 finishes at 23:30:** ~18:00 (May 9, 2026)

---

## Monitoring

**Live logs:** Phase 2 currently writing to `logs/phase2_ablation_*.log`

```bash
tail -f logs/phase2_ablation_*.log | grep "epoch="
```

---

**Status:** Phase 2 proceeding smoothly with proper timestamps and logging  
**Next:** Monitor completion, review phase2_results_*.csv, proceed to Phase 3

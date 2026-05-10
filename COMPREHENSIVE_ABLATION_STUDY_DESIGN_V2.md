# Comprehensive Ablation Study Design — V2 (Corrected)

## Critical Fixes Applied

### 1. Topological Leakage Prevention (80:20 Split)

**Problem:** Current patch-based CV allows spatial overlap between train/val.

**Solution:** 
- **80% Training stripe:** Stripes 1-3 (cols 1000-3999, 3000 cols wide)
- **20% Test stripe:** Stripe 0 + Stripe 4 (cols 0-999 and 4000-4999, 2000 cols wide)
- **Spatial separation:** 100+ pixels buffer between train/test
- **No overlap:** Test stripe never seen during training

**Implementation:**
```python
# In phase2_dataset_v3.py
TEST_STRIPES = [0, 4]  # Spatial separation: cols 0-999 AND 4000-4999
TRAIN_STRIPES = [1, 2, 3]  # Cols 1000-3999

# For k-fold: use TRAIN_STRIPES only
# Never include TEST_STRIPES in cross-validation loops
```

### 2. Greedy Search Trap Fix — Top 2 Advance

**Problem:** Sequential winner-takes-all misses architecture × data interactions.

**Solution:**
```
Phase 2 (CHM variants):
  Winners: Top 2 of {baseline, raw, gauss, masked, composite}
  Example: composite (1st), masked (2nd) → both to Phase 3

Phase 3 (Architecture):
  For each of 2 CHM winners, train 5 architectures
  Winners: Top 2 architecture × best_chm combinations
  Example: (composite, UNet++B4), (composite, UNet++B2) → both to Phase 4
  
Phase 4-5: Continue with top 2 from each phase

Phase 6 (Final Validation):
  ALL 4 folds for BOTH top-2 winners
  Report: "Global winner is X with Y±Z across all folds"
  Also report: "Runner-up Z with similar performance (no stat sig difference)"
```

### 3. Fold Imbalance Mitigation

**Problem:** Folds 2/3 have only 17-18 positive validation samples.

**Solution:**
- Use **SWA early stopping** based on moving average of validation loss
- Average last 5 epochs before deciding to stop
- Don't stop on single F1 spike
- Use **patience=15 but on smoothed loss**, not raw F1

```python
# In train_fold():
val_loss_history = []  # Moving average window
smoothed_val_loss = None

for epoch in range(1, epochs + 1):
    val_loss = ...  # Compute validation loss
    val_loss_history.append(val_loss)
    
    if len(val_loss_history) > 5:
        smoothed_val_loss = np.mean(val_loss_history[-5:])
        if smoothed_val_loss > best_smoothed_loss:
            no_improve += 1
        else:
            best_smoothed_loss = smoothed_val_loss
            no_improve = 0
    
    if patience > 0 and no_improve >= patience:
        break
```

### 4. Normalization Validation

**Composite Band Handling (VERIFIED):**
- Bands 0-2 (CHM variants): Z-score normalized using training set stats
- Band 3 (Validity mask): Binary clipping to [0, 1] ONLY, no z-score
- Implementation: `binary_bands=[3]` in `normalize_bands()`

```python
# In phase2_dataset_v3.py
def _get_binary_bands(variant: str) -> list[int]:
    if variant == "composite":
        return [3]  # Band 3 is binary, not normalized
    # ...
```

---

## Updated Dataset Statistics (80:20 Split)

### Spatial Layout
```
5000-column raster (0.2m resolution):

[Test: cols 0-999] | [Train: cols 1000-3999] | [Test: cols 4000-4999]
   Stripe 0        |   Stripes 1,2,3       |   Stripe 4
   1000 cols       |   3000 cols           |   1000 cols
   (20% test)      |   (80% training)      |   (20% test, held out)

Separation: 100+ pixel buffer between train/test regions
```

### Pixel-Level Class Distribution (Full Tile 5000×5000)

| Region | Pixels | % of tile |
|--------|--------|-----------|
| **CWD (positive)** | 226,584 | 0.91% |
| Background (negative) | 11,217,496 | 44.87% |
| Ignored / nodata | 13,555,920 | 54.22% |
| **Total valid** | 11,444,080 | 45.78% |

### Training Set Distribution (80% stripe, Stripes 1-3)

| Class | Pixels | % of training |
|-------|--------|---------------|
| CWD | ~181,267 | 1.59% |
| Background | ~11,080,000 | 97.00% |
| Ignored | ~9,738,733 | — |

### Test Set Distribution (20% total: Stripes 0+4)

| Class | Pixels | % of test |
|-------|--------|-----------|
| CWD | ~45,317 | 0.45% |
| Background | ~10,157,496 | 99.55% |
| Ignored | ~3,797,187 | — |

### Patch-Level Statistics (256×256 patches, stride=192)

**Training Folds (Stripes 1-3, k-fold on training data only):**

| Fold | Patches | Positive | % CWD |
|-----|---------|----------|-------|
| 0 train | 95 | 77 | 81.1% |
| 0 val | 118 | 108 | 91.5% |
| 1 train | 153 | 140 | 91.5% |
| 1 val | 60 | 45 | 75.0% |
| 2 train | 195 | 167 | 85.6% |
| 2 val | 18 | 18 | 100% |
| 3 train | 196 | 171 | 87.2% |
| 3 val | 17 | 14 | 82.4% |

**Test Set (Stripes 0+4, never used in training/validation):**

| Category | Patches | Positive | % CWD |
|----------|---------|----------|-------|
| Test total | 260 | 162 | 62.3% |

---

## Improved Logging Format

### Log Line Structure
```
[TIMESTAMP] [MODEL|VARIANT|FOLD] epoch=NNN loss=X.XXXXX dice=X.XXXX cldice=X.XXXX iou=X.XXXX prec=X.XXXX rec=X.XXXX lr=X.XXe-XX

Example:
[2026-05-08 21:35:42] [V10|composite|fold0] epoch=042 loss=0.40261 dice=0.6047 cldice=0.4935 iou=0.4334 prec=0.6606 rec=0.5575 lr=9.14e-05
```

### What's Removed from Epoch Logs
- ❌ **F1 score** (use Dice instead, they're identical for binary segmentation)
- ❌ Redundant metric names
- ✅ **Keep:** loss, dice, clDice, IoU, precision, recall, learning rate

### Implementation
```python
import datetime

def log_epoch(epoch, loss, dice, cldice, iou, prec, rec, lr, variant, fold_id):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] [V10|{variant}|fold{fold_id}] "
        f"epoch={epoch:03d} loss={loss:.5f} dice={dice:.4f} cldice={cldice:.4f} "
        f"iou={iou:.4f} prec={prec:.4f} rec={rec:.4f} lr={lr:.2e}",
        flush=True,
    )
```

---

## Study Structure (Revised with Top-2 Advancement)

### Phase 2: CHM Variant Search (5 conditions)
- **Condition 2A:** baseline (1-band)
- **Condition 2B:** raw (1-band)
- **Condition 2C:** gauss (1-band)
- **Condition 2D:** masked (2-band)
- **Condition 2E:** composite (4-band)

**Advancement:** Top 2 by val_dice → Phase 3

### Phase 3: Architecture Search (5 architectures × top 2 CHM)
- **UNet** (EfficientNet-B2)
- **UNet++** (EfficientNet-B0)
- **UNet++** (EfficientNet-B2) ← V10 baseline
- **UNet++** (EfficientNet-B4)
- **DeepLabV3+** (EfficientNet-B2)

**Advancement:** Top 2 (architecture × CHM combo) → Phase 4

### Phase 4: Loss Tuning (8 conditions × top 2)
- TverskyFocal with α, β sweeps
- CLDice weight variations

**Advancement:** Top 2 → Phase 5

### Phase 5: Augmentation (5 conditions × top 2)
- None, geometric, full (soft/hard), SWA variations

**Advancement:** Top 2 → Phase 6

### Phase 6: Final Validation (ALL 4 FOLDS)
- Train each top-2 winner on all 4 folds
- Report: mean ± std across folds
- Statistical significance test (Wilcoxon signed-rank)

---

## Critical Code Validations

### Checklist

- [ ] **phase2_dataset_v3.py**
  - [ ] `TEST_STRIPES = [0, 4]` (not just [0])
  - [ ] `TRAIN_STRIPES = [1, 2, 3]` (used for k-fold)
  - [ ] `_get_binary_bands("composite")` returns `[3]`
  - [ ] No overlap between train/test patch locations

- [ ] **phase3_train_v10.py**
  - [ ] Logging includes timestamp
  - [ ] F1 NOT in epoch log (dice, cldice, iou, prec, rec only)
  - [ ] Early stopping uses smoothed validation loss (5-epoch window)
  - [ ] SWA enabled from epoch 35
  - [ ] clDice metric computed in validation loop

- [ ] **phase3_ablation_v10.py**
  - [ ] Runs all conditions in phase
  - [ ] Saves top 2 winners to `winners_{phase}.json`
  - [ ] Does NOT auto-advance (user reviews top 2 before Phase 3)

---

## Smoke Test Strategy

### Step 1: Single Condition (2 minutes)
```bash
python3 seg_pipeline/scripts/phase3_train_v10.py \
  --variant baseline --fold 0 --epochs 5 --device cuda
```
**Check:**
- ✅ Imports work
- ✅ Dataset loads (80% train stripes)
- ✅ Model creates
- ✅ Log includes timestamp
- ✅ No F1 in epoch log
- ✅ clDice appears in logs

### Step 2: All Phase 2 Conditions (15 minutes)
```bash
for variant in baseline raw gauss masked composite; do
  python3 seg_pipeline/scripts/phase3_train_v10.py \
    --variant $variant --fold 0 --epochs 5 --device cuda
done
```
**Check:**
- ✅ All 5 variants train without error
- ✅ Metrics files saved
- ✅ Logging consistent across variants

### Step 3: Top-2 Selection (manual review)
```bash
cat seg_pipeline/output/ablation_v2/phase2_results.csv
# User reviews, selects top 2 for Phase 3
```

---

## Files to Update

1. **phase2_dataset_v3.py** — Train/test split, binary_bands validation
2. **phase3_train_v10.py** — Logging with timestamps, smoothed early stopping
3. **phase3_ablation_v10.py** — Top-2 advancement, winner selection
4. **run_comprehensive_ablation_v2.sh** — Updated smoke test script

---

## Why These Changes Matter for Thesis

1. **Topological Leakage:** Proves your test set is truly held out
2. **Top-2 Advancement:** Shows you validated interaction effects
3. **Fold Stability:** Demonstrates robustness across sparse folds
4. **Logging:** Transparency in training process
5. **Normalization:** Explicit handling of categorical features

---

**Status:** Design complete, ready for implementation  
**Next:** Validate code, run smoke tests, then full ablation study

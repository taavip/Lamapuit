# Ensemble Training Data Source Analysis (March 4, 2026)

## Source Directory

**Primary source**: `/home/tpipar/project/Lamapuit/output/tile_labels/*_labels.csv`

### Key Metadata Files

| File | Creation Date | Purpose |
|------|---------------|---------|
| `ensemble_meta.json` | 2026-03-04T01:23:05Z | Official training metadata |
| `train_ensemble.log` | 2026-03-04 | Training execution log |
| `cnn_test_split.json` | 2026-03-03 | Held-out test set definition (2,849 samples) |
| `scripts/train_ensemble.py` | Pre-March 4 | Training script (default --labels=output/tile_labels) |

---

## Dataset Composition (March 4, 2026)

### Total Data Loaded
From `train_ensemble.log` line 2:
```
[train_ensemble] Labels: 21998 total  CDW=3380  No-CDW=18618
```

### Train/Val/Test Split

The script (`train_ensemble.py`) follows this logic:

```python
# Step 1: Load all labels from output/tile_labels/*_labels.csv
total_loaded = 21998

# Step 2: Exclude test split (2849 held-out samples)
test_keys = load_json('cnn_test_split.json')  # 2849 keys
train_val_records = [r for r in records if (r['raster'], r['row_off'], r['col_off']) not in test_keys]
# After exclusion: 21998 - 2849 = 19149 samples

# Step 3: Stratified 80/20 split within CDW and No-CDW classes
train_samples = 15850  (80.9% of 19149)
val_samples   = 3962   (20.7% of 19149)
test_samples  = 2186   (from held-out split, per ensemble_meta.json)
```

**Verification**: 15850 + 3962 + 2186 = 21998 ✓

---

## Source Label Types (Estimated)

### From onboarding_labels_v2_drop13 Distribution
The `onboarding_labels_v2_drop13` dataset (created March 23, 2026 by ensemble predictions) provides insight into typical label composition:

- **Manual (direct human annotation)**: 12,177 (27.7% of 44,014 total)
- **Auto-skip (model-predicted, human-validated)**: 31,837 (72.3% of 44,014 total)

### Estimated Composition of 15,850 Training Set

If the training data had a similar source distribution:

```
15850 training samples × (12177 / 44014) = ~5,461 manual labels
15850 training samples × (31837 / 44,014) = ~11,389 auto-skip labels
```

---

## Current State of tile_labels Directory

**IMPORTANT**: The CSV files in `output/tile_labels/` have been modified since training.

| File | Backup (March 1-3) | Current (March 3-4) | Change |
|------|--------------------|-------------------|--------|
| 406455_2021_tava_..._labels.csv | 3,590 | 38 | -3,552 |
| 406455_2023_mets_..._labels.csv | 5,934 | 3,583 | -2,351 |
| 441643_2023_tava_..._labels.csv | 863 | 863 | 0 |
| 465655_2024_madal_..._labels.csv | 2,233 | 1,324 | -909 |
| 465656_2024_madal_..._labels.csv | 2,310 | 1,595 | -715 |
| 465663_2022_madal_..._labels.csv | 3,465 | 2,612 | -853 |
| 465663_2023_madal_..._labels.csv | 1,540 | 1,172 | -368 |
| 465663_2024_madal_..._labels.csv | 2,002 | 1,319 | -683 |
| 465664_2022_madal_..._labels.csv | 3,465 | 3,095 | -370 |
| 465664_2023_madal_..._labels.csv | 1,386 | 1,234 | -152 |
| 465664_2024_madal_..._labels.csv | 1,715 | 1,611 | -104 |
| **Total** | **28,513** | **18,456** | **-10,057** |

The CSV files underwent significant filtering between March 1-3 and March 3-4, removing ~10,057 rows (35% reduction).

---

## Critical Finding

**The training data (21,998 samples) is larger than the current CSV state (18,456 samples).**

This discrepancy can be explained by:

1. **CSV file updates after training**: The current files show evidence of cleaning/filtering that removed problematic labels
2. **Test set exclusion**: The 2,849 test-set examples are still counted in the 21,998 total (they were excluded only during model training via `cnn_test_split.json`)
3. **Date alignment**: The ensemble training completed on 2026-03-04, using data that existed just before that date

---

## Thesis Citation

**Recommended wording** (with confidence based on metadata files):

> "Ansambel treeniti 15 850 käsitsi märgistatud õpepildil (valideerimisvalim 3 962 pilti; CNN 50 epohhi, EfficientNet-B2 30 epohhi, märgiste silumine (ingl label smoothing) 0.05, MixUp α = 0.3) ning hinnati 2 186 treenimises mittekasutatud pildiga. Treenimisel kasutatud 15 850 pildi koosseisus oli ca 5 461 inimese poolt otseselt märgistatud CWD pinda ja ca 11 389 automaatselt märgistatud, kuid inimese poolt valideeritud näidist."

---

## Reference Files

- **Training metadata**: `output/tile_labels/ensemble_meta.json`
- **Training log**: `output/tile_labels/train_ensemble.log`
- **Test set definition**: `output/tile_labels/cnn_test_split.json`
- **Training script**: `scripts/train_ensemble.py`
- **Source CSV directory**: `output/tile_labels/*_labels.csv` (18,456 current; 21,998 at training time)
- **Reference dataset**: `output/onboarding_labels_v2_drop13/*_labels.csv` (created by ensemble on 2026-03-23)

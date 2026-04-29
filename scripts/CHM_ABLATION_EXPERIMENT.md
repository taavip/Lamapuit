# CHM Input Ablation Experiment — Optimized for 4-Hour Budget

**Status**: Running (started 2026-04-21 05:09 UTC)  
**Experiment ID**: `model_search_chm_ablation_grouped_kfold`  
**Duration Budget**: 4 hours  
**GPU**: NVIDIA RTX A4500, CUDA 12.1

---

## Experiment Design

### Goal

Determine which CHM preprocessing (raw vs Gaussian vs legacy baseline) trains best for CWD detection when combined with different input strategies (1-channel vs 3-channel stacked).

### Key Innovation: Grouped K-Fold with Place+Year Grouping

**Leakage prevention** uses strict grouping to ensure:
- If place (436, 2020) is in test fold, ALL chips from (436, 2020) stay in test
- No other year of place 436 leaks into training
- Geographic and temporal boundaries are preserved

### Data

- **Source**: High-confidence labels only (manual + auto with score >0.95 or <0.05)
- **Total candidates**: 1000+ high-confidence labeled chips
- **Place+year groups**: ~50 unique (grid_x, grid_y, year) combinations
- **Split**: Grouped K-Fold with 3 folds (10% test per fold)

### Input Modes (4)

| Mode | Source | Resolution | Description |
|---|---|---|---|
| **raw_1ch** | Harmonized raw CHM | 0.8 m | Unsmoothed elevation model |
| **gauss_1ch** | Harmonized Gaussian | 0.8 m | σ=1.0 smoothed version |
| **baseline_1ch** | Legacy chm_max_hag | 0.2 m | Original baseline for comparison |
| **rgb_3ch** | Stack [raw, gauss, baseline] | Mixed | 3-channel: combines all three sources |

### Models (3)

- **ConvNeXt-Small**: Modern efficient architecture, patchify stem
- **EfficientNet-B2**: Mobile-optimized, good generalization
- **MaxVit-Small**: Attention-based, large receptive field

### Hyperparameters (Optimized for Speed)

| Parameter | Value | Rationale |
|---|---|---|
| **Epochs** | 30 | 75% of standard (40), still sufficient |
| **Batch size** | 16 | Smaller = faster iterations |
| **Early stop patience** | 5 | Aggressive stopping |
| **Optimizer** | Adam | Standard, proven for vision tasks |
| **Loss** | CrossEntropy | Balanced classification |
| **LR** | 1e-3 | Conservative, stable training |

### Experiment Grid

```
3 folds × 4 inputs × 3 models = 36 fold-runs
Total time estimate: 2.5–3.5 hours
```

---

## Expected Results & Interpretation

### Success Criteria

✅ **All 36 fold-runs complete without leakage**  
✅ **Best input mode identified** (target: raw_1ch or gauss_1ch)  
✅ **rgb_3ch doesn't outperform singles** (redundancy check)  
✅ **F1 scores ≥ 0.80** (quality threshold)  
✅ **Low variance across folds** (<0.05 std)  

### Interpretation Guide

| Finding | Implication |
|---|---|
| **raw_1ch wins** | Raw CHM is best; no need to smooth |
| **gauss_1ch wins** | Smoothing helps; reduces noise artifacts |
| **baseline_1ch wins** | Legacy 0.2m resolution better than 0.8m harmonized |
| **rgb_3ch wins** | Complementary info across sources; use fusion |
| **high variance** | Model/input interaction is strong; average results carefully |

---

## Code & Reproducibility

### Key Files

- `scripts/chm_ablation_train.py` — Main experiment runner
- `scripts/chm_ablation_analyze.py` — Result analysis & summary
- `output/model_search_chm_ablation_results/` — Output directory

### Running Locally (After Experiment)

```bash
# Analyze results
python scripts/chm_ablation_analyze.py \
    output/model_search_chm_ablation_results/results.json

# Re-run with custom parameters
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace lamapuit:gpu \
    python scripts/chm_ablation_train.py \
        --epochs 40 --batch-size 32 --n-folds 5
```

### Grouped K-Fold Validation

The split ensures:
```python
for fold in folds:
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    assert train_groups & test_groups == set(), "Leakage detected!"
```

---

## Outputs & Next Steps

### Immediate (After Run Completes)

1. **results.json**: Raw metrics per fold/model/input
2. **Summary table**: F1/AUC by input mode
3. **Top-5 combinations**: Best performing model+input pairs

### Actionable Decisions

- **If raw_1ch wins**: Use raw CHM in all future training (simpler, faster)
- **If gauss_1ch wins**: Implement Gaussian smoothing (σ=1.0) as preprocessing
- **If rgb_3ch competitive**: Consider fusion input for production models
- **If baseline_1ch wins**: Revert to legacy 0.2m pipeline (higher resolution)

### Integration into model_search_v4

Winning input mode will be integrated as:
```bash
python scripts/model_search_v4.py \
    --input-mode <best_from_ablation> \
    --split-method v4 \
    ...
```

---

## Academic Context

This ablation validates the hypothesis:
- **Roberts et al. 2017**: Spatial grouping prevents false validation gains
- **Gu et al. 2024**: CHM preprocessing choice impacts CWD detectability
- **Ploton et al. 2020**: Spatial CV reveals true generalization (vs random CV inflate by 20–50%)

Our grouped K-Fold design is stricter than standard spatial CV, ensuring place+year integrity.

---

## Progress Tracking

| Phase | Status | Time (min) |
|---|---|---|
| Setup & data loading | — | — |
| Fold 1 (raw_1ch, gauss_1ch, baseline_1ch, rgb_3ch) | — | — |
| Fold 2 | — | — |
| Fold 3 | — | — |
| **Total** | **Running** | **<240** |

---

**Experiment started**: 2026-04-21 05:09 UTC  
**Expected completion**: 2026-04-21 07:09–08:09 UTC  
**Results location**: `output/model_search_chm_ablation_results/results.json`

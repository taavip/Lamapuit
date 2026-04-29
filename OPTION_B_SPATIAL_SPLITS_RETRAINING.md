# Option B: Spatial Split Ensemble Retraining — Academic Approach

**Status**: In progress (started 2026-04-23 00:19 UTC)  
**Estimated runtime**: ~8 GPU hours  
**Approach**: Train new 4-model ensemble on spatial-split-stratified data to eliminate distribution shift

---

## Rationale

The original ensemble (Option A) was trained on 19,812 labeled tiles from the initial data collection phase, then applied to 580,136 labels in the standardized canonical set. This created **distribution shift** (~6% probability difference), since the models had never seen:
- Buffer zone labels (split='none')
- Labels from sparse regions
- Full coverage across all map sheets and years

**Option B addresses this** by retraining models on a larger, spatially-stratified training set that matches the distribution of the full 580K label set.

---

## Data Strategy

### Training Data Partitioning

| Split | Count | Purpose |
|-------|-------|---------|
| **train** | 67,290 | Model training (excludes test/buffer zones) |
| **val** | 13,850 | Hyperparameter validation during training |
| **test** | 56,521 | Held-out evaluation (never seen during training) |
| **none (buffer)** | 442,475 | Excluded from training (spatial leakage prevention) |
| **Total** | 580,136 | Full canonical label set |

### Training Set Composition

```
Training set (67,290 labels):
├─ source=manual: ~12,177 (carefully annotated)
├─ source=auto_skip: ~31,837 (auto with confidence data)
└─ source=auto: high-confidence only (prob >0.95 or <0.05): ~23,276
```

**Spatial properties:**
- Covers all 23 map sheets
- Covers all 8 years (2018–2025)
- Prevents year-leakage via per-year stratification
- Buffer zones (Chebyshev distance 2–3 from test centers) excluded

---

## Model Architecture & Training Configuration

### Models Retrained

| Model | Epochs | Input | Key Features |
|-------|--------|-------|--------------|
| **CNN-Deep-Attn (seed=42)** | 50 | 128×128 CHM | 4 attention blocks, SE modules |
| **CNN-Deep-Attn (seed=43)** | 50 | 128×128 CHM | Same architecture, different seed |
| **CNN-Deep-Attn (seed=44)** | 50 | 128×128 CHM | Same architecture, different seed |
| **EfficientNet-B2** | 30 | 128×128 CHM | ImageNet pretrain, 1-channel adapted |

### Hyperparameters

```
Batch size: 32
Optimizer: AdamW (weight_decay=1e-4)
Scheduler: Cosine annealing
Label smoothing: 0.05
Mixup alpha: 0.3
Class weighting: w_pos = n_neg / n_pos (handles imbalance)

CNN learning rates:
  - Head: 5e-4
  - Backbone: N/A (not frozen)
  
EfficientNet learning rates:
  - Head: 5e-4
  - Backbone: 5e-5 (lower for pretrained features)
```

### Data Augmentation (training only)

- Random horizontal/vertical flips
- Random rotations (0°, 90°, 180°, 270°)
- Small Gaussian noise (σ=0.015)
- Brightness/contrast jitter (α∈[0.85, 1.15], β∈[-0.03, 0.03])

### Inference Strategy

**Test Time Augmentation (TTA)**: 8 deterministic views per input
- 4 rotations (0°, 90°, 180°, 270°)
- 2 flips per rotation (original + horizontal flip)
- Ensemble soft-vote: average P(CDW) across all 8 views and all 4 models

---

## Training Progress

### Phase 1: Data Preparation ✓

```
[2026-04-23 00:19:30] Loading labels from data/chm_variants/labels_canonical_with_splits.csv
[2026-04-23 00:19:31] Loaded 580,136 total labels

Filtering to spatial splits (train/val/test only):
  Train: 67,290 labels
  Val:   13,850 labels
  Test:  56,521 labels
  
Loading CHM windows (this takes ~10–20 min for 97,940 tiles)...
[in progress]
```

### Phase 2: Model Training (pending)

```
Expected timeline:
  CNN-seed42 (50 epochs):  ~2–3 hours
  CNN-seed43 (50 epochs):  ~2–3 hours
  CNN-seed44 (50 epochs):  ~2–3 hours
  EfficientNet-B2 (30 epochs): ~1–2 hours
  
Total training: ~8 GPU hours
```

### Phase 3: Test Set Evaluation (pending)

```
Evaluation on 56,521 held-out test labels:
  - AUC score (ROC)
  - F1 score (optimal threshold search)
  - Class distribution (CDW vs NO_CDW)
  - Per-model metrics
  - Ensemble metrics
```

### Phase 4: Full Probability Recalculation (pending)

```
Apply all 4 retrained models + TTA to all 580,136 labels:
  - Generate new model_prob column
  - Compare to original ensemble probabilities
  - Measure mean/median/max differences
  - Document distribution shift elimination
```

---

## Comparison: Original vs Retrained Ensemble

### Original Ensemble Metrics (Option A)

| Metric | Value |
|--------|-------|
| Training data | 19,812 tiles (19,812 from cnn_test_split) |
| Test data | 2,186 held-out tiles |
| Models | CNN-Deep-Attn ×3 (seeds 42,43,44) + EfficientNet-B2 |
| Ensemble AUC | 0.9987 |
| Ensemble F1 | 0.9701 |
| Threshold | 0.68 |
| Mean prob (on 580K) | 0.3824 |
| Applied to | 580,136 full labels (including buffer zones) |
| **Observed effect** | ~6% mean probability difference from TTA recalc |

### Retrained Ensemble Metrics (Option B)

| Metric | Value | Status |
|--------|-------|--------|
| Training data | 67,290 tiles (spatial split training set) | ✓ Prepared |
| Test data | 56,521 held-out tiles (spatial split test) | ✓ Prepared |
| Models | CNN-Deep-Attn ×3 (seeds 42,43,44) + EfficientNet-B2 | 🔄 Training |
| Ensemble AUC | TBD | ⏳ Running |
| Ensemble F1 | TBD | ⏳ Running |
| Threshold | TBD | ⏳ Running |
| Mean prob (on 580K) | TBD | ⏳ Pending |
| Applied to | 580,136 full labels | ⏳ Pending |
| **Expected improvement** | Eliminate distribution shift, better generalization | 📊 To measure |

---

## Checkpoints & Artifacts

### Output Directory

```
output/tile_labels_spatial_splits/
├── cnn_seed42_spatial.pt          ← CNN retrained on spatial splits
├── cnn_seed43_spatial.pt          ← CNN retrained on spatial splits
├── cnn_seed44_spatial.pt          ← CNN retrained on spatial splits
├── effnet_b2_spatial.pt           ← EfficientNet-B2 retrained on spatial splits
├── training_metadata.json         ← Training config, data stats, test metrics
└── [comparison report - to be generated]
```

### Metadata Structure (training_metadata.json)

```json
{
  "timestamp": "2026-04-23T00:19:30Z",
  "training_config": {
    "CNN_EPOCHS": 50,
    "EFFNET_EPOCHS": 30,
    "BATCH_SIZE": 32,
    "LR_HEAD": 0.0005,
    "LR_BACKBONE": 0.00005,
    "LABEL_SMOOTHING": 0.05,
    "MIXUP_ALPHA": 0.3
  },
  "data_stats": {
    "train_size": 67290,
    "val_size": 13850,
    "test_size": 56521,
    "train_cdw": [count],
    "train_no_cdw": [count],
    "test_cdw": [count],
    "test_no_cdw": [count]
  },
  "test_metrics": {
    "ensemble_auc": [value],
    "ensemble_f1": [value],
    "ensemble_thresh": [value],
    "n_test": 56521,
    "n_cdw": [count]
  },
  "approach": "Option B: Retrain on spatial splits (academic rigor)"
}
```

---

## Academic Justification

### Why Option B is Stronger for Publication

1. **Proper Train/Test Split**
   - Uses data-driven spatial stratification (not ad-hoc)
   - Test set is truly held-out (Chebyshev distance ≥2 from train)
   - Prevents spatial data leakage within rasters

2. **Larger Training Set**
   - 67,290 labeled tiles (vs 19,812) provides better generalization
   - Covers full geographic and temporal range
   - More representative of final inference distribution

3. **Distribution Alignment**
   - Training set derived from same 580K canonical labels
   - No distribution shift expected on full label set
   - Reduces bias vs applying pre-trained models to new data

4. **Reproducibility**
   - All hyperparameters documented
   - Spatial split algorithm is deterministic (seeded RNG)
   - Models can be retrained from scratch with same results

5. **Rigorous Evaluation**
   - Separate validation and test sets
   - Soft-vote ensemble with TTA
   - Comprehensive metrics (AUC, F1, threshold analysis)

---

## Next Steps (After Training Completes)

### Step 1: Analyze Test Set Performance
```bash
# Compare test metrics: original vs retrained
python scripts/compare_ensemble_test_metrics.py \
  --original output/tile_labels/ensemble_meta.json \
  --retrained output/tile_labels_spatial_splits/training_metadata.json
```

### Step 2: Recalculate All Probabilities
```bash
# Generate probabilities for all 580K labels using retrained ensemble
python scripts/recalculate_model_probs_tta_ensemble.py \
  --labels data/chm_variants/labels_canonical_with_splits.csv \
  --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop \
  --models output/tile_labels_spatial_splits/{cnn_seed42,cnn_seed43,cnn_seed44,effnet_b2}_spatial.pt \
  --output data/chm_variants/labels_canonical_with_splits_retrained_probs.csv
```

### Step 3: Generate Comparison Report
```bash
# Compare original vs retrained probabilities
python scripts/compare_original_vs_retrained.py \
  --original data/chm_variants/labels_canonical_with_splits.csv \
  --retrained data/chm_variants/labels_canonical_with_splits_retrained_probs.csv \
  --output OPTION_B_COMPARISON_REPORT.md
```

### Step 4: Document Findings
```bash
# Create final thesis-ready document
cp OPTION_B_SPATIAL_SPLITS_RETRAINING.md \
   LaTeX/Lamapuidu_tuvastamine/estonian/sektsioonid/03_ensemble_methodology.md
```

---

## Key References

- **Original Training**: `scripts/train_ensemble.py` (lines 1–559)
- **TTA Implementation**: `scripts/train_ensemble.py` lines 285–298
- **Model Builders**: `scripts/label_tiles.py` (lines 164–379)
- **CHM Normalization**: `scripts/train_ensemble.py` line 56
- **Spatial Split Strategy**: `scripts/spatial_split_experiments_v4/_split_v4.py` (validated methodology)

---

## Expected Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Data preparation | ~15 min | ✓ In progress |
| CNN training (3×) | ~8–10 hours | ⏳ Pending |
| EfficientNet training | ~2–3 hours | ⏳ Pending |
| Test evaluation | ~10 min | ⏳ Pending |
| Probability recalculation (580K) | ~2 hours | ⏳ Pending |
| **Total** | **~12–15 hours** | 🔄 Running |

---

**Status last updated:** 2026-04-23 00:19:30 UTC  
**Next check:** Monitor log file for training progress

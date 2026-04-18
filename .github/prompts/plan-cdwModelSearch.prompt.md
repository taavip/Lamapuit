## Plan: Comprehensive CDW Model Search for F1 > 0.99

**TL;DR** — Create a single self-contained script `scripts/model_search.py` that: (A) analyses the training data for bias, balance, and label quality; (B) trains 15+ model architectures with 5-fold CV on training data only (never touching the test set); (C) applies systematic regularization/anti-overfit techniques; (D) adds raster-level leave-one-out cross-validation for spatial generalization analysis; (E) selects the top-5 models by mean CV F1 and evaluates them (+ their ensemble) on the held-out test set with TTA; (F) logs every experiment to JSON, CSV, and saves all checkpoints. Output goes to `output/model_search/`. The existing test split ([cnn_test_split.json](output/tile_labels/cnn_test_split.json)) is used unchanged. Reuses existing data loading from [fine_tune_cnn.py](scripts/fine_tune_cnn.py) (`_load_labels`, `_build_arrays`, `_norm_tile`) and model builders.

---

**Steps**

### Phase 0 — Data Analysis (inline, at script start)

1. Load all labels via `_load_labels()` from [fine_tune_cnn.py](scripts/fine_tune_cnn.py), excluding test-split keys from [cnn_test_split.json](output/tile_labels/cnn_test_split.json). Count:
   - Overall class balance (CDW vs no_CDW in training set only)
   - Per-raster distribution (table + Gini coefficient of CDW ratio across rasters)
   - Label `source` distribution (manual / auto_reviewed / auto / auto_skip) and quality-weight histogram
   - Tiles per raster bar chart
2. Build arrays via `_build_arrays()`. Compute pixel-level statistics:
   - Mean, std, min, max of normalized tile values per class
   - Fraction of near-zero (nodata-like) pixels per tile — flag tiles with >50% zeros as potentially noisy
3. Cross-check: use existing ensemble predictions (`model_prob` column in CSVs) vs actual labels to estimate label noise rate (label ≠ prediction at high confidence). Save noisy-tile candidates to `output/model_search/data_analysis/suspect_labels.csv`.
4. Save all analysis outputs:
   - `output/model_search/data_analysis/class_balance.json`
   - `output/model_search/data_analysis/per_raster_stats.csv`
   - `output/model_search/data_analysis/source_distribution.json`
   - `output/model_search/data_analysis/pixel_stats.json`
   - Matplotlib plots: `class_balance.png`, `per_raster_cdw.png`, `source_weights.png`, `pixel_histograms.png`

### Phase 1 — Training Data Selection Strategies

5. Define three data variants (each will be tried with each model architecture):
   - **full**: All training tiles with source-quality weights (current approach)
   - **manual_only**: Only tiles with `source in ("manual", "auto_reviewed")` — highest label quality
   - **balanced**: Oversample CDW tiles to match no_CDW count via weighted random sampling in the DataLoader (no synthetic data, just repeated sampling with augmentation diversity)

### Phase 2 — Model Zoo (15 architectures)

6. Implement model builder functions in the script, each returning `(model, param_groups_fn)`. All adapted for 1-channel 128×128 input (average RGB→1ch for pretrained first conv, same pattern as existing [EfficientNet-B2 builder](scripts/train_ensemble.py#L78-L108)):

   | # | Architecture | Source | Notes |
   |---|---|---|---|
   | 1 | CNN-Deep-Attn | existing `_build_deep_cnn_attn_net()` | Baseline, from scratch |
   | 2 | CNN-Deep-Attn-Wide | variant: channels 1→64→128→256→512 | Wider baseline |
   | 3 | EfficientNet-B0 | `torchvision.models.efficientnet_b0` | Lighter pretrained |
   | 4 | EfficientNet-B2 | existing `_build_effnet_b2()` | Current best pretrained |
   | 5 | EfficientNet-B4 | `torchvision.models.efficientnet_b4` | Heavier pretrained |
   | 6 | ResNet-18 | `torchvision.models.resnet18` | Light pretrained CNN |
   | 7 | ResNet-34 | `torchvision.models.resnet34` | Medium pretrained CNN |
   | 8 | ResNet-50 | `torchvision.models.resnet50` | Heavy pretrained CNN |
   | 9 | ConvNeXt-Tiny | `timm.create_model("convnext_tiny")` | Modern CNN |
   | 10 | ConvNeXt-Small | `timm.create_model("convnext_small")` | Larger modern CNN |
   | 11 | DenseNet-121 | `torchvision.models.densenet121` | Dense connections |
   | 12 | MobileNetV3-Large | `torchvision.models.mobilenet_v3_large` | Efficient |
   | 13 | RegNetY-400MF | `timm.create_model("regnety_004")` | AutoML-designed |
   | 14 | Swin-Tiny | `timm.create_model("swin_tiny_patch4_window7_224", img_size=128)` | Vision Transformer |
   | 15 | MaxViT-Tiny | `timm.create_model("maxvit_tiny_tf_128")` | Multi-axis ViT |

### Phase 3 — Training Configuration Matrix

7. Define a config matrix. Each experiment is `(model, data_strategy, loss, regularization)`:
   - **Loss functions**: CrossEntropy (class-weighted + label-smoothing 0.05), Focal Loss (gamma=2, alpha=0.25)
   - **Regularization combos**: (a) Mixup α=0.3 only, (b) CutMix α=1.0 only, (c) Mixup + SWA (last 10 epochs), (d) None (for comparison)
   - **Optimizers**: AdamW (wd=1e-4) with CosineAnnealingLR — keep constant to isolate architecture effects
   - **Epochs**: 60 for from-scratch models, 40 for pretrained models (with early stopping patience=10 on val F1)
   - **Batch size**: 32 (consistent with existing)
   - **LR**: Head 5e-4, backbone 5e-5 for pretrained; flat 5e-4 for from-scratch

   Total initial experiments: 15 models × `full` data × CE loss × Mixup = **15 baseline runs**. Then the top-8 models are expanded to the full matrix (3 data × 2 loss × 3 reg = 18 combos each → **144 extended runs**).

### Phase 4 — 5-Fold Stratified CV (No Test Leakage)

8. Create 5 stratified folds from the training set only (excluding test-split keys). Stratification by class label, with fold assignment additionally considering raster origin to avoid having all tiles from one raster in the same fold.
9. For each experiment config: train on 4 folds, validate on the held-out fold. Report per-fold metrics: AUC, F1, precision, recall, threshold. Calculate mean ± std across 5 folds.
10. **Early stopping**: Monitor val F1; if no improvement for `patience` epochs, stop training and record best epoch.
11. Save per-fold checkpoints to `output/model_search/checkpoints/{experiment_id}/fold{k}.pt`.

### Phase 5 — Raster-Level Leave-One-Out CV

12. For the top-5 models (by 5-fold CV mean F1), run raster-level leave-one-out CV:
    - 11 rasters → 11 folds, each time training on 10 rasters and validating on the held-out raster.
    - Report per-raster F1, AUC, and identify which rasters are hardest to generalize to.
    - Save results to `output/model_search/raster_cv_results.csv`.

### Phase 6 — Iterative Improvement of Top Models

13. After the first round of results, identify failure patterns:
    - Examine false positives and false negatives from the best models — are they concentrated in specific rasters, low-confidence tiles, or label-noise suspects from Phase 0?
    - Try targeted improvements on the top-5:
      - **Hard-example mining**: Increase weight on misclassified tiles, retrain.
      - **Progressive resizing**: Train at 96×96 for 20 epochs, then fine-tune at 128×128 for 40 epochs.
      - **Test-time augmentation (TTA)**: 8 views (4 rot × 2 hflip) — measure F1 lift.
      - **Stochastic Weight Averaging (SWA)**: Average weights from last 15 epochs.
      - **Threshold calibration**: Platt scaling on validation set.

### Phase 7 — Final Top-5 Test Evaluation

14. Select the 5 best experiments by mean 5-fold CV F1 (with std as tiebreaker — prefer lower variance).
15. Retrain each of the 5 on the **full training set** (all non-test tiles) using the winning hyperparameters.
16. Evaluate each on the held-out test set ([cnn_test_split.json](output/tile_labels/cnn_test_split.json)) with TTA (8 views). Report: AUC, F1, precision, recall, confusion matrix, per-class accuracy.
17. Build a **top-5 soft-vote ensemble** and evaluate it on test with TTA.
18. Build a **stacking ensemble**: train a simple logistic regression meta-learner on the 5 models' out-of-fold predictions from Phase 4. Evaluate on test.
19. Save final models to `output/model_search/final_models/{model_name}.pt` with full metadata.

### Phase 8 — Logging & Documentation

20. **Per-experiment JSON log**: `output/model_search/experiments/{experiment_id}.json` containing: model arch, hyperparams, data strategy, per-fold metrics, training time, best epoch, early-stop info.
21. **Summary CSV**: `output/model_search/experiment_summary.csv` — one row per experiment, columns: `experiment_id, model, data_strategy, loss, regularization, mean_cv_auc, std_cv_auc, mean_cv_f1, std_cv_f1, mean_cv_precision, mean_cv_recall, best_fold_f1, worst_fold_f1, epochs_used, training_time_s`.
22. **Final results table**: `output/model_search/final_test_results.csv` — the 5 models + 2 ensembles evaluated on test.
23. **Research paper report**: `output/model_search/RESEARCH_REPORT.md` — auto-generated markdown with:
    - Data analysis summary (class balance, label quality, spatial distribution)
    - Methodology (CV strategy, model zoo, training details)
    - Results tables (top-10 by CV F1, top-5 test results, ensemble vs best single model)
    - Key findings (which architecture wins, does focal loss help, does balance matter, spatial generalization gaps)
    - Comparison vs existing ensemble baseline (F1=0.9701)
24. **Console logging**: Use Python `logging` module to write timestamped progress to both stdout and `output/model_search/model_search.log`.

---

**Verification**

- **No test leakage**: Assert at startup that no training tile key exists in `-test_split.json` keys. Log the assertion.
- **Smoke test**: Run with `--smoke-test` flag → train 1 model (CNN-Deep-Attn), 2 folds, 3 epochs, verify pipeline end-to-end.
- **Reproducibility**: All random seeds logged. Each fold assignment deterministic (seed=2026). Model init seeds logged per experiment.
- **Post-run validation**: After final test evaluation, compare test F1 to best CV F1 — flag if gap > 5% (potential overfitting to validation).
- **Run command**:
  ```bash
  docker exec -w /workspace lamapuit-dev-x11 conda run -n cwd-detect --no-capture-output \
    python scripts/model_search.py \
      --labels output/tile_labels \
      --chm-dir chm_max_hag \
      --test-split output/tile_labels/cnn_test_split.json \
      --output output/model_search 2>&1 | tee output/model_search/model_search.log
  ```
  Smoke test:
  ```bash
  docker exec -w /workspace lamapuit-dev-x11 conda run -n cwd-detect --no-capture-output \
    python scripts/model_search.py --smoke-test \
      --labels output/tile_labels --chm-dir chm_max_hag \
      --test-split output/tile_labels/cnn_test_split.json \
      --output output/model_search
  ```

**Decisions**
- **5-fold CV** for model selection, raster-level LOOCV for spatial generalization (per user choice)
- **No time limit** — full matrix runs overnight
- **Output** → `output/model_search/` to keep production models in `output/tile_labels/` untouched
- **Baseline round first** (15 runs) → expand top-8 to full matrix → avoids wasting GPU time on weak architectures
- **Reuse** `_load_labels`, `_build_arrays`, `_norm_tile`, `_compute_metrics`, `_build_deep_cnn_attn_net` from [fine_tune_cnn.py](scripts/fine_tune_cnn.py) to maintain consistency with existing pipeline

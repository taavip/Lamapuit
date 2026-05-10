# Ablation and Validation Protocol for Thesis Reporting

## Purpose

This protocol prevents test-set leakage during the CWD segmentation ablation study. The ablation phases are used for model selection, so they must use only training/development data and cross-validation metrics. The held-out test stripe remains locked until the final configuration is chosen.

## Data Splitting

- The spatial split is based on vertical stripes.
- Stripe 0 is the held-out test stripe.
- The remaining stripes are used for training and validation folds.
- The held-out test stripe is not used to choose CHM variants, architectures, loss functions, augmentation settings, thresholds, or SWA behavior.

## Model Selection Procedure

Phases 2-6 use validation-only metrics:

- Phase 2: CHM/input representation selection.
- Phase 3: architecture selection using the top Phase 2 candidates.
- Phase 4: loss and loss-parameter selection using the top Phase 3 candidates.
- Phase 5: augmentation and regularization selection using the top Phase 4 candidates.
- Phase 6: final cross-validation of the top configurations across folds.

The winner selection metric is validation F1. Test F1, test clDice, Boundary IoU, and AP@IoU are not computed during ablation selection.

## Final Test Evaluation

After Phase 6 identifies the final configuration, run the held-out test evaluation once with:

```bash
python3 seg_pipeline/scripts/phase3_ablation_v10.py \
  --phase 6 \
  --evaluate-test \
  --output-dir seg_pipeline/output/ablation_v10_top2_cv
```

This final result can be reported as the locked held-out test performance. It should not be used to revise the selected model.

## Code Safeguards Added

- `phase3_ablation_v10.py` now defaults to validation-only model selection.
- Held-out test evaluation requires the explicit `--evaluate-test` flag.
- Validation metrics are written separately from test metrics:
  - `phaseN_results_val.csv`
  - `phaseN_results_test.csv`
- Cached test-metric files are not reused during validation-only ablations.
- Carried winner combinations receive stable IDs such as `2A__3C__4E`, avoiding overwritten or skipped runs when multiple top candidates are tested.
- AP@IoU on the held-out test stripe is bounded by `--max-ap-component-pairs` to avoid CPU stalls from dense connected-component matching.

## Thesis Wording

Use language similar to:

> Hyperparameter and design choices were selected using spatial cross-validation on the development area only. The westernmost spatial stripe was reserved as an independent held-out test region and was not used during ablation, threshold selection, or model selection. After the final configuration was fixed, the model was evaluated once on the held-out test stripe to estimate generalization performance.


# Ablation Refactor Protocol (2026-05-10)

This note documents the refactor applied to the Top-2 ablation pipeline to improve methodological correctness and thesis-grade traceability.

## 1. What Was Fixed

1. Carry-forward chain reconstruction:
- Fixed chain parsing so labels like `2E__3E__4F__5E` now correctly re-apply phases 2, 3, 4, and 5 settings.
- This prevents mislabeled runs where effective architecture/loss/augmentation differed from the run label.

2. Fold-safe CSV persistence:
- Phase result rows now include fold identity and use fold-aware merge keys.
- This prevents fold overwrites in `phase*_results.csv`.

3. Selection metric logging:
- `selection_metric` now logs the real metric used (e.g., `val_cldice`).
- Added `selection_metric_value` and effective config provenance fields for auditability.

4. Phase 5 augmentation validity:
- `aug_none`, `aug_geometric`, and `aug_full` are now real augmentation modes.
- Batch augment controls (`mixup/cutmix/gridmask`) now follow `batch_aug`.

5. Cross-validation in Phases 2-5:
- Orchestrator now runs all configured CV folds for each phase and selects top-2 by aggregated cross-fold mean of the chosen metric.

6. Fold-balance preflight:
- Added explicit preflight fold statistics logging (`--validate`) using selected CV version.
- Test stripe remains untouched.

7. Thesis final protocol mode:
- Added final mode to train on all non-test stripes and evaluate on locked test stripe (`--final-train-all --evaluate-test`).
- Orchestrator phase 7 compares: top-2 selected configs + legacy V10 comparator chain.

## 2. Files Changed

- `seg_pipeline/scripts/phase3_ablation_v10.py`
- `seg_pipeline/scripts/phase3_train_v10.py`
- `seg_pipeline/scripts/phase2_dataset_v3.py`
- `run_full_ablation_automated_top2.sh`

## 3. Smoke Validation Performed

1. Phase 2 cross-fold smoke (CV v4, 1 epoch, condition `2A`) completed.
2. Aggregated winner summary by mean `val_cldice` verified.
3. Carry-forward chain reconstruction verified in-container for `5:2E__3E__4F__5E`.
4. Phase 5 one-condition smoke run completed (`5A`) with effective config logged.
5. Final protocol path (`--final-train-all --evaluate-test`) executed once as functional smoke.

## 4. Recommended Full Rerun Command

```bash
CV_VERSION=4 \
SELECTION_METRIC=val_cldice \
OUTPUT_BASE=/home/tpipar/project/Lamapuit/seg_pipeline/output/ablation_v10_top2_cv_refactor \
DATASET_DIR=/home/tpipar/project/Lamapuit/seg_pipeline/output/phase2_dataset_v10_reconstructed \
REBUILD_DATASET=false \
EPOCHS=100 \
SWA_START=35 \
bash /home/tpipar/project/Lamapuit/run_full_ablation_automated_top2.sh
```

## 5. Final Thesis Protocol Only (if Phases 2-6 already exist)

```bash
CV_VERSION=4 \
SELECTION_METRIC=val_cldice \
OUTPUT_BASE=/home/tpipar/project/Lamapuit/seg_pipeline/output/ablation_v10_top2_cv_refactor \
LEGACY_V10_CHAIN=2E__3C__4H__5D \
REBUILD_DATASET=false \
EPOCHS=100 \
SWA_START=35 \
bash /home/tpipar/project/Lamapuit/run_full_ablation_automated_top2.sh 7
```

## 6. Notes for Reporting

- Phases 2-5 now select by cross-fold mean metric rather than single-fold winners.
- Keep stripe 0 as locked test until phase 7 only.
- Report both cross-fold validation results (selection stage) and locked-test final comparison (phase 7).

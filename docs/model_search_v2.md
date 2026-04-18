# Model Search V2 Protocol

This document describes the v2 experimental protocol implemented in `scripts/model_search_v2/model_search_v2.py`.

## Objective

Create a stronger and more trustworthy comparison pipeline by:

- keeping the original baseline training pool,
- adding only reliable labels from drop13,
- restricting model families to top performers from previous experiments,
- improving test robustness with extra external test samples,
- and building a diverse ensemble from non-similar model families.

## Data policy

Training labels are merged from:

- `output/tile_labels` (baseline)
- `output/onboarding_labels_v2_drop13` (additional)

Drop13 rows are included if they satisfy at least one:

1. Manual/reviewed source (`manual` or `auto_reviewed`), or
2. High-confidence threshold gate:
   - `cdw` with `model_prob >= 0.9995`
   - `no_cdw` with `model_prob <= 0.0698`

This keeps pseudo-labels very conservative and consistent with the stated precision constraints.

## Validation design

- Base split starts from `output/tile_labels/cnn_test_split.json`.
- Additional stratified test keys are sampled from drop13 accepted rows.
- Final split is written to `prepared/cnn_test_split_v2.json`.
- Base model search uses grouped spatial CV (inherited from `model_search.py`):
  raster + coarse spatial block grouping with `StratifiedGroupKFold` fallback.

## Model selection policy

- Read `output/model_search/experiment_summary.csv`.
- Sort by `mean_cv_f1` desc, `std_cv_f1` asc.
- Keep top unique `model_name` values, default `n=12`.
- If prior results are missing, fallback list is used.

## Ensemble policy

After final test evaluation:

- choose the best single models from different architecture families,
- keep up to 3 diverse members,
- evaluate soft-vote top3 ensemble and save:
  - `diverse_ensemble_top3.csv`
  - `diverse_ensemble_top3.json`

## Reproducibility notes

- Script writes curation/test metadata to:
  - `prepared/prepared_dataset_summary.json`
  - `prepared/prepared_dataset_counts.csv`
- Full run analysis is written to:
  - `RESULTS_ANALYSIS_V2.md`

## Run command (Docker + conda)

```bash
cd /home/tpipar/project/Lamapuit

docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc "\
source /opt/conda/etc/profile.d/conda.sh && \
conda activate cwd-detect && \
python scripts/model_search_v2/model_search_v2.py \
  --output output/model_search_v2 \
  --n-models 12 \
  --t-high 0.9995 \
  --t-low 0.0698\
"
```

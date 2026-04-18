# Thesis Note: Drop-Above-1.3m CHM Pipeline (v2)

## Objective

- Enforce strict point filtering: discard points with HAG > 1.3 m before CHM rasterization.
- Re-run onboarding labeling on filtered CHM outputs.
- Compute prediction histogram and select a manual-review policy:
  - low-confidence band for manual review
  - plus 5% random spot-check from auto-labeled remainder

## Refactor Summary

The CHM pipeline was refactored to support strict filtering semantics (drop-above threshold) rather than value saturation semantics.

- Previous behavior: all points contributed after clipping `HAG := min(HAG, 1.3)`.
- New behavior: points with `HAG > 1.3` are removed before raster max aggregation.

## Executed Pipeline (Docker + conda)

```bash
docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python scripts/run_phase_a_laz_to_chm.py --input-dir data/lamapuit/laz --pattern '*.laz' --out data/lamapuit/chm_max_hag_13_drop --resolution 0.2 --hag-max 1.3 --drop-above-hag-max --workers 4 --report-json analysis/onboarding_new_laz/phase_a_laz_to_chm_drop13_report.json --report-csv analysis/onboarding_new_laz/phase_a_laz_to_chm_drop13_report.csv"

docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python scripts/label_all_rasters.py --chm-dir data/lamapuit/chm_max_hag_13_drop --output output/onboarding_labels_v2_drop13 --pattern '*_chm_max_hag_20cm.tif' --resume --max-nodata-pct 75 --model-path output/tile_labels/ensemble_model.pt --auto-advance 0.5 --review-pct 0.0 --no-finetune"

docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python scripts/tmp_diag/prediction_histogram_recommendation.py --labels-dir output/onboarding_labels_v2_drop13 --out-hist-csv analysis/onboarding_new_laz/drop13_prediction_histogram.csv --out-json analysis/onboarding_new_laz/drop13_prediction_confidence_recommendation.json --out-md analysis/onboarding_new_laz/drop13_prediction_confidence_recommendation.md"

docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python scripts/recalculate_manual_review_queue.py --labels-dir output/onboarding_labels_v2_drop13 --out output/onboarding_labels_v2_drop13/manual_review_queue_pre_split.csv --low-min 0.39 --low-max 0.61 --spotcheck-frac 0.05 --seed 2026"
```

## Results

- CHM rasters generated (drop13): 119
- Rasters labeled: 100
- Rasters skipped by nodata rule (>75%): 19
- Total labeled tiles: 586124
- CDW tiles: 166539
- No-CDW tiles: 419585
- Unknown: 0
- Auto predictions used for histogram: 553937

## Confidence Policy from Histogram

- Recommended low-confidence band: [0.39, 0.61]
- Low-confidence manual bucket: 29628
- 5% random spot-check from remainder: 26215
- Total manual-review queue: 55843 (10.08% of auto-labeled tiles)

## Produced Artifacts

- `analysis/onboarding_new_laz/phase_a_laz_to_chm_drop13_report.json`
- `analysis/onboarding_new_laz/phase_a_laz_to_chm_drop13_report.csv`
- `analysis/onboarding_new_laz/drop13_prediction_histogram.csv`
- `analysis/onboarding_new_laz/drop13_prediction_confidence_recommendation.json`
- `analysis/onboarding_new_laz/drop13_prediction_confidence_recommendation.md`
- `output/onboarding_labels_v2_drop13/manual_review_queue_pre_split.csv`

## Next Required Step

Manual review must be completed from `output/onboarding_labels_v2_drop13/manual_review_queue_pre_split.csv` before any split/training stage.

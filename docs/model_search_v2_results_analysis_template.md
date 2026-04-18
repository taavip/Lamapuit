# Model Search V2 - Results Analysis Template

Use this template after a full run of `scripts/model_search_v2/model_search_v2.py`.

## 1. Run Metadata

- Date:
- Commit / branch:
- Output folder:
- GPU / runtime:
- Key arguments:

## 2. Data Curation Summary

Source files:

- `prepared/prepared_dataset_summary.json`
- `prepared/prepared_dataset_counts.csv`

Report:

- Base labeled rows:
- Drop13 labeled rows:
- Drop13 manual/reviewed kept:
- Drop13 threshold-gated kept:
- Curated rows after dedup:
- Curated raster count:

## 3. Test Split Summary

Source file:

- `prepared/cnn_test_split_v2.json`

Report:

- Base test keys:
- Extra test keys:
- Total test keys:
- Extra fraction and cap:

## 4. Model Ranking (CV)

Source file:

- `experiment_summary.csv`

Report:

- Top 12 unique model names by CV F1:
- Best CV F1 +- std:
- Stability comments (variance across folds):

## 5. Final Test Results

Source file:

- `final_test_results.csv`

Report:

- Best single model:
- Best single-model F1 / AUC:
- Soft-vote / stacking rows (if present):
- Error profile from confusion matrices:

## 6. Diverse Top-3 Ensemble

Source files:

- `diverse_ensemble_top3.csv`
- `diverse_ensemble_top3.json`

Report:

- Member models and architecture families:
- Ensemble threshold:
- Ensemble F1 / AUC / precision / recall:
- Compare against best single model:

## 7. Statistical Credibility Checks

- Class imbalance handling used:
- Leakage controls used:
- External test enrichment impact:
- Any bootstrap/CI analysis added:

## 8. Conclusions

- Did v2 improve over previous model_search baseline?
- Which model families are most robust?
- Recommended production model(s):

## 9. Next Experiments

- Calibration improvement experiment:
- Per-raster threshold experiment:
- Domain-shift stress test:

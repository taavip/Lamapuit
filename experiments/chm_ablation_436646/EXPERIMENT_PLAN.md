# CHM Ablation Plan (Tile 436646)

## Aim
Identify which CHM representation gives the best CDW tile-classification performance when evaluated with top models from model_search_v2 and model_search_v3.

## Research Question
For tile 436646 (years 2018, 2020, 2022, 2024), which CHM variant maximizes mean F1 across strong classification models?

## Experimental Design
1. Data scope:
     - Tile: 436646.
     - Years: 2018, 2020, 2022, 2024.
     - CHMs: all discovered rasters containing tile ID and year under configured data roots.
     - Labels: year-matched tile labels from curated v2/v3 label directories.
2. Model scope:
     - Select top-ranked checkpoints from model_search_v2 and model_search_v3 ranking CSVs.
     - Use fold1 checkpoint per experiment (or first available fold).
     - Keep 3-5 best models total, with representation from both v2 and v3.
3. Evaluation:
     - For each CHM, extract exactly the labeled tile windows.
     - Run model inference and compute Accuracy, Precision, Recall, F1, AUC.
     - Aggregate by CHM and by variant across years.

## Validity Controls
- Same tile windows across CHMs of the same year.
- Year-based label matching to avoid spatial-temporal mismatch.
- Model-specific decision threshold loaded from checkpoint metadata.
- Reproducible run artifacts written into timestamped result folders.

## Outputs
- selected_models.csv
- discovered_chms.csv
- labels_by_year.csv
- metrics_detailed.csv
- summary_chm.csv
- summary_variant.csv
- summary_model.csv
- experiment_report.md
- run_metadata.json

## Execution
- Dry run (planning/verification):
    - ./run.sh --dry-run --top-k 5
- Full run (docker + conda):
    - ./run.sh --top-k 5

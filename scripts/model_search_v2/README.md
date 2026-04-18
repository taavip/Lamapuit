# Model Search V2

This folder contains a standalone v2 workflow that extends the original model search without changing existing code.

## What it does

- Uses prior model-search results to pick the top 12 model families/architectures.
- Builds a curated label set by combining:
  - baseline labels from `output/tile_labels`
  - additional drop13 labels from `output/onboarding_labels_v2_drop13` with strict inclusion:
    - manual / auto_reviewed labels, and
    - high-confidence pseudo-labels using:
      - `t_high = 0.9995` for `cdw`
      - `t_low = 0.0698` for `no_cdw`
- Adds extra stratified drop13 samples to the test split.
- Merges CHM sources into one searchable directory via symlinks:
  - `chm_max_hag`
  - `data/lamapuit/chm_max_hag_13_drop`
- Runs the existing `scripts/model_search.py` pipeline on this prepared dataset.
- Builds an extra diverse top-3 ensemble (different architecture families).
- Writes CSV outputs and markdown analysis docs.

## Main script

- `model_search_v2.py`

## Recommended run (Docker + conda)

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

## Quick preparation-only check

```bash
cd /home/tpipar/project/Lamapuit

docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc "\
source /opt/conda/etc/profile.d/conda.sh && \
conda activate cwd-detect && \
python scripts/model_search_v2/model_search_v2.py \
  --output output/model_search_v2_smoke \
  --prepare-only\
"
```

## Useful options

- `--smoke-test`: forward smoke mode to base model search.
- `--prepare-only`: only build curated data + test split + metadata (no training).
- `--stage2-pilot --stage2-pilot-top-models 6`: reduce stage2 compute.
- `--top-k-final 12`: retrain/evaluate up to 12 finalists.

## Outputs

In `output/model_search_v2`:

- `prepared/prepared_dataset_summary.json`
- `prepared/prepared_dataset_counts.csv`
- `prepared/labels_curated_v2/*_labels.csv`
- `prepared/cnn_test_split_v2.json`
- `prepared/chm_merged/*.tif` (symlinks)
- `experiment_summary.csv`
- `final_test_results.csv`
- `diverse_ensemble_top3.csv`
- `diverse_ensemble_top3.json`
- `RESULTS_ANALYSIS_V2.md`

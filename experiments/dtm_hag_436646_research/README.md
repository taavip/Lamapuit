# 436646 DTM/HAG Research Test

This directory contains a dedicated experiment for improving DTM and HAG generation for tile 436646 across years 2018, 2020, 2022, and 2024.

## Hypothesis

Compared with the current baseline (class-2 + IDW k=3), the following workflow should improve DTM stability and HAG usefulness for CDW detection:

1. CSF ground reclassification.
2. Multi-year harmonized ground surface (lowest valid ground per cell).
3. Interpolation with stronger terrain models (IDW k=12, TIN, TPS).
4. Gaussian smoothing on final CHM.

## Implemented Script

- `run_experiment.py`

## Methods Tested

- `baseline_idw3_drop13` (existing CHM in `data/lamapuit/chm_max_hag_13_drop`)
- `idw_k12_raw`
- `idw_k12_gauss`
- `tin_linear_raw`
- `tin_linear_gauss`
- `tps_raw`
- `tps_gauss`

All new methods use:
- CSF-based ground labels
- harmonized multi-year DEM
- HAG drop mode: keep only `0 <= HAG <= 1.3`

## Run

```bash
cd /home/tpipar/project/Lamapuit

docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python experiments/dtm_hag_436646_research/run_experiment.py --tile-id 436646 --years 2018,2020,2022,2024 --out-dir experiments/dtm_hag_436646_research/results"

# Faster pilot (recommended first): reuse CSF and process 25% of points for CHM stage
docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python experiments/dtm_hag_436646_research/run_experiment.py --tile-id 436646 --years 2018,2020,2022,2024 --out-dir experiments/dtm_hag_436646_research/results --reuse-csf --tps-max-samples 8000 --tps-neighbors 32 --tps-smoothing 0.1 --point-sample-rate 0.25"
```

## Outputs

Inside `experiments/dtm_hag_436646_research/results`:

- `scratch/`: CSF-reclassified LAZ files and PDAL pipelines
- `dtm/`: yearly CSF DEM, harmonized DEM, interpolated DEMs
- `chm/`: per-year CHM rasters for each tested method
- `eval/method_summary.csv`: compact method comparison table
- `experiment_report.json`: full machine-readable report
- `experiment_report.md`: analysis and recommendation

## Fast evaluation of existing outputs

If CHM files are already generated and you want a quick comparative score pass:

```bash
cd /home/tpipar/project/Lamapuit

docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python experiments/dtm_hag_436646_research/quick_evaluate_existing_outputs.py --tile-id 436646 --years 2018,2020,2022,2024 --results-dir experiments/dtm_hag_436646_research/results --max-tiles-per-year 10000"
```

This writes:

- `results/eval/method_summary.csv`
- `results/eval/quick_eval_report.json`
- `results/eval/quick_eval_report.md`

# 436646 SOTA DTM/HAG Experiment (2026-04-14)

This experiment implements a SOTA-oriented DTM and CHM workflow for tile 436646
across years 2018, 2020, 2022, and 2024.

## Pipeline

1. Ground classification with PDAL SMRF (class 2).
2. Multi-temporal vertical alignment to a reference year using stable low-slope cells.
3. Ground super-cloud stacking across years.
4. Statistical Outlier Removal (SOR) on the stacked super-cloud.
5. Harmonized DTM anchors via per-cell 10th percentile aggregation.
6. DTM interpolation methods:
   - idw_k6
   - tin_linear
   - natural_neighbor_linear
   - tin_linear_bilateral
   - natural_neighbor_bilateral
7. CHM generation from last returns only, with class exclusions and strict HAG range [0, 1.3] m.
8. Evaluation against label CSVs and aggregate comparison.

## Required constraints implemented

- Last returns only (`--return-mode last` default).
- Building/water exclusion (`--exclude-classes 6,9` default).
- HAG range constrained to 0-1.3 m (`--hag-max 1.3` default).
- CHM output on 0.2 m grid (derived from baseline 20 cm CHM grid).

## Run (Docker + conda)

```bash
cd /home/tpipar/project/Lamapuit

docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc \
"source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && \
python experiments/dtm_hag_436646_sota_2026-04-14/run_experiment.py \
  --tile-id 436646 \
  --years 2018,2020,2022,2024 \
  --out-dir experiments/dtm_hag_436646_sota_2026-04-14/results \
  --reuse-smrf"
```

## Outputs

- `results/scratch/`: SMRF-classified LAZ and PDAL pipelines
- `results/dtm/`: aligned/harmonized and interpolated DTM rasters
- `results/chm/{year}/`: CHM GeoTIFF outputs per method
- `results/eval/method_summary.csv`: compact comparison table
- `results/experiment_report.json`: machine-readable report
- `results/experiment_report.md`: academic narrative and improvement ideas

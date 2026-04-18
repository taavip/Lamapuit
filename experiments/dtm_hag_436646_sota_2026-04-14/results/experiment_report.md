# SOTA DTM/HAG Experiment for Tile 436646

## Abstract
This experiment benchmarks a practical SOTA-inspired terrain workflow for CHM generation: SMRF ground classification, temporal super-cloud harmonization with SOR and 10th-percentile aggregation, TIN/Natural-Neighbor interpolation, and bilateral edge-preserving DTM filtering. CHM is generated at 0.2 m using last returns only, with class exclusions and strict HAG range 0-1.3 m.

## Methods
1. Ground reclassification with PDAL SMRF for each year.
2. Vertical alignment to a reference year on stable low-slope surfaces.
3. Stacked ground super-cloud, then Statistical Outlier Removal (SOR).
4. Harmonized DTM anchors via per-cell 10th-percentile elevation.
5. Interpolation methods: IDW-k6, TIN linear, Natural Neighbor linear.
6. Post-processing variants: bilateral filter on interpolated DTMs.
7. CHM from point-wise HAG with last-return filtering and class exclusion.

## Core Equations
- HAG at point i: `HAG_i = z_i - z_ground(x_i, y_i)`
- CHM pixel p: `CHM[p] = max(HAG_i)` for points in pixel p
- HAG constraint: keep only `0 <= HAG_i <= 1.3` m
- Temporal harmonization: `z_cell = percentile_10({z_t})` over stacked years
- TIN/Natural-Neighbor local surface: `z = Ax + By + C` inside each triangle

## Input Data
- Tile ID: 436646
- Years: [2018, 2020, 2022, 2024]
- LAZ directory: data/lamapuit/laz
- Labels directory: output/onboarding_labels_v2_drop13

## SMRF Diagnostics
- 2018: class2 ratio=53.22% | DEM-valid=96.64% | class2 points=17266448
- 2020: class2 ratio=45.56% | DEM-valid=97.19% | class2 points=19442906
- 2022: class2 ratio=43.07% | DEM-valid=97.28% | class2 points=28966049
- 2024: class2 ratio=38.19% | DEM-valid=95.95% | class2 points=15410111

## Vertical Alignment and SOR
- Reference year: 2018
- 2020: overlap_cells=84344 | shift_to_ref=0.0300 m
- 2022: overlap_cells=83512 | shift_to_ref=0.0300 m
- 2024: overlap_cells=83422 | shift_to_ref=-0.0300 m
- SOR: input=3600000 working=1500000 output=1472986 kept=98.20%

## Aggregate Evaluation
- idw_k6: AUC(tile_max)=0.5921113244004566, J(tile_max)=0.17227690567086595 @thr=1.2795852661132812, CDW>=15cm=1.0, NoCDW>=15cm=0.981056358807908
- tin_linear: AUC(tile_max)=0.591972316632137, J(tile_max)=0.17884093285028801 @thr=1.2783203125, CDW>=15cm=1.0, NoCDW>=15cm=0.9809383298908233
- natural_neighbor_linear: AUC(tile_max)=0.591972316632137, J(tile_max)=0.17884093285028801 @thr=1.2783203125, CDW>=15cm=1.0, NoCDW>=15cm=0.9809383298908233
- tin_linear_bilateral: AUC(tile_max)=0.5840248618997609, J(tile_max)=0.1636629300866056 @thr=1.278253936767578, CDW>=15cm=1.0, NoCDW>=15cm=0.9788728238418413
- natural_neighbor_bilateral: AUC(tile_max)=0.5840248618997609, J(tile_max)=0.1636629300866056 @thr=1.278253936767578, CDW>=15cm=1.0, NoCDW>=15cm=0.9788728238418413
- baseline_idw3_drop13: AUC(tile_max)=0.6149085635417983, J(tile_max)=0.1855900742056047 @thr=1.295918345451355, CDW>=15cm=1.0, NoCDW>=15cm=0.9603422838595456

## Best Method
**baseline_idw3_drop13**

## Interpretation
- Bilateral variants should reduce micro-jitter while preserving terrain edges in DTM.
- Natural-Neighbor/TIN methods are expected to avoid IDW pockmark artifacts in sparse-ground zones.
- Last-return and class exclusion reduce canopy/building/water contamination in CHM.

## Improvement Ideas
1. Add strict PTD implementation (e.g., lidR ptd) and compare against SMRF.
2. Add explicit stable-surface masks (roads/rock polygons) for stronger vertical datum alignment.
3. Evaluate quantiles 5th/10th/15th for harmonization sensitivity.
4. Introduce Kriging interpolation as an additional comparator.
5. Run uncertainty maps (ensemble spread among methods) to guide manual QA.

## Output Artifacts
- JSON report: experiments/dtm_hag_436646_sota_2026-04-14/results/experiment_report.json
- CSV summary: experiments/dtm_hag_436646_sota_2026-04-14/results/eval/method_summary.csv
- This Markdown report: experiments/dtm_hag_436646_sota_2026-04-14/results/experiment_report.md

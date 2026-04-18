# 436646 DTM/HAG Research Experiment

## Goal
Evaluate a research-oriented workflow: CSF classification + multi-year harmonized ground + TIN/TPS interpolation + Gaussian-refined CHM/HAG.

## Tested Workflow
1. CSF reclassification for each year (ground class=2).
2. Build yearly ground DEM from per-cell minimum class-2 z.
3. Harmonize years by lowest valid cell value (MAD-guarded).
4. Interpolate harmonized DTM with IDW-k12, TIN-linear, TPS.
5. Compute HAG and CHM (drop mode: keep only 0 <= HAG <= 1.3 m).
6. Apply Gaussian smoothing to final CHM and compare raw vs smoothed.

## Core Formulas
- HAG per point: `HAG = z - z_ground(x, y)`
- Drop filter: keep points where `0 <= HAG <= HAG_max`.
- CHM per pixel p: `CHM[p] = max(HAG_i)` over points in p.
- IDW-k12: `z_ground = sum(w_i z_i) / sum(w_i)`, `w_i = 1/(d_i+eps)^p`, with `k=12`, `p=2`.
- Harmonized ground validity: for yearly values z_t in one cell, valid if `z_t >= median(z_t) - max(floor, mad_factor*MAD)`.
- Harmonized value: minimum of valid yearly values, fallback minimum of all yearly values.

## Input
- Tile: 436646
- Years: [2018]
- LAZ directory: data/lamapuit/laz
- Labels directory: output/onboarding_labels_v2_drop13

## CSF and Harmonization Diagnostics
- 2018: class2 ratio=53.47% | DEM valid=52.44% | class2 points=17347992
- Harmonized DEM valid=52.44% | median MAD=0.0 | p95 MAD=0.0

## DTM Ground-Fit (CSF points)

## Aggregate CHM/Label Evaluation
- harmonized_dem_raw: AUC(tile_max)=0.639938687843486, J(tile_max)=0.20739098474444528 @thr=1.2899932861328125, J(frac>=15cm)=0.3373463082948199 @thr=0.008110894142031502, CDW>=15cm=1.0, NoCDW>=15cm=0.9555160142348754
- harmonized_dem_gauss: AUC(tile_max)=0.6363006552463165, J(tile_max)=0.22557121411169856 @thr=1.2799605131149292, J(frac>=15cm)=0.30136115408780817 @thr=0.01015459091599268, CDW>=15cm=1.0, NoCDW>=15cm=0.9537366548042705
- baseline_idw3_drop13: AUC(tile_max)=0.6266596127627132, J(tile_max)=0.20051319557059943 @thr=1.2954013347625732, J(frac>=15cm)=0.36052190611240387 @thr=0.03514151304214795, CDW>=15cm=1.0, NoCDW>=15cm=0.9532918149466192

## Recommendation
**Best method from aggregate score: baseline_idw3_drop13**

## Interpretation
- If TIN/TPS + Gaussian improves Youden/AUC while keeping NoCDW false-high low, it should replace baseline IDW3.
- If smoothed versions consistently beat raw variants, keep Gaussian refinement in production.
- If one method has better DTM residuals but worse CHM separability, prioritize CHM-label separability for CDW detection tasks.

## Output Artifacts
- JSON report: experiments/dtm_hag_436646_reproduce_single_2018/dem_resolution_fasttest_2018_436646/r0p2/experiment_report.json
- CSV summary: experiments/dtm_hag_436646_reproduce_single_2018/dem_resolution_fasttest_2018_436646/r0p2/eval/method_summary.csv
- This Markdown report: experiments/dtm_hag_436646_reproduce_single_2018/dem_resolution_fasttest_2018_436646/r0p2/experiment_report.md

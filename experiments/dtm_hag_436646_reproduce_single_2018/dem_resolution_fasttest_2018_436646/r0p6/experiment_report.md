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
- 2018: class2 ratio=53.47% | DEM valid=90.58% | class2 points=17347992
- Harmonized DEM valid=90.58% | median MAD=0.0 | p95 MAD=0.0

## DTM Ground-Fit (CSF points)
- idw_k3: RMSE=0.07953146121796364 | MAE=0.05233165471961859 | p95|res|=0.1505166105597638 | n=60000
- idw_k6: RMSE=0.07962144801145372 | MAE=0.05245956116618668 | p95|res|=0.14995605344574367 | n=60000
- tin_linear: RMSE=0.08976073363184524 | MAE=0.05452816266011769 | p95|res|=0.15449850082784802 | n=60000
- tps: RMSE=0.17513678146538447 | MAE=0.10020025575817323 | p95|res|=0.3209027896261421 | n=60000

## Aggregate CHM/Label Evaluation
- harmonized_dem_raw: AUC(tile_max)=0.6473144772557385, J(tile_max)=0.22779339563367795 @thr=1.2944488525390625, J(frac>=15cm)=0.32903931974579864 @thr=0.056433128846960785, CDW>=15cm=1.0, NoCDW>=15cm=0.9822064056939501
- harmonized_dem_gauss: AUC(tile_max)=0.6495555947381623, J(tile_max)=0.2298701427709332 @thr=1.2872485550712136, J(frac>=15cm)=0.3144843955268915 @thr=0.053303733396466005, CDW>=15cm=1.0, NoCDW>=15cm=0.9817615658362989
- baseline_idw3_drop13: AUC(tile_max)=0.6266596127627132, J(tile_max)=0.20051319557059943 @thr=1.2954013347625732, J(frac>=15cm)=0.36052190611240387 @thr=0.03514151304214795, CDW>=15cm=1.0, NoCDW>=15cm=0.9532918149466192

## Recommendation
**Best method from aggregate score: harmonized_dem_raw**

## Interpretation
- If TIN/TPS + Gaussian improves Youden/AUC while keeping NoCDW false-high low, it should replace baseline IDW3.
- If smoothed versions consistently beat raw variants, keep Gaussian refinement in production.
- If one method has better DTM residuals but worse CHM separability, prioritize CHM-label separability for CDW detection tasks.

## Output Artifacts
- JSON report: experiments/dtm_hag_436646_reproduce_single_2018/dem_resolution_fasttest_2018_436646/r0p6/experiment_report.json
- CSV summary: experiments/dtm_hag_436646_reproduce_single_2018/dem_resolution_fasttest_2018_436646/r0p6/eval/method_summary.csv
- This Markdown report: experiments/dtm_hag_436646_reproduce_single_2018/dem_resolution_fasttest_2018_436646/r0p6/experiment_report.md

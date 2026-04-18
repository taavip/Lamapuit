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
- idw_k3: RMSE=0.02173019707164498 | MAE=0.0104610456641243 | p95|res|=0.03701940753037424 | n=60000
- idw_k6: RMSE=0.022528715059009417 | MAE=0.011357401365518225 | p95|res|=0.038472970482099786 | n=60000
- tin_linear: RMSE=0.05864272635398439 | MAE=0.03144422320217437 | p95|res|=0.10010933380392809 | n=60000
- tps: RMSE=0.1726126172533752 | MAE=0.0928723908650294 | p95|res|=0.31155187928122513 | n=60000

## Aggregate CHM/Label Evaluation
- idw_k3_raw: AUC(tile_max)=0.6428330183548437, J(tile_max)=0.21161199658283514 @thr=1.2933197021484375, J(frac>=15cm)=0.30620208581156416 @thr=0.017092544474427467, CDW>=15cm=1.0, NoCDW>=15cm=0.9572953736654805
- idw_k3_gauss: AUC(tile_max)=0.6428330183548437, J(tile_max)=0.21161199658283514 @thr=1.2933197021484375, J(frac>=15cm)=0.30620208581156416 @thr=0.017092544474427467, CDW>=15cm=1.0, NoCDW>=15cm=0.9572953736654805
- idw_k6_raw: AUC(tile_max)=0.6350536756872202, J(tile_max)=0.19521066100781526 @thr=1.290008544921875, J(frac>=15cm)=0.30804259908163695 @thr=0.017092802273403186, CDW>=15cm=1.0, NoCDW>=15cm=0.9577402135231317
- idw_k6_gauss: AUC(tile_max)=0.6350536756872202, J(tile_max)=0.19521066100781526 @thr=1.290008544921875, J(frac>=15cm)=0.30804259908163695 @thr=0.017092802273403186, CDW>=15cm=1.0, NoCDW>=15cm=0.9577402135231317
- tin_linear_raw: AUC(tile_max)=0.6322440137530949, J(tile_max)=0.19689875531759016 @thr=1.294769287109375, J(frac>=15cm)=0.3568408795722584 @thr=0.0253848951741255, CDW>=15cm=1.0, NoCDW>=15cm=0.9626334519572953
- tin_linear_gauss: AUC(tile_max)=0.6322440137530949, J(tile_max)=0.19689875531759016 @thr=1.294769287109375, J(frac>=15cm)=0.3568408795722584 @thr=0.0253848951741255, CDW>=15cm=1.0, NoCDW>=15cm=0.9626334519572953
- tps_raw: AUC(tile_max)=0.6474711645677759, J(tile_max)=0.23879705120532035 @thr=1.2955780029296875, J(frac>=15cm)=0.17916088240333972 @thr=0.06716391263290078, CDW>=15cm=1.0, NoCDW>=15cm=0.9875444839857651
- tps_gauss: AUC(tile_max)=0.6474711645677759, J(tile_max)=0.23879705120532035 @thr=1.2955780029296875, J(frac>=15cm)=0.17916088240333972 @thr=0.06716391263290078, CDW>=15cm=1.0, NoCDW>=15cm=0.9875444839857651
- baseline_idw3_drop13: AUC(tile_max)=0.6266596127627132, J(tile_max)=0.20051319557059943 @thr=1.2954013347625732, J(frac>=15cm)=0.36052190611240387 @thr=0.03514151304214795, CDW>=15cm=1.0, NoCDW>=15cm=0.9532918149466192

## Recommendation
**Best method from aggregate score: baseline_idw3_drop13**

## Interpretation
- If TIN/TPS + Gaussian improves Youden/AUC while keeping NoCDW false-high low, it should replace baseline IDW3.
- If smoothed versions consistently beat raw variants, keep Gaussian refinement in production.
- If one method has better DTM residuals but worse CHM separability, prioritize CHM-label separability for CDW detection tasks.

## Output Artifacts
- JSON report: experiments/dtm_hag_436646_research/results/res_0.2/experiment_report.json
- CSV summary: experiments/dtm_hag_436646_research/results/res_0.2/eval/method_summary.csv
- This Markdown report: experiments/dtm_hag_436646_research/results/res_0.2/experiment_report.md

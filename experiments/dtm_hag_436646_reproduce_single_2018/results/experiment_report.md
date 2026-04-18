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
- 2018: class2 ratio=53.47% | DEM valid=93.70% | class2 points=17347992
- Harmonized DEM valid=93.70% | median MAD=0.0 | p95 MAD=0.0

## DTM Ground-Fit (CSF points)
- idw_k3: RMSE=0.12912779015122816 | MAE=0.09038759563961642 | p95|res|=0.24886955392387772 | n=60000
- idw_k6: RMSE=0.12955383268394172 | MAE=0.09051160815806318 | p95|res|=0.24797667171045878 | n=60000
- tin_linear: RMSE=0.128518066215812 | MAE=0.09050634403436073 | p95|res|=0.24459812918632337 | n=60000
- tps: RMSE=0.19498429022042507 | MAE=0.12021651929465972 | p95|res|=0.3563419763252741 | n=60000

## Aggregate CHM/Label Evaluation
- idw_k3_raw: AUC(tile_max)=0.6267493257556379, J(tile_max)=0.20053787440429338 @thr=1.2934646606445312, J(frac>=15cm)=0.24151979397675033 @thr=0.12126166959638177, CDW>=15cm=1.0, NoCDW>=15cm=0.9913256227758007
- idw_k3_gauss: AUC(tile_max)=0.5492428099226916, J(tile_max)=0.16494432206778198 @thr=0.8129341138110442, J(frac>=15cm)=0.23669997119250608 @thr=0.17541787152719254, CDW>=15cm=1.0, NoCDW>=15cm=0.9806494661921709
- idw_k6_raw: AUC(tile_max)=0.6290307209572035, J(tile_max)=0.20140877337194196 @thr=1.29547119140625, J(frac>=15cm)=0.24151979397675033 @thr=0.121162921089086, CDW>=15cm=1.0, NoCDW>=15cm=0.9913256227758007
- idw_k6_gauss: AUC(tile_max)=0.5492481647639648, J(tile_max)=0.16770509197289107 @thr=0.8113348227388718, J(frac>=15cm)=0.23762022782754239 @thr=0.17552398366185779, CDW>=15cm=1.0, NoCDW>=15cm=0.9806494661921709
- tin_linear_raw: AUC(tile_max)=0.6303333443017038, J(tile_max)=0.20255144993580398 @thr=1.295255091050092, J(frac>=15cm)=0.24232022633749972 @thr=0.11225449786382168, CDW>=15cm=1.0, NoCDW>=15cm=0.9913256227758007
- tin_linear_gauss: AUC(tile_max)=0.5452532203549778, J(tile_max)=0.16678483533785482 @thr=0.8114606501074398, J(frac>=15cm)=0.23647755126368047 @thr=0.1754150390625, CDW>=15cm=1.0, NoCDW>=15cm=0.9806494661921709
- tps_raw: AUC(tile_max)=0.6259071411542368, J(tile_max)=0.21711972245469646 @thr=1.295684814453125, J(frac>=15cm)=0.18824005458523413 @thr=0.22050704122474768, CDW>=15cm=1.0, NoCDW>=15cm=0.9986654804270463
- tps_gauss: AUC(tile_max)=0.535876350040852, J(tile_max)=0.14911944679678046 @thr=0.8869017362594604, J(frac>=15cm)=0.21228655137046681 @thr=0.261720133800736, CDW>=15cm=1.0, NoCDW>=15cm=0.994661921708185
- baseline_idw3_drop13: AUC(tile_max)=0.6266596127627132, J(tile_max)=0.20051319557059943 @thr=1.2954013347625732, J(frac>=15cm)=0.36052190611240387 @thr=0.03514151304214795, CDW>=15cm=1.0, NoCDW>=15cm=0.9532918149466192

## Recommendation
**Best method from aggregate score: baseline_idw3_drop13**

## Interpretation
- If TIN/TPS + Gaussian improves Youden/AUC while keeping NoCDW false-high low, it should replace baseline IDW3.
- If smoothed versions consistently beat raw variants, keep Gaussian refinement in production.
- If one method has better DTM residuals but worse CHM separability, prioritize CHM-label separability for CDW detection tasks.

## Output Artifacts
- JSON report: experiments/dtm_hag_436646_reproduce_single_2018/results/experiment_report.json
- CSV summary: experiments/dtm_hag_436646_reproduce_single_2018/results/eval/method_summary.csv
- This Markdown report: experiments/dtm_hag_436646_reproduce_single_2018/results/experiment_report.md

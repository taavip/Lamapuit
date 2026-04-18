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
- idw_k3_raw: AUC(tile_max)=0.6267493257556379, J(tile_max)=0.20053787440429338 @thr=1.2934646606445312, J(frac>=15cm)=0.24139996970246325 @thr=0.11233034099195825, CDW>=15cm=1.0, NoCDW>=15cm=0.9913256227758007
- idw_k3_gauss: AUC(tile_max)=0.5959783900336004, J(tile_max)=0.1944383221621514 @thr=1.2636816501617432, J(frac>=15cm)=0.22786044755918572 @thr=0.12351016208163512, CDW>=15cm=1.0, NoCDW>=15cm=0.9897686832740213
- idw_k6_raw: AUC(tile_max)=0.6290307209572035, J(tile_max)=0.20140877337194196 @thr=1.29547119140625, J(frac>=15cm)=0.24151979397675033 @thr=0.12123136416175563, CDW>=15cm=1.0, NoCDW>=15cm=0.9913256227758007
- idw_k6_gauss: AUC(tile_max)=0.5994314865908567, J(tile_max)=0.19005945891579523 @thr=1.2630638229145723, J(frac>=15cm)=0.22863620108624116 @thr=0.12546582117079447, CDW>=15cm=1.0, NoCDW>=15cm=0.9897686832740213
- tin_linear_raw: AUC(tile_max)=0.6303333443017038, J(tile_max)=0.20255144993580398 @thr=1.295255091050092, J(frac>=15cm)=0.24232022633749972 @thr=0.11225449786382168, CDW>=15cm=1.0, NoCDW>=15cm=0.9913256227758007
- tin_linear_gauss: AUC(tile_max)=0.5920328032920013, J(tile_max)=0.19466074209097706 @thr=1.2631965884040384, J(frac>=15cm)=0.2277159444512048 @thr=0.12543546641844067, CDW>=15cm=1.0, NoCDW>=15cm=0.9897686832740213
- tps_raw: AUC(tile_max)=0.6259071411542368, J(tile_max)=0.21711972245469646 @thr=1.295684814453125, J(frac>=15cm)=0.18363877141005236 @thr=0.22344067295241504, CDW>=15cm=1.0, NoCDW>=15cm=0.9986654804270463
- tps_gauss: AUC(tile_max)=0.5790340425109207, J(tile_max)=0.1636857015493962 @thr=1.2568027316822725, J(frac>=15cm)=0.18731979795019782 @thr=0.23283946482338647, CDW>=15cm=1.0, NoCDW>=15cm=0.9977758007117438
- baseline_idw3_drop13: AUC(tile_max)=0.6266596127627132, J(tile_max)=0.20051319557059943 @thr=1.2954013347625732, J(frac>=15cm)=0.36052190611240387 @thr=0.03514151304214795, CDW>=15cm=1.0, NoCDW>=15cm=0.9532918149466192

## Recommendation
**Best method from aggregate score: baseline_idw3_drop13**

## Interpretation
- If TIN/TPS + Gaussian improves Youden/AUC while keeping NoCDW false-high low, it should replace baseline IDW3.
- If smoothed versions consistently beat raw variants, keep Gaussian refinement in production.
- If one method has better DTM residuals but worse CHM separability, prioritize CHM-label separability for CDW detection tasks.

## Output Artifacts
- JSON report: experiments/dtm_hag_436646_reproduce_single_2018/improve_rounds_2026_04_15/r1_sharp_negative/experiment_report.json
- CSV summary: experiments/dtm_hag_436646_reproduce_single_2018/improve_rounds_2026_04_15/r1_sharp_negative/eval/method_summary.csv
- This Markdown report: experiments/dtm_hag_436646_reproduce_single_2018/improve_rounds_2026_04_15/r1_sharp_negative/experiment_report.md

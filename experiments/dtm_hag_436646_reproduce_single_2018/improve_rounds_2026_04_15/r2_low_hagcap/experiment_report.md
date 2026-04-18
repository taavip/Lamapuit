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
- idw_k3_raw: AUC(tile_max)=0.6163425409699682, J(tile_max)=0.2091091903355825 @thr=0.883209228515625, J(frac>=15cm)=0.24180880019271228 @thr=0.10558946350994865, CDW>=15cm=1.0, NoCDW>=15cm=0.9911032028469751
- idw_k3_gauss: AUC(tile_max)=0.5574382004753237, J(tile_max)=0.1671936658281037 @thr=0.7364046573638916, J(frac>=15cm)=0.21056943226886338 @thr=0.08386674245054847, CDW>=15cm=1.0, NoCDW>=15cm=0.9853202846975089
- idw_k6_raw: AUC(tile_max)=0.6126171235717319, J(tile_max)=0.20863377355819734 @thr=0.8835916137695312, J(frac>=15cm)=0.2427290568277486 @thr=0.10558087822748562, CDW>=15cm=1.0, NoCDW>=15cm=0.9911032028469751
- idw_k6_gauss: AUC(tile_max)=0.5586943375890611, J(tile_max)=0.16627340919306732 @thr=0.7362359762191772, J(frac>=15cm)=0.21148968890389974 @thr=0.08384561807428344, CDW>=15cm=1.0, NoCDW>=15cm=0.9853202846975089
- tin_linear_raw: AUC(tile_max)=0.6158673570117689, J(tile_max)=0.20924826099589489 @thr=0.8839874267578125, J(frac>=15cm)=0.2427290568277486 @thr=0.10557918058797819, CDW>=15cm=1.0, NoCDW>=15cm=0.9911032028469751
- tin_linear_gauss: AUC(tile_max)=0.5593758769522664, J(tile_max)=0.16768041313919735 @thr=0.7580105398682987, J(frac>=15cm)=0.21148968890389974 @thr=0.08362555830695381, CDW>=15cm=1.0, NoCDW>=15cm=0.9853202846975089
- tps_raw: AUC(tile_max)=0.593825355809801, J(tile_max)=0.1792274686904759 @thr=0.88848876953125, J(frac>=15cm)=0.17354062725834607 @thr=0.19655540349366352, CDW>=15cm=1.0, NoCDW>=15cm=0.9986654804270463
- tps_gauss: AUC(tile_max)=0.5228761923446569, J(tile_max)=0.13052790353462984 @thr=0.7725944519042969, J(frac>=15cm)=0.17276487373129062 @thr=0.19714284497766552, CDW>=15cm=1.0, NoCDW>=15cm=0.9959964412811388
- baseline_idw3_drop13: AUC(tile_max)=0.6266596127627132, J(tile_max)=0.20051319557059943 @thr=1.2954013347625732, J(frac>=15cm)=0.36052190611240387 @thr=0.03514151304214795, CDW>=15cm=1.0, NoCDW>=15cm=0.9532918149466192

## Recommendation
**Best method from aggregate score: baseline_idw3_drop13**

## Interpretation
- If TIN/TPS + Gaussian improves Youden/AUC while keeping NoCDW false-high low, it should replace baseline IDW3.
- If smoothed versions consistently beat raw variants, keep Gaussian refinement in production.
- If one method has better DTM residuals but worse CHM separability, prioritize CHM-label separability for CDW detection tasks.

## Output Artifacts
- JSON report: experiments/dtm_hag_436646_reproduce_single_2018/improve_rounds_2026_04_15/r2_low_hagcap/experiment_report.json
- CSV summary: experiments/dtm_hag_436646_reproduce_single_2018/improve_rounds_2026_04_15/r2_low_hagcap/eval/method_summary.csv
- This Markdown report: experiments/dtm_hag_436646_reproduce_single_2018/improve_rounds_2026_04_15/r2_low_hagcap/experiment_report.md

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
- 2018: class2 ratio=53.47% | DEM valid=89.24% | class2 points=17347992
- Harmonized DEM valid=89.24% | median MAD=0.0 | p95 MAD=0.0

## DTM Ground-Fit (CSF points)
- idw_k3: RMSE=0.0680937373133694 | MAE=0.0416419821568342 | p95|res|=0.12429581394724835 | n=60000
- idw_k6: RMSE=0.06787651257676515 | MAE=0.041770987006641315 | p95|res|=0.12274639698651121 | n=60000
- tin_linear: RMSE=0.07490179174882021 | MAE=0.04569387920072829 | p95|res|=0.13373192340657097 | n=60000
- tps: RMSE=0.17635822091409833 | MAE=0.09568697105179111 | p95|res|=0.31114264370682176 | n=60000

## Aggregate CHM/Label Evaluation
- idw_k3_raw: AUC(tile_max)=0.6271457392226447, J(tile_max)=0.2073658402723798 @thr=1.2928009033203125, J(frac>=15cm)=0.3254076508730409 @thr=0.036897841341207153, CDW>=15cm=1.0, NoCDW>=15cm=0.972864768683274
- idw_k3_gauss: AUC(tile_max)=0.6271457392226447, J(tile_max)=0.2073658402723798 @thr=1.2928009033203125, J(frac>=15cm)=0.3254076508730409 @thr=0.036897841341207153, CDW>=15cm=1.0, NoCDW>=15cm=0.972864768683274
- idw_k6_raw: AUC(tile_max)=0.6195625855222476, J(tile_max)=0.20604808368080307 @thr=1.2929534912109375, J(frac>=15cm)=0.3244873942380046 @thr=0.036791443747491805, CDW>=15cm=1.0, NoCDW>=15cm=0.972864768683274
- idw_k6_gauss: AUC(tile_max)=0.6195625855222476, J(tile_max)=0.20604808368080307 @thr=1.2929534912109375, J(frac>=15cm)=0.3244873942380046 @thr=0.036791443747491805, CDW>=15cm=1.0, NoCDW>=15cm=0.972864768683274
- tin_linear_raw: AUC(tile_max)=0.6295178563002735, J(tile_max)=0.20052654387058477 @thr=1.293243408203125, J(frac>=15cm)=0.3172698442656945 @thr=0.04197345350074225, CDW>=15cm=1.0, NoCDW>=15cm=0.9788701067615658
- tin_linear_gauss: AUC(tile_max)=0.6295178563002735, J(tile_max)=0.20052654387058477 @thr=1.293243408203125, J(frac>=15cm)=0.3172698442656945 @thr=0.04197345350074225, CDW>=15cm=1.0, NoCDW>=15cm=0.9788701067615658
- tps_raw: AUC(tile_max)=0.6317202481914606, J(tile_max)=0.22098126147022523 @thr=1.295294584386489, J(frac>=15cm)=0.16026900239151864 @thr=0.08317427111196582, CDW>=15cm=1.0, NoCDW>=15cm=0.9953291814946619
- tps_gauss: AUC(tile_max)=0.6317202481914606, J(tile_max)=0.22098126147022523 @thr=1.295294584386489, J(frac>=15cm)=0.16026900239151864 @thr=0.08317427111196582, CDW>=15cm=1.0, NoCDW>=15cm=0.9953291814946619
- baseline_idw3_drop13: AUC(tile_max)=0.6266596127627132, J(tile_max)=0.20051319557059943 @thr=1.2954013347625732, J(frac>=15cm)=0.36052190611240387 @thr=0.03514151304214795, CDW>=15cm=1.0, NoCDW>=15cm=0.9532918149466192

## Recommendation
**Best method from aggregate score: baseline_idw3_drop13**

## Interpretation
- If TIN/TPS + Gaussian improves Youden/AUC while keeping NoCDW false-high low, it should replace baseline IDW3.
- If smoothed versions consistently beat raw variants, keep Gaussian refinement in production.
- If one method has better DTM residuals but worse CHM separability, prioritize CHM-label separability for CDW detection tasks.

## Output Artifacts
- JSON report: experiments/dtm_hag_436646_research/results/res_0.5/experiment_report.json
- CSV summary: experiments/dtm_hag_436646_research/results/res_0.5/eval/method_summary.csv
- This Markdown report: experiments/dtm_hag_436646_research/results/res_0.5/experiment_report.md

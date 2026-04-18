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
- 2018: class2 ratio=53.47% | DEM valid=16.19% | class2 points=17347992
- Harmonized DEM valid=16.19% | median MAD=0.0 | p95 MAD=0.0

## DTM Ground-Fit (CSF points)
- idw_k3: RMSE=0.011060287487524993 | MAE=0.004293439430128581 | p95|res|=0.01718205684357343 | n=60000
- idw_k6: RMSE=0.011643588422050691 | MAE=0.004929797129192764 | p95|res|=0.018441537304217093 | n=60000
- tin_linear: RMSE=0.058692939074036034 | MAE=0.03085331019457178 | p95|res|=0.1008017414055679 | n=60000
- tps: RMSE=0.17028391168577542 | MAE=0.09399693256984204 | p95|res|=0.31855193742320903 | n=60000

## Aggregate CHM/Label Evaluation
- idw_k3_raw: AUC(tile_max)=0.6299063539149633, J(tile_max)=0.23791280393768643 @thr=1.2949986715877757, J(frac>=15cm)=0.31252405798253174 @thr=0.015843964490352653, CDW>=15cm=1.0, NoCDW>=15cm=0.9524021352313167
- idw_k3_gauss: AUC(tile_max)=0.6299063539149633, J(tile_max)=0.23791280393768643 @thr=1.2949986715877757, J(frac>=15cm)=0.31252405798253174 @thr=0.015843964490352653, CDW>=15cm=1.0, NoCDW>=15cm=0.9524021352313167
- idw_k6_raw: AUC(tile_max)=0.6277360910714153, J(tile_max)=0.217836339908561 @thr=1.2935198974609374, J(frac>=15cm)=0.31830992517501794 @thr=0.01737012014818334, CDW>=15cm=1.0, NoCDW>=15cm=0.9524021352313167
- idw_k6_gauss: AUC(tile_max)=0.6277360910714153, J(tile_max)=0.217836339908561 @thr=1.2935198974609374, J(frac>=15cm)=0.31830992517501794 @thr=0.01737012014818334, CDW>=15cm=1.0, NoCDW>=15cm=0.9524021352313167
- tin_linear_raw: AUC(tile_max)=0.6236220208456986, J(tile_max)=0.19138854604108047 @thr=1.2959188124712775, J(frac>=15cm)=0.3577611362072947 @thr=0.024716366880391392, CDW>=15cm=1.0, NoCDW>=15cm=0.9619661921708185
- tin_linear_gauss: AUC(tile_max)=0.6236220208456986, J(tile_max)=0.19138854604108047 @thr=1.2959188124712775, J(frac>=15cm)=0.3577611362072947 @thr=0.024716366880391392, CDW>=15cm=1.0, NoCDW>=15cm=0.9619661921708185
- tps_raw: AUC(tile_max)=0.6254726229471557, J(tile_max)=0.21487782890832008 @thr=1.2923812866210938, J(frac>=15cm)=0.16961607184986327 @thr=0.06237320012853205, CDW>=15cm=1.0, NoCDW>=15cm=0.9866548042704626
- tps_gauss: AUC(tile_max)=0.6254726229471557, J(tile_max)=0.21487782890832008 @thr=1.2923812866210938, J(frac>=15cm)=0.16961607184986327 @thr=0.06237320012853205, CDW>=15cm=1.0, NoCDW>=15cm=0.9866548042704626
- baseline_idw3_drop13: AUC(tile_max)=0.6266596127627132, J(tile_max)=0.20051319557059943 @thr=1.2954013347625732, J(frac>=15cm)=0.36052190611240387 @thr=0.03514151304214795, CDW>=15cm=1.0, NoCDW>=15cm=0.9532918149466192

## Recommendation
**Best method from aggregate score: idw_k3_raw**

## Interpretation
- If TIN/TPS + Gaussian improves Youden/AUC while keeping NoCDW false-high low, it should replace baseline IDW3.
- If smoothed versions consistently beat raw variants, keep Gaussian refinement in production.
- If one method has better DTM residuals but worse CHM separability, prioritize CHM-label separability for CDW detection tasks.

## Output Artifacts
- JSON report: experiments/dtm_hag_436646_research/results/res_0.1/experiment_report.json
- CSV summary: experiments/dtm_hag_436646_research/results/res_0.1/eval/method_summary.csv
- This Markdown report: experiments/dtm_hag_436646_research/results/res_0.1/experiment_report.md

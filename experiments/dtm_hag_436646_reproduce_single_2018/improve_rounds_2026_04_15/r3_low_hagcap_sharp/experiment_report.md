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
- idw_k3_raw: AUC(tile_max)=0.6199640434049465, J(tile_max)=0.2145641438586645 @thr=0.6831008731617648, J(frac>=15cm)=0.24103304666565672 @thr=0.09917372995103853, CDW>=15cm=1.0, NoCDW>=15cm=0.9911032028469751
- idw_k3_gauss: AUC(tile_max)=0.6183921258688811, J(tile_max)=0.2138549766187453 @thr=0.6787462830543518, J(frac>=15cm)=0.22555786581171322 @thr=0.08768210448008963, CDW>=15cm=1.0, NoCDW>=15cm=0.9908807829181495
- idw_k6_raw: AUC(tile_max)=0.6233803545308476, J(tile_max)=0.2161822371999116 @thr=0.68280029296875, J(frac>=15cm)=0.24180880019271228 @thr=0.10171930413989808, CDW>=15cm=1.0, NoCDW>=15cm=0.9911032028469751
- idw_k6_gauss: AUC(tile_max)=0.6220884408688936, J(tile_max)=0.21522007311143287 @thr=0.678547556961284, J(frac>=15cm)=0.2253886838700384 @thr=0.09958218227751132, CDW>=15cm=1.0, NoCDW>=15cm=0.9908807829181495
- tin_linear_raw: AUC(tile_max)=0.6232419823280925, J(tile_max)=0.21537047430545375 @thr=0.683990478515625, J(frac>=15cm)=0.24180880019271228 @thr=0.10171899274053335, CDW>=15cm=1.0, NoCDW>=15cm=0.9911032028469751
- tin_linear_gauss: AUC(tile_max)=0.6224755415684687, J(tile_max)=0.217836339908561 @thr=0.6794771456718445, J(frac>=15cm)=0.22446842723500204 @thr=0.09920232397391379, CDW>=15cm=1.0, NoCDW>=15cm=0.9908807829181495
- tps_raw: AUC(tile_max)=0.5638903961775436, J(tile_max)=0.15465231714070726 @thr=0.6832711612477023, J(frac>=15cm)=0.17339612415036515 @thr=0.1935506157216678, CDW>=15cm=1.0, NoCDW>=15cm=0.9986654804270463
- tps_gauss: AUC(tile_max)=0.548507644540359, J(tile_max)=0.15678183662674183 @thr=0.6793064061333152, J(frac>=15cm)=0.1746053870013634 @thr=0.19169943158960517, CDW>=15cm=1.0, NoCDW>=15cm=0.9986654804270463
- baseline_idw3_drop13: AUC(tile_max)=0.6266596127627132, J(tile_max)=0.20051319557059943 @thr=1.2954013347625732, J(frac>=15cm)=0.36052190611240387 @thr=0.03514151304214795, CDW>=15cm=1.0, NoCDW>=15cm=0.9532918149466192

## Recommendation
**Best method from aggregate score: baseline_idw3_drop13**

## Interpretation
- If TIN/TPS + Gaussian improves Youden/AUC while keeping NoCDW false-high low, it should replace baseline IDW3.
- If smoothed versions consistently beat raw variants, keep Gaussian refinement in production.
- If one method has better DTM residuals but worse CHM separability, prioritize CHM-label separability for CDW detection tasks.

## Output Artifacts
- JSON report: experiments/dtm_hag_436646_reproduce_single_2018/improve_rounds_2026_04_15/r3_low_hagcap_sharp/experiment_report.json
- CSV summary: experiments/dtm_hag_436646_reproduce_single_2018/improve_rounds_2026_04_15/r3_low_hagcap_sharp/eval/method_summary.csv
- This Markdown report: experiments/dtm_hag_436646_reproduce_single_2018/improve_rounds_2026_04_15/r3_low_hagcap_sharp/experiment_report.md

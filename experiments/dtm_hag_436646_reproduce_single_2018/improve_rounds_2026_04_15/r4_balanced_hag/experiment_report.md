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
- idw_k3_raw: AUC(tile_max)=0.6289783366404005, J(tile_max)=0.21115629182984708 @thr=1.093994140625, J(frac>=15cm)=0.24180880019271228 @thr=0.11041696525831907, CDW>=15cm=1.0, NoCDW>=15cm=0.9911032028469751
- idw_k3_gauss: AUC(tile_max)=0.6136619384711665, J(tile_max)=0.2059814973936669 @thr=1.0840765237808228, J(frac>=15cm)=0.22601993428911293 @thr=0.11401123396194535, CDW>=15cm=1.0, NoCDW>=15cm=0.9908807829181495
- idw_k6_raw: AUC(tile_max)=0.6278674786985966, J(tile_max)=0.20962992924780155 @thr=1.093994140625, J(frac>=15cm)=0.24103304666565672 @thr=0.10816150696224518, CDW>=15cm=1.0, NoCDW>=15cm=0.9911032028469751
- idw_k6_gauss: AUC(tile_max)=0.6124560126951646, J(tile_max)=0.20536157750830075 @thr=1.0845968776590684, J(frac>=15cm)=0.22786044755918572 @thr=0.11399623318822691, CDW>=15cm=1.0, NoCDW>=15cm=0.9908807829181495
- tin_linear_raw: AUC(tile_max)=0.6256794439905332, J(tile_max)=0.21038891979348007 @thr=1.0936126708984375, J(frac>=15cm)=0.2408885435576758 @thr=0.11041696525831907, CDW>=15cm=1.0, NoCDW>=15cm=0.9911032028469751
- tin_linear_gauss: AUC(tile_max)=0.6103128344835635, J(tile_max)=0.2044413208732644 @thr=1.0847245822233311, J(frac>=15cm)=0.22601993428911293 @thr=0.11388734192894633, CDW>=15cm=1.0, NoCDW>=15cm=0.9908807829181495
- tps_raw: AUC(tile_max)=0.5829423005763982, J(tile_max)=0.17467554318268175 @thr=1.08673095703125, J(frac>=15cm)=0.17906216706856426 @thr=0.199786603920469, CDW>=15cm=1.0, NoCDW>=15cm=0.9986654804270463
- tps_gauss: AUC(tile_max)=0.5843608678754225, J(tile_max)=0.1764494701656183 @thr=1.0821440304026884, J(frac>=15cm)=0.1816784338656925 @thr=0.20870401475989703, CDW>=15cm=1.0, NoCDW>=15cm=0.9986654804270463
- baseline_idw3_drop13: AUC(tile_max)=0.6266596127627132, J(tile_max)=0.20051319557059943 @thr=1.2954013347625732, J(frac>=15cm)=0.36052190611240387 @thr=0.03514151304214795, CDW>=15cm=1.0, NoCDW>=15cm=0.9532918149466192

## Recommendation
**Best method from aggregate score: baseline_idw3_drop13**

## Interpretation
- If TIN/TPS + Gaussian improves Youden/AUC while keeping NoCDW false-high low, it should replace baseline IDW3.
- If smoothed versions consistently beat raw variants, keep Gaussian refinement in production.
- If one method has better DTM residuals but worse CHM separability, prioritize CHM-label separability for CDW detection tasks.

## Output Artifacts
- JSON report: experiments/dtm_hag_436646_reproduce_single_2018/improve_rounds_2026_04_15/r4_balanced_hag/experiment_report.json
- CSV summary: experiments/dtm_hag_436646_reproduce_single_2018/improve_rounds_2026_04_15/r4_balanced_hag/eval/method_summary.csv
- This Markdown report: experiments/dtm_hag_436646_reproduce_single_2018/improve_rounds_2026_04_15/r4_balanced_hag/experiment_report.md

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
- Years: [2018, 2020, 2022, 2024]
- LAZ directory: data/lamapuit/laz
- Labels directory: output/onboarding_labels_v2_drop13

## CSF and Harmonization Diagnostics
- 2018: class2 ratio=52.63% | DEM valid=92.69% | class2 points=17075152
- 2020: class2 ratio=44.94% | DEM valid=93.96% | class2 points=19178183
- 2022: class2 ratio=42.34% | DEM valid=94.53% | class2 points=28478822
- 2024: class2 ratio=37.25% | DEM valid=87.82% | class2 points=15027353
- Harmonized DEM valid=96.40% | median MAD=0.029647022485733032 | p95 MAD=0.08895237743854523

## DTM Ground-Fit (CSF points)
- idw_k3: RMSE=0.1793947746294638 | MAE=0.1392546462851674 | p95|res|=0.33455115367217586 | n=240000
- idw_k6: RMSE=0.17887283563599 | MAE=0.13905092120996046 | p95|res|=0.3321717665018191 | n=240000
- tin_linear: RMSE=0.1775773885275083 | MAE=0.1391682518104688 | p95|res|=0.32930057533772084 | n=240000
- tps: RMSE=0.2883961444382655 | MAE=0.20108162826440015 | p95|res|=0.5991516574510065 | n=240000

## Aggregate CHM/Label Evaluation
- idw_k3_raw: AUC(tile_max)=0.6155242235351944, J(tile_max)=0.18427436816245346 @thr=1.282012939453125, J(frac>=15cm)=0.17471773007958058 @thr=0.2119689551859108, CDW>=15cm=1.0, NoCDW>=15cm=0.9989967542047802
- idw_k3_gauss: AUC(tile_max)=0.5499097722007908, J(tile_max)=0.14128044815079532 @thr=1.0523313105106353, J(frac>=15cm)=0.17450201484565314 @thr=0.2706293086401309, CDW>=15cm=1.0, NoCDW>=15cm=0.995868987902036
- idw_k6_raw: AUC(tile_max)=0.616174136493008, J(tile_max)=0.184628899417826 @thr=1.28759765625, J(frac>=15cm)=0.17471773007958058 @thr=0.21190503996227555, CDW>=15cm=1.0, NoCDW>=15cm=0.9989967542047802
- idw_k6_gauss: AUC(tile_max)=0.5503650446493492, J(tile_max)=0.14148715128161649 @thr=1.0523307740688324, J(frac>=15cm)=0.17470871797647436 @thr=0.27068786621093754, CDW>=15cm=1.0, NoCDW>=15cm=0.995868987902036
- tin_linear_raw: AUC(tile_max)=0.6180810940239951, J(tile_max)=0.18699130807059539 @thr=1.287200927734375, J(frac>=15cm)=0.17471773007958058 @thr=0.21179725210803824, CDW>=15cm=1.0, NoCDW>=15cm=0.9989967542047802
- tin_linear_gauss: AUC(tile_max)=0.5526683124649544, J(tile_max)=0.1437608857206495 @thr=1.0494307637214662, J(frac>=15cm)=0.17574223363058022 @thr=0.2703270451036834, CDW>=15cm=1.0, NoCDW>=15cm=0.995868987902036
- tps_raw: AUC(tile_max)=0.5985792951114701, J(tile_max)=0.16938306183936935 @thr=1.279730987548828, J(frac>=15cm)=0.12523958445140282 @thr=0.3399587621836504, CDW>=15cm=1.0, NoCDW>=15cm=0.999645913248746
- tps_gauss: AUC(tile_max)=0.5280590435692042, J(tile_max)=0.130531885348094 @thr=1.0760288774967195, J(frac>=15cm)=0.1315257878410605 @thr=0.3224219462914774, CDW>=15cm=1.0, NoCDW>=15cm=0.9995278843316613
- baseline_idw3_drop13: AUC(tile_max)=0.6149085635417983, J(tile_max)=0.1855900742056047 @thr=1.295918345451355, J(frac>=15cm)=0.2919859611654184 @thr=0.05042476689706324, CDW>=15cm=1.0, NoCDW>=15cm=0.9603422838595456

## Recommendation
**Best method from aggregate score: baseline_idw3_drop13**

## Interpretation
- If TIN/TPS + Gaussian improves Youden/AUC while keeping NoCDW false-high low, it should replace baseline IDW3.
- If smoothed versions consistently beat raw variants, keep Gaussian refinement in production.
- If one method has better DTM residuals but worse CHM separability, prioritize CHM-label separability for CDW detection tasks.

## Output Artifacts
- JSON report: experiments/dtm_hag_436646_research/results/experiment_report.json
- CSV summary: experiments/dtm_hag_436646_research/results/eval/method_summary.csv
- This Markdown report: experiments/dtm_hag_436646_research/results/experiment_report.md

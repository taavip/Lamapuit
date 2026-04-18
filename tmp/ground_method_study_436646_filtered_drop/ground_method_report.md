# Ground Estimation Study: 436646_2024_madal

## Objective
Assess alternatives to unreliable LAS ground tags for LAZ->CHM generation and select the best method for CDW visibility.

Configured CHM mode: **drop**

## Methods
- class2_idw_baseline: Existing LAS class 2 ground points + IDW interpolation (drop mode).
- smrf_idw: PDAL SMRF ground reclassification, then same IDW CHM logic (drop mode).
- csf_idw: PDAL CSF ground reclassification, then same IDW CHM logic (drop mode).
- allpoints_surface: No class tags: coarse minimum surface + morphological opening + smoothing, then CHM (drop mode).
- class2_surface_fixed_low: Class2 ground DEM with fixed low smoothing (sigma=0.8), then CHM (drop mode).
- class2_surface_fixed_high: Class2 ground DEM with fixed high smoothing (sigma=2.4), then CHM (drop mode).
- class2_surface_adaptive: Class2 ground DEM with adaptive roughness-based smoothing, then CHM (drop mode).
- class2_filter_local_residual: Class2 filter local residual (n=5, thr=2.0) + fixed_high surface smoothing, then CHM (drop mode).
- class2_filter_mad: Class2 filter MAD (n=5, thr=2.5) + fixed_high surface smoothing, then CHM (drop mode).
- class2_filter_slope_curvature: Class2 filter slope-curvature (slope_thr=0.6) + fixed_high surface smoothing, then CHM (drop mode).

## Raster Statistics
- class2_idw_baseline: max=1.2999999523162842 p95=0.864553689956665 mean=0.10299110412597656 std=0.26261356472969055
- smrf_idw: max=1.2999999523162842 p95=0.893893837928772 mean=0.07843755930662155 std=0.26786598563194275
- csf_idw: max=1.2999999523162842 p95=0.9126774668693542 mean=0.08437337726354599 std=0.2736928164958954
- allpoints_surface: max=1.2999998331069946 p95=0.7791322469711304 mean=0.18888890743255615 std=0.24072016775608063
- class2_surface_fixed_low: max=1.299999475479126 p95=0.954949140548706 mean=0.21059559285640717 std=0.2843547761440277
- class2_surface_fixed_high: max=1.2999998331069946 p95=1.0811998844146729 mean=0.307621031999588 std=0.3323136866092682
- class2_surface_adaptive: max=1.2999999523162842 p95=1.0887290239334106 mean=0.31436893343925476 std=0.3360998332500458
- class2_filter_local_residual: max=1.2999999523162842 p95=1.168668508529663 mean=0.4770176410675049 std=0.3515603244304657
- class2_filter_mad: max=1.2999998331069946 p95=1.1482762098312378 mean=0.41676798462867737 std=0.35293343663215637
- class2_filter_slope_curvature: max=1.2999993562698364 p95=1.0973405838012695 mean=0.31401821970939636 std=0.3392166197299957

## CDW Validation (tile labels)
- class2_idw_baseline: cdw_detect_rate_15cm=1.0000, no_false_high_rate_15cm=0.9715, best_tile_max_j=0.1524 at thr=1.2947, best_frac15_j=0.3113 at thr=0.0974
  auc_tile_max=0.577340357118189 auc_frac15=0.6769455802537778 cohens_d_tile_max=0.2901514818486875
- smrf_idw: cdw_detect_rate_15cm=1.0000, no_false_high_rate_15cm=0.9499, best_tile_max_j=0.1623 at thr=1.2952, best_frac15_j=0.2426 at thr=0.0421
  auc_tile_max=0.5846391386811398 auc_frac15=0.6381862744467139 cohens_d_tile_max=0.28981138224233105
- csf_idw: cdw_detect_rate_15cm=1.0000, no_false_high_rate_15cm=0.9506, best_tile_max_j=0.1581 at thr=1.2952, best_frac15_j=0.2470 at thr=0.0507
  auc_tile_max=0.5872806729654212 auc_frac15=0.6506773885351684 cohens_d_tile_max=0.2870953458922933
- allpoints_surface: cdw_detect_rate_15cm=1.0000, no_false_high_rate_15cm=0.9961, best_tile_max_j=0.1168 at thr=1.2926, best_frac15_j=0.1585 at thr=0.2664
  auc_tile_max=0.54429281014459 auc_frac15=0.6068090857882474 cohens_d_tile_max=0.28102198001032813
- class2_surface_fixed_low: cdw_detect_rate_15cm=1.0000, no_false_high_rate_15cm=0.9922, best_tile_max_j=0.1491 at thr=1.2991, best_frac15_j=0.3577 at thr=0.3828
  auc_tile_max=0.5993102773691438 auc_frac15=0.7379087251178389 cohens_d_tile_max=0.2778526103480927
- class2_surface_fixed_high: cdw_detect_rate_15cm=0.9956, no_false_high_rate_15cm=0.9915, best_tile_max_j=0.1383 at thr=1.2979, best_frac15_j=0.3490 at thr=0.6846
  auc_tile_max=0.5788015720716605 auc_frac15=0.7385538519488085 cohens_d_tile_max=0.25561115660526795
- class2_surface_adaptive: cdw_detect_rate_15cm=1.0000, no_false_high_rate_15cm=0.9944, best_tile_max_j=0.1751 at thr=1.2984, best_frac15_j=0.3495 at thr=0.6229
  auc_tile_max=0.6098531406039717 auc_frac15=0.7403862695559769 cohens_d_tile_max=0.27999862251717156
- class2_filter_local_residual: cdw_detect_rate_15cm=0.9506, no_false_high_rate_15cm=0.9776, best_tile_max_j=0.0817 at thr=1.2985, best_frac15_j=0.1474 at thr=0.9814
  auc_tile_max=0.5427868706876248 auc_frac15=0.5913819206610904 cohens_d_tile_max=-0.013558567623835964
- class2_filter_mad: cdw_detect_rate_15cm=0.9718, no_false_high_rate_15cm=0.9822, best_tile_max_j=0.1063 at thr=1.2961, best_frac15_j=0.2465 at thr=0.8779
  auc_tile_max=0.5540310613285558 auc_frac15=0.6569981299365877 cohens_d_tile_max=0.09905542960226105
- class2_filter_slope_curvature: cdw_detect_rate_15cm=0.9718, no_false_high_rate_15cm=0.9888, best_tile_max_j=0.1431 at thr=1.2987, best_frac15_j=0.3447 at thr=0.6743
  auc_tile_max=0.5830424631811155 auc_frac15=0.7187735018149354 cohens_d_tile_max=0.1124144791186642

## Ground Filter Diagnostics
- class2_filter_local_residual: n_class2_input=11230061 n_class2_filtered=727356 filter_pct=6.477 reject_cell_pct=6.362 roughness_reduction_pct=-49.962
- class2_filter_mad: n_class2_input=11230061 n_class2_filtered=456343 filter_pct=4.064 reject_cell_pct=4.280 roughness_reduction_pct=-49.962
- class2_filter_slope_curvature: n_class2_input=11230061 n_class2_filtered=102375 filter_pct=0.912 reject_cell_pct=2.955 roughness_reduction_pct=0.000

## Recommended Method
**class2_surface_adaptive**

## Thesis-Oriented Interpretation
- Ground tags (class 2) can suppress CDW when misclassified points define the local ground surface.
- Recomputing ground with physical filters (SMRF or CSF) reduces dependence on delivered class tags.
- The selected method is the one with strongest CDW/no-CDW separability on independent tile labels and acceptable false-high behavior.
- For thesis reproducibility, report filter parameters, HAG cap, grid resolution, and label-source version.

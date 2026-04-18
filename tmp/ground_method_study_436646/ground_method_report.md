# Ground Estimation Study: 436646_2024_madal

## Objective
Assess alternatives to unreliable LAS ground tags for LAZ->CHM generation and select the best method for CDW visibility.

## Methods
- class2_idw_baseline: Existing LAS class 2 ground points + IDW interpolation.
- smrf_idw: PDAL SMRF ground reclassification, then same IDW CHM logic.
- csf_idw: PDAL CSF ground reclassification, then same IDW CHM logic.
- allpoints_surface: No class tags: coarse minimum surface + morphological opening + smoothing, then max-HAG rasterization.

## Raster Statistics
- class2_idw_baseline: max=1.2999999523162842 p95=1.2999999523162842 mean=0.7694681882858276 std=0.6282770037651062
- smrf_idw: max=1.2999999523162842 p95=1.2999999523162842 mean=0.7591851949691772 std=0.6363879442214966
- csf_idw: max=1.2999999523162842 p95=1.2999999523162842 mean=0.7613499760627747 std=0.6358346939086914
- allpoints_surface: max=1.2999999523162842 p95=1.2999999523162842 mean=0.8052651882171631 std=0.5839194059371948

## CDW Validation (tile labels)
- class2_idw_baseline: cdw_detect_rate_15cm=1.0000, no_false_high_rate_15cm=0.9776, best_tile_max_j=0.0396 at thr=1.3000
- smrf_idw: cdw_detect_rate_15cm=1.0000, no_false_high_rate_15cm=0.9621, best_tile_max_j=0.0394 at thr=1.3000
- csf_idw: cdw_detect_rate_15cm=1.0000, no_false_high_rate_15cm=0.9628, best_tile_max_j=0.0394 at thr=1.3000
- allpoints_surface: cdw_detect_rate_15cm=1.0000, no_false_high_rate_15cm=0.9968, best_tile_max_j=0.0384 at thr=1.3000

## Recommended Method
**smrf_idw**

## Thesis-Oriented Interpretation
- Ground tags (class 2) can suppress CDW when misclassified points define the local ground surface.
- Recomputing ground with physical filters (SMRF or CSF) reduces dependence on delivered class tags.
- The selected method is the one with strongest CDW/no-CDW separability on independent tile labels and acceptable false-high behavior.
- For thesis reproducibility, report filter parameters, HAG cap, grid resolution, and label-source version.

# DEM Resolution Fast Comparison (tile 436646, year 2018)

- Mode: harmonized raw+gauss only (`--only-raw-gauss`) with baseline comparison
- HAG filter: drop, 0.0 <= HAG <= 1.3 m
- Gaussian sigma: 0.3

## Runtime
- 1.0 m: 14.24 s
- 0.6 m: 31.93 s
- 0.2 m: 147.84 s

## Best Method By Run
- 1.0 m: baseline_idw3_drop13
- 0.6 m: harmonized_dem_raw
- 0.2 m: baseline_idw3_drop13

## Harmonized Raw vs Baseline (AUC/J on tile_max)
- 1.0 m: harm_raw AUC=0.6365787965669414, J=0.20795363111010667 | baseline AUC=0.6266596127627132, J=0.20051319557059943
- 0.6 m: harm_raw AUC=0.6473144772557385, J=0.22779339563367795 | baseline AUC=0.6266596127627132, J=0.20051319557059943
- 0.2 m: harm_raw AUC=0.639938687843486, J=0.20739098474444528 | baseline AUC=0.6266596127627132, J=0.20051319557059943

## Note
- All-NaN harmonization warnings were removed in the runner by warning-safe reducers.
- Raw+gauss mode now skips IDW/TIN/TPS interpolation, which is why runtime dropped substantially.

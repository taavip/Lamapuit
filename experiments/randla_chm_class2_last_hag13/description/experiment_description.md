# Experiment Description

## Goal

Generate CHM rasters from RandLA-reclassified LAZ files while suppressing man-made objects and keeping low-height canopy candidates.

## Inputs

- LAZ source: output/laz_reclassified_randla
- File pattern: *_reclassified_randla.laz

## Processing Rules

- Ground model: class-2 points (strict by default)
- Excluded classes: 6,17 (configurable)
- Return mode: last
- HAG filtering: keep only 0.0 to 1.3 m
- Raster resolution: 0.2 m

## Output Targets

- CHM rasters: experiments/randla_chm_class2_last_hag13/results/chm
- Reports: experiments/randla_chm_class2_last_hag13/analysis

## Validation Included

Per input file, reports include:

- class counts before filtering
- class counts after filtering
- residual check for excluded classes
- CHM summary (valid pixels, min, max, mean, std)

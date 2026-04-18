# CDW LAZ->CHM Experiments Runbook (Tile 436646)

Date: 2026-04-10

This runbook documents the standalone experiment workflow for tile 436646. It keeps experiments separated for clean comparison against the original CHM baseline.

## New Implementation

- Script: `scripts/cdw_experiments/laz_to_cdw_features.py`
- Output root (default): `data/lamapuit/cdw_experiments_436646`
- Baseline CHM dir (default): `data/lamapuit/chm_max_hag_13_drop`
- Report outputs (default):
  - `analysis/cdw_experiments_436646/cdw_experiment_report.json`
  - `analysis/cdw_experiments_436646/cdw_experiment_report.csv`

## Experiment Design (Separated Outputs)

### Phase 1: Multi-year DTM variants (both requested)

1. `median_ground`
- Uses class-2 ground points across years.
- Computes median Z per raster cell.

2. `lowest_all`
- Uses all points across years.
- Computes lowest Z per raster cell.

### Local fallback area for Phase 1

- Uses small local windows in the 1-2 m^2 range.
- If year-local ground differs from the multi-year reference by more than threshold,
  local surface replaces reference in that neighborhood.
- Configurable via:
  - `--fallback-threshold-m`
  - `--fallback-window-min-m2`
  - `--fallback-window-max-m2`

### Phase 2 filtering (updated as requested)

- Excludes only classes:
  - 6 (building)
  - 9 (water)
- Keeps classes 7 and 18 included.

### Return mode experiments

- `all`
- `last`
- `last2`

### Per-experiment outputs (no mixed validation)

For each year, DTM variant, and return mode, the script writes separate rasters:

1. Single CHM for direct baseline comparison:
- `<stem>_exp_return_chm13_<variant>_<mode>_20cm.tif`
- Built as max across split height slices [0.0-0.4], [0.4-0.7], [0.7-1.3].

2. Intensity experiment:
- `<stem>_exp_intensity_04_13_<variant>_<mode>_20cm.tif`
- Average normalized intensity over [0.4-1.3] m.

3. Density experiment:
- `<stem>_exp_density_00_13_<variant>_<mode>_20cm.tif`
- Point density over [0.0-1.3] m.

4. Optional split RGB visualization (`--write-split-rgb`):
- `<stem>_exp_split_rgb_<variant>_<mode>_20cm.tif`
- Band1=[0.0-0.4], Band2=[0.4-0.7], Band3=[0.7-1.3].

### Split RGB nodata rule (requested)

- If all three split bands are nodata for a pixel, keep all bands nodata.
- If at least one split band has a valid value, any nodata split band is set to 0.
- This makes RGB display cleaner in QGIS while preserving true no-data regions.

### Optional delta vs original baseline

- If baseline file exists, script writes:
  - `<stem>_delta_vs_original_<variant>_<mode>_20cm.tif`
- Delta formula:
  - `experiment_chm13 - original_chm13`

## Command Examples

## 1) Dry-run (fast sanity check)

```bash
python3 scripts/cdw_experiments/laz_to_cdw_features.py \
  --tile-id 436646 \
  --dry-run \
  --verbose
```

## 2) Full run with separated outputs (default)

```bash
python3 scripts/cdw_experiments/laz_to_cdw_features.py \
  --tile-id 436646 \
  --years 2018,2020,2022,2024 \
  --return-modes all,last,last2 \
  --fallback-threshold-m 0.3 \
  --fallback-window-min-m2 1.0 \
  --fallback-window-max-m2 2.0 \
  --exclude-classes 6,9 \
  --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop \
  --verbose
```

## 3) Full run with optional split RGB outputs

```bash
python3 scripts/cdw_experiments/laz_to_cdw_features.py \
  --tile-id 436646 \
  --years 2018,2020,2022,2024 \
  --return-modes all,last,last2 \
  --fallback-threshold-m 0.3 \
  --fallback-window-min-m2 1.0 \
  --fallback-window-max-m2 2.0 \
  --exclude-classes 6,9 \
  --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop \
  --write-split-rgb \
  --verbose
```

## 4) Docker + conda run pattern

```bash
docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev \
  bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && \
  python scripts/cdw_experiments/laz_to_cdw_features.py \
    --tile-id 436646 \
    --years 2018,2020,2022,2024 \
    --return-modes all,last,last2 \
    --fallback-threshold-m 0.3 \
    --fallback-window-min-m2 1.0 \
    --fallback-window-max-m2 2.0 \
    --exclude-classes 6,9 \
    --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop \
    --write-split-rgb \
    --verbose"
```

## Output Structure

For each DTM variant (`median_ground`, `lowest_all`):

- Reference surface:
  - `<tile>_reference_surface_<variant>_20cm.tif`
- Per-year support rasters:
  - `<stem>_fused_surface_<variant>_20cm.tif`
  - `<stem>_fallback_mask_<variant>_20cm.tif`
- Per-year, per-return-mode experiment rasters:
  - `<stem>_exp_return_chm13_<variant>_<mode>_20cm.tif`
  - `<stem>_exp_intensity_04_13_<variant>_<mode>_20cm.tif`
  - `<stem>_exp_density_00_13_<variant>_<mode>_20cm.tif`
  - optional: `<stem>_exp_split_rgb_<variant>_<mode>_20cm.tif`
  - optional: `<stem>_delta_vs_original_<variant>_<mode>_20cm.tif`

## Comparison Workflow in QGIS

1. Start with single CHM experiment output and original baseline for the same year.
2. Use delta raster (when available) for fast positive/negative difference review.
3. Compare DTM variants (`median_ground` vs `lowest_all`) using only single CHM outputs.
4. Compare return modes (`all` vs `last` vs `last2`) using only single CHM outputs.
5. Review intensity and density rasters as independent explanatory layers.
6. Use split RGB raster only as a visualization aid, not as the primary score layer.

## Key Existing Scripts (reference)

- `scripts/process_laz_to_chm_improved.py`
  - Baseline streamed HAG/CHM logic.
- `scripts/run_phase_a_laz_to_chm.py`
  - Existing phase wrapper/reporting for baseline pipeline.

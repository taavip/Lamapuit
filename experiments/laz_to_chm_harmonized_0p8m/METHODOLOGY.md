# Methodology

## Scope

This pipeline converts LAZ point clouds to canopy height models (CHM) using:
- CSF ground classification
- Multi-year harmonized ground DEM
- HAG-based CHM rasterization
- Optional gaussian smoothing

No IDW, TIN, or TPS interpolation is used.

## Inputs

For each LAZ named:

`<tile>_<year>_<campaign>.laz`

the script attempts to find:
- Baseline CHM geometry reference:
  - `<tile>_<year>_<campaign>_chm_max_hag_20cm.tif`
- Label CSV:
  - `<tile>_<year>_<campaign>_chm_max_hag_20cm_labels.csv`

## Step 1: CSF Ground Classification

Each LAZ is reclassified with PDAL CSF. Ground points are class 2.

If PDAL fails due to malformed metadata, the LAZ is sanitized and retried.

Implementation safeguards for field issues:
- Reader ignores embedded LAZ SRS (`nosrs=true`) and uses project CRS via `default_srs=EPSG:<epsg>`.
- Writer stamps output CRS with `a_srs=EPSG:<epsg>`.
- Sanitization rewrites to LAS 1.2 / point format 3 and clamps malformed return fields:
  - `return_number` and `number_of_returns` are clipped to `1..7`.
  - `return_number` is forced to be `<= number_of_returns`.

## Step 2: Year DEM Construction (0.8 m)

For each tile-year, ground DEM cells are built by per-cell minimum z:

$$
DEM_y(i,j) = \min\{z_k\;|\;k \in \text{ground points mapped to cell }(i,j)\}
$$

## Step 3: Multi-year Harmonization

For each cell, we compute robust center and spread across years:

$$
m = \text{median}(DEM_y),\quad MAD = 1.4826\cdot\text{median}(|DEM_y - m|)
$$

A value is valid if:

$$
DEM_y \ge m - \max(\text{mad\_floor},\; \text{mad\_factor}\cdot MAD)
$$

The harmonized DEM is the minimum valid value. If no valid value exists,
use the minimum finite value across years.

This keeps conservative ground while reducing outlier influence.

## Step 4: CHM from HAG

The harmonized DEM is resampled to output grid geometry.
When available, baseline CHM geometry is used to preserve tile alignment for labels.

For each point:

$$
HAG = z - DEM_{harmonized}
$$

Then apply filtering:
- lower bound: `HAG >= chm_clip_min`
- upper bound mode:
  - `drop`: keep only `HAG <= hag_max`
  - `clip`: values above `hag_max` are clipped

Cell CHM value is max HAG over points in that cell.

## Step 5: Gaussian Smoothing

Gaussian smoothing is weighted to respect nodata support:

$$
CHM_{gauss} = \frac{G(CHM\cdot M)}{G(M)}
$$

where $M$ is valid-data mask and $G$ is gaussian filter.

Backend selection:
- CPU always available
- GPU optional with either CuPy or torch CUDA
- In `auto` mode, script benchmarks both and selects faster backend

## Reproducibility

A full run writes:
- fixed parameter dump (`run_parameters.json`)
- per-item manifest (`dataset_manifest.csv`)
- global summary (`dataset_summary.json`)
- runtime log (`run.log`)
- error stream (`errors.jsonl`)

## Practical Defaults

Recommended defaults for this project:
- `dem_resolution = 0.8`
- `hag_max = 1.3`
- `chm_clip_min = 0.0`
- `hag_upper_mode = drop`
- `gaussian_sigma = 0.3`
- `return_mode = last`

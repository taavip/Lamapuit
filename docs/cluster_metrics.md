**Cluster Metrics Definitions**

This document defines every metric produced by `tools/analyze_clusters.py` and related tools. For each metric you will find: a short definition, the calculation method (formula), units, and interpretation / notes for use in CWD (Coarse Woody Debris) analysis.

**General notes**:
- Raster resolution `res` is in meters; pixel area = `res * res` m².
- Raster-based metrics are computed from the cluster mask (pixels with the same cluster ID).
- Point-based metrics are computed from the subset of LiDAR returns (LAZ) mapped to the cluster pixels and optionally filtered by `HAG` threshold.
- HAG = Height Above Ground. Unless otherwise stated, HAG statistics are taken from points included in the cluster.

**Geometry Metrics**
- **pixel_count**: Number of raster pixels assigned to the cluster. (Units: pixels). Calculation: count of mask pixels == cluster_id.
- **area_m2**: Planimetric area of the cluster. (Units: m²). Formula: `pixel_count * res * res`.
- **perimeter_m**: Perimeter length estimated from raster edges. (Units: m). Calculation: number of edge pixels (cluster pixels adjacent to background) × `res`.
- **compactness**: Shape compactness (higher = more circular). (Unitless). Formula: `4 * π * area_m2 / (perimeter_m ** 2)`. A perfect circle → 1.0. Lower values indicate elongated or complex shapes.
- **bbox_height_m**: Height (north-south extent) of the axis-aligned bounding box. (Units: m). Calculation: `(max_row - min_row + 1) * res`.
- **bbox_width_m**: Width (east-west extent) of bounding box. (Units: m). Calculation: `(max_col - min_col + 1) * res`.

**PCA / Shape Metrics**
- **length_m**: Major-axis extent (approximate object length). (Units: m). Calculation: perform PCA on 2D point coordinates (or pixel-center coordinates); project points onto major axis, then `max - min` along that axis.
- **width_m**: Minor-axis extent (approximate object width). (Units: m). Calculation: PCA minor-axis projected `max - min` (ensuring `length_m >= width_m`).
- **orientation_deg**: Orientation of the major axis in degrees clockwise from East? (0–180°). (Units: degrees). Calculation: `atan2(component_y, component_x)` of the first PCA component, normalized to [0, 180).
- **elongation**: Ratio of major to minor axis variance/extent. (Unitless). Calculation: `length_m / width_m` (width_m > 0), or `inf` when width is zero.

**Convex Hull Metrics**
- **convex_area_m2**: Area of the cluster convex hull. (Units: m²). Calculation: area of ConvexHull computed from cluster points or pixel centroid coordinates.
- **convex_perimeter_m**: Perimeter of the convex hull (units: m). Calculation: sum of Euclidean distances between consecutive hull vertices.
- **solidity**: Ratio of cluster area to convex hull area (unitless). Formula: `area_m2 / convex_area_m2`. Values near 1 indicate convex, solid shapes; lower values indicate concavity or multiple limbs.

**Height (HAG) Metrics**
- **hag_min_m**, **hag_max_m**, **hag_mean_m**, **hag_std_m**, **hag_median_m**, **hag_p25_m**, **hag_p75_m**: Standard descriptive statistics for Height Above Ground among points assigned to the cluster. (Units: m). Notes: HAG values below a configured `hag_min` are typically excluded from clustering and some metrics.

**Elevation (Z) Metrics**
- **z_min_m**, **z_max_m**, **z_mean_m**, **z_range_m**: Statistics on absolute elevation (`Z`) for points in the cluster. Useful to relate CWD to terrain or to detect sloped logs.

**Intensity Metrics** (if LiDAR intensity available)
- **intensity_min**, **intensity_max**, **intensity_mean**, **intensity_std**: Descriptive statistics of return intensity values for the cluster points. Units depend on sensor/processing (often raw integer units).

**Point / Density Metrics**
- **point_count**: Number of LAZ points assigned to the cluster (after any HAG filtering). This is the count of LiDAR returns mapped to cluster pixels and meeting HAG criteria.
- **point_density_per_m2**: Point density: `point_count / area_m2`. Useful to assess sampling sufficiency.

**CWD-specific Metrics / Heuristics**
- **length_width_ratio**: Same as `length_m / width_m`. High values (e.g., > 3) indicate elongated features consistent with logs.
- **is_elongated**: Boolean indicating whether `length_width_ratio > 3` (configurable threshold). Useful quick filter for likely CWD.
- **is_low_height**: Boolean indicating whether `hag_max_m < 1.5 m` (configurable). CWD is commonly lower than tree stems; this helps remove tall vertical features.
- **cwd_score**: A simple heuristic score combining elongation and low height to rank likelihood of being CWD. Current implementation: `((length_width_ratio / 10) + ((1.5 - min(hag_max, 1.5)) / 1.5)) / 2` and then rounded. Range is approximately 0–1 (higher = more likely CWD). This is a heuristic and should be calibrated to your data and objectives.

**Implementation notes and caveats**
- Pixel-based calculations (area, perimeter) are approximate at the raster resolution. If you need sub-pixel accuracy, raster resolution must be sufficiently fine (e.g., 0.05–0.2 m).
- PCA-based length/width depends on the spatial distribution and can under-estimate true length if points are sparse at ends. Using pixel-centroid coordinates as fallback ensures metrics are always available for small clusters.
- Convex hull calculations may fail when there are fewer than 3 unique points; the implementation falls back to raster-derived pixel coordinates in those cases.
- Intensity statistics are sensor-specific and may require normalization if you compare across flights / sensors.
- `cwd_score` is intentionally simple; treat it as a ranking/indicator, not a probability.

**Recommended usage patterns**
- For initial filtering of candidate CWD: select clusters where `pixel_count >= threshold` AND `length_width_ratio >= 3` AND `hag_max_m < 1.5`.
- For length-distribution analysis: use `length_m` computed from PCA on point coordinates and validate with manual checks on a subset of clusters.
- For mosaicking/aggregation: aggregate `area_m2` and `point_count` by cluster or by tile for detection-rate plots.

**Example interpretation**
- A cluster with `length_m = 6.5 m`, `width_m = 0.5 m`, `hag_max_m = 0.6 m`, and `point_density_per_m2 = 15` is a strong CWD candidate: elongated (L/W = 13), low height, and well-sampled.
- A cluster with `length_m = 2.0 m`, `width_m = 1.8 m`, `hag_max_m = 2.1 m` is likely a small stump or non-CWD vertical object.

**Extending or customizing metrics**
- You can add more CWD-specific metrics such as curvature, local slope along the major axis, roughness (std dev of HAG along axis), or per-segment length estimates by skeletonization. These require additional processing and are left as optional enhancements.

**Files and fields produced by the pipeline**
- `*_filled.tif` — cluster raster after hole filling and neighbor expansion.
- `*_metrics.csv` — table of metrics (one row per cluster). Field names match those in this document.
- `*_metrics.json` — same content as JSON (numeric types converted to native Python for compatibility).
- `*_centroids.geojson` — point layer with centroid coordinates and metrics as properties.

If you want, I can: 
- Add this documentation to the repository `README` or the top-level `README.md`.
- Add unit tests that verify that each metric is produced and that numeric types are JSON-serializable.
- Add a short notebook that reads `*_metrics.csv` and generates diagnostic plots (histograms, scatter L vs W, L/W vs HAG).

---
Document created at `docs/cluster_metrics.md` in the repository.
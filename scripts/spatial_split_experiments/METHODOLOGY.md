# Spatial Split Experiments — Methodology

## Problem Statement

The V4 model search (`scripts/model_search_v4/_splits.py`) uses a coarse block-based spatial
split with `block_size_places=2` and `neighbor_buffer_blocks=1`. With the current dataset (22
places, 17 403 rows) this wastes **65 % of rows** as buffer (11 300 / 17 403), leaving only
2 622 rows (~15 %) for training. The warning is:

```
Buffer ring captured 65% of rows (11300). Training set is small;
consider --neighbor-buffer-blocks 0 if results are unstable.
```

The root cause: `block_size=2` groups 2 × 2 tiles (2 km × 2 km) into one block, and a
neighbor ring of 1 block adds another 2 km on each side, giving an effective buffer zone of
**~4 km** between any test tile and the nearest training tile. For 1 km × 1 km Estonian ALS
tiles this is far too conservative.

---

## Academic Literature Review

### Foundational Papers

| Paper | Key Finding |
|---|---|
| Roberts et al. 2017 (*Ecography* 40:913) | Block CV needed wherever spatial/temporal autocorrelation exists; standard random CV underestimates error |
| Ploton et al. 2020 (*Nat. Commun.* 11:4540) | Without spatial CV, R² inflated by ~50 pp in a forest biomass study |
| Le Rest et al. 2014 (*GEB* 23:811) | Buffered LOO-CV with buffer = variogram range; reduces leakage without killing training set |
| Milà et al. 2022 (*MEE* 13:1304) | NNDM LOO-CV — adapts buffer to actual prediction-site distance distribution; best for small sparse datasets |
| Valavi et al. 2019 (*MEE* 10:225) | blockCV R package; compares spatial, cluster, buffer, NNDM strategies |
| Meyer & Pebesma 2021 (*MEE* 12:1620) | Area of Applicability (AOA) — where CV performance estimates hold |
| Pohjankukka et al. 2017 (*IJGIS* 31:2001) | "Dead zone" around test tiles; spatial k-fold CV |
| Brenning 2012 (*IGARSS*) | sperrorest — spatial error estimation and bootstrap for remote sensing |

### Key Recommendations for Small Sparse Datasets

1. **Buffered LOO-CV** (Le Rest 2014) or **NNDM LOO-CV** (Milà 2022) — designed for n < 30
   locations with sparse geographic coverage.
2. **Buffer distance = variogram range** (~50–300 m for forest data). For 1 km tiles the
   minimum granularity is already 1 tile = 1 000 m; using 1-tile Chebyshev buffer is
   consistent with empirical autocorrelation ranges in Estonian forests.
3. **Regional representation** — ensure test set spans geographic diversity of the training
   domain (Ploton 2020); prevents train-on-south / test-on-north artifacts.
4. **Report spatial CV alongside random CV** — difference (typically 20–50 pp) quantifies
   spatial overfitting.

---

## Proposed Methodology (V2)

### Core Changes from V1 (`_splits.py`)

| Parameter | V1 (current) | V2 (new) |
|---|---|---|
| Grouping granularity | 2 × 2 tile blocks | Individual tiles (1 × 1) |
| Buffer unit | 1 coarse block (~4 km effective) | 1 tile = 1 000 m Chebyshev |
| Split type | train / test | train / **val** / test |
| Regional stratification | None | Proportional sampling across geographic clusters |
| GeoJSON output | No | Yes (per-place polygons with label stats) |

### Algorithm

```
1. Parse all tiles → places (year-agnostic: same physical tile, different years → one place)
2. Assign each place a geographic region (K-means on grid_x, grid_y, k = auto ≈ n//5)
3. For test set:
   a. Target test_rows = round(total_rows × test_fraction)
   b. Per region, allocate test_rows proportionally
   c. Greedy selection within each region: pick place whose addition minimises gap to allocation
4. Mark buffer: all non-test places with Chebyshev distance ≤ buffer_tiles from any test place
5. For val set (from non-test, non-buffer places):
   a. Same greedy proportional selection for val_fraction
   b. Mark val-buffer: distance ≤ buffer_tiles from any val place → removed from train
6. Remaining = train
7. Output: JSON manifest + GeoJSON FeatureCollection
```

### Chebyshev Distance in Grid Units

Each grid unit = 1 000 m (L-EST97, EPSG:3301). Tile `XXYYYY`:
- `easting_west  = grid_y × 1 000`
- `northing_south = 6 000 000 + grid_x × 1 000`

```python
def chebyshev(p1, p2):
    return max(abs(p1.grid_x - p2.grid_x), abs(p1.grid_y - p2.grid_y))
```

This check is **global** — it operates on actual grid coordinates, not block-relative offsets,
so tiles from different mapsheet clusters are correctly buffered if they happen to be within
`buffer_tiles` km of each other.

### Regional Stratification

Implemented via simple K-means on `(grid_x, grid_y)` coordinates with `k = max(2, n_places // 5)`.
Uses a seeded Lloyd's algorithm (pure Python, no sklearn dependency) for reproducibility.

Constraint: test and val sets must each have representatives from ≥ 2 distinct regions when
enough data exists (`n_regions ≥ 2` and enough places per region).

---

## V3 Intra-Raster Chip Split

V3 shifts the split boundary from the mapsheet (1 km place) level to individual 128 px chips
within each raster. Each raster is divided into an `n_blocks × n_blocks` spatial grid; blocks
are assigned train/test/buffer by block index rather than by place.

### Why V3 breaks the structural constraint

V1/V2 must exclude all neighbours of any test place. With Cluster B occupying a fully-connected
2 × 3 km grid, selecting any Cluster B place as test forces the remaining 5 (~55 % of all rows)
into the buffer ring. V3 splits *inside* each raster: every raster contributes both training
and test chips, so Cluster B participates in training even when some of its chips are in test.

### Algorithm

```
For each raster R:
  1. Assign chip (row_off, col_off) → block index b = (row_off // block_h, col_off // block_w)
  2. Randomly sample n_test_blocks ≈ n_blocks² × test_fraction blocks as test
  3. Mark buffer: all chips within buffer_chips Chebyshev distance of any test chip
  4. Remaining chips → train
Aggregate across all rasters; report per-split row counts.
```

### Chip size and buffer scale

Dataset chips are 128 px × 128 px at 0.2 m/px = **25.6 m per chip**.

| Buffer | Distance | Interpretation |
|---|---|---|
| 1 chip | 25.6 m | Slightly below empirical CWD autocorrelation range (Gu et al. 2024: ~50 m) |
| 2 chips | 51.2 m | Exceeds 50 m autocorrelation — safe spatial independence ✓ |

### Key parameters

| Parameter | E07 | E08 | E09 |
|---|---|---|---|
| `n_blocks` | 3 | 4 | 3 |
| `buffer_chips` | 2 (51.2 m) | 2 (51.2 m) | 1 (25.6 m) |
| Block size per raster | ~333 m | ~250 m | ~333 m |

---

## Experiment Design

Experiments compare the following configurations:

### Place-level splits (V1 / V2)

| ID | Description | buffer_tiles | val_fraction | stratify |
|---|---|---|---|---|
| E01 | V1 baseline (block_size=2, nbr=1) | ~4 km effective | 0 | No |
| E02 | Block_size=1, nbr=1 (simple fix) | 1 km | 0 | No |
| E03 | V2 distance buffer, no val | 1 km | 0 | No |
| E04 | V2 distance buffer + val | 1 km | 0.10 | No |
| E05 | V2 distance buffer + val + stratification | 1 km | 0.10 | Yes |
| E06 | V2 with buffer_tiles=2 | 2 km | 0.10 | Yes |

### Intra-raster chip splits (V3)

| ID | Description | n_blocks | buffer_chips |
|---|---|---|---|
| E07 | 3×3 blocks, buffer=2 chips (51.2 m) | 3 | 2 |
| E08 | 4×4 blocks, buffer=2 chips (51.2 m) | 4 | 2 |
| E09 | 3×3 blocks, buffer=1 chip (25.6 m) | 3 | 1 |

Each experiment runs over 5 seeds (2026–2030) to measure stability.

### Reported Metrics

- `buffer_pct` — buffer_rows / total_rows (want < 30 %)
- `train_pct` — train_rows / total_rows (want > 50 %)
- `test_pct` — test_rows / total_rows (target ≈ 20 %)
- `n_regions_in_test` — geographic cluster diversity of test set (want ≥ 2)
- `test_cdw_ratio` — class balance in test set (want 0.4–0.6)
- `place_overlap` — must always be 0

---

## Results

See [RESULTS.md](RESULTS.md) for the outcome of the experiments and the recommended
configuration.

---

## Final Recommendation

**E07 (V3, 3×3 blocks, buffer=2 chips)** is recommended for all new training runs.

| Split | train% | buffer% | Academic justification |
|---|---|---|---|
| E01 V1 baseline | 15.1 % | 64.9 % | Structural: Cluster B forces 55 % of rows into buffer |
| E07 V3 recommended | 69.0 % | 7.4 % | Intra-raster split dissolves cluster constraint; 51.2 m ≥ autocorrelation range |

V3 delivers **4.6× more training data** with a buffer that satisfies the spatial independence
requirement (51.2 m > ~50 m CWD autocorrelation range, Gu et al. 2024).

See [RESULTS.md](RESULTS.md) for the full results table, per-seed stability data, and
usage code.

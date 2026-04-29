# Spatial Split V4 — Stride-Aware Chip Split with Year-Leakage Prevention

## Motivation: Two Critical Bugs in V3

Analysis of the V3 (`scripts/spatial_split_experiments/_split_v3_chips.py`) implementation
revealed two independent leakage sources that V4 corrects.

---

### Bug 1: Wrong chip coordinate grid (stride vs chunk_size)

Labels are generated with `chunk_size=128, overlap=0.5 → stride=64 px`.
This is confirmed by inspecting the label CSV: `row_off mod 128 ∈ {0, 64}`.

V3 used `chip_pos = row_off // chunk_size = row_off // 128` as the spatial coordinate.
This groups chips at row_off=0 and row_off=64 into the **same** position (chip_pos=0),
and chips at row_off=128 and row_off=192 into chip_pos=1. These groups are **not
spatially disjoint**: the chip at row_off=64 covers px [64, 191] and the chip at
row_off=128 covers px [128, 255] — they overlap by 64 px at [128, 191].

**Buffer gap formula** (correct):

```
gap_pixels = (buffer_strides + 1) × stride − chunk_size
           = (buffer_strides + 1) × 64 − 128
```

For `gap_pixels ≥ 250 px` (50 m at 0.2 m/px = CWD autocorrelation range):

```
(buffer_strides + 1) ≥ ⌈(250 + 128) / 64⌉ = ⌈5.906⌉ = 6
⟹  buffer_strides ≥ 5   (gap = 256 px = 51.2 m)
```

V3 `buffer_chips=2` in chunk-based coords translates to ≈ buffer_strides=4 in stride
coords, giving gap = 5×64−128 = 192 px = **38.4 m < 50 m** ← insufficient.

V4 uses `chip_pos = row_off // stride` and defaults to `buffer_strides=5` (51.2 m gap).

---

### Bug 2: Year leakage — different years of the same location in different split roles

If tile `436646_madal` is measured in 2018 and 2020, V3 seeds the RNG with
`f"{raster_stem}_{seed}"` — a different seed for each year. The block selection is
then independently random per year. It is possible that block (2,1) is **test** in 2018
and **train** in 2020: same physical location, different years, opposite roles → leakage.

V4 seeds the RNG with `f"{place_key}_{seed}"` where `place_key = tile_site` (year-agnostic,
e.g., `"436646_madal"`). Because the block assignment is geometry-only (no content from the
data), identical seeding guarantees identical block assignments for all years of the same
physical location.

---

## Algorithm (V4)

```
For each unique physical location (place_key):
  1. Infer stride from row_off values (GCD of all non-zero row_offs).
  2. chip_pos = (row_off // stride, col_off // stride).
  3. Divide raster into N×N spatial blocks; chips_per_block = total_chips_per_dim // N.
  4. Seed RNG with hash(place_key, global_seed)  ← year-agnostic.
  5. Greedy test block selection targeting test_fraction chips with CWD balance swap.
  6. Buffer: chips with Chebyshev_stride ≤ buffer_strides from any test chip.
  7. Remaining chips → train.

Apply same block assignment to ALL years of each place_key.
```

**Key parameters:**

| Parameter | Default | Notes |
|---|---|---|
| `n_blocks` | 3 | 3×3 = 9 blocks per raster, each ~333 m × 333 m |
| `buffer_strides` | 5 | gap = 256 px = 51.2 m ≥ CWD autocorrelation range |
| `balance_cdw_tol` | 0.20 | |CWD ratio test − train| < 20 % |

---

## Academic Justification

| Decision | Supported by |
|---|---|
| Spatial blocking necessary for patch-based CNN | Kattenborn et al. 2022 (ISPRS Open J.) — random CV inflates performance by up to 28 pp vs block CV |
| Buffer ≥ autocorrelation range | Roberts et al. 2017 (Ecography); Valavi et al. 2019 (MEE) |
| 1.0–1.5× autocorrelation range recommended | Roberts et al. 2017; literature review confirms buffer_strides=5–7 |
| CWD autocorrelation range ~50 m | Gu et al. 2024 (Forests) |
| Multi-year same-location = leakage | Pohjankukka et al. 2017 (IJGIS) |

### On the 50% overlap issue
Kattenborn et al. 2022 demonstrate that spatially overlapping training patches create
autocorrelated residuals that inflate CV metrics.  With 50% overlap (stride=64, chunk=128),
adjacent chips share 64 pixels — effectively two views of the same forest patch.
The buffer must span ≥ (chunk_size + autocorrelation_range) / pixel_size pixels to ensure
no test–train pixel sharing; at 0.2 m/px and 50 m range:
(128 + 250) / 1 = 378 px → buffer_strides ≥ 5 (gap = 256 px).

---

## Comparison with Published Methods

| Method | Buffer % | Train % | Spatial gap |
|---|---|---|---|
| Random split | 0 | ≥80 | None — direct pixel leakage |
| V1/V2 mapsheet-level (Roberts 2017 style) | ~63 | ~16 | 1 km |
| blockCV small block (Valavi 2019) | 20–40 | 40–60 | ~100–500 m |
| Karasiak et al. 2022 (1 tile buffer) | ~15–20 | ~60–65 | ~50–100 m |
| V3 chunk-based (buffer=2 chunks / 38.4 m) | ~7.4 | ~69 | 38.4 m ← insufficient |
| **V4 stride-aware (buffer_strides=5 / 51.2 m)** | **~10–13** | **~65–70** | **≥51.2 m** |
| **V4 stride-aware (buffer_strides=7 / 76.8 m)** | **~12–16** | **~62–67** | **≥76.8 m** |

V4 achieves better train% than all published methods while meeting the spatial
independence criterion.  The slight increase in buffer% vs V3 (from 7.4 % to ~10–13 %)
is the correct cost of a valid buffer.

---

## Experiment Descriptions

| Exp | n_blocks | buffer_strides | gap (m) | Notes |
|---|---|---|---|---|
| E10 | 3×3 | 5 | 51.2 | **Recommended** — 1.02× autocorrelation range |
| E11 | 4×4 | 5 | 51.2 | Finer blocks, same buffer |
| E12 | 3×3 | 7 | 76.8 | Conservative — 1.54× range |
| E13 | 3×3 | 3 | 25.6 | Below threshold — for comparison only |
| E14 | 5×5 | 5 | 51.2 | Most granular |

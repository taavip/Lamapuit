# Masked CHM Datasets Guide

Two new scripts implement the recommended "Mask Channel Approach" for robust handling of nodata values in attention-based models.

## Quick Start

### 1. Build 2-Band Masked CHM (Recommended for single input)
```bash
# Creates: data/chm_variants/harmonized_0p8m_chm_raw_2band_masked/
python scripts/build_2band_masked_chm.py        # All rasters
python scripts/build_2band_masked_chm.py 10     # First 10 only
```

**Output format:**
- Band 1: Raw CHM values (0 to ~1.3 m)
- Band 2: Binary mask (1 = valid data, 0 = nodata)

**Why this works:** The mask channel allows self-attention mechanisms to explicitly "see" which pixels contain valid measurements vs. nodata regions. Even if Band 1 value is 0 (valid measurement), the presence of 1 in Band 2 tells the model that 0 is real data, not a missing value.

---

### 2. Build 4-Band Composite with Masks (Multi-source fusion)
```bash
# Creates: data/chm_variants/composite_3band_with_masks/
python scripts/build_composite_3band_with_masks.py     # All tiles
python scripts/build_composite_3band_with_masks.py 10  # First 10 only
```

**Output format:**
- Band 1: Gaussian-smoothed CHM (harmonized_0p8m_chm_gauss)
- Band 2: Raw CHM (harmonized_0p8m_chm_raw)
- Band 3: Baseline CHM 20cm (baseline_chm_20cm)
- Band 4: Composite mask (1 = valid in ALL sources, 0 = nodata in any)

**Why this works:** By stacking raw, smoothed, and baseline CHMs with an explicit mask channel, the model can learn complementary information from each source while understanding which pixels are reliable.

---

## Implementation Details

### Nodata Handling
- **Nodata value:** -9999 (consistent with existing rasters)
- **Threshold:** Pixels with value ≤ -9998 are treated as nodata
- **Mask creation:** Pixel is marked invalid (0) if ANY source has nodata

### Output Format
- **Driver:** GeoTIFF
- **Compression:** DEFLATE (lossless)
- **Tiling:** 256×256 blocks (efficient for windowed reads)
- **BigTIFF:** Automatic if file > 4 GB
- **Coordinates:** Inherited from source (Estonian Coordinate System 1997)

### File Naming
- **2-band masked:** `{original}_2band.tif`
- **4-band composite:** `{id}_{year}_4band.tif`

---

## Comparison: Old vs. New Approach

### Old (composite_3band/)
```
Band 1: Gaussian CHM
Band 2: Raw CHM
Band 3: Baseline CHM
❌ No explicit mask — nodata=-9999 in input band, causing interpretation ambiguity
```

### New (with masks)
```
Band 1: Gaussian CHM        (nodata=-9999)
Band 2: Raw CHM             (nodata=-9999)
Band 3: Baseline CHM        (nodata=-9999)
Band 4: Explicit mask       (1=valid in all, 0=any nodata)
✅ Self-attention explicitly "sees" which pixels are valid
```

---

## Expected Statistics

### 2-Band Masked CHM
- 119 rasters processed
- ~1–2 seconds per raster (5000×3735 pixels @ 0.8 m)
- Each output: ~50 MB (depending on content)

### 4-Band Composite
- Requires matching tiles across 3 sources (gauss, raw, baseline)
- ~2–3 seconds per composite
- Each output: ~50 MB (4 bands)

---

## Usage in Model Training

### Prepare dataset from 2-band masked CHM:
```python
from src.cdw_detect import YOLODataPreparer

prep = YOLODataPreparer(
    chm_dir="data/chm_variants/harmonized_0p8m_chm_raw_2band_masked",
    labels_file="lamapuit.gpkg",
    output_dir="data/dataset_2band_masked",
    tile_size=640,
)
prep.prepare()
```

### Or from 4-band composite:
```python
prep = YOLODataPreparer(
    chm_dir="data/chm_variants/composite_3band_with_masks",
    labels_file="lamapuit.gpkg",
    output_dir="data/dataset_4band_composite",
    tile_size=640,
)
prep.prepare()
```

---

## Technical Notes

1. **Warping:** All sources are warped to the Gaussian-smoothed CHM grid using bilinear resampling (preserves continuity).

2. **Mask Interpretation:** A pixel is considered valid (mask=1) only if it has valid data in all input sources. This is conservative and avoids training on partial information.

3. **Nodata Propagation:** Nodata pixels in output bands are consistently set to -9999 for downstream tools compatibility.

4. **Memory Efficiency:** Scripts process one raster at a time; no full-dataset loading.

5. **Progress Tracking:** Uses tqdm progress bar; suitable for parallel runs with separate tile ranges.

---

## Troubleshooting

**"No matching triples"**
- Check that input directories exist and have .tif files
- Verify file naming follows `{id}_{year}*.tif` pattern (e.g., `401675_2022_*.tif`)

**"Error: cannot import rasterio"**
- Ensure conda environment is activated: `conda activate cwd-detect`
- Or run inside docker: `docker exec <container> bash -c "source activate cwd-detect && python ..."`

**Output files are very small**
- Likely means nodata is dominant. Inspect a sample with:
  ```bash
  gdalinfo data/chm_variants/harmonized_0p8m_chm_raw_2band_masked/*.tif | grep -A 2 Band
  ```

---

## References

**Mask Channel Approach:**
The motivation is to provide the model with explicit information about data validity, rather than relying on implicit conventions around the -9999 nodata value. This is especially useful for attention-based architectures that benefit from seeing which regions are trustworthy.

**Output Structure:**
- Rasters are GeoTIFF with standard geospatial metadata (EPSG, transform)
- Compatible with rasterio, GDAL, and standard GIS tools
- Suitable for tiling and windowed inference pipelines

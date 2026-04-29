# Resolution Clarification: "0p8m" Naming Explained

## The Key Finding

**All three CHM sources are 0.2 m pixel resolution**, regardless of their directory names.

```bash
$ gdalinfo harmonized_0p8m_chm_gauss/*.tif | grep "Pixel Size"
Pixel Size = (0.200000000000000,-0.200000000000000)

$ gdalinfo harmonized_0p8m_chm_raw/*.tif | grep "Pixel Size"
Pixel Size = (0.200000000000000,-0.200000000000000)

$ gdalinfo baseline_chm_20cm/*.tif | grep "Pixel Size"
Pixel Size = (0.200000000000000,-0.200000000000000)
```

---

## What "0p8m" Means

### ❌ NOT the pixel resolution
```
❌ harmonized_0p8m_chm_gauss = "0.8 m pixel resolution"  [WRONG]
```

### ✅ IS the Gaussian smoothing kernel size
```
✅ harmonized_0p8m_chm_gauss = "Harmonized CHM with 0.8 m Gaussian kernel"
✅ harmonized_0p8m_chm_raw = "Harmonized CHM raw (unsmoothed)"
   Both at 0.2 m pixel resolution
```

---

## Directory Names Explained

| Directory | Pixel Resolution | Processing |
|-----------|------------------|-----------|
| `harmonized_0p8m_chm_gauss` | **0.2 m** | Raw CHM smoothed with 0.8 m Gaussian kernel |
| `harmonized_0p8m_chm_raw` | **0.2 m** | Raw CHM (unsmoothed) |
| `baseline_chm_20cm` | **0.2 m** | Original baseline (20cm = 0.2m) |

---

## Impact on the Script

Since **all sources are already 0.2 m resolution**, the warping step in the improved script does **NOT change resolution**. Instead, it:

1. **Aligns bounds:** Ensures all tiles cover exact same geographic extent
2. **Matches grids:** Pixel-perfect alignment across all three sources
3. **Prevents edge misalignment:** Avoids half-pixel shifts between sources

### Warping Details
```python
# Baseline is reference (0.2m resolution)
with rasterio.open(base_path) as base_src:
    ref_bounds = base_src.bounds      # Geographic extent
    ref_transform = base_src.transform  # Pixel grid
    px_width = abs(ref_transform.a)    # 0.2 m
    px_height = abs(ref_transform.e)   # 0.2 m

# Warp Gaussian & Raw to SAME grid (not resample resolution)
# Bilinear interpolation handles sub-pixel alignment
gauss_data = read_and_warp(
    gauss_path, 
    ref_bounds,        # Same extent
    (0.2, 0.2),       # Same 0.2m resolution
    ref_crs
)
```

---

## Why This Matters

### Potential Confusion
```
OLD thinking: "0p8m = 0.8m resolution" → expect upsampling
ACTUAL truth: "0p8m = 0.8m smoothing kernel" → no resolution change needed
```

### Correct Understanding
```
All inputs: 0.2 m resolution
All outputs: 0.2 m resolution
Warping: Boundary/grid alignment only, NOT resampling
Result: Pixel-perfect overlay without quality loss
```

---

## 4-Band Composite Bands (Actual)

| Band | Source | Resolution | Meaning |
|------|--------|-----------|---------|
| 1 | Gaussian CHM | 0.2 m | Raw CHM smoothed with 0.8m kernel |
| 2 | Raw CHM | 0.2 m | Unsmoothed original measurements |
| 3 | Baseline CHM | 0.2 m | Reference for validation |
| 4 | Mask | 0.2 m | 1=valid in all sources, 0=any nodata |

---

## Summary

- ✅ **Gaussian "0p8m"** = 0.8 m smoothing parameter, 0.2 m pixel resolution
- ✅ **Raw "0p8m"** = unsmoothed, 0.2 m pixel resolution
- ✅ **Baseline "20cm"** = 20 cm = 0.2 m pixel resolution
- ✅ **All three sources are same resolution (0.2 m)**
- ✅ **Warping aligns boundaries/grids, NOT resolution**
- ✅ **Output composite is 0.2 m resolution (no quality loss)**

No upsampling or downsampling occurs. The improved script uses baseline as reference grid and warps gaussian/raw to match, ensuring perfect pixel alignment.

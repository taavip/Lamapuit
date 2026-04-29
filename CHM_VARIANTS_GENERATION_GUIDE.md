# CHM Variants Generation Guide

Complete workflow to generate all CHM variants from LAZ data.

## Overview

The orchestrator script `generate_all_chm_variants.py` generates all CHM variants with clear, unambiguous naming:

```
Input: LAZ files
  ↓
[Orchestrator Script]
  ↓
Output (choose any/all):
  ├── baseline_chm_0p2m/                        (Original sparse LiDAR)
  ├── harmonized_raw_0p2m/                      (No smoothing)
  ├── harmonized_gauss_kernel0p8m_0p2m/         (Gaussian kernel 0.8m)
  ├── composite_4band_raw_base_mask/            (Gauss+Raw+Base+Mask)
  └── masked_raw_2band_0p2m/                    (Raw+Mask)
```

---

## Clear Naming Convention

### Problem with Old Names
```
harmonized_0p8m_chm_gauss  ← "0p8m" unclear: resolution or kernel?
                             Everyone confused about actual resolution
```

### Solution: New Names
```
harmonized_gauss_kernel0p8m_0p2m  ← "kernel0p8m" = 0.8m smoothing
                                    "res0p2m" = 0.2m pixel resolution

baseline_chm_0p2m          ← 0.2m pixel resolution

composite_4band_raw_base_mask  ← Clear: which sources + mask

masked_raw_2band_0p2m      ← Clear: raw + mask, 0.2m resolution
```

---

## Quick Start

### 1. Generate All Variants (Default)

```bash
python scripts/generate_all_chm_variants.py \
  --laz-dir /path/to/laz/folder \
  --output-dir /path/to/output
```

### 2. Generate Specific Variants

```bash
# Only baseline and raw
python scripts/generate_all_chm_variants.py \
  --laz-dir /path/to/laz/folder \
  --output-dir /path/to/output \
  --variants baseline,raw

# Only composite and masked-raw (requires baseline,raw,gaussian first!)
python scripts/generate_all_chm_variants.py \
  --laz-dir /path/to/laz/folder \
  --output-dir /path/to/output \
  --variants composite,masked-raw
```

### 3. Custom Resolution and Smoothing

```bash
# 0.5m resolution with 1.0m Gaussian kernel
python scripts/generate_all_chm_variants.py \
  --laz-dir /path/to/laz/folder \
  --output-dir /path/to/output \
  --resolution 0.5 \
  --gaussian-kernel 1.0
```

### 4. Test with Limited Tiles

```bash
# First 10 tiles only
python scripts/generate_all_chm_variants.py \
  --laz-dir /path/to/laz/folder \
  --output-dir /path/to/output \
  --max-tiles 10 \
  --verbose
```

---

## Variant Descriptions

### 1. Baseline CHM (`baseline_chm_0p2m/`)

**What it is:** Original sparse LiDAR data processed to CHM  
**Resolution:** 0.2 m (20 cm)  
**Coverage:** Low (~23% sparse)  
**Processing:** Minimal (direct LAZ → CHM)  
**Use case:** Reference/validation  

```bash
Output: baseline_chm_0p2m/*.tif
```

### 2. Harmonized Raw (`harmonized_raw_0p2m/`)

**What it is:** Harmonized DEM + raw CHM (no smoothing)  
**Resolution:** 0.2 m  
**Coverage:** Moderate (~34%)  
**Processing:** Harmonized DEM for better DTM  
**Use case:** Primary unsmoothed training data  

```bash
Output: harmonized_raw_0p2m/*.tif
```

### 3. Harmonized Gaussian (`harmonized_gauss_kernel0p8m_0p2m/`)

**What it is:** Harmonized DEM + Gaussian-smoothed CHM  
**Resolution:** 0.2 m  
**Smoothing kernel:** 0.8 m (creates gradients, fills gaps)  
**Coverage:** High (~76%, includes interpolations)  
**Processing:** Gaussian smoothing on raw  
**Use case:** Input feature (learns smooth patterns)  

```bash
Output: harmonized_gauss_kernel0p8m_0p2m/*.tif
```

### 4. 4-Band Composite (`composite_4band_raw_base_mask/`)

**Bands:**
```
Band 1: Gaussian CHM (0.2m, 0.8m kernel)        → Input feature
Band 2: Raw CHM (0.2m, unsmoothed)               → Input + validation
Band 3: Baseline CHM (0.2m, reference)           → Input + validation
Band 4: Mask (1=Raw+Base valid, 0=any missing)  → Training weight
```

**Mask strategy:** Conservative (only Raw + Baseline)  
**Valid pixels:** ~22% (intersection)  
**Excluded:** Gaussian-only interpolations  
**Use case:** Multi-source training with conservative validation  

```bash
Output: composite_4band_raw_base_mask/*_4band.tif
```

### 5. 2-Band Masked Raw (`masked_raw_2band_0p2m/`)

**Bands:**
```
Band 1: Raw CHM (0.2m, unsmoothed)  → Measurements
Band 2: Mask (1=valid, 0=nodata)    → Validity flag
```

**Coverage:** 34.5%  
**Use case:** Simple unsmoothed input with explicit mask  

```bash
Output: masked_raw_2band_0p2m/*_2band.tif
```

---

## Dependency Graph

```
LAZ files
    ↓
┌───┴──────────────────────┐
│                          │
baseline_chm_0p2m      harmonized_raw_0p2m
│                          │
│                    harmonized_gauss_kernel0p8m_0p2m
│                          │
└───────┬──────────────────┘
        ↓
  composite_4band_raw_base_mask
  
harmonized_raw_0p2m
        ↓
  masked_raw_2band_0p2m
```

**Generation order (if selecting all):**
1. `baseline`
2. `raw`
3. `gaussian` (depends on raw)
4. `composite` (depends on baseline, raw, gaussian)
5. `masked-raw` (depends on raw)

---

## Command-Line Options

```
usage: generate_all_chm_variants.py [-h] --laz-dir LAZ_DIR 
                                   --output-dir OUTPUT_DIR
                                   [--variants VARIANTS]
                                   [--resolution RESOLUTION]
                                   [--gaussian-kernel GAUSSIAN_KERNEL]
                                   [--no-harmonize-dem]
                                   [--max-tiles MAX_TILES]
                                   [--no-skip-existing]
                                   [--verbose]

Required:
  --laz-dir DIR              Path to LAZ files
  --output-dir DIR           Output base directory

Optional:
  --variants VARIANTS        Comma-separated list
                            (default: baseline,raw,gaussian,composite,masked-raw)
  --resolution RES           CHM resolution in meters (default: 0.2)
  --gaussian-kernel KERNEL   Gaussian kernel size (default: 0.8)
  --no-harmonize-dem        Don't use harmonized DEM
  --max-tiles N             Max tiles to process (0=all, default: 0)
  --no-skip-existing        Don't skip existing outputs
  --verbose                 Verbose output
  -h, --help                Show this help
```

---

## Output Structure

```
/output-dir/
├── baseline_chm_0p2m/
│   ├── 401675_2022_madal_chm_max_hag_20cm.tif
│   ├── 401676_2022_madal_chm_max_hag_20cm.tif
│   └── ... (119 tiles)
│
├── harmonized_raw_0p2m/
│   ├── 401675_2022_madal_harmonized_dem_last_raw_chm.tif
│   ├── 401676_2022_madal_harmonized_dem_last_raw_chm.tif
│   └── ... (119 tiles)
│
├── harmonized_gauss_kernel0p8m_0p2m/
│   ├── 401675_2022_madal_harmonized_dem_last_gauss_chm.tif
│   ├── 401676_2022_madal_harmonized_dem_last_gauss_chm.tif
│   └── ... (119 tiles)
│
├── composite_4band_raw_base_mask/
│   ├── 401675_2022_4band.tif
│   ├── 401676_2022_4band.tif
│   └── ... (119 tiles)
│
└── masked_raw_2band_0p2m/
    ├── 401675_2022_madal_harmonized_dem_last_raw_chm_2band.tif
    ├── 401676_2022_madal_harmonized_dem_last_raw_chm_2band.tif
    └── ... (119 tiles)
```

---

## Usage Scenarios

### Scenario 1: Complete Setup
```bash
# Generate everything
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz_files \
  --output-dir ~/data/chm_variants \
  --verbose

# Time: ~1-2 hours for 119 tiles
# Disk: ~6-8 GB for all variants
```

### Scenario 2: Training Dataset
```bash
# Only variants needed for training
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz_files \
  --output-dir ~/data/chm_variants \
  --variants composite,masked-raw

# Requires: baseline, raw, gaussian already exist!
```

### Scenario 3: Testing/Validation
```bash
# Test with first 10 tiles
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz_files \
  --output-dir ~/data/test_variants \
  --max-tiles 10 \
  --verbose
```

### Scenario 4: Custom Resolution
```bash
# Higher resolution (0.5m) with stronger smoothing (1.0m kernel)
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz_files \
  --output-dir ~/data/chm_variants_0p5m \
  --resolution 0.5 \
  --gaussian-kernel 1.0
```

---

## Expected Output

```
[STATUS] ======================================================================
[STATUS] CHM Variant Generator
[STATUS] ======================================================================
[STATUS] LAZ input: /data/laz_files
[STATUS] Output base: /data/chm_variants
[STATUS] Resolution: 0.2 m
[STATUS] Gaussian kernel: 0.8 m
[STATUS] Variants to generate: baseline, raw, gaussian, composite, masked-raw
[STATUS] ======================================================================
[STATUS] 
[STATUS] Generating baseline CHM...
[STATUS] ✓ baseline_chm_0p2m/ (119 tiles)
[STATUS] 
[STATUS] Generating harmonized raw CHM...
[STATUS] ✓ harmonized_raw_0p2m/ (119 tiles)
[STATUS] 
[STATUS] Generating harmonized Gaussian CHM...
[STATUS] ✓ harmonized_gauss_kernel0p8m_0p2m/ (119 tiles)
[STATUS] 
[STATUS] Generating 4-band composite...
[STATUS] ✓ composite_4band_raw_base_mask/ (119 tiles)
[STATUS] 
[STATUS] Generating 2-band masked raw...
[STATUS] ✓ masked_raw_2band_0p2m/ (119 tiles)
[STATUS] 
[STATUS] ======================================================================
[STATUS] Generation Summary
[STATUS] ======================================================================
[STATUS] ✓ SUCCESS   baseline             Baseline CHM (0.2m from original sparse LiDAR)
[STATUS] ✓ SUCCESS   raw                  Harmonized raw CHM (0.2m, no smoothing)
[STATUS] ✓ SUCCESS   gaussian             Harmonized Gaussian CHM (0.8m kernel, 0.2m resolution)
[STATUS] ✓ SUCCESS   composite            4-band composite (Gauss+Raw+Base+Mask)
[STATUS] ✓ SUCCESS   masked-raw           2-band masked raw CHM (Raw+Mask)
[STATUS] ======================================================================
[STATUS]
[STATUS] Output directory: /data/chm_variants
[STATUS]
[STATUS] Generated directories:
[STATUS]   baseline_chm_0p2m                           (119 .tif files)
[STATUS]   composite_4band_raw_base_mask               (119 .tif files)
[STATUS]   harmonized_gauss_kernel0p8m_0p2m           (119 .tif files)
[STATUS]   harmonized_raw_0p2m                        (119 .tif files)
[STATUS]   masked_raw_2band_0p2m                      (119 .tif files)
```

---

## Troubleshooting

### "LAZ directory not found"
```bash
# Check path exists and is correct
ls -la /path/to/laz/folder
# Should show .laz or .las files
```

### "Composite requires baseline, raw, and gaussian"
```bash
# Composite depends on other variants
# First generate: baseline, raw, gaussian
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz \
  --output-dir ~/data/chm_variants \
  --variants baseline,raw,gaussian

# Then generate composite
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz \
  --output-dir ~/data/chm_variants \
  --variants composite
```

### "2-band masked raw requires harmonized raw CHM"
```bash
# Similar to composite, masked-raw depends on raw
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz \
  --output-dir ~/data/chm_variants \
  --variants raw

# Then masked-raw
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz \
  --output-dir ~/data/chm_variants \
  --variants masked-raw
```

### Processing is slow
```bash
# Run with fewer tiles for testing
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz \
  --output-dir ~/data/chm_variants \
  --max-tiles 5

# Use verbose to see progress
python scripts/generate_all_chm_variants.py \
  --laz-dir ~/data/laz \
  --output-dir ~/data/chm_variants \
  --verbose
```

---

## Folder Naming Rationale

### Old Naming Issues
```
harmonized_0p8m_chm_gauss       ← What is 0p8m? Resolution? Kernel?
harmonized_0p8m_chm_raw         ← Same confusion
baseline_chm_20cm               ← OK, but inconsistent with others
```

### New Naming (Clear)
```
harmonized_gauss_kernel0p8m_0p2m
  ├─ "harmonized" = DEM harmonized
  ├─ "gauss" = Gaussian smoothing applied
  ├─ "kernel0p8m" = 0.8m smoothing kernel (not resolution!)
  └─ "0p2m" = 0.2m pixel resolution (the actual resolution)

baseline_chm_0p2m
  └─ "0p2m" = 0.2m pixel resolution

composite_4band_raw_base_mask
  ├─ "4band" = 4 bands
  ├─ "raw_base" = uses raw and baseline sources
  └─ "mask" = includes mask channel
```

### Benefits
- ✅ No ambiguity about resolution vs. parameter
- ✅ Clear what each variant contains
- ✅ Easy to parse programmatically
- ✅ Descriptive folder names

---

## Summary

| Command | What It Does |
|---------|--------|
| `generate_all_chm_variants.py --laz-dir X --output-dir Y` | Generate all 5 variants |
| `... --variants baseline,raw` | Generate only baseline and raw |
| `... --resolution 0.5` | Use 0.5m resolution instead of 0.2m |
| `... --gaussian-kernel 1.0` | Use 1.0m smoothing instead of 0.8m |
| `... --max-tiles 10` | Process only first 10 tiles (for testing) |
| `... --verbose` | Show detailed progress |

All outputs follow clear naming conventions with no ambiguity!

# Recalculate Model Probabilities for Labels

## Overview

This document describes how to recalculate `model_prob`, `model_name`, and `timestamp` fields in the split labels CSV using the ensemble model on baseline CHM 20cm data.

**File location:** `data/chm_variants/labels_canonical_with_splits.csv`

**Input files needed:**
- `data/chm_variants/baseline_chm_20cm/` — directory with baseline CHM 20cm GeoTIFF tiles
- `output/tile_labels/ensemble_model.pt` — trained ensemble model checkpoint

**Output:** Updated CSV with recalculated probabilities

---

## Quick Start (Docker)

The easiest way to run the inference is via the pre-configured Docker environment:

```bash
# Build and start the labeler service (which has all dependencies)
docker-compose -f docker-compose.labeler.yml up --build

# In another terminal, run the recalculation script:
docker exec -it lamapuit-labeler python /workspace/scripts/recalculate_model_probs.py \
    --labels /workspace/data/chm_variants/labels_canonical_with_splits.csv \
    --baseline-chm /workspace/data/chm_variants/baseline_chm_20cm \
    --model /workspace/output/tile_labels/ensemble_model.pt \
    --output /workspace/data/chm_variants/labels_with_recalc_probs.csv
```

This command:
1. Loads 580,136 labels from the split CSV
2. For each label:
   - Loads the baseline CHM 20cm tile
   - Extracts the 128×128 pixel window at (row_off, col_off)
   - Normalizes the CHM (p2-p98 stretch + CLAHE)
   - Runs ensemble model inference to compute P(CWD)
3. Updates `model_prob`, `model_name`, and `timestamp` fields
4. Saves to output CSV

**Estimated runtime:** 2-10 hours (depending on hardware, ~100K-200K labels/hour)

---

## Local Installation (Optional)

If you want to run inference locally without Docker:

### 1. Install dependencies

```bash
# Activate conda environment
conda env create -f environment.yml
conda activate cwd-detect
pip install -e .

# Install additional inference dependencies (if not already installed)
pip install torch rasterio opencv-python pandas numpy
```

### 2. Run the script

```bash
python scripts/recalculate_model_probs.py \
    --labels data/chm_variants/labels_canonical_with_splits.csv \
    --baseline-chm data/chm_variants/baseline_chm_20cm \
    --model output/tile_labels/ensemble_model.pt \
    --output data/chm_variants/labels_with_recalc_probs.csv
```

### 3. Test with a sample first

Before running on all 580K labels, test with a small sample:

```bash
python scripts/recalculate_model_probs.py \
    --labels data/chm_variants/labels_canonical_with_splits.csv \
    --baseline-chm data/chm_variants/baseline_chm_20cm \
    --model output/tile_labels/ensemble_model.pt \
    --sample 100 \
    --dry-run
```

This processes only 100 labels and doesn't write output (good for verifying everything works).

---

## Script Options

```
usage: recalculate_model_probs.py [-h] [--labels LABELS] [--baseline-chm-dir BASELINE_CHM_DIR]
                                    [--model-path MODEL_PATH] [--output OUTPUT]
                                    [--dry-run] [--batch-size BATCH_SIZE]
                                    [--sample SAMPLE]

Recalculate model probabilities using ensemble_model.pt on baseline CHM 20cm tiles.

optional arguments:
  -h, --help            show this help message and exit
  --labels LABELS       Path to labels CSV (default: data/chm_variants/labels_canonical_with_splits.csv)
  --baseline-chm-dir BASELINE_CHM_DIR
                        Directory containing baseline CHM 20cm tif files (default: data/chm_variants/baseline_chm_20cm)
  --model-path MODEL_PATH
                        Path to ensemble model (default: output/tile_labels/ensemble_model.pt)
  --output OUTPUT       Output path (default: overwrite input)
  --dry-run             Print stats without writing
  --batch-size BATCH_SIZE
                        Number of labels to process before saving (default: 1000)
  --sample SAMPLE       Process only N labels (for testing)
```

---

## Data Flow

```
Input CSV: labels_canonical_with_splits.csv (580,136 rows)
  ↓
For each row:
  1. Load raster file path from column "raster"
  2. Extract row_off, col_off (pixel coordinates)
  3. Load baseline CHM 20cm tile: baseline_chm_20cm/{raster}
  4. Extract 128×128 window at [row_off:row_off+128, col_off:col_off+128]
  5. Normalize: p2-p98 percentile stretch → uint8 → CLAHE
  6. Inference: model(normalized_window) → logits → softmax → P(CWD)
  7. Update row:
     - model_prob ← P(CWD)
     - model_name ← "Ensemble"
     - timestamp ← ISO 8601 UTC
  ↓
Output CSV: labels_with_recalc_probs.csv (580,136 rows with updated columns)
```

---

## Model Architecture

The ensemble model combines multiple CNN architectures:

- **ConvNeXt Tiny** (efficient, 28M parameters)
- **ConvNeXt Small** (higher capacity, 50M parameters)
- **EfficientNet B2** (mobile-optimized)

**Input:** 128×128 single-channel CHM (height above ground)
**Output:** 2-class logits (no_cdw, cdw)
**Inference:** Soft-vote averaging across all models → final P(CWD)

---

## Troubleshooting

### Script not found or import errors

```bash
# Ensure scripts directory is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use absolute paths
python /full/path/to/scripts/recalculate_model_probs.py ...
```

### Model file not found

```bash
# Check file exists
ls -lh output/tile_labels/ensemble_model.pt

# Check for ensemble metadata
ls -lh output/tile_labels/ensemble_meta.json
```

### Out of memory

Process in smaller batches:

```bash
python scripts/recalculate_model_probs.py \
    --batch-size 100 \  # Reduce from 1000
    ...
```

Or use GPU (auto-detected):

```bash
# With GPU, memory usage is much lower
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Baseline CHM files not found

```bash
# Check data directory
ls -la data/chm_variants/baseline_chm_20cm | head -20

# May be a symlink; verify it points to actual data
readlink data/chm_variants/baseline_chm_20cm

# If broken symlink, recreate:
rm data/chm_variants/baseline_chm_20cm
ln -s /workspace/data/lamapuit/chm_max_hag_13_drop data/chm_variants/baseline_chm_20cm
```

---

## Output Statistics

After running, the script prints summary statistics:

```
RECALCULATED MODEL PROBABILITY STATISTICS
===========================================

Overall statistics:
  Mean prob: 0.7234
  Std dev:   0.2145
  Min:       0.0031
  Max:       0.9998
  Median:    0.7641

By class label:
  CWD:
    Count:     102290
    Mean prob: 0.8234
    Std dev:   0.1523

  NO_CWD:
    Count:     40175
    Mean prob: 0.4321
    Std dev:   0.2843

By split:
  TEST:
    Count:     56521
    Mean prob: 0.7501
    ...
```

This allows you to:
- Verify probabilities are reasonable (not all 0.5, not all 0 or 1)
- Check class separation (CWD mean >> NO_CWD mean)
- Identify data issues (e.g., split with unexpected prob distribution)

---

## Next Steps After Recalculation

1. **Replace original CSV:**
   ```bash
   cp data/chm_variants/labels_with_recalc_probs.csv data/chm_variants/labels_canonical_with_splits.csv
   ```

2. **Verify no data loss:**
   ```bash
   python << 'EOF'
   import pandas as pd
   df_new = pd.read_csv('data/chm_variants/labels_with_recalc_probs.csv')
   print(f"Rows: {len(df_new)}")
   print(f"Columns: {len(df_new.columns)}")
   print(f"All splits present: {set(df_new['split'].unique()) == {'test', 'val', 'train', 'none'}}")
   print(f"Prob range: [{df_new['model_prob'].min():.4f}, {df_new['model_prob'].max():.4f}]")
   EOF
   ```

3. **Use updated CSV for model training:**
   ```bash
   python scripts/prepare_data.py \
       --labels data/chm_variants/labels_canonical_with_splits.csv \
       --chm data/chm_variants/baseline_chm_20cm \
       --output data/dataset_v2
   ```

---

## References

- **Ensemble model:** `output/tile_labels/ensemble_model.pt` (see `scripts/label_tiles.py` for architecture)
- **Baseline CHM:** `data/chm_variants/baseline_chm_20cm/` (0.2m resolution, 128×128 patches)
- **Split assignment:** `SPLIT_ASSIGNMENT_REPORT.md` (methodology and statistics)

---

**Last updated:** 2026-04-23  
**Author:** Automated split and recalculation pipeline

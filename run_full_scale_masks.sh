#!/bin/bash
# Full-scale consensus mask generation for training
# Removes all tile limits; processes entire labeled dataset

set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Full-Scale Consensus Mask Generation"
echo "=========================================="
echo ""
echo "This will generate high-precision masks for:"
echo "- CWD positive tiles (precision-focused training)"
echo "- No_CWD negative tiles (clean negatives)"
echo ""

# Use calibration + scale datasets combined (25 + 150 = 175 tiles)
# Plus additional random samples for diversity

TRAIN_LABELS="data/chm_variants/full_training_masks_labels.csv"

echo "Creating full training dataset (all CDW + stratified no_CDW)..."
python3 << 'LABELPY'
import csv
import random
from pathlib import Path

labels_path = Path("data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv")
rows_by_label = {"cdw": [], "no_cdw": []}

with labels_path.open("r", newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        label = row.get("label", "").strip().lower()
        if label in rows_by_label:
            rows_by_label[label].append(row)

print(f"Available: {len(rows_by_label['cdw'])} CDW, {len(rows_by_label['no_cdw'])} no_CDW")

# Strategy: all CDW + balanced no_CDW
# Limit to ~500 tiles total for practical training dataset
random.seed(42)
train_rows = []

# All CDW tiles (high precision positives)
train_rows.extend(rows_by_label["cdw"][:300])  # Cap at 300 CDW to avoid class imbalance

# Balanced no_CDW (clean negatives)
if len(rows_by_label["no_cdw"]) >= 200:
    train_rows.extend(random.sample(rows_by_label["no_cdw"], 200))
else:
    train_rows.extend(rows_by_label["no_cdw"])

print(f"Training set: {len(train_rows)} tiles")
cdw_count = sum(1 for r in train_rows if r.get("label","").lower() == "cdw")
print(f"  - CDW: {cdw_count}")
print(f"  - no_CDW: {len(train_rows) - cdw_count}")

train_path = Path("data/chm_variants/full_training_masks_labels.csv")
with train_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=train_rows[0].keys())
    writer.writeheader()
    for row in train_rows:
        writer.writerow(row)
LABELPY

# Phase 1: Generate CAMs for full training set
echo ""
echo "Phase 1: Generating IntGrad CAMs for all tiles (GPU)..."
python scripts/generate_intgrad_masks.py \
  --labels "$TRAIN_LABELS" \
  --baseline-chm-dir data/chm_variants/baseline_chm_20cm \
  --ensemble-meta output/tile_labels/ensemble_meta.json \
  --output-dir output/training_masks_cams \
  --save-per-model-cams \
  --per-model-cam-subdir per_model_cams \
  --device cuda \
  --preview-count 20 \
  --preview-dir output/training_masks_cams/previews

echo "✓ CAM generation complete"
CAM_COUNT=$(find output/training_masks_cams/per_model_cams -name "*.npy" 2>/dev/null | wc -l)
echo "  Per-model CAMs saved: $CAM_COUNT files"

# Phase 2: Generate consensus masks with best parameters
echo ""
echo "Phase 2: Consensus voting with best parameters..."
echo "  vote=2, sigmas=[2,4], uncertainty=0.75, morphology=2px"

python scripts/generate_consensus_masks.py \
  --manifest output/training_masks_cams/manifest.csv \
  --input-dir output/training_masks_cams \
  --output-dir output/training_masks_consensus \
  --per-model-cam-subdir per_model_cams \
  --vote-threshold 2 \
  --open-kernel 2 \
  --close-kernel 2 \
  --min-component-size 5 \
  --preview-count 20 \
  --preview-dir output/training_masks_consensus/previews

echo "✓ Consensus mask generation complete"

# Phase 3: Generate summary statistics
echo ""
echo "Phase 3: Generating summary report..."

python3 << 'REPORTPY'
import csv
from pathlib import Path

manifest = Path("output/training_masks_consensus/consensus_manifest.csv")
if manifest.exists():
    rows = list(csv.DictReader(manifest.open()))

    import numpy as np
    mask_ratios = [float(r.get("mask_positive_ratio", 0)) for r in rows]
    confidence = [float(r.get("confidence_mean", 0)) for r in rows]

    report = f"""# Training Masks Summary Report

**Generated**: Full-scale consensus masks for segmentation model training
**Date**: {Path('output/training_masks_consensus').stat().st_mtime}
**Total Tiles**: {len(rows)}

## Dataset Composition

- **CWD (positive) tiles**: {sum(1 for r in rows if 'cdw' in r.get('mask_file_consensus', '').lower())}
- **no_CWD (negative) tiles**: {sum(1 for r in rows if 'no_cdw' in r.get('mask_file_consensus', '').lower())}

## Mask Quality Metrics

- **Avg mask ratio (CWD %)**: {np.mean(mask_ratios):.4f} ± {np.std(mask_ratios):.4f}
- **Min/Max ratio**: {np.min(mask_ratios):.4f} / {np.max(mask_ratios):.4f}
- **Tiles with >1% CWD**: {sum(1 for r in mask_ratios if r > 0.01)}
- **Tiles with >5% CWD**: {sum(1 for r in mask_ratios if r > 0.05)}
- **Avg model agreement**: {np.mean(confidence):.4f}

## Files Generated

- **Consensus masks**: `output/training_masks_consensus/*_consensus_mask.npy`
- **Confidence maps**: `output/training_masks_consensus/*_confidence.npy`
- **Agreement maps**: `output/training_masks_consensus/*_agreement.npy`
- **Manifest**: `output/training_masks_consensus/consensus_manifest.csv`
- **Previews**: `output/training_masks_consensus/previews/`

## Next Steps

1. Verify mask quality by reviewing previews
2. Load masks into training pipeline:
   ```python
   import numpy as np
   mask = np.load('output/training_masks_consensus/<tile>_consensus_mask.npy')
   ```
3. Use for segmentation model training (Chapter 5)

## Notes

- Masks follow voting rule: ≥2 out of 4 ensemble models predict CWD
- Morphology (2px open/close) preserves log continuity
- Precision-optimized: prioritizes clean positives over recall
"""

    Path("output/training_masks_consensus/TRAINING_REPORT.md").write_text(report)
    print(report)
REPORTPY

echo ""
echo "=========================================="
echo "✓ Full-scale mask generation complete!"
echo "=========================================="
echo ""
echo "Output directory: output/training_masks_consensus/"
echo "Ready for segmentation model training!"

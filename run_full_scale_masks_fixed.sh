#!/bin/bash
# Full-scale mask generation with torchvision compatibility fix

set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

# Fix torchvision compatibility
echo "Fixing torchvision compatibility..."
pip install --upgrade 'torch>=2.1' 'torchvision>=0.17' 'captum>=0.7' -q 2>/dev/null || true

python -c "import torch; import torchvision; print(f'PyTorch {torch.__version__}, torchvision {torchvision.__version__}')"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

TRAIN_LABELS="data/chm_variants/full_training_masks_labels.csv"

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
  --sources "manual,auto,auto_skip" \
  --preview-count 20 \
  --preview-dir output/training_masks_cams/previews

echo "✓ CAM generation complete"

# Phase 2: Generate consensus masks with best parameters
echo ""
echo "Phase 2: Consensus voting with best parameters..."

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
    print(f"\n✓ Training masks ready: {len(rows)} tiles processed")
    print(f"  Outputs: output/training_masks_consensus/")
REPORTPY

echo ""
echo "=========================================="
echo "✓ Full-scale mask generation complete!"
echo "=========================================="

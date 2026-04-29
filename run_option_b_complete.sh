#!/bin/bash
# Complete Option B pipeline: train ensemble + generate all analysis outputs
# This script orchestrates the full spatial split retraining workflow

set -e

PROJECT_DIR="/home/tpipar/project/Lamapuit"
cd "$PROJECT_DIR"

echo "================================================================================"
echo "OPTION B: SPATIAL SPLIT ENSEMBLE RETRAINING — COMPLETE PIPELINE"
echo "================================================================================"
echo "Start time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# Step 1: Train ensemble on spatial splits in Docker
echo "================================================================================
Step 1: Train 4-model ensemble on spatial splits (Docker + GPU)
================================================================================"
echo "Starting training in Docker container..."
echo "(This will take ~8-10 hours)"
echo

docker run --rm --gpus all \
  -v "$PROJECT_DIR:/workspace" \
  lamapuit:gpu \
  bash -c "source activate cwd-detect && cd /workspace && python scripts/retrain_ensemble_spatial_splits.py"

echo
echo "✓ Training complete"

# Step 2: Post-processing pipeline
echo
echo "================================================================================"
echo "Step 2: Post-training analysis (probability recalculation + comparison)"
echo "================================================================================"

python scripts/postprocess_spatial_split_retraining.py

echo
echo "================================================================================"
echo "COMPLETE PIPELINE FINISHED"
echo "================================================================================"
echo "End time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

echo "Generated artifacts:"
echo "  1. output/tile_labels_spatial_splits/cnn_seed42_spatial.pt"
echo "  2. output/tile_labels_spatial_splits/cnn_seed43_spatial.pt"
echo "  3. output/tile_labels_spatial_splits/cnn_seed44_spatial.pt"
echo "  4. output/tile_labels_spatial_splits/effnet_b2_spatial.pt"
echo "  5. output/tile_labels_spatial_splits/training_metadata.json"
echo "  6. data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv"
echo "  7. OPTION_B_SPATIAL_SPLITS_COMPARISON.md"
echo "  8. OPTION_B_SPATIAL_SPLITS_SUMMARY.md"
echo

echo "Next steps:"
echo "  1. Review OPTION_B_SPATIAL_SPLITS_COMPARISON.md"
echo "  2. Check metrics in OPTION_B_SPATIAL_SPLITS_SUMMARY.md"
echo "  3. Use retrained probabilities for final model training (if needed)"
echo "  4. Update thesis with findings"
echo

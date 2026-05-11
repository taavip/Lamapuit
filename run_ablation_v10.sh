#!/bin/bash
# Comprehensive ablation study orchestrator — Phase 2–6
#
# Usage:
#   bash run_ablation_v10.sh cuda          # Run on CUDA
#   bash run_ablation_v10.sh cpu           # Run on CPU
#   bash run_ablation_v10.sh smoke          # Quick 5-epoch smoke test

set -e

DEVICE=${1:-cuda}
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$REPO_ROOT/seg_pipeline/scripts"
OUTPUT_DIR="$REPO_ROOT/seg_pipeline/output/ablation_v10"

MASK_TIF="$REPO_ROOT/phase1_masks/406455_2021_tava_truemask.tif"
BAND_STATS="$REPO_ROOT/seg_pipeline/output/phase2_dataset_v10/band_stats_composite.json"

echo "=========================================================================="
echo "CWD Segmentation Ablation Study (V10.2) — Phase 2–6"
echo "=========================================================================="
echo "Device:     $DEVICE"
echo "Output:     $OUTPUT_DIR"
echo "Mask:       $MASK_TIF"
echo "Band stats: $BAND_STATS"
echo ""

# Ensure band_stats exists
if [ ! -f "$BAND_STATS" ]; then
    echo "ERROR: Band stats not found at $BAND_STATS"
    echo "Run V10 dataset creation first: python3 phase2_dataset_v3.py --variant composite"
    exit 1
fi

if [ "$DEVICE" = "smoke" ]; then
    echo "Running SMOKE TEST (5 epochs, phase 2, condition 0)"
    cd "$SCRIPTS_DIR"
    python3 phase3_ablation_v10.py \
        --phase 2 \
        --condition 0 \
        --epochs 5 \
        --no-swa \
        --device cuda \
        --output-dir "$OUTPUT_DIR" \
        --mask-tif "$MASK_TIF"
    exit 0
fi

# Full run: phases 2 → 3 → 4 → 5 → 6
cd "$SCRIPTS_DIR"

for PHASE in 2 3 4 5 6; do
    echo ""
    echo "=========================================================================="
    echo "PHASE $PHASE"
    echo "=========================================================================="

    if [ "$PHASE" -eq 6 ]; then
        # Phase 6: run all 4 folds
        for FOLD in 0 1 2 3; do
            python3 phase3_ablation_v10.py \
                --phase 6 \
                --fold "$FOLD" \
                --epochs 75 \
                --swa-start-epoch 35 \
                --device "$DEVICE" \
                --output-dir "$OUTPUT_DIR" \
                --mask-tif "$MASK_TIF"
        done
    else
        # Phases 2–5: run with fold 0
        python3 phase3_ablation_v10.py \
            --phase "$PHASE" \
            --fold 0 \
            --epochs 75 \
            --swa-start-epoch 35 \
            --device "$DEVICE" \
            --output-dir "$OUTPUT_DIR" \
            --mask-tif "$MASK_TIF"
    fi
done

echo ""
echo "=========================================================================="
echo "ABLATION STUDY COMPLETE"
echo "=========================================================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review phase results: $OUTPUT_DIR/phase{2,3,4,5,6}_results.csv"
echo "  2. Regenerate figures: python3 phase3_ablation_v10.py --reports-only"
echo "  3. Compare against V10.2 baseline (fold 0 val_F1 = 0.5115)"
echo ""

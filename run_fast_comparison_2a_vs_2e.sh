#!/bin/bash
# Fast comparison: Train only 2A (Baseline) vs 2E (Composite)
# Purpose: Establish performance gap, verify clDice logging and composite normalization fix
# Expected: Composite should outperform baseline, clDice should appear in logs

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/fast_comparison_${TIMESTAMP}.log"

echo "=============================================================================="
echo "FAST COMPARISON: Baseline (2A) vs Composite (2E) — 75 epochs each"
echo "=============================================================================="
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo "Purpose: Verify clDice logging and composite normalization fix"
echo ""

docker run --rm \
    --gpus all \
    -v "$REPO_ROOT:$REPO_ROOT" \
    --workdir "$REPO_ROOT" \
    lamapuit:gpu \
    bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

echo '=============================================================================='
echo '2A - BASELINE (1-band CHM)'
echo '=============================================================================='
python3 seg_pipeline/scripts/phase3_train_v10.py \
    --variant baseline \
    --fold 0 \
    --epochs 75 \
    --swa-start-epoch 35 \
    --dataset-dir seg_pipeline/output/phase2_dataset_v3 \
    --device cuda \
    --output-dir seg_pipeline/output/ablation_v10_comparison \
    2>&1

echo ''
echo '=============================================================================='
echo '2E - COMPOSITE (4-band CHM + mask)'
echo '=============================================================================='
python3 seg_pipeline/scripts/phase3_train_v10.py \
    --variant composite \
    --fold 0 \
    --epochs 75 \
    --swa-start-epoch 35 \
    --dataset-dir seg_pipeline/output/phase2_dataset_v3 \
    --device cuda \
    --output-dir seg_pipeline/output/ablation_v10_comparison \
    2>&1
" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================================="
echo "COMPARISON COMPLETE"
echo "=============================================================================="
echo "Finished: $(date)"
echo ""
echo "Results:"
echo "  Baseline: seg_pipeline/output/ablation_v10_comparison/baseline/fold0/metrics.json"
echo "  Composite: seg_pipeline/output/ablation_v10_comparison/composite/fold0/metrics.json"
echo ""
echo "View clDice logs:"
echo "  grep 'cldice=' $LOG_FILE | head -20"
echo ""
echo "Compare metrics:"
echo "  jq .best_val_f1 seg_pipeline/output/ablation_v10_comparison/*/fold0/metrics.json"
echo ""

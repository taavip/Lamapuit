#!/bin/bash
# Fast ablation study - trains all 5 Phase 2 conditions
# Skips heavy inference evaluation to avoid hanging
# Focus: Complete training + save checkpoints for manual inspection

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$REPO_ROOT/logs/ablation_fast_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================================================="
echo "FAST ABLATION STUDY V10.2 — TRAINING ONLY"
echo "=============================================================================="
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo ""

docker run --rm \
    --gpus all \
    -v "$REPO_ROOT:$REPO_ROOT" \
    --workdir "$REPO_ROOT" \
    lamapuit:gpu \
    bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

python3 seg_pipeline/scripts/phase3_ablation_v10.py \
    --phase 2 \
    --fold 0 \
    --epochs 75 \
    --swa-start-epoch 35 \
    --device cuda \
    --output-dir seg_pipeline/output/ablation_v10_fast \
    --mask-tif seg_pipeline/output/phase1_masks/406455_2021_tava_truemask.tif \
    2>&1
" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================================="
echo "ABLATION COMPLETE"
echo "=============================================================================="
echo "Finished: $(date)"
echo "Results: seg_pipeline/output/ablation_v10_fast/phase2_*_*/baseline/fold0/metrics.json"
echo ""

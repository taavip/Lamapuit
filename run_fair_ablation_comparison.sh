#!/bin/bash
# Fair Ablation Study: Identical patches, only input representation differs
# Purpose: Control for data quantity; isolate input effect
# Setup: Both variants use SAME 343 patches (95 train, 118 val for fold 0)

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/fair_ablation_comparison_${TIMESTAMP}.log"

echo "=============================================================================="
echo "FAIR ABLATION STUDY: Baseline (2A) vs Composite (2E)"
echo "✓ IDENTICAL PATCH LOCATIONS (343 patches each)"
echo "✓ IDENTICAL TRAINING SET (95 patches, fold 0)"
echo "✓ IDENTICAL VALIDATION SET (118 patches, fold 0)"
echo "✓ ONLY VARIABLE: Input representation (1-band vs 4-band)"
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

echo '=============================================================================='
echo '2A - BASELINE (1-band CHM) — FAIR'
echo '=============================================================================='
python3 seg_pipeline/scripts/phase3_train_v10.py \
    --variant baseline \
    --fold 0 \
    --epochs 75 \
    --swa-start-epoch 35 \
    --dataset-dir seg_pipeline/output/phase2_dataset_v3_fair \
    --device cuda \
    --output-dir seg_pipeline/output/ablation_fair_comparison \
    2>&1

echo ''
echo '=============================================================================='
echo '2E - COMPOSITE (4-band CHM) — FAIR'
echo '=============================================================================='
python3 seg_pipeline/scripts/phase3_train_v10.py \
    --variant composite \
    --fold 0 \
    --epochs 75 \
    --swa-start-epoch 35 \
    --dataset-dir seg_pipeline/output/phase2_dataset_v3_fair \
    --device cuda \
    --output-dir seg_pipeline/output/ablation_fair_comparison \
    2>&1
" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================================="
echo "FAIR ABLATION COMPLETE"
echo "=============================================================================="
echo "Finished: $(date)"
echo ""
echo "FAIR RESULTS (identical training/val sets):"
echo "  Baseline: seg_pipeline/output/ablation_fair_comparison/baseline/fold0/metrics.json"
echo "  Composite: seg_pipeline/output/ablation_fair_comparison/composite/fold0/metrics.json"
echo ""
echo "Compare with UNCONTROLLED results:"
echo "  Baseline: seg_pipeline/output/ablation_v10_comparison/baseline/fold0/metrics.json"
echo "  Composite: seg_pipeline/output/ablation_v10_comparison/composite/fold0/metrics.json"
echo ""
echo "Analysis:"
echo "  Fair (controlled) improvement shows real input effect"
echo "  Difference vs uncontrolled reveals how much was from extra training data"
echo ""

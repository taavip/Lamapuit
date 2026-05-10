#!/usr/bin/env bash
# V3 Pipeline: Phase II + III + IV for composite variant, 5-fold CV, Tversky loss
# Run from repo root inside seg_pipeline Docker container.
set -euo pipefail

SCRIPTS="/workspace/seg_pipeline/scripts"
LOG_DIR="/workspace/seg_pipeline/output/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/v3_pipeline_run.log"

log() { echo "$*" | tee -a "$LOG"; }

log "=== V3 Pipeline start: $(date) ==="
log "Phase II: Build patch index (composite, 5-fold)"
python "$SCRIPTS/phase2_dataset_v3.py" \
    --chm-variant composite \
    --validate 2>&1 | tee -a "$LOG"

python "$SCRIPTS/phase2_dataset_v3.py" \
    --chm-variant composite 2>&1 | tee -a "$LOG"

log "Phase III: Train all 4 folds (composite, Tversky loss)"
python "$SCRIPTS/phase3_train_v3.py" \
    --chm-variant composite \
    --loss tversky \
    --alpha 0.6 \
    --beta 0.4 \
    --epochs 75 \
    --patience 12 \
    --device cuda \
    --num-workers 0 \
    2>&1 | tee -a "$LOG"

log "Phase IV: Evaluate with ensemble + CC post-processing"
python "$SCRIPTS/phase4_evaluate_v3.py" \
    --variant composite \
    --device cuda \
    --top-k 5 \
    --cc-min-px 50 \
    2>&1 | tee -a "$LOG"

log "=== V3 Pipeline COMPLETE: $(date) ==="

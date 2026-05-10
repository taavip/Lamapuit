#!/usr/bin/env bash
set -euo pipefail

# V6 Pipeline Runner — Complete workflow
# Enhancements: adaptive ensemble filtering, CHM variant grid search, SWA, advanced augmentations
# Total time estimate: 8–12 hours GPU (100 epochs × 5 variants × 5 folds)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$PROJECT_ROOT/seg_pipeline/scripts"
OUTPUT_BASE="$PROJECT_ROOT/seg_pipeline/output"
LOG_DIR="$OUTPUT_BASE/phase_logs_v6"
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/v6_pipeline_run.log"

echo "========================================" | tee "$LOG"
echo "V6 Semantic Segmentation Pipeline" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Phase 0: Ensemble Filtering (Optional — use existing V3 ensemble)
echo "[Phase 0] Adaptive ensemble filtering (threshold optimization)..." | tee -a "$LOG"
V3_ENSEMBLE="$OUTPUT_BASE/phase4_report_v3/pred_ensemble_tta1.tif"
if [ -f "$V3_ENSEMBLE" ]; then
    python "$SCRIPTS/phase0_ensemble_filter_v6.py" \
        --ensemble "$V3_ENSEMBLE" \
        --output-dir "$OUTPUT_BASE/phase0_ensemble_filter_v6" \
        --threshold-min 0.05 --threshold-max 0.30 --threshold-step 0.01 \
        2>&1 | tee -a "$LOG"
    echo "✓ Phase 0 complete" | tee -a "$LOG"
else
    echo "⚠ V3 ensemble not found at $V3_ENSEMBLE — skipping Phase 0" | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"

# Phase III: Training (single variant vs all variants grid search)
echo "[Phase III] V6 Model Training..." | tee -a "$LOG"
echo "Training config: 100 epochs, patience=12, SWA (start=70), augmentations=(mixup, cutmix, gridmask)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Option 1: Train composite variant only (faster smoke test)
if [ "${TRAIN_ALL_VARIANTS:-false}" = "false" ]; then
    echo "[Phase III] Training composite variant (smoke test)..." | tee -a "$LOG"
    python "$SCRIPTS/phase3_train_v6.py" \
        --variant composite \
        --epochs 100 \
        --patience 12 \
        --num-workers 0 \
        --device cuda \
        --output-dir "$OUTPUT_BASE/phase3_runs_v6" \
        2>&1 | tee -a "$LOG"
    echo "✓ Composite training complete" | tee -a "$LOG"
else
    # Option 2: Grid search — train all 5 variants
    echo "[Phase III] Training all 5 CHM variants (grid search)..." | tee -a "$LOG"
    echo "Variants: baseline, raw, gauss, masked, composite" | tee -a "$LOG"
    for VARIANT in baseline raw gauss masked composite; do
        echo "" | tee -a "$LOG"
        echo "  Training variant: $VARIANT" | tee -a "$LOG"
        python "$SCRIPTS/phase3_train_v6.py" \
            --variant "$VARIANT" \
            --epochs 100 \
            --patience 12 \
            --num-workers 0 \
            --device cuda \
            --output-dir "$OUTPUT_BASE/phase3_runs_v6" \
            2>&1 | tee -a "$LOG"
        echo "  ✓ $VARIANT complete" | tee -a "$LOG"
    done
    echo "✓ All variants trained" | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"

# Phase IV: Evaluation
echo "[Phase IV] V6 Model Evaluation..." | tee -a "$LOG"
python "$SCRIPTS/phase4_evaluate_v6.py" \
    --runs-dir "$OUTPUT_BASE/phase3_runs_v6" \
    --output-dir "$OUTPUT_BASE/phase4_report_v6" \
    --device cuda \
    2>&1 | tee -a "$LOG"
echo "✓ Phase IV complete" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Summary
echo "========================================" | tee -a "$LOG"
echo "V6 Pipeline Complete!" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Output directories:" | tee -a "$LOG"
echo "  Phase 0 (ensemble filter):  $OUTPUT_BASE/phase0_ensemble_filter_v6" | tee -a "$LOG"
echo "  Phase III (training):       $OUTPUT_BASE/phase3_runs_v6" | tee -a "$LOG"
echo "  Phase IV (evaluation):      $OUTPUT_BASE/phase4_report_v6" | tee -a "$LOG"
echo "  Logs:                       $LOG_DIR" | tee -a "$LOG"
echo "End: $(date)" | tee -a "$LOG"

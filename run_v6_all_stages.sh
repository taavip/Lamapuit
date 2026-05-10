#!/usr/bin/env bash
set -euo pipefail

# V6 Full Testing Pipeline — 3 stages
# Stage 1: Smoke test (3 epochs, 1 fold) — ~15-20 minutes
# Stage 2: Full single variant (100 epochs, 4 folds) — ~3-4 hours
# Stage 3: Complete grid search (100 epochs, 5 variants × 4 folds) — ~8-12 hours
# Total: ~12-16 hours GPU time

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$PROJECT_ROOT/seg_pipeline/scripts"
OUTPUT_BASE="$PROJECT_ROOT/seg_pipeline/output"
LOG_DIR="$OUTPUT_BASE/phase_logs_v6"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/v6_all_stages_${TIMESTAMP}.log"

echo "========================================" | tee "$MASTER_LOG"
echo "V6 Full Testing Pipeline — All 3 Stages" | tee -a "$MASTER_LOG"
echo "========================================" | tee -a "$MASTER_LOG"
echo "Start: $(date)" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Stage 1: Smoke test (3 epochs, 1 fold)
echo "╔════════════════════════════════════════╗" | tee -a "$MASTER_LOG"
echo "║ STAGE 1: SMOKE TEST (3 epochs, fold 0) ║" | tee -a "$MASTER_LOG"
echo "╚════════════════════════════════════════╝" | tee -a "$MASTER_LOG"
STAGE1_START=$(date +%s)

python "$SCRIPTS/phase3_train_v6.py" \
    --variant composite \
    --fold 0 \
    --epochs 3 \
    --patience 12 \
    --num-workers 0 \
    --device cuda \
    --output-dir "$OUTPUT_BASE/phase3_runs_v6" \
    2>&1 | tee -a "$MASTER_LOG"

STAGE1_END=$(date +%s)
STAGE1_DURATION=$((STAGE1_END - STAGE1_START))
echo "" | tee -a "$MASTER_LOG"
echo "✓ Stage 1 complete in $((STAGE1_DURATION / 60)) minutes" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Stage 2: Full single variant (100 epochs, all 4 folds)
echo "╔══════════════════════════════════════════════════════╗" | tee -a "$MASTER_LOG"
echo "║ STAGE 2: FULL SINGLE VARIANT (100 epochs, 4 folds)   ║" | tee -a "$MASTER_LOG"
echo "╚══════════════════════════════════════════════════════╝" | tee -a "$MASTER_LOG"
STAGE2_START=$(date +%s)

python "$SCRIPTS/phase3_train_v6.py" \
    --variant composite \
    --epochs 100 \
    --patience 12 \
    --num-workers 0 \
    --device cuda \
    --output-dir "$OUTPUT_BASE/phase3_runs_v6" \
    2>&1 | tee -a "$MASTER_LOG"

STAGE2_END=$(date +%s)
STAGE2_DURATION=$((STAGE2_END - STAGE2_START))
echo "" | tee -a "$MASTER_LOG"
echo "✓ Stage 2 complete in $((STAGE2_DURATION / 3600)) hours $((($STAGE2_DURATION % 3600) / 60)) minutes" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Stage 3: Complete pipeline (all 5 variants grid search)
echo "╔══════════════════════════════════════════════════════════════╗" | tee -a "$MASTER_LOG"
echo "║ STAGE 3: COMPLETE PIPELINE (all 5 variants, 100 epochs each) ║" | tee -a "$MASTER_LOG"
echo "╚══════════════════════════════════════════════════════════════╝" | tee -a "$MASTER_LOG"
STAGE3_START=$(date +%s)

for VARIANT in baseline raw gauss masked composite; do
    echo "" | tee -a "$MASTER_LOG"
    echo "  Training variant: $VARIANT" | tee -a "$MASTER_LOG"
    VARIANT_START=$(date +%s)

    python "$SCRIPTS/phase3_train_v6.py" \
        --variant "$VARIANT" \
        --epochs 100 \
        --patience 12 \
        --num-workers 0 \
        --device cuda \
        --output-dir "$OUTPUT_BASE/phase3_runs_v6" \
        2>&1 | tee -a "$MASTER_LOG"

    VARIANT_END=$(date +%s)
    VARIANT_DURATION=$((VARIANT_END - VARIANT_START))
    echo "  ✓ $VARIANT complete in $((VARIANT_DURATION / 3600)) hours $((($VARIANT_DURATION % 3600) / 60)) minutes" | tee -a "$MASTER_LOG"
done

STAGE3_END=$(date +%s)
STAGE3_DURATION=$((STAGE3_END - STAGE3_START))
echo "" | tee -a "$MASTER_LOG"
echo "✓ Stage 3 complete in $((STAGE3_DURATION / 3600)) hours $((($STAGE3_DURATION % 3600) / 60)) minutes" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Phase IV: Evaluation (after all training done)
echo "╔═══════════════════════════════════════╗" | tee -a "$MASTER_LOG"
echo "║ PHASE IV: EVALUATION & COMPARISON     ║" | tee -a "$MASTER_LOG"
echo "╚═══════════════════════════════════════╝" | tee -a "$MASTER_LOG"
EVAL_START=$(date +%s)

python "$SCRIPTS/phase4_evaluate_v6.py" \
    --runs-dir "$OUTPUT_BASE/phase3_runs_v6" \
    --output-dir "$OUTPUT_BASE/phase4_report_v6" \
    --device cuda \
    2>&1 | tee -a "$MASTER_LOG"

EVAL_END=$(date +%s)
EVAL_DURATION=$((EVAL_END - EVAL_START))
echo "✓ Evaluation complete in $((EVAL_DURATION / 60)) minutes" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Summary
TOTAL_DURATION=$((EVAL_END - STAGE1_START))
echo "========================================" | tee -a "$MASTER_LOG"
echo "All Stages Complete! 🎉" | tee -a "$MASTER_LOG"
echo "========================================" | tee -a "$MASTER_LOG"
echo "Timeline:" | tee -a "$MASTER_LOG"
echo "  Stage 1 (smoke test):      $((STAGE1_DURATION / 60)) min" | tee -a "$MASTER_LOG"
echo "  Stage 2 (single variant):  $((STAGE2_DURATION / 3600)) h $((($STAGE2_DURATION % 3600) / 60)) min" | tee -a "$MASTER_LOG"
echo "  Stage 3 (grid search):     $((STAGE3_DURATION / 3600)) h $((($STAGE3_DURATION % 3600) / 60)) min" | tee -a "$MASTER_LOG"
echo "  Phase IV (evaluation):     $((EVAL_DURATION / 60)) min" | tee -a "$MASTER_LOG"
echo "  ─────────────────────────────────────" | tee -a "$MASTER_LOG"
echo "  Total:                     $((TOTAL_DURATION / 3600)) h $((($TOTAL_DURATION % 3600) / 60)) min" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Output directories:" | tee -a "$MASTER_LOG"
echo "  Training:                  $OUTPUT_BASE/phase3_runs_v6" | tee -a "$MASTER_LOG"
echo "  Evaluation:                $OUTPUT_BASE/phase4_report_v6" | tee -a "$MASTER_LOG"
echo "  Logs:                      $LOG_DIR" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Key results files:" | tee -a "$MASTER_LOG"
echo "  - thesis_table_v6.csv (per-fold metrics)" | tee -a "$MASTER_LOG"
echo "  - v6_variant_comparison.csv (ranking by F1)" | tee -a "$MASTER_LOG"
echo "  - v6_results.json (structured results)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "End: $(date)" | tee -a "$MASTER_LOG"

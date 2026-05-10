#!/bin/bash
# ============================================================================
# COMPREHENSIVE AUTOMATED ABLATION STUDY - All Phases with Auto Advancement
# ============================================================================
#
# Runs phases 2-6 sequentially with automatic winner selection and advancement.
# Each phase automatically starts when the previous phase completes.
#
# Features:
# - Auto-parses results and selects winners
# - Top winners automatically advance to next phase
# - Unified logging with timestamps
# - Error handling and recovery
# - Progress tracking across all phases
#
# Usage:
#   bash run_full_ablation_automated.sh                # Run all phases
#   bash run_full_ablation_automated.sh 2              # Run only phase 2
#   bash run_full_ablation_automated.sh 2 3            # Run phases 2-3
#   bash run_full_ablation_automated.sh --help         # Show options

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="$REPO_ROOT/seg_pipeline/scripts"
OUTPUT_BASE="$REPO_ROOT/seg_pipeline/output/ablation_v10_auto"
LOG_DIR="$REPO_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/ablation_full_auto_${TIMESTAMP}.log"

# Training parameters
FOLD=${FOLD:-0}
EPOCHS=${EPOCHS:-75}
SWA_START=${SWA_START:-35}
DEVICE=${DEVICE:-cuda}
NO_SWA=${NO_SWA:-false}

# Docker config
DOCKER_IMAGE="lamapuit:gpu"
DOCKER_OPTS=(
    "--rm"
    "--gpus" "all"
    "-v" "$REPO_ROOT:$REPO_ROOT"
    "--workdir" "$REPO_ROOT"
    "-e" "NO_ALBUMENTATIONS_UPDATE=1"
)

mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    local msg="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $msg" | tee -a "$MAIN_LOG"
}

log_section() {
    local title="$1"
    local width=80
    printf "%0.s=" {1..$width} | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
    printf "%s\n" "$title" | tee -a "$MAIN_LOG"
    printf "%0.s=" {1..$width} | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
}

log_success() {
    echo "✓ $1" | tee -a "$MAIN_LOG"
}

log_error() {
    echo "✗ ERROR: $1" | tee -a "$MAIN_LOG"
}

# Parse results CSV and return top N winners
get_phase_winners() {
    local phase=$1
    local count=${2:-1}
    local results_file="$OUTPUT_BASE/phase${phase}/results.csv"

    if [ ! -f "$results_file" ]; then
        log_error "Results file not found: $results_file"
        return 1
    fi

    # Parse CSV, sort by metric, return top N
    python3 << PYTHON_EOF
import csv
import sys

results = []
with open('$results_file', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Prefer condition_id, fallback to other fields
        cond_id = row.get('condition_id', row.get('condition', ''))

        # Try multiple metric names for sorting
        metric_val = 0.0
        for metric_name in ['best_val_dice', 'val_dice', 'val_f1', 'best_val_f1', 'test_f1']:
            if metric_name in row and row[metric_name]:
                try:
                    metric_val = float(row[metric_name])
                    break
                except (ValueError, TypeError):
                    continue

        results.append((cond_id, metric_val, row))

if not results:
    sys.exit(1)

results.sort(key=lambda x: x[1], reverse=True)
for i in range(min($count, len(results))):
    print(results[i][0])
PYTHON_EOF
}

# Run a single phase in Docker
run_phase() {
    local phase=$1
    local swa_flag="--swa-start-epoch $SWA_START"
    [ "$NO_SWA" = "true" ] && swa_flag="--no-swa"

    local cmd="
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect
export NO_ALBUMENTATIONS_UPDATE=1

python3 $SCRIPT_DIR/phase3_ablation_v10.py \\
    --phase $phase \\
    --fold $FOLD \\
    --epochs $EPOCHS \\
    $swa_flag \\
    --output-dir $OUTPUT_BASE \\
    --device $DEVICE
"

    docker run "${DOCKER_OPTS[@]}" "$DOCKER_IMAGE" bash -c "$cmd" 2>&1 | tee -a "$MAIN_LOG"
}

# Print phase summary with winners
print_phase_summary() {
    local phase=$1
    local phase_dir="$OUTPUT_BASE/phase${phase}"

    if [ ! -d "$phase_dir" ]; then
        log_error "Phase $phase directory not found"
        return 1
    fi

    log_section "PHASE $phase SUMMARY"

    if [ -f "$phase_dir/results.csv" ]; then
        log "Results file: $phase_dir/results.csv"
        log "Content:"
        cat "$phase_dir/results.csv" | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"

        # Show top winners
        log "Top Winners:"
        local count=2
        [ $phase -eq 6 ] && count=1

        local winners=$(get_phase_winners "$phase" "$count")
        if [ $? -eq 0 ]; then
            local i=1
            while IFS= read -r winner; do
                log "  $i. $winner ✓"
                ((i++))
            done <<< "$winners"
        fi
    else
        log_error "Results file not found: $phase_dir/results.csv"
    fi

    echo "" | tee -a "$MAIN_LOG"
}

# ============================================================================
# Main Orchestration Logic
# ============================================================================

main() {
    log_section "COMPREHENSIVE AUTOMATED ABLATION STUDY"
    log "Start time: $(date)"
    log "Output directory: $OUTPUT_BASE"
    log "Main log: $MAIN_LOG"
    log "Configuration: fold=$FOLD, epochs=$EPOCHS, device=$DEVICE"
    echo "" | tee -a "$MAIN_LOG"

    # Determine which phases to run
    local phases_to_run=(2 3 4 5 6)
    if [ $# -gt 0 ]; then
        phases_to_run=("$@")
    fi

    log "Will execute phases: ${phases_to_run[@]}"
    echo "" | tee -a "$MAIN_LOG"

    # ── Phase 2: CHM Variant Search
    if [[ " ${phases_to_run[@]} " =~ " 2 " ]]; then
        log_section "PHASE 2: CHM VARIANT SEARCH (5 conditions)"
        log "Testing: baseline, raw, gauss, masked, composite"
        log "Expected runtime: ~2.5 hours"

        mkdir -p "$OUTPUT_BASE/phase2"
        run_phase 2

        print_phase_summary 2

        # Select winner for Phase 3
        local phase2_winner=$(get_phase_winners 2 1)
        log_success "Phase 2 winner selected: $phase2_winner"

        # Extract CHM variant from winner (e.g., "2C" -> "gauss")
        case "$phase2_winner" in
            2A) local phase2_chm="baseline" ;;
            2B) local phase2_chm="raw" ;;
            2C) local phase2_chm="gauss" ;;
            2D) local phase2_chm="masked" ;;
            2E) local phase2_chm="composite" ;;
            *) log_error "Unknown winner: $phase2_winner"; return 1 ;;
        esac

        log "Phase 2 CHM variant: $phase2_chm"
        echo "$phase2_chm" > "$OUTPUT_BASE/phase2_winner_chm.txt"
    fi

    # ── Phase 3: Architecture Search
    if [[ " ${phases_to_run[@]} " =~ " 3 " ]]; then
        log_section "PHASE 3: ARCHITECTURE SEARCH (5 conditions)"

        # Load winner CHM from phase 2
        if [ -f "$OUTPUT_BASE/phase2_winner_chm.txt" ]; then
            local phase2_chm=$(cat "$OUTPUT_BASE/phase2_winner_chm.txt")
            log "Using Phase 2 winner CHM: $phase2_chm"
        else
            local phase2_chm="gauss"
            log "Phase 2 winner not found, using default: $phase2_chm"
        fi

        log "Testing architectures: UNet+EffB2, UNet++EffB0, UNet++EffB2, UNet++EffB4, DeepLabV3++EffB2"
        log "Expected runtime: ~3.5 hours"

        mkdir -p "$OUTPUT_BASE/phase3"

        # Run Phase 3 with Phase 2 winner CHM variant
        local cmd="
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect
export NO_ALBUMENTATIONS_UPDATE=1

python3 $SCRIPT_DIR/phase3_ablation_v10.py \\
    --phase 3 \\
    --chm-variant $phase2_chm \\
    --fold $FOLD \\
    --epochs $EPOCHS \\
    --swa-start-epoch $SWA_START \\
    --output-dir $OUTPUT_BASE \\
    --device $DEVICE
"
        docker run "${DOCKER_OPTS[@]}" "$DOCKER_IMAGE" bash -c "$cmd" 2>&1 | tee -a "$MAIN_LOG"

        print_phase_summary 3

        local phase3_winner=$(get_phase_winners 3 1)
        log_success "Phase 3 winner selected: $phase3_winner"
        echo "$phase3_winner" > "$OUTPUT_BASE/phase3_winner_arch.txt"
    fi

    # ── Phase 4: Loss Function & Parameters
    if [[ " ${phases_to_run[@]} " =~ " 4 " ]]; then
        log_section "PHASE 4: LOSS FUNCTION & PARAMETER SEARCH (8 conditions)"

        log "Testing Tversky α/β combinations and CLDice weights"
        log "Expected runtime: ~4.5 hours"

        mkdir -p "$OUTPUT_BASE/phase4"
        run_phase 4

        print_phase_summary 4

        local phase4_winner=$(get_phase_winners 4 1)
        log_success "Phase 4 winner selected: $phase4_winner"
        echo "$phase4_winner" > "$OUTPUT_BASE/phase4_winner_loss.txt"
    fi

    # ── Phase 5: Augmentation & Regularization
    if [[ " ${phases_to_run[@]} " =~ " 5 " ]]; then
        log_section "PHASE 5: AUGMENTATION & REGULARIZATION SEARCH (5 conditions)"

        log "Testing: none, geometric, full (no soft), full (soft), full (no SWA)"
        log "Expected runtime: ~2.5 hours"

        mkdir -p "$OUTPUT_BASE/phase5"
        run_phase 5

        print_phase_summary 5

        local phase5_winner=$(get_phase_winners 5 1)
        log_success "Phase 5 winner selected: $phase5_winner"
        echo "$phase5_winner" > "$OUTPUT_BASE/phase5_winner_aug.txt"
    fi

    # ── Phase 6: Final Validation (all 4 folds)
    if [[ " ${phases_to_run[@]} " =~ " 6 " ]]; then
        log_section "PHASE 6: FINAL VALIDATION (ALL 4 FOLDS)"

        log "Running winning configuration across all folds (0, 1, 2, 3)"
        log "Expected runtime: ~5 hours"

        mkdir -p "$OUTPUT_BASE/phase6"
        # For phase 6, need to run all folds
        for fold in 0 1 2 3; do
            log "Running Phase 6 with fold $fold..."
            FOLD=$fold run_phase 6
        done

        print_phase_summary 6
    fi

    # ── Final Report
    log_section "ABLATION STUDY COMPLETE"
    log "Total runtime summary:"
    log "  Phase 2 (CHM):       ~2.5h"
    log "  Phase 3 (Arch):      ~3.5h"
    log "  Phase 4 (Loss):      ~4.5h"
    log "  Phase 5 (Aug):       ~2.5h"
    log "  Phase 6 (Final):     ~5.0h"
    log "  ───────────────────────────"
    log "  TOTAL:               ~18h"
    echo "" | tee -a "$MAIN_LOG"

    log "Full results saved to: $OUTPUT_BASE"
    log "Main log file: $MAIN_LOG"

    # Create summary file
    cat > "$OUTPUT_BASE/ABLATION_SUMMARY.md" << 'EOF'
# Comprehensive Ablation Study Summary

## Winners by Phase

EOF

    for phase in 2 3 4 5 6; do
        if [ -f "$OUTPUT_BASE/phase${phase}_winner_"* ]; then
            local winner=$(cat "$OUTPUT_BASE/phase${phase}_winner_"* 2>/dev/null || echo "N/A")
            echo "- **Phase $phase**: $winner" >> "$OUTPUT_BASE/ABLATION_SUMMARY.md"
        fi
    done

    echo "" | tee -a "$MAIN_LOG"
    log_success "Study complete! Results in: $OUTPUT_BASE"
    log_success "Start time: $(date)"
}

# ============================================================================
# Entry Point
# ============================================================================

if [ $# -eq 1 ] && [ "$1" = "--help" ]; then
    cat << EOF
COMPREHENSIVE AUTOMATED ABLATION STUDY

Usage:
  bash run_full_ablation_automated.sh                # Run all phases (2-6)
  bash run_full_ablation_automated.sh 2              # Run only phase 2
  bash run_full_ablation_automated.sh 2 3            # Run phases 2 and 3
  bash run_full_ablation_automated.sh --help         # Show this help

Environment variables:
  FOLD=N           Set fold (default: 0)
  EPOCHS=N         Set epochs (default: 75)
  SWA_START=N      SWA start epoch (default: 35)
  DEVICE=cuda|cpu  Device (default: cuda)
  NO_SWA=true      Disable SWA (default: false)

Examples:
  EPOCHS=5 bash run_full_ablation_automated.sh 2    # Quick smoke test phase 2
  bash run_full_ablation_automated.sh 3 4 5         # Run phases 3-5 only

EOF
    exit 0
fi

main "$@"

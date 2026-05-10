#!/bin/bash
# ============================================================================
# COMPREHENSIVE AUTOMATED ABLATION STUDY - Top 2 Winners Per Phase Strategy
# ============================================================================
#
# Runs phases 2-6 sequentially where EACH phase tests ALL conditions with
# TOP 2 WINNERS from previous phase, creating a more thorough search.
#
# Strategy:
# - Phase 2: Test 5 CHM variants → Select top 2 (e.g., Gauss + Baseline)
# - Phase 3: Test 5 arch × 2 CHM variants = 10 conditions → Select top 2 combos
# - Phase 4: Test 8 loss × 2 arch/CHM combos = 16 conditions → Select top 2
# - Phase 5: Test 5 aug × 2 loss configs = 10 conditions → Select top 2
# - Phase 6: Validate both top 2 configurations across all 4 folds
#
# Usage:
#   bash run_full_ablation_automated_top2.sh                # All phases
#   bash run_full_ablation_automated_top2.sh 2              # Phase 2 only
#   EPOCHS=5 bash run_full_ablation_automated_top2.sh 2     # Quick test

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="$REPO_ROOT/seg_pipeline/scripts"
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/seg_pipeline/output/ablation_v10_top2_cv}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/seg_pipeline/output/phase2_dataset_v10_reconstructed}"
SELECTION_METRIC="${SELECTION_METRIC:-val_cldice}"
CHM_SOURCE_DIR="${CHM_SOURCE_DIR:-}"
PHASE2_CONDITIONS="${PHASE2_CONDITIONS:-2A,2B,2C,2D,2E}"
REBUILD_DATASET="${REBUILD_DATASET:-true}"
LOG_DIR="$REPO_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/ablation_top2_${TIMESTAMP}.log"

FOLD=${FOLD:-0}
EPOCHS=${EPOCHS:-100}
WARMUP=${WARMUP:-25}
SEED=${SEED:-42}
SWA_START=${SWA_START:-35}
DEVICE=${DEVICE:-cuda}
NO_SWA=${NO_SWA:-false}

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
    printf "%*s" "$width" "" | tr " " "=" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
    printf "%s\n" "$title" | tee -a "$MAIN_LOG"
    printf "%*s" "$width" "" | tr " " "=" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
}

log_success() {
    echo "✓ $1" | tee -a "$MAIN_LOG"
}

log_error() {
    echo "✗ ERROR: $1" | tee -a "$MAIN_LOG"
}

join_by_comma() {
    local IFS=", "
    echo "$*"
}

prepare_inputs_and_dataset() {
    log_section "PREFLIGHT: CHM INPUTS + DATASET ASSETS"
    log "Dataset dir: $DATASET_DIR"
    log "Rebuild dataset assets: $REBUILD_DATASET"
    if [ -n "$CHM_SOURCE_DIR" ]; then
        log "Linking seg_pipeline/input to CHM source dir: $CHM_SOURCE_DIR"
        mkdir -p "$REPO_ROOT/seg_pipeline/input"
        ln -sfn "$CHM_SOURCE_DIR/baseline_chm_0p2m/406455_2021_tava_chm_max_hag_20cm.tif" "$REPO_ROOT/seg_pipeline/input/baseline_chm.tif"
        ln -sfn "$CHM_SOURCE_DIR/harmonized_raw_0p2m/406455_2021_tava_harmonized_dem_last_raw_chm.tif" "$REPO_ROOT/seg_pipeline/input/raw_chm.tif"
        ln -sfn "$CHM_SOURCE_DIR/harmonized_gauss_kernel0p8m_0p2m/406455_2021_tava_harmonized_dem_last_gauss_chm.tif" "$REPO_ROOT/seg_pipeline/input/gauss_chm.tif"
        ln -sfn "$CHM_SOURCE_DIR/masked_raw_2band_0p2m/406455_2021_tava_harmonized_dem_last_raw_chm_2band.tif" "$REPO_ROOT/seg_pipeline/input/masked_chm.tif"
        ln -sfn "$CHM_SOURCE_DIR/composite_4band_raw_base_mask/406455_2021_4band.tif" "$REPO_ROOT/seg_pipeline/input/composite_4band.tif"
    fi

    if [ "$REBUILD_DATASET" = "true" ]; then
        local cmd="
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect
set -e
mkdir -p '$DATASET_DIR'
for v in baseline raw gauss masked composite; do
  python3 $SCRIPT_DIR/phase2_dataset_v3.py --chm-variant \$v --cv-version 3 --output-dir '$DATASET_DIR'
done
"
        docker run "${DOCKER_OPTS[@]}" "$DOCKER_IMAGE" bash -c "$cmd" 2>&1 | tee -a "$MAIN_LOG"
    fi
}

# Get top N winners from phase results, return condition IDs
get_phase_winners() {
    local phase=$1
    local count=${2:-1}
    local results_file1="$OUTPUT_BASE/phase${phase}/results.csv"
    local results_file2="$OUTPUT_BASE/phase${phase}_results.csv"
    local results_file=""

    if [ -f "$results_file1" ]; then
        results_file="$results_file1"
    elif [ -f "$results_file2" ]; then
        results_file="$results_file2"
    else
        log_error "Results file not found: checked $results_file1 and $results_file2"
        return 1
    fi

    python3 << PYTHON_EOF
import csv
import sys

results = []
with open('$results_file', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cond_id = row.get('run_id') or row.get('condition_id') or row.get('condition', '')
        metric_val = 0.0
        # Selection is validation-only. The held-out test stripe is not used here.
        for metric_name in ['val_cldice', 'best_val_cldice', 'val_f1', 'best_val_f1', 'threshold_f1', 'best_val_dice', 'val_dice']:
            if metric_name in row and row[metric_name]:
                try:
                    metric_val = float(row[metric_name])
                    break
                except (ValueError, TypeError):
                    continue
        results.append((cond_id, metric_val))

if not results:
    sys.exit(1)

results.sort(key=lambda x: x[1], reverse=True)
for i in range(min($count, len(results))):
    print(results[i][0])
PYTHON_EOF
}

base_condition_id() {
    local cond_id="$1"
    echo "${cond_id##*__}"
}

# Map condition ID to parameter (e.g., "2C" -> "gauss")
get_chm_from_condition() {
    local cond_id="$1"
    cond_id="$(base_condition_id "$cond_id")"
    case "$cond_id" in
        2A) echo "baseline" ;;
        2B) echo "raw" ;;
        2C) echo "gauss" ;;
        2D) echo "masked" ;;
        2E) echo "composite" ;;
        *) echo "gauss" ;;
    esac
}

get_arch_from_condition() {
    local cond_id="$1"
    cond_id="$(base_condition_id "$cond_id")"
    case "$cond_id" in
        3A) echo "unet_effb2" ;;
        3B) echo "unetpp_effb0" ;;
        3C) echo "unetpp_effb2" ;;
        3D) echo "unetpp_effb4" ;;
        3E) echo "deeplabv3p_effb2" ;;
        *) echo "unetpp_effb2" ;;
    esac
}

# Run a single phase with specific CHM variants (Phase 3+)
run_phase_with_winners() {
    local phase=$1
    shift
    local winners=("$@")  # Array of condition IDs from previous phase

    local swa_flag="--swa-start-epoch $SWA_START"
    [ "$NO_SWA" = "true" ] && swa_flag="--no-swa"

    # For Phase 2, just run normally (no winners needed)
    if [ $phase -eq 2 ]; then
        local cmd="
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect
export NO_ALBUMENTATIONS_UPDATE=1

python3 $SCRIPT_DIR/phase3_ablation_v10.py \\
    --phase $phase \\
    --fold $FOLD \\
    --epochs $EPOCHS \\
    --selection-metric $SELECTION_METRIC \\
    $swa_flag \\
    --warmup-epochs $WARMUP \\
    --seed $SEED \\
    --dataset-dir $DATASET_DIR \\
    --condition-ids $PHASE2_CONDITIONS \\
    --output-dir $OUTPUT_BASE \\
    --device $DEVICE
"
        docker run "${DOCKER_OPTS[@]}" "$DOCKER_IMAGE" bash -c "$cmd" 2>&1 | tee -a "$MAIN_LOG"
        return
    fi

    # For Phase 3+, test all conditions with each winning CHM variant
    if [ $phase -eq 3 ]; then
        for winner in "${winners[@]}"; do
            local chm=$(get_chm_from_condition "$winner")
            log "  Testing Phase $phase architectures with CHM variant: $chm (from $winner)"

            local cmd="
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect
export NO_ALBUMENTATIONS_UPDATE=1

python3 $SCRIPT_DIR/phase3_ablation_v10.py \\
    --phase $phase \\
    --chm-variant $chm \\
    --fold $FOLD \\
    --epochs $EPOCHS \\
    --selection-metric $SELECTION_METRIC \\
    $swa_flag \\
    --warmup-epochs $WARMUP \\
    --seed $SEED \\
    --dataset-dir $DATASET_DIR \\
    --carry-winner 2:${winner} \\
    --output-dir $OUTPUT_BASE \\
    --device $DEVICE
"
            docker run "${DOCKER_OPTS[@]}" "$DOCKER_IMAGE" bash -c "$cmd" 2>&1 | tee -a "$MAIN_LOG"
        done
        return
    fi

    # For Phase 4+, run with all winner combinations
    local prev_phase=$((phase - 1))
    for winner in "${winners[@]}"; do
        log "  Testing Phase $phase with winning configuration: $winner"

        local cmd="
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect
export NO_ALBUMENTATIONS_UPDATE=1

python3 $SCRIPT_DIR/phase3_ablation_v10.py \\
    --phase $phase \\
    --fold $FOLD \\
    --epochs $EPOCHS \\
    --selection-metric $SELECTION_METRIC \\
    $swa_flag \\
    --warmup-epochs $WARMUP \\
    --seed $SEED \\
    --dataset-dir $DATASET_DIR \\
    --carry-winner ${prev_phase}:${winner} \\
    --output-dir $OUTPUT_BASE \\
    --device $DEVICE
"
        docker run "${DOCKER_OPTS[@]}" "$DOCKER_IMAGE" bash -c "$cmd" 2>&1 | tee -a "$MAIN_LOG"
    done
}

# Print summary with top 2 winners
print_phase_summary() {
    local phase=$1
    local phase_dir="$OUTPUT_BASE/phase${phase}"
    local results_file1="$phase_dir/results.csv"
    local results_file2="$OUTPUT_BASE/phase${phase}_results.csv"
    local results_file=""

    # Only error if neither a phase dir nor a flat results file exists
    if [ ! -d "$phase_dir" ] && [ ! -f "$results_file2" ]; then
        log_error "Phase $phase directory not found and no flat results file present"
        return 1
    fi

    log_section "PHASE $phase SUMMARY - TOP 2 WINNERS"

    if [ -f "$results_file1" ]; then
        results_file="$results_file1"
    elif [ -f "$results_file2" ]; then
        results_file="$results_file2"
    else
        log_error "Results file not found (checked $results_file1 and $results_file2)"
        return 1
    fi

    log "Results file: $results_file"
    log ""
    log "Top Winners:"

    local winners=$(get_phase_winners "$phase" 2)
    if [ $? -eq 0 ]; then
        local i=1
        while IFS= read -r winner; do
            # Get metric value
            local metric=$(python3 << PYTHON_EOF
import csv
with open('$results_file', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('condition_id') == '$winner':
            for m in ['val_cldice', 'best_val_cldice', 'val_f1', 'best_val_f1', 'threshold_f1', 'best_val_dice', 'val_dice']:
                if m in row and row[m]:
                    print(row[m])
                    break
            break
PYTHON_EOF
)
            log "  $i. $winner → metric=$metric ✓✓ ADVANCE TO PHASE $((phase+1))"
            ((i++))
        done <<< "$winners"
    fi

    echo "" | tee -a "$MAIN_LOG"
}

# ============================================================================
# Main Orchestration Logic
# ============================================================================

main() {
    log_section "COMPREHENSIVE AUTOMATED ABLATION STUDY - TOP 2 WINNERS STRATEGY"
    log "Start time: $(date)"
    log "Output directory: $OUTPUT_BASE"
    log "Main log: $MAIN_LOG"
    log "Configuration: fold=$FOLD, epochs=$EPOCHS, device=$DEVICE"
    log "Selection metric: $SELECTION_METRIC"
    echo "" | tee -a "$MAIN_LOG"

    local phases_to_run=(2 3 4 5 6)
    if [ $# -gt 0 ]; then
        phases_to_run=("$@")
    fi

    log "Will execute phases: ${phases_to_run[*]}"
    log "Strategy: Select top 2 winners per phase, test all with both in next phase"
    echo "" | tee -a "$MAIN_LOG"
    prepare_inputs_and_dataset

    # Track winners across phases
    declare -a phase2_winners=()
    declare -a phase3_winners=()
    declare -a phase4_winners=()
    declare -a phase5_winners=()

    # ── Phase 2: CHM Variant Search
    if [[ " ${phases_to_run[@]} " =~ " 2 " ]]; then
        log_section "PHASE 2: CHM VARIANT SEARCH (5 conditions)"
        log "Testing: baseline, raw, gauss, masked, composite"
        log "Expected runtime: ~2.5 hours"
        log "Strategy: Select top 2 variants for Phase 3"

        mkdir -p "$OUTPUT_BASE/phase2"
        run_phase_with_winners 2

        print_phase_summary 2

        # Get top 2 winners (capture safely to avoid set -e exiting on missing files)
        if winners_out=$(get_phase_winners 2 2); then
            readarray -t phase2_winners <<<"$winners_out"
        else
            phase2_winners=()
        fi
        log ""
        log_success "Phase 2 complete. Top winners: $(join_by_comma "${phase2_winners[@]}")"
        log "These will be tested with all 5 Phase 3 architectures"
    fi

    # ── Phase 3: Architecture Search with Top 2 CHMs
    if [[ " ${phases_to_run[@]} " =~ " 3 " ]]; then
        log_section "PHASE 3: ARCHITECTURE SEARCH (5 arch × 2 CHM = 10 conditions)"

        if [ ${#phase2_winners[@]} -eq 0 ]; then
            # Load from file if not in memory
            if winners_out=$(get_phase_winners 2 2); then
                readarray -t phase2_winners <<<"$winners_out"
            else
                phase2_winners=()
            fi
        fi

        log "Testing 5 architectures with each of top 2 Phase 2 CHM winners:"
        for winner in "${phase2_winners[@]}"; do
            local chm=$(get_chm_from_condition "$winner")
            log "  - $chm (from condition $winner)"
        done
        log "Expected runtime: ~7 hours (5 arch × 2 CHMs × 1.4h per run)"
        log "Strategy: Select top 2 architecture/CHM combinations for Phase 4"

        mkdir -p "$OUTPUT_BASE/phase3"
        run_phase_with_winners 3 "${phase2_winners[@]}"

        print_phase_summary 3

        if winners_out=$(get_phase_winners 3 2); then
            readarray -t phase3_winners <<<"$winners_out"
        else
            phase3_winners=()
        fi
        log ""
        log_success "Phase 3 complete. Top combinations: $(join_by_comma "${phase3_winners[@]}")"
    fi

    # ── Phase 4: Loss Function & Parameters with Top 2 Winners
    if [[ " ${phases_to_run[@]} " =~ " 4 " ]]; then
        log_section "PHASE 4: LOSS FUNCTION SEARCH (8 loss × 2 winners = 16 conditions)"

            if [ ${#phase3_winners[@]} -eq 0 ]; then
                if winners_out=$(get_phase_winners 3 2); then
                    readarray -t phase3_winners <<<"$winners_out"
                else
                    phase3_winners=()
                fi
            fi

        log "Testing 8 loss configurations with each of top 2 Phase 3 winners:"
        for winner in "${phase3_winners[@]}"; do
            log "  - Configuration $winner"
        done
        log "Expected runtime: ~9 hours (8 loss × 2 combos × 1.125h per run)"

        mkdir -p "$OUTPUT_BASE/phase4"
        run_phase_with_winners 4 "${phase3_winners[@]}"

        print_phase_summary 4

        if winners_out=$(get_phase_winners 4 2); then
            readarray -t phase4_winners <<<"$winners_out"
        else
            phase4_winners=()
        fi
        log ""
        log_success "Phase 4 complete. Top loss configs: $(join_by_comma "${phase4_winners[@]}")"
    fi

    # ── Phase 5: Augmentation & Regularization with Top 2 Winners
    if [[ " ${phases_to_run[@]} " =~ " 5 " ]]; then
        log_section "PHASE 5: AUGMENTATION SEARCH (5 aug × 2 winners = 10 conditions)"

            if [ ${#phase4_winners[@]} -eq 0 ]; then
                if winners_out=$(get_phase_winners 4 2); then
                    readarray -t phase4_winners <<<"$winners_out"
                else
                    phase4_winners=()
                fi
            fi

        log "Testing 5 augmentation strategies with each of top 2 Phase 4 winners:"
        for winner in "${phase4_winners[@]}"; do
            log "  - Configuration $winner"
        done
        log "Expected runtime: ~5 hours (5 aug × 2 combos × 1h per run)"

        mkdir -p "$OUTPUT_BASE/phase5"
        run_phase_with_winners 5 "${phase4_winners[@]}"

        print_phase_summary 5

        if winners_out=$(get_phase_winners 5 2); then
            readarray -t phase5_winners <<<"$winners_out"
        else
            phase5_winners=()
        fi
        log ""
        log_success "Phase 5 complete. Top augmentation configs: $(join_by_comma "${phase5_winners[@]}")"
    fi

    # ── Phase 6: Final Validation with Top 2 Winners (all 4 folds)
    if [[ " ${phases_to_run[@]} " =~ " 6 " ]]; then
        log_section "PHASE 6: FINAL VALIDATION - TOP 2 CONFIGURATIONS × 4 FOLDS = 8 runs"

            if [ ${#phase5_winners[@]} -eq 0 ]; then
                if winners_out=$(get_phase_winners 5 2); then
                    readarray -t phase5_winners <<<"$winners_out"
                else
                    phase5_winners=()
                fi
            fi

        log "Validating top 2 winning configurations across all folds:"
        for winner in "${phase5_winners[@]}"; do
            log "  - Configuration $winner → testing on folds 0,1,2,3"
        done
        log "Expected runtime: ~10 hours (2 configs × 4 folds × 1.25h per fold)"

        mkdir -p "$OUTPUT_BASE/phase6"

        for fold_id in 0 1 2 3; do
            log ""
            log "Running Phase 6 - Fold $fold_id (with both top 2 winners)"
            FOLD=$fold_id run_phase_with_winners 6 "${phase5_winners[@]}"
        done

        print_phase_summary 6
    fi

    # ── Final Report
    log_section "ABLATION STUDY COMPLETE - TOP 2 STRATEGY"
    log "Estimated total runtime:"
    log "  Phase 2 (5 conditions):        ~2.5h"
    log "  Phase 3 (10 conditions):       ~7h"
    log "  Phase 4 (16 conditions):       ~9h"
    log "  Phase 5 (10 conditions):       ~5h"
    log "  Phase 6 (8 folds):             ~10h"
    log "  ──────────────────────────────────"
    log "  TOTAL (49 conditions):         ~33.5h"
    echo "" | tee -a "$MAIN_LOG"

    log "Full results saved to: $OUTPUT_BASE"
    log "Main log file: $MAIN_LOG"

    # Create comprehensive summary
# NOTE: use unquoted EOF so variables expand
    cat > "$OUTPUT_BASE/TOP2_ABLATION_SUMMARY.md" << EOF
# Top 2 Winners Ablation Study Summary

## Overview
Instead of greedy single-winner advancement, this study selects the **top 2 results from each phase** and tests ALL next-phase conditions with BOTH winners. This explores parameter interactions more thoroughly.

## Reproducibility & Setup

- **Selection metric**: validation F1 from spatial CV only.
- **Held-out test use**: the test stripe is locked during Phases 2-6 and is evaluated only after the final configuration is fixed with \`--evaluate-test\`.
- **Seed**: $SEED
- **Epochs**: $EPOCHS
- **Warmup (linear LR)**: $WARMUP
- **SWA start epoch**: $SWA_START (set `NO_SWA=true` to disable)
- **Cross-validation**: Vertical-stripe spatial CV (see seg_pipeline/scripts/phase2_dataset_v3.py) — stripe 0 is held-out test; folds rotate among other vertical stripes.


## Winners by Phase

### Phase 2: CHM Variants (5 conditions tested)
EOF

    for winner in "${phase2_winners[@]}"; do
        echo "- **$winner** ✓✓" >> "$OUTPUT_BASE/TOP2_ABLATION_SUMMARY.md"
    done

    cat >> "$OUTPUT_BASE/TOP2_ABLATION_SUMMARY.md" << 'EOF'

### Phase 3: Architectures (5 × 2 = 10 conditions tested with both Phase 2 winners)
EOF

    for winner in "${phase3_winners[@]}"; do
        echo "- **$winner** ✓✓" >> "$OUTPUT_BASE/TOP2_ABLATION_SUMMARY.md"
    done

    cat >> "$OUTPUT_BASE/TOP2_ABLATION_SUMMARY.md" << 'EOF'

### Phase 4: Loss Functions (8 × 2 = 16 conditions tested with both Phase 3 winners)
EOF

    for winner in "${phase4_winners[@]}"; do
        echo "- **$winner** ✓✓" >> "$OUTPUT_BASE/TOP2_ABLATION_SUMMARY.md"
    done

    cat >> "$OUTPUT_BASE/TOP2_ABLATION_SUMMARY.md" << 'EOF'

### Phase 5: Augmentation (5 × 2 = 10 conditions tested with both Phase 4 winners)
EOF

    for winner in "${phase5_winners[@]}"; do
        echo "- **$winner** ✓✓" >> "$OUTPUT_BASE/TOP2_ABLATION_SUMMARY.md"
    done

    cat >> "$OUTPUT_BASE/TOP2_ABLATION_SUMMARY.md" << 'EOF'

### Phase 6: Final Validation (both top 2 configurations, all 4 folds = 8 runs)

## Results Files

All detailed results in: `seg_pipeline/output/ablation_v10_top2/`

## Strategy Rationale

**Greedy Single-Winner** (original):
- Phase 2: Best of 5 CHMs
- Phase 3: Best arch with best CHM (1 combination)
- Total combinations tested: 5 + 5 + 8 + 5 = 23

**Top 2 Multi-Winner** (this study):
- Phase 2: Top 2 of 5 CHMs
- Phase 3: All 5 arch × 2 CHMs = 10 combinations tested
- Phase 4: All 8 loss × 2 arch/CHM combos = 16 combinations
- Phase 5: All 5 aug × 2 loss combos = 10 combinations
- Total combinations tested: 5 + 10 + 16 + 10 = 41 (1.8× more comprehensive)

**Benefit**: Discovers parameter interactions (e.g., "Gauss works best with UNet++B4, but Baseline works best with UNet++B2"). Single-winner greedy approach would miss this.

EOF

    echo "" | tee -a "$MAIN_LOG"
    log_success "Study complete! Results in: $OUTPUT_BASE"
    log_success "Summary: $OUTPUT_BASE/TOP2_ABLATION_SUMMARY.md"
}

# ============================================================================
# Entry Point
# ============================================================================

if [ $# -eq 1 ] && [ "$1" = "--help" ]; then
    cat << EOF
COMPREHENSIVE AUTOMATED ABLATION STUDY - TOP 2 WINNERS STRATEGY

Tests top 2 results from each phase with all next-phase conditions.
More thorough than greedy single-winner approach.

Usage:
  bash run_full_ablation_automated_top2.sh           # All phases (33.5h)
  bash run_full_ablation_automated_top2.sh 2         # Phase 2 only (2.5h)
  bash run_full_ablation_automated_top2.sh 2 3       # Phases 2-3 (9.5h)
  EPOCHS=5 bash run_full_ablation_automated_top2.sh 2 # Quick test (15 min)

Environment variables:
  FOLD=N           Fold to train (default: 0)
  EPOCHS=N         Epochs per condition (default: 75)
  SWA_START=N      SWA start epoch (default: 35)
  DEVICE=cuda|cpu  Device (default: cuda)
  NO_SWA=true      Disable SWA (default: false)

EOF
    exit 0
fi

main "$@"

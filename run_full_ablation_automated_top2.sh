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
# - Phase 6: Final locked-test estimation (Top 2 from Phase 5 + legacy V10 baseline)
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
PHASE3_CONDITIONS="${PHASE3_CONDITIONS:-3B,3C,3E}"
PHASE4_CONDITIONS="${PHASE4_CONDITIONS:-4A,4D,4F,4H}"
PHASE5_CONDITIONS="${PHASE5_CONDITIONS:-5A,5D,5E}"
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
CV_VERSION=${CV_VERSION:-4}
LEGACY_V10_CHAIN=${LEGACY_V10_CHAIN:-2A__3C__4H__5D}
PARALLEL_JOBS=${PARALLEL_JOBS:-1}

DOCKER_IMAGE="lamapuit:gpu"
DOCKER_OPTS=(
    "--rm"
    "--gpus" "all"
    "-v" "$REPO_ROOT:$REPO_ROOT"
    "--workdir" "$REPO_ROOT"
    "-e" "NO_ALBUMENTATIONS_UPDATE=1"
)

mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

# Locked thesis protocol assumptions
if [ "$SELECTION_METRIC" != "val_cldice" ]; then
    echo "ERROR: Locked protocol requires SELECTION_METRIC=val_cldice (got '$SELECTION_METRIC')." >&2
    exit 1
fi
if [ "$CV_VERSION" -ne 4 ]; then
    echo "ERROR: Locked protocol requires CV_VERSION=4 (balanced 2-fold)." >&2
    exit 1
fi

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

cv_folds_for_version() {
    if [ "$CV_VERSION" -eq 4 ]; then
        echo "0 1"
    else
        echo "0 1 2 3"
    fi
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
  python3 $SCRIPT_DIR/phase2_dataset_v3.py --chm-variant \$v --cv-version $CV_VERSION --output-dir '$DATASET_DIR'
done
"
        docker run "${DOCKER_OPTS[@]}" "$DOCKER_IMAGE" bash -c "$cmd" 2>&1 | tee -a "$MAIN_LOG"
    fi

    log "Balanced-fold preflight check (variant=composite, cv_version=$CV_VERSION)"
    local validate_cmd="
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect
set -e
python3 $SCRIPT_DIR/phase2_dataset_v3.py --chm-variant composite --cv-version $CV_VERSION --output-dir '$DATASET_DIR' --validate
"
    docker run "${DOCKER_OPTS[@]}" "$DOCKER_IMAGE" bash -c "$validate_cmd" 2>&1 | tee -a "$MAIN_LOG"
}

# Get top N winners from phase results, return condition IDs.
# Tie-break is fixed before run start:
#   1) higher mean(selection metric), 2) higher mean(val_f1), 3) lower std(selection metric)
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
import statistics
from collections import defaultdict

rows = []
with open('$results_file', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

if not rows:
    sys.exit(1)

metric_priority = ['$SELECTION_METRIC', 'val_cldice', 'best_val_cldice', 'val_f1', 'best_val_f1', 'threshold_f1', 'best_val_dice', 'val_dice']
grouped = defaultdict(list)
folds = defaultdict(set)
f1_grouped = defaultdict(list)

for row in rows:
    run_id = row.get('run_id') or row.get('condition_id') or row.get('condition', '')
    if not run_id:
        continue
    metric_val = None
    for metric_name in metric_priority:
        if metric_name in row and row[metric_name]:
            try:
                metric_val = float(row[metric_name])
                break
            except (ValueError, TypeError):
                continue
    if metric_val is None:
        continue
    grouped[run_id].append(metric_val)
    f1_val = None
    for f1_name in ('val_f1', 'best_val_f1'):
        if f1_name in row and row[f1_name]:
            try:
                f1_val = float(row[f1_name])
                break
            except (ValueError, TypeError):
                continue
    if f1_val is not None:
        f1_grouped[run_id].append(f1_val)
    folds[run_id].add(str(row.get('fold_id', 'NA')))

aggregated = []
for run_id, vals in grouped.items():
    if not vals:
        continue
    mean_sel = sum(vals) / len(vals)
    mean_f1 = (sum(f1_grouped[run_id]) / len(f1_grouped[run_id])) if f1_grouped[run_id] else float('-inf')
    std_sel = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    aggregated.append((run_id, mean_sel, mean_f1, std_sel, len(folds[run_id]), len(vals)))

if not aggregated:
    sys.exit(1)

# Primary: mean selection metric (higher better)
# Tie-break 1: mean val_f1 (higher better)
# Tie-break 2: std(selection metric) (lower better)
# Tie-break 3: fold coverage / row count (higher better)
aggregated.sort(key=lambda x: (-x[1], -x[2], x[3], -x[4], -x[5], x[0]))
for i in range(min($count, len(aggregated))):
    print(aggregated[i][0])
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
    --cv-version $CV_VERSION \\
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
	    --cv-version $CV_VERSION \\
	    --condition-ids $PHASE3_CONDITIONS \\
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
    local condition_ids_arg=""
    if [ "$phase" -eq 4 ]; then
        condition_ids_arg="--condition-ids $PHASE4_CONDITIONS"
    elif [ "$phase" -eq 5 ]; then
        condition_ids_arg="--condition-ids $PHASE5_CONDITIONS"
    fi
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
	    --cv-version $CV_VERSION \\
	    $condition_ids_arg \
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

run_final_locked_test() {
    local winner="$1"
    local swa_flag="--swa-start-epoch $SWA_START"
    [ "$NO_SWA" = "true" ] && swa_flag="--no-swa"
    log "  Final locked-test run: $winner (train on all non-test stripes)"
    local cmd="
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect
export NO_ALBUMENTATIONS_UPDATE=1

python3 $SCRIPT_DIR/phase3_ablation_v10.py \\
    --phase 6 \\
    --condition-ids 6 \\
    --fold 0 \\
    --epochs $EPOCHS \\
    --selection-metric $SELECTION_METRIC \\
    --cv-version $CV_VERSION \\
    $swa_flag \\
    --warmup-epochs $WARMUP \\
    --seed $SEED \\
    --dataset-dir $DATASET_DIR \\
    --carry-winner 5:${winner} \\
    --evaluate-test \\
    --final-train-all \\
    --max-ap-component-pairs 50000 \\
    --output-dir $OUTPUT_BASE \\
    --device $DEVICE
"
    docker run "${DOCKER_OPTS[@]}" "$DOCKER_IMAGE" bash -c "$cmd" 2>&1 | tee -a "$MAIN_LOG"
}

run_phase_across_folds() {
    local phase=$1
    shift
    local winners=("$@")
    local -a pids=()
    local fail=0

    for fold_id in "${cv_folds[@]}"; do
        log "  Running Phase $phase on fold $fold_id"
        if [ "$PARALLEL_JOBS" -gt 1 ]; then
            (
                FOLD=$fold_id run_phase_with_winners "$phase" "${winners[@]}"
            ) &
            pids+=($!)
            while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL_JOBS" ]; do
                sleep 2
            done
        else
            FOLD=$fold_id run_phase_with_winners "$phase" "${winners[@]}"
        fi
    done

    if [ "$PARALLEL_JOBS" -gt 1 ]; then
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                fail=1
            fi
        done
        if [ "$fail" -ne 0 ]; then
            log_error "One or more parallel fold jobs failed in Phase $phase"
            exit 1
        fi
    fi
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
            # Get aggregated cross-fold metric value for this run_id
            local metric=$(python3 << PYTHON_EOF
import csv
from collections import defaultdict
with open('$results_file', 'r') as f:
    reader = csv.DictReader(f)
    vals = []
    for row in reader:
        if (row.get('run_id') or row.get('condition_id')) == '$winner':
            for m in ['$SELECTION_METRIC', 'val_cldice', 'best_val_cldice', 'val_f1', 'best_val_f1', 'threshold_f1', 'best_val_dice', 'val_dice']:
                if m in row and row[m]:
                    try:
                        vals.append(float(row[m]))
                        break
                    except Exception:
                        pass
if vals:
    print(f"{sum(vals)/len(vals):.6f}")
else:
    print("N/A")
PYTHON_EOF
)
            log "  $i. $winner → mean($SELECTION_METRIC)=$metric ✓✓ ADVANCE TO PHASE $((phase+1))"
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
    log "Configuration: epochs=$EPOCHS, device=$DEVICE, cv_version=$CV_VERSION"
    log "Selection metric: $SELECTION_METRIC"
    log "Locked assumptions: legacy=$LEGACY_V10_CHAIN, selection_metric=val_cldice, cv_version=4"
    echo "" | tee -a "$MAIN_LOG"

    local phases_to_run=(2 3 4 5 6)
    if [ $# -gt 0 ]; then
        phases_to_run=("$@")
    fi

    log "Will execute phases: ${phases_to_run[*]}"
    log "Strategy: Select top 2 winners per phase, then run locked test once with Top2 + legacy baseline"
    log "Selection tie-break (locked): mean($SELECTION_METRIC) > mean(val_f1) > lower std($SELECTION_METRIC)"
    log "Academic guardrail: test stripe is untouched in Phases 2-5 and used only in final Phase 6"
    log "Pruning (locked defaults): phase3=$PHASE3_CONDITIONS phase4=$PHASE4_CONDITIONS phase5=$PHASE5_CONDITIONS"
    log "Parallel speed mode: PARALLEL_JOBS=$PARALLEL_JOBS (same configs/seeds/metrics; wall-clock only)"
    echo "" | tee -a "$MAIN_LOG"
    prepare_inputs_and_dataset
    local -a cv_folds
    read -r -a cv_folds <<< "$(cv_folds_for_version)"
    log "CV folds used for model-selection phases (2-5): ${cv_folds[*]}"

    # Track winners across phases
    declare -a phase2_winners=()
    declare -a phase3_winners=()
    declare -a phase4_winners=()
    declare -a phase5_winners=()

    # ── Phase 2: CHM Variant Search
    if [[ " ${phases_to_run[@]} " =~ " 2 " ]]; then
        log_section "PHASE 2: CHM VARIANT SEARCH (5 conditions)"
        log "Testing: baseline, raw, gauss, masked, composite"
        log "Strategy: run all CV folds, aggregate by mean $SELECTION_METRIC, select top 2 variants"
        log "Execution order (Phase 2): per-condition cross-validation (each condition runs on folds ${cv_folds[*]} before next condition)"

        mkdir -p "$OUTPUT_BASE/phase2"
        IFS=',' read -r -a phase2_ids <<< "$PHASE2_CONDITIONS"
        for cond in "${phase2_ids[@]}"; do
            cond="$(echo "$cond" | xargs)"
            [ -z "$cond" ] && continue
            log "  Condition $cond: running folds ${cv_folds[*]}"
            for fold_id in "${cv_folds[@]}"; do
                log "    fold $fold_id"
                FOLD=$fold_id PHASE2_CONDITIONS="$cond" run_phase_with_winners 2
            done
        done

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
        log "Candidate pruning: keeping phase3 IDs [$PHASE3_CONDITIONS], dropping others"
        log "Strategy: run all CV folds, aggregate by mean $SELECTION_METRIC, select top 2 architecture/CHM combinations"

        mkdir -p "$OUTPUT_BASE/phase3"
        run_phase_across_folds 3 "${phase2_winners[@]}"

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
        log "Candidate pruning: keeping phase4 IDs [$PHASE4_CONDITIONS], dropping others"
        log "Strategy: run all CV folds, aggregate by mean $SELECTION_METRIC, select top 2 loss configurations"

        mkdir -p "$OUTPUT_BASE/phase4"
        run_phase_across_folds 4 "${phase3_winners[@]}"

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
        log "Candidate pruning: keeping phase5 IDs [$PHASE5_CONDITIONS], dropping others"
        log "Strategy: run all CV folds, aggregate by mean $SELECTION_METRIC, select top 2 augmentation strategies"

        mkdir -p "$OUTPUT_BASE/phase5"
        run_phase_across_folds 5 "${phase4_winners[@]}"

        print_phase_summary 5

        if winners_out=$(get_phase_winners 5 2); then
            readarray -t phase5_winners <<<"$winners_out"
        else
            phase5_winners=()
        fi
        log ""
        log_success "Phase 5 complete. Top augmentation configs: $(join_by_comma "${phase5_winners[@]}")"
    fi

    # ── Phase 6: Thesis Final Protocol (Top2 + Legacy V10 on locked test stripe)
    if [[ " ${phases_to_run[@]} " =~ " 6 " ]]; then
        log_section "PHASE 6: THESIS FINAL PROTOCOL (TOP2 + LEGACY V10, LOCKED TEST)"

        if [ ${#phase5_winners[@]} -eq 0 ]; then
            if winners_out=$(get_phase_winners 5 2); then
                readarray -t phase5_winners <<<"$winners_out"
            else
                phase5_winners=()
            fi
        fi

        local -a final_candidates=("${phase5_winners[@]}")
        local has_legacy=0
        for cfg in "${final_candidates[@]}"; do
            if [ "$cfg" = "$LEGACY_V10_CHAIN" ]; then
                has_legacy=1
                break
            fi
        done
        if [ "$has_legacy" -eq 0 ]; then
            final_candidates+=("$LEGACY_V10_CHAIN")
        fi

        log "Final candidate set (3-way locked-test comparison): $(join_by_comma "${final_candidates[@]}")"
        log "No parameter edits are allowed after Phase 5 winner selection."
        for cfg in "${final_candidates[@]}"; do
            run_final_locked_test "$cfg"
        done
    fi

    # ── Final Report
    log_section "ABLATION STUDY COMPLETE - TOP 2 STRATEGY"
    log "Runtime note: depends on CV_VERSION and number of conditions per phase."
    log "  CV_VERSION=$CV_VERSION uses folds: ${cv_folds[*]}"
    log "  Final protocol (Phase 6): top2 + legacy V10 chain on locked test stripe."
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

- **Selection metric**: \`$SELECTION_METRIC\` (cross-fold mean across spatial CV folds).
- **Held-out test use**: the test stripe is locked during Phases 2-5 and is evaluated only in final Phase 6 with \`--evaluate-test\`.
- **Seed**: $SEED
- **Epochs**: $EPOCHS
- **Warmup (linear LR)**: $WARMUP
- **SWA start epoch**: $SWA_START (set `NO_SWA=true` to disable)
- **Cross-validation**: Spatial CV version \`$CV_VERSION\` (see seg_pipeline/scripts/phase2_dataset_v3.py) — stripe 0 is held-out test and remains untouched in Phases 2-5.
- **Conservative candidate pruning**:
  - Phase 3: \`$PHASE3_CONDITIONS\`
  - Phase 4: \`$PHASE4_CONDITIONS\`
  - Phase 5: \`$PHASE5_CONDITIONS\`
- **Parallel wall-clock speedup**: \`PARALLEL_JOBS=$PARALLEL_JOBS\` (no methodological changes; same seeds/configs/metrics).

## Locked Assumptions

- **Legacy comparator** is locked to \`2A__3C__4H__5D\`.
- **Selection metric** across Phases 2-5 is \`val_cldice\` only.
- **CV protocol** is balanced 2-fold (\`CV_VERSION=4\`).


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

### Phase 6: Thesis Final Protocol (Top 2 from Phase 5 + legacy V10 chain on locked test stripe)

## Results Files

All detailed results in: `$OUTPUT_BASE`

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
  bash run_full_ablation_automated_top2.sh           # All phases (2-6)
  bash run_full_ablation_automated_top2.sh 2         # Phase 2 only (2.5h)
  bash run_full_ablation_automated_top2.sh 2 3       # Phases 2-3
  bash run_full_ablation_automated_top2.sh 6         # Final locked-test protocol only
  EPOCHS=5 bash run_full_ablation_automated_top2.sh 2 # Quick test (15 min)

Environment variables:
  EPOCHS=N         Epochs per condition (default: 75)
  SWA_START=N      SWA start epoch (default: 35)
  DEVICE=cuda|cpu  Device (default: cuda)
  NO_SWA=true      Disable SWA (default: false)
  CV_VERSION=4     Locked to balanced 2-fold protocol
  LEGACY_V10_CHAIN Run-id chain for baseline thesis comparator (default: 2A__3C__4H__5D)
  PHASE3_CONDITIONS Conservative pruning set (default: 3B,3C,3E)
  PHASE4_CONDITIONS Conservative pruning set (default: 4A,4D,4F,4H)
  PHASE5_CONDITIONS Conservative pruning set (default: 5A,5D,5E)
  PARALLEL_JOBS=N  Parallel fold jobs for wall-clock speedup (default: 1)

EOF
    exit 0
fi

main "$@"

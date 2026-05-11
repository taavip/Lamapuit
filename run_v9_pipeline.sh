#!/usr/bin/env bash
# V9 pipeline runner — Phase 1 bypass + ReduceLROnPlateau + SWA@25 + α=0.4/β=0.6
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/seg_pipeline/output/phase_logs_v9"
TRAIN_SCRIPT="${SCRIPT_DIR}/seg_pipeline/scripts/phase3_train_v9.py"
PREDICT_SCRIPT="${SCRIPT_DIR}/seg_pipeline/scripts/phase5_predict_v9.py"
PHASE1_SCRIPT="${SCRIPT_DIR}/seg_pipeline/scripts/phase1_mask_synthesis.py"
PHASE2_SCRIPT="${SCRIPT_DIR}/seg_pipeline/scripts/phase2_dataset_v3.py"
OUTPUT_DIR="${SCRIPT_DIR}/seg_pipeline/output/phase3_runs_v9"
DATASET_OUTPUT="${SCRIPT_DIR}/seg_pipeline/output/phase2_dataset_v9"

mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/v9_run_${TIMESTAMP}.log"

_interrupted=0
_current_stage="(not started)"
trap '_interrupted=1; echo "" | tee -a "${MASTER_LOG}"; echo "=== INTERRUPTED at stage: ${_current_stage} at $(date) ===" | tee -a "${MASTER_LOG}"' SIGINT SIGTERM

log() {
    echo "$*" | tee -a "${MASTER_LOG}"
}

run_stage() {
    local stage_name="$1"; shift
    _current_stage="${stage_name}"
    local stage_log="${LOG_DIR}/${stage_name}_${TIMESTAMP}.log"

    log ""
    log "========================================================================"
    log "  STAGE: ${stage_name}"
    log "  START: $(date)"
    log "  LOG  : ${stage_log}"
    log "========================================================================"

    if "$@" 2>&1 | tee -a "${stage_log}" | tee -a "${MASTER_LOG}"; then
        log "  DONE: ${stage_name} at $(date)"
        return 0
    else
        local rc=$?
        log "  FAILED: ${stage_name} (exit ${rc}) at $(date)"
        return ${rc}
    fi
}

log ""
log "###################################################################"
log "#  V9 PIPELINE RUN — $(date)"
log "#  Fixes: Phase1 bypass | ReduceLROnPlateau | SWA@25 | α=0.4/β=0.6"
log "#  MASTER LOG: ${MASTER_LOG}"
log "###################################################################"
log ""

# ------------------------------------------------------------------ STAGE 1
# Verify new labels are in place
# ------------------------------------------------------------------ STAGE 1
run_stage "verify_labels" \
    bash -c "test -L ${SCRIPT_DIR}/seg_pipeline/input/cdw_labels_MP.gpkg && echo 'Labels symlinked OK' || (echo 'Labels not found'; exit 1)"

if [[ ${_interrupted} -eq 1 ]]; then exit 130; fi

# ------------------------------------------------------------------ STAGE 2
# Phase 1: Rasterize labels with ensemble bypass
#   --neg-threshold 1.0  → all non-GPKG valid CHM pixels become confirmed negatives
#                          (condition: ensemble_prob < 1.0 is always True for any prob in [0,1])
#   --noisy-threshold 1.0 → no pixels excluded as noisy (prob > 1.0 impossible)
# Together: all valid CHM pixels = either positive (from GPKG) or negative (everything else)
# This bypasses the broken Phase 0 ensemble which outputs prob≈0 on the new composite CHM.
# ------------------------------------------------------------------ STAGE 2
run_stage "phase1_rasterize_bypass" \
    python3 "${PHASE1_SCRIPT}" \
        --device cuda \
        --neg-threshold 1.0 \
        --noisy-threshold 1.0

if [[ ${_interrupted} -eq 1 ]]; then exit 130; fi

# ------------------------------------------------------------------ STAGE 3
# Phase 2: Rebuild patch dataset from new mask
# ------------------------------------------------------------------ STAGE 3
run_stage "phase2_rebuild_dataset" \
    python3 "${PHASE2_SCRIPT}" \
        --output-dir "${DATASET_OUTPUT}"

if [[ ${_interrupted} -eq 1 ]]; then exit 130; fi

# ------------------------------------------------------------------ STAGE 4
# Training smoke test (fold 0, 5 epochs)
# ------------------------------------------------------------------ STAGE 4
run_stage "smoke_test_fold0_5ep" \
    python3 "${TRAIN_SCRIPT}" \
        --fold 0 \
        --epochs 5 \
        --config v7c \
        --dataset-dir "${DATASET_OUTPUT}" \
        --output-dir "${OUTPUT_DIR}" \
        --device cuda

if [[ ${_interrupted} -eq 1 ]]; then exit 130; fi

log ""
log "Smoke test PASSED. Proceeding to full 4-fold training (100 epochs)."
log ""

# ------------------------------------------------------------------ STAGE 5
# Full training (all 4 folds, 100 epochs)
#   ReduceLROnPlateau + warmup=5ep + SWA start=25 + α=0.4/β=0.6
# ------------------------------------------------------------------ STAGE 5
for FOLD in 0 1 2 3; do
    if [[ ${_interrupted} -eq 1 ]]; then break; fi
    run_stage "full_fold${FOLD}_v7c_100ep" \
        python3 "${TRAIN_SCRIPT}" \
            --fold "${FOLD}" \
            --epochs 100 \
            --config v7c \
            --dataset-dir "${DATASET_OUTPUT}" \
            --output-dir "${OUTPUT_DIR}" \
            --device cuda
done

if [[ ${_interrupted} -eq 1 ]]; then exit 130; fi

# ------------------------------------------------------------------ STAGE 6
# Prediction: TTA + 4-fold ensemble + adaptive threshold
# ------------------------------------------------------------------ STAGE 6
run_stage "predict_v9_adaptive_threshold" \
    python3 "${PREDICT_SCRIPT}" \
        --runs-dir "${OUTPUT_DIR}" \
        --output-dir "${SCRIPT_DIR}/seg_pipeline/output/phase5_predict_v9" \
        --device cuda

if [[ ${_interrupted} -eq 1 ]]; then exit 130; fi

# ------------------------------------------------------------------ STAGE 7
# Summary
# ------------------------------------------------------------------ STAGE 7
log ""
log "========================================================================"
log "  V9 FULL PIPELINE SUMMARY — $(date)"
log "========================================================================"

python3 - "${OUTPUT_DIR}" <<'PYEOF'
import json, pathlib, statistics, sys

base = pathlib.Path(sys.argv[1]) / "composite"
f1s = []
for fold in range(4):
    mf = base / f"fold{fold}" / "metrics.json"
    if mf.exists():
        d = json.load(open(mf))
        f1 = d.get("best_val_f1", 0)
        ep = d.get("n_epochs_trained", "?")
        swa_ran = d.get("swa_enabled", False)
        swa_ep = d.get("swa_epochs", 0)
        swa_f1 = d.get("swa_val_f1", 0)
        use_swa = d.get("use_swa_for_inference", False)
        thr = d.get("best_threshold", "?")
        f1s.append(f1)
        swa_str = (f"SWA {swa_ep}ep, swa_f1={swa_f1:.4f}, "
                   f"use_for_inference={use_swa}") if swa_ran else "no SWA"
        print(f"  fold{fold}: best_f1={f1:.4f}  epochs={ep}  thr={thr}  ({swa_str})")
    else:
        print(f"  fold{fold}: metrics.json NOT FOUND")

if f1s:
    m = statistics.mean(f1s)
    s = statistics.stdev(f1s) if len(f1s) > 1 else 0
    print(f"")
    print(f"  MEAN F1 = {m:.4f}  (stdev={s:.4f})")
    print(f"")
    print(f"  V6 baseline: F1=0.2289")
    print(f"  V7 result:   F1=0.2420")
    print(f"  V8 result:   F1=0.3894  (broken inference, max_prob=0.21)")
    print(f"  V9 target:   F1>=0.40, inference max_prob>=0.50")

# Show threshold summary if it exists
thr_file = pathlib.Path(sys.argv[1]).parent.parent / "phase5_predict_v9" / "threshold_summary.json"
if thr_file.exists():
    t = json.load(open(thr_file))
    print(f"")
    print(f"  Inference threshold summary:")
    print(f"    ensemble max_prob = {t.get('ensemble_max_prob', '?'):.4f}")
    print(f"    F1-optimal thr    = {t.get('threshold_f1_optimal', '?'):.3f}")
    print(f"    Prec>=0.50 thr    = {t.get('threshold_p50', '?'):.3f}")
    print(f"    Best F1           = {t.get('best_f1', '?'):.4f}")
PYEOF

log ""
log "Full logs in: ${LOG_DIR}"
log "Models  in:   ${OUTPUT_DIR}"
log "Predictions: ${SCRIPT_DIR}/seg_pipeline/output/phase5_predict_v9/"
log ""
log "Done at $(date)"

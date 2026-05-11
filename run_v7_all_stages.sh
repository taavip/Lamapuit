#!/usr/bin/env bash
# V7 pipeline runner — crash-proof logging, never writes to /tmp/.
# Stages: smoke test → fold-0 ablation (v7a/v7b/v7c) → full 4-fold run (best config).
# All logs written to seg_pipeline/output/phase_logs_v7/ which lives inside the
# project directory and survives WSL/Docker restarts.
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths (all relative to repo root so they work both on host and in Docker)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/seg_pipeline/output/phase_logs_v7"
TRAIN_SCRIPT="${SCRIPT_DIR}/seg_pipeline/scripts/phase3_train_v7.py"
OUTPUT_DIR="${SCRIPT_DIR}/seg_pipeline/output/phase3_runs_v7"

mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/v7_run_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# Interrupt handler — record where training was when killed
# ---------------------------------------------------------------------------
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

    # Run command, tee both to stage log and master log
    if "$@" 2>&1 | tee -a "${stage_log}" | tee -a "${MASTER_LOG}"; then
        log "  DONE: ${stage_name} at $(date)"
        return 0
    else
        local rc=$?
        log "  FAILED: ${stage_name} (exit ${rc}) at $(date)"
        return ${rc}
    fi
}

# ---------------------------------------------------------------------------
# Helper: read best_f1 from a metrics.json
# ---------------------------------------------------------------------------
best_f1() {
    local d="${OUTPUT_DIR}/composite/${1}"
    if [[ -f "${d}/metrics.json" ]]; then
        python3 -c "import json,sys; d=json.load(open('${d}/metrics.json')); print(d.get('best_f1', 0))"
    else
        echo "0"
    fi
}

# ---------------------------------------------------------------------------
log ""
log "###################################################################"
log "#  V7 PIPELINE RUN — $(date)"
log "#  MASTER LOG: ${MASTER_LOG}"
log "###################################################################"
log ""

# ------------------------------------------------------------------ STAGE 1
# Smoke test: fold 0, 5 epochs, default v7b config
# Expected: runs to completion, best.pt written, f1 printed each epoch
# ------------------------------------------------------------------ STAGE 1
run_stage "smoke_test_fold0_5ep" \
    python3 "${TRAIN_SCRIPT}" \
        --fold 0 \
        --epochs 5 \
        --config v7b \
        --device cuda

if [[ ${_interrupted} -eq 1 ]]; then exit 130; fi

log ""
log "Smoke test PASSED. Proceeding to fold-0 ablation (60 epochs each)."
log ""

# ------------------------------------------------------------------ STAGE 2
# Ablation: fold 0 only, 60 epochs, three configs
# v7a: flip Tversky only (no CLDice, no soft targets)
# v7b: flip Tversky + CLDice(λ=0.5)
# v7c: flip Tversky + CLDice + soft distance-transform targets
# ------------------------------------------------------------------ STAGE 2

run_stage "ablation_v7a_fold0_60ep" \
    python3 "${TRAIN_SCRIPT}" \
        --fold 0 \
        --epochs 60 \
        --config v7a \
        --device cuda \
        --output-dir "${OUTPUT_DIR}_ablation"

if [[ ${_interrupted} -eq 1 ]]; then exit 130; fi

run_stage "ablation_v7b_fold0_60ep" \
    python3 "${TRAIN_SCRIPT}" \
        --fold 0 \
        --epochs 60 \
        --config v7b \
        --device cuda \
        --output-dir "${OUTPUT_DIR}_ablation"

if [[ ${_interrupted} -eq 1 ]]; then exit 130; fi

run_stage "ablation_v7c_fold0_60ep" \
    python3 "${TRAIN_SCRIPT}" \
        --fold 0 \
        --epochs 60 \
        --config v7c \
        --device cuda \
        --output-dir "${OUTPUT_DIR}_ablation"

if [[ ${_interrupted} -eq 1 ]]; then exit 130; fi

# ------------------------------------------------------------------ STAGE 3
# Pick the ablation winner based on best_f1 from metrics.json
# ------------------------------------------------------------------ STAGE 3
log ""
log "--- Ablation note ---"
log "  Each ablation config (v7a/v7b/v7c) writes to the same fold0 dir;"
log "  compare val_f1 lines in the individual stage logs above to pick the best."
log "  Default full-run config: v7b (flip Tversky + CLDice)."
log "  To override: V7_CONFIG=v7a bash run_v7_all_stages.sh"

# The full run defaults to v7b unless user overrides via V7_CONFIG env variable.
BEST_CONFIG="${V7_CONFIG:-v7b}"
log ""
log "Selected config for full run: ${BEST_CONFIG}"
log "(Override with: V7_CONFIG=v7a|v7b|v7c bash run_v7_all_stages.sh)"
log ""

# ------------------------------------------------------------------ STAGE 4
# Full run: all 4 folds, 100 epochs, best config
# ------------------------------------------------------------------ STAGE 4
for FOLD in 0 1 2 3; do
    if [[ ${_interrupted} -eq 1 ]]; then break; fi
    run_stage "full_fold${FOLD}_${BEST_CONFIG}_100ep" \
        python3 "${TRAIN_SCRIPT}" \
            --fold "${FOLD}" \
            --epochs 100 \
            --config "${BEST_CONFIG}" \
            --device cuda \
            --output-dir "${OUTPUT_DIR}"
done

if [[ ${_interrupted} -eq 1 ]]; then exit 130; fi

# ------------------------------------------------------------------ STAGE 5
# Summary: collect best_f1 from all folds
# ------------------------------------------------------------------ STAGE 5
log ""
log "========================================================================"
log "  V7 FULL RUN SUMMARY — $(date)"
log "========================================================================"

python3 - "${OUTPUT_DIR}" <<'PYEOF' | tee -a "${MASTER_LOG}"
import json, pathlib, statistics, sys

base = pathlib.Path(sys.argv[1]) / "composite"
f1s = []
for fold in range(4):
    mf = base / f"fold{fold}" / "metrics.json"
    if mf.exists():
        d = json.load(open(mf))
        f1 = d.get("best_f1", 0)
        ep = d.get("best_epoch", "?")
        thr = d.get("best_threshold", "?")
        f1s.append(f1)
        print(f"  fold{fold}: best_f1={f1:.4f}  epoch={ep}  threshold={thr}")
    else:
        print(f"  fold{fold}: metrics.json NOT FOUND")

if f1s:
    print(f"  MEAN F1 = {statistics.mean(f1s):.4f}  (stdev={statistics.stdev(f1s):.4f})" if len(f1s) > 1 else f"  MEAN F1 = {f1s[0]:.4f}")
PYEOF

log ""
log "  V6 baseline: mean F1=0.2289, precision≈0.11, recall≈0.46"
log "  V7 target:   precision>0.25, F1>0.25"
log ""
log "Full logs in: ${LOG_DIR}"
log "Models  in:   ${OUTPUT_DIR}"
log "Done at $(date)"

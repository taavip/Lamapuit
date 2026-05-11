#!/bin/bash
# Smoke test for ablation study V10.2 infrastructure
# Tests: imports, single condition training (Phase 2, condition 0, 5 epochs)

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/smoke_test_$(date +%Y%m%d_%H%M%S).log"

echo "==============================================================================" | tee "$LOG_FILE"
echo "ABLATION STUDY V10.2 — SMOKE TEST" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run inside Docker container
DOCKER_IMAGE="lamapuit:gpu"

echo "[$(date)] Starting Docker container..." | tee -a "$LOG_FILE"

docker run --rm \
    --gpus all \
    -v "$REPO_ROOT:$REPO_ROOT" \
    --workdir "$REPO_ROOT" \
    "$DOCKER_IMAGE" \
    bash -c "
set -e
cd $REPO_ROOT

echo 'Activating conda environment...' >> $LOG_FILE
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

echo '[Docker] Environment setup complete' >> $LOG_FILE
echo '[Docker] Python:' \$(python3 --version) >> $LOG_FILE
echo '[Docker] PyTorch:' \$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A') >> $LOG_FILE
echo '[Docker] CUDA available:' \$(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A') >> $LOG_FILE
echo '' >> $LOG_FILE

echo '[Docker] Test 1: Import extended_metrics...' >> $LOG_FILE
python3 -c '
import sys
sys.path.insert(0, \"seg_pipeline/scripts\")
from common.extended_metrics import boundary_iou, cldice_metric, ap_at_iou
print(\"✓ extended_metrics imports OK\")
' >> $LOG_FILE 2>&1

echo '[Docker] Test 2: Check architecture configs...' >> $LOG_FILE
python3 -c '
import sys
sys.path.insert(0, \"seg_pipeline/scripts\")
from phase3_train_v10 import _ARCH_CONFIGS
print(f\"✓ _ARCH_CONFIGS: {list(_ARCH_CONFIGS.keys())}\")
assert \"unet_effb2\" in _ARCH_CONFIGS, \"Missing unet_effb2\"
assert \"unetpp_effb4\" in _ARCH_CONFIGS, \"Missing unetpp_effb4\"
assert \"deeplabv3p_effb2\" in _ARCH_CONFIGS, \"Missing deeplabv3p_effb2\"
print(\"✓ All new architectures registered\")
' >> $LOG_FILE 2>&1

echo '[Docker] Test 3: Import ablation runner...' >> $LOG_FILE
python3 -c '
import sys
sys.path.insert(0, \"seg_pipeline/scripts\")
from phase3_ablation_v10 import AblationConfig, run_phase, DEVICE
print(f\"✓ phase3_ablation_v10 imports OK (device: {DEVICE})\")
' >> $LOG_FILE 2>&1

echo '[Docker] Test 4: Smoke test - Phase 2, Condition 0 (5 epochs)...' >> $LOG_FILE
python3 seg_pipeline/scripts/phase3_ablation_v10.py \
    --phase 2 \
    --condition 0 \
    --epochs 5 \
    --no-swa \
    --device cuda \
    --output-dir seg_pipeline/output/ablation_v10_smoke \
    --mask-tif seg_pipeline/output/phase1_masks/406455_2021_tava_truemask.tif \
    >> $LOG_FILE 2>&1

echo '[Docker] All tests passed!' >> $LOG_FILE
" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"
echo "SMOKE TEST COMPLETE" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Next steps:" | tee -a "$LOG_FILE"
echo "  1. Review log: tail -100 $LOG_FILE" | tee -a "$LOG_FILE"
echo "  2. If passed, run full test: bash run_ablation_v10.sh cuda 2>&1 | tee ablation_full_test.log" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

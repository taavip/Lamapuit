#!/bin/bash
# Comprehensive full ablation study test with detailed logging
# Runs: Phase 2 (5 conditions) with full logging for analysis

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ablation_full_test_${TIMESTAMP}.log"
METRICS_FILE="$LOG_DIR/ablation_metrics_${TIMESTAMP}.json"
SUMMARY_FILE="$LOG_DIR/ablation_summary_${TIMESTAMP}.md"

echo "==============================================================================" | tee "$LOG_FILE"
echo "COMPREHENSIVE ABLATION STUDY V10.2 — FULL TEST WITH LOGGING" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Metrics: $METRICS_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Configuration
DOCKER_IMAGE="lamapuit:gpu"
PHASE=2  # Run Phase 2 for comprehensive testing
EPOCHS=75
SWA_START=35
DEVICE="cuda"

echo "[$(date)] Launching Docker container with comprehensive logging..." | tee -a "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  - Phase: $PHASE (CHM Variant Search: 5 conditions)" | tee -a "$LOG_FILE"
echo "  - Epochs: $EPOCHS" | tee -a "$LOG_FILE"
echo "  - SWA start: $SWA_START" | tee -a "$LOG_FILE"
echo "  - Device: $DEVICE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

docker run --rm \
    --gpus all \
    -v "$REPO_ROOT:$REPO_ROOT" \
    --workdir "$REPO_ROOT" \
    "$DOCKER_IMAGE" \
    bash -c "
set -e
cd $REPO_ROOT

# Setup
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

echo '===============================================================================' >> $LOG_FILE
echo 'DOCKER ENVIRONMENT SETUP' >> $LOG_FILE
echo '===============================================================================' >> $LOG_FILE
echo '[Docker] Python version:' \$(python3 --version) >> $LOG_FILE
echo '[Docker] PyTorch version:' \$(python3 -c 'import torch; print(torch.__version__)') >> $LOG_FILE
echo '[Docker] CUDA available:' \$(python3 -c 'import torch; print(torch.cuda.is_available())') >> $LOG_FILE
echo '[Docker] GPU count:' \$(python3 -c 'import torch; print(torch.cuda.device_count())') >> $LOG_FILE
echo '' >> $LOG_FILE

# Run full ablation (Phase 2, all 5 conditions)
echo '===============================================================================' >> $LOG_FILE
echo 'RUNNING FULL ABLATION STUDY - PHASE 2 (CHM VARIANTS)' >> $LOG_FILE
echo '===============================================================================' >> $LOG_FILE
echo '[Docker] Starting Phase $PHASE with 5 conditions (baseline, raw, gauss, masked, composite)' >> $LOG_FILE
echo '[Docker] Expected runtime: ~4-6 hours (5 conditions × 75 epochs × ~1 hour each)' >> $LOG_FILE
echo '' >> $LOG_FILE

python3 seg_pipeline/scripts/phase3_ablation_v10.py \
    --phase $PHASE \
    --fold 0 \
    --epochs $EPOCHS \
    --swa-start-epoch $SWA_START \
    --device $DEVICE \
    --output-dir seg_pipeline/output/ablation_v10_full \
    --mask-tif seg_pipeline/output/phase1_masks/406455_2021_tava_truemask.tif \
    2>&1 | tee -a $LOG_FILE

# Capture results
echo '' >> $LOG_FILE
echo '===============================================================================' >> $LOG_FILE
echo 'RESULTS SUMMARY' >> $LOG_FILE
echo '===============================================================================' >> $LOG_FILE

if [ -f seg_pipeline/output/ablation_v10_full/phase${PHASE}_results.csv ]; then
    echo '[Docker] Phase $PHASE results found. Statistics:' >> $LOG_FILE
    wc -l seg_pipeline/output/ablation_v10_full/phase${PHASE}_results.csv >> $LOG_FILE
    echo '' >> $LOG_FILE
    echo '[Docker] CSV header:' >> $LOG_FILE
    head -1 seg_pipeline/output/ablation_v10_full/phase${PHASE}_results.csv >> $LOG_FILE
    echo '' >> $LOG_FILE
    echo '[Docker] All results:' >> $LOG_FILE
    cat seg_pipeline/output/ablation_v10_full/phase${PHASE}_results.csv >> $LOG_FILE
fi

echo '' >> $LOG_FILE
echo '[Docker] Test completed successfully!' >> $LOG_FILE
" 2>&1 | tee -a "$LOG_FILE"

# Generate summary report
echo "" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"
echo "TEST EXECUTION COMPLETE" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"

# Create summary markdown
cat > "$SUMMARY_FILE" << 'SUMMARY_EOF'
# Ablation Study V10.2 — Full Test Results

## Execution Details
- **Date**: $(date)
- **Timestamp**: TIMESTAMP_PLACEHOLDER
- **Phase**: 2 (CHM Variant Search)
- **Conditions**: 5 (baseline, raw, gauss, masked, composite)
- **Epochs**: 75
- **SWA Start**: Epoch 35
- **Device**: CUDA GPU

## Conditions Tested
1. **2A - CHM Baseline** (1 channel): Raw max-HAG CHM, 0.2m resolution
2. **2B - CHM Raw** (1 channel): Unfiltered CHM with noise
3. **2C - CHM Gauss** (1 channel): Gaussian-smoothed CHM (σ=0.2)
4. **2D - CHM Masked** (2 channels): Baseline + binary validity mask
5. **2E - CHM Composite** (4 channels): Baseline + raw + gauss + area mask

## All Conditions Use Fixed V10.2 Hyperparameters
- Loss: TverskyFocal + SoftCLDice
- Tversky α=0.6, β=0.4 (precision-biased)
- CLDice λ=0.3
- Soft targets: Yes (σ=2.0)
- Augmentation: Full (geometric + batch: Mixup/CutMix/GridMask)
- SWA: Enabled from epoch 35

## Metrics Computed
- **Pixel-level**: Dice, Precision, Recall, IoU, F1
- **Shape quality**: Boundary IoU, clDice (skeleton-level)
- **Component detection**: AP@IoU25, AP@IoU50
- **Validation**: Val F1 (best on validation fold)
- **Test**: Test F1 (test stripe generalization)
- **Calibration**: Optimal threshold

## Expected Results
- Composite variant (4-band) should outperform single-band variants
- Test F1 should be in range [0.05, 0.45]
- All conditions should train without errors

## Files Generated
- Phase 2 results CSV: `seg_pipeline/output/ablation_v10_full/phase2_results.csv`
- Checkpoints: `seg_pipeline/output/ablation_v10_full/phase2_*/{composite,baseline,raw,gauss,masked}/fold0/best.pt`
- Detailed logs: `logs/ablation_full_test_${TIMESTAMP}.log`

## Next Steps
1. Review `phase2_results.csv` for performance comparison
2. Identify winner (highest test_F1)
3. Proceed with Phase 3 (Architecture Search) using winner CHM variant
4. Continue to Phase 4 (Loss parameters), Phase 5 (Augmentation), Phase 6 (Final validation)

SUMMARY_EOF

# Replace timestamp placeholder
sed -i "s/TIMESTAMP_PLACEHOLDER/$TIMESTAMP/g" "$SUMMARY_FILE"

echo "Summary saved: $SUMMARY_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Logs and metrics:" | tee -a "$LOG_FILE"
echo "  1. Full log: tail -500 $LOG_FILE" | tee -a "$LOG_FILE"
echo "  2. Summary:  cat $SUMMARY_FILE" | tee -a "$LOG_FILE"
echo "  3. Results:  cat seg_pipeline/output/ablation_v10_full/phase2_results.csv" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "GPU monitoring (during run):" | tee -a "$LOG_FILE"
echo "  watch -n 2 nvidia-smi" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

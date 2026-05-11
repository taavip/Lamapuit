#!/bin/bash
# Comprehensive Ablation Study Phase 3: Model Architecture Search
# Tests 5 architectures with each of the top 2 CHM variants from Phase 2
# Advancement: Top 2 (architecture × CHM combination) move to Phase 4

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$REPO_ROOT/seg_pipeline/output/ablation_v2_comprehensive"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase3_ablation_${TIMESTAMP}.log"
RESULTS_FILE="$OUTPUT_DIR/phase3_results_${TIMESTAMP}.csv"

echo "=============================================================================="
echo "COMPREHENSIVE ABLATION STUDY — PHASE 3: Architecture Search"
echo "=============================================================================="
echo "Date: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Architectures to test (with each of top 2 CHM variants):"
echo "  3A) UNet + EfficientNet-B2"
echo "  3B) UNet++ + EfficientNet-B0"
echo "  3C) UNet++ + EfficientNet-B2 (V10.2 current)"
echo "  3D) UNet++ + EfficientNet-B4"
echo "  3E) DeepLabV3+ + EfficientNet-B2"
echo ""
echo "Training configuration (all conditions):"
echo "  Fold: 0 (train=95 patches, val=118 patches)"
echo "  Epochs: 75 (SWA from epoch 35)"
echo "  Loss: TverskyFocal (α=0.6, β=0.4) + SoftCLDice (λ=0.3)"
echo "  Augmentation: Full (geometric + Mixup/CutMix/GridMask)"
echo "  Metrics: Dice, clDice, IoU, Precision, Recall"
echo ""

# Initialize results CSV
echo "chm_variant,architecture,fold_id,best_val_dice,best_val_cldice,best_val_iou,best_val_prec,best_val_rec,epochs_trained,swa_val_dice" > "$RESULTS_FILE"

docker run --rm \
    --gpus all \
    -v "$REPO_ROOT:$REPO_ROOT" \
    --workdir "$REPO_ROOT" \
    -e NO_ALBUMENTATIONS_UPDATE=1 \
    lamapuit:gpu \
    bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect
export NO_ALBUMENTATIONS_UPDATE=1

echo '=============================================================================='
echo 'Starting Phase 3 training loop...'
echo '=============================================================================='

# Read top 2 CHM variants from Phase 2 results
# This is set by the wrapper script or user review
CHM_WINNERS=(\${CHM_WINNERS[@]})

for chm_variant in \"\${CHM_WINNERS[@]}\"; do
  echo ''
  echo \"========================================\"
  echo \"CHM Variant: \$chm_variant\"
  echo \"========================================\"

  for arch in unet_effb2 unetpp_effb0 unetpp_effb2 unetpp_effb4 deeplabv3p_effb2; do
    echo ''
    echo \"Training \$chm_variant × \$arch\"
    echo \"[$(date)] Starting training...\"

    python3 seg_pipeline/scripts/phase3_train_v10.py \
      --variant \$chm_variant \
      --architecture \$arch \
      --fold 0 \
      --epochs 75 \
      --swa-start-epoch 35 \
      --dataset-dir seg_pipeline/output/phase2_dataset_v3 \
      --device cuda \
      --output-dir $OUTPUT_DIR 2>&1 | tee -a $LOG_FILE

    echo \"[$(date)] Completed \$chm_variant × \$arch\"

    # Extract metrics
    if [ -f \"$OUTPUT_DIR/\${chm_variant}_\${arch}/fold0/metrics.json\" ]; then
      python3 -c \"
import json
with open('$OUTPUT_DIR/\${chm_variant}_\${arch}/fold0/metrics.json') as f:
    m = json.load(f)
print(f\\\"\$chm_variant,\$arch,0,{m['best_val_dice']:.4f},{m.get('best_val_cldice', 0):.4f},{m.get('best_val_iou', 0):.4f},{m.get('best_val_prec', 0):.4f},{m.get('best_val_rec', 0):.4f},{m['n_epochs_trained']},{m.get('swa_val_dice', m['best_val_dice']):.4f}\\\")
      \"
    fi >> $RESULTS_FILE
  done
done

echo ''
echo '=============================================================================='
echo 'PHASE 3 COMPLETE'
echo '=============================================================================='

" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=============================================================================="
echo "PHASE 3 RESULTS SUMMARY"
echo "=============================================================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Top 2 winners (by best_val_dice):"
python3 << PYTHON_EOF
import csv
results = []
with open('$RESULTS_FILE', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append((f"{row['chm_variant']}×{row['architecture']}", float(row['best_val_dice'])))

results.sort(key=lambda x: x[1], reverse=True)
for i, (combo, dice) in enumerate(results[:2], 1):
    print(f"  {i}. {combo:30} → dice={dice:.4f} ✓ ADVANCE TO PHASE 4")

for i, (combo, dice) in enumerate(results[2:], 3):
    print(f"  {i}. {combo:30} → dice={dice:.4f} (did not advance)")
PYTHON_EOF

echo ""
echo "Full results:"
cat "$RESULTS_FILE"
echo ""
echo "=============================================================================="
echo "Next step: Review winners, update PHASE3_WINNERS.md, then run Phase 4"
echo "=============================================================================="
echo ""

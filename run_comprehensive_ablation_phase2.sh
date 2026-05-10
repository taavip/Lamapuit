#!/bin/bash
# Comprehensive Ablation Study Phase 2: CHM Variant Search
# All 5 conditions train with IDENTICAL dataset (80:20 train/test split)
# Logging: timestamps + metrics (no F1 in epoch output)
# Advancement: Top 2 winners move to Phase 3 (user decision)

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$REPO_ROOT/seg_pipeline/output/ablation_v2_comprehensive"
LOG_DIR="$REPO_ROOT/logs"
DATASET_DIR="$REPO_ROOT/seg_pipeline/output/phase2_dataset_v3"
MASK_TIF="$REPO_ROOT/source/406455_2021_tava/phase1_masks/406455_2021_tava_truemask.tif"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$DATASET_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase2_ablation_${TIMESTAMP}.log"
RESULTS_FILE="$OUTPUT_DIR/phase2_results_${TIMESTAMP}.csv"

echo "=============================================================================="
echo "COMPREHENSIVE ABLATION STUDY — PHASE 2: CHM Variant Search"
echo "=============================================================================="
echo "Date: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Ensure all CHM variant datasets exist
echo "Checking CHM variant datasets..."
for variant in baseline raw gauss masked composite; do
    if [ ! -f "$DATASET_DIR/patch_index_${variant}.csv" ]; then
        echo "Generating dataset for variant: $variant"
        docker run --rm \
            -v "$REPO_ROOT:$REPO_ROOT" \
            --workdir "$REPO_ROOT" \
            lamapuit:gpu \
            bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect
export NO_ALBUMENTATIONS_UPDATE=1
python3 seg_pipeline/scripts/phase2_dataset_v3.py \
    --variant $variant \
    --mask-tif $MASK_TIF \
    --output-dir $DATASET_DIR
" 2>&1 | grep -v "UserWarning\|albumentations\|pip as the\|venv\|--root-user-action\|verbose parameter" || true
    fi
done
echo ""
echo "Variants to test:"
echo "  2A) Baseline (1-band max-HAG CHM)"
echo "  2B) Raw (1-band unfiltered CHM)"
echo "  2C) Gauss (1-band Gaussian-smoothed CHM)"
echo "  2D) Masked (2-band: baseline + validity mask)"
echo "  2E) Composite (4-band: baseline + raw + gauss + validity mask)"
echo ""
echo "Training configuration (all variants):"
echo "  Fold: 0 (train=95 patches, val=118 patches)"
echo "  Epochs: 75 (SWA from epoch 35)"
echo "  Loss: TverskyFocal (α=0.6, β=0.4) + SoftCLDice (λ=0.3)"
echo "  Augmentation: Full (geometric + Mixup/CutMix/GridMask)"
echo "  Metrics: Dice, clDice, IoU, Precision, Recall (no F1 in epoch logs)"
echo ""
echo "Log format:"
echo "  [TIMESTAMP] [V10|VARIANT|fold0] epoch=NNN loss=X dice=X cldice=X ..."
echo ""

# Initialize results CSV
echo "variant,fold_id,best_val_dice,best_val_cldice,best_val_iou,best_val_prec,best_val_rec,epochs_trained,swa_val_dice,convergence_epoch" > "$RESULTS_FILE"

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
echo 'Starting Phase 2 training loop...'
echo '=============================================================================='

for variant in baseline raw gauss masked composite; do
  echo ''
  echo \"Training variant: \$variant\"
  echo \"[$(date)] Starting \$variant training...\"

  python3 seg_pipeline/scripts/phase3_train_v10.py \
    --variant \$variant \
    --fold 0 \
    --epochs 75 \
    --swa-start-epoch 35 \
    --dataset-dir seg_pipeline/output/phase2_dataset_v3 \
    --device cuda \
    --output-dir $OUTPUT_DIR

  echo \"[$(date)] Completed \$variant training\"

  # Extract metrics
  if [ -f \"$OUTPUT_DIR/\$variant/fold0/metrics.json\" ]; then
    python3 -c \"
import json
with open('$OUTPUT_DIR/\$variant/fold0/metrics.json') as f:
    m = json.load(f)
print(f\\\"\$variant,0,{m['best_val_dice']:.4f},{m.get('best_val_cldice', 0):.4f},{m.get('best_val_iou', 0):.4f},{m.get('best_val_prec', 0):.4f},{m.get('best_val_rec', 0):.4f},{m['n_epochs_trained']},{m['swa_val_f1']:.4f},35\\\")
    \"
  fi >> $RESULTS_FILE
done

echo ''
echo '=============================================================================='
echo 'PHASE 2 COMPLETE'
echo '=============================================================================='

" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=============================================================================="
echo "PHASE 2 RESULTS SUMMARY"
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
        results.append((row['variant'], float(row['best_val_dice'])))

results.sort(key=lambda x: x[1], reverse=True)
for i, (variant, dice) in enumerate(results[:2], 1):
    print(f"  {i}. {variant:12} → dice={dice:.4f} ✓ ADVANCE TO PHASE 3")

for i, (variant, dice) in enumerate(results[2:], 3):
    print(f"  {i}. {variant:12} → dice={dice:.4f} (did not advance)")
PYTHON_EOF

echo ""
echo "Full results:"
cat "$RESULTS_FILE"
echo ""
echo "=============================================================================="
echo "Next step: Review winners, update PHASE2_WINNERS.md, then run Phase 3"
echo "=============================================================================="
echo ""

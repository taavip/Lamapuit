#!/bin/bash
# Comprehensive Ablation Study Phase 4: Loss Function & Parameter Search
# Tests 8 loss configurations with top CHM variant × architecture from Phases 2-3
# Advancement: Top 2 move to Phase 5

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$REPO_ROOT/seg_pipeline/output/ablation_v2_comprehensive"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase4_ablation_${TIMESTAMP}.log"
RESULTS_FILE="$OUTPUT_DIR/phase4_results_${TIMESTAMP}.csv"

echo "=============================================================================="
echo "COMPREHENSIVE ABLATION STUDY — PHASE 4: Loss Function & Parameter Search"
echo "=============================================================================="
echo "Date: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Loss configurations to test (8 conditions):"
echo "  4A) DiceFocal (baseline loss, no Tversky)"
echo "  4B) TverskyFocal (α=0.3, β=0.7) — extreme recall bias"
echo "  4C) TverskyFocal (α=0.4, β=0.6) — V9 setting"
echo "  4D) TverskyFocal (α=0.5, β=0.5) — symmetric (Dice-like)"
echo "  4E) TverskyFocal (α=0.6, β=0.4) — V10 precision bias"
echo "  4F) TverskyFocal (α=0.7, β=0.3) — extreme precision bias"
echo "  4G) TverskyFocal (best α/β) + CLDice λ=0.1 — weak skeleton"
echo "  4H) TverskyFocal (best α/β) + CLDice λ=0.3 — V10 full"
echo ""
echo "Training configuration:"
echo "  CHM variant: [From Phase 2 winner]"
echo "  Architecture: [From Phase 3 winner]"
echo "  Fold: 0 (train=95 patches, val=118 patches)"
echo "  Epochs: 75 (SWA from epoch 35)"
echo "  Augmentation: Full (geometric + Mixup/CutMix/GridMask)"
echo ""

# Initialize results CSV
echo "condition,loss_config,fold_id,best_val_dice,best_val_cldice,best_val_iou,best_val_prec,best_val_rec,epochs_trained,swa_val_dice" > "$RESULTS_FILE"

docker run --rm \
    --gpus all \
    -v "$REPO_ROOT:$REPO_ROOT" \
    --workdir "$REPO_ROOT" \
    lamapuit:gpu \
    bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

echo '=============================================================================='
echo 'Starting Phase 4 training loop...'
echo '=============================================================================='

# These will be set from Phase 3 results
CHM_VARIANT=\${CHM_VARIANT}
ARCHITECTURE=\${ARCHITECTURE}

echo \"Using CHM variant: \$CHM_VARIANT\"
echo \"Using architecture: \$ARCHITECTURE\"
echo \"\"

for loss_config in 4A 4B 4C 4D 4E 4F 4G 4H; do
  echo ''
  echo \"Training \$loss_config\"
  echo \"[$(date)] Starting \$loss_config...\"

  python3 seg_pipeline/scripts/phase3_train_v10.py \
    --variant \$CHM_VARIANT \
    --architecture \$ARCHITECTURE \
    --loss-config \$loss_config \
    --fold 0 \
    --epochs 75 \
    --swa-start-epoch 35 \
    --dataset-dir seg_pipeline/output/phase2_dataset_v3 \
    --device cuda \
    --output-dir $OUTPUT_DIR \
    2>&1 | tee -a $LOG_FILE

  echo \"[$(date)] Completed \$loss_config\"

  # Extract metrics
  if [ -f \"$OUTPUT_DIR/loss_\${loss_config}/fold0/metrics.json\" ]; then
    python3 -c \"
import json
with open('$OUTPUT_DIR/loss_\${loss_config}/fold0/metrics.json') as f:
    m = json.load(f)
print(f\\\"\$loss_config,\${loss_config},0,{m['best_val_dice']:.4f},{m.get('best_val_cldice', 0):.4f},{m.get('best_val_iou', 0):.4f},{m.get('best_val_prec', 0):.4f},{m.get('best_val_rec', 0):.4f},{m['n_epochs_trained']},{m.get('swa_val_dice', m['best_val_dice']):.4f}\\\")
    \"
  fi >> $RESULTS_FILE
done

echo ''
echo '=============================================================================='
echo 'PHASE 4 COMPLETE'
echo '=============================================================================='

" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=============================================================================="
echo "PHASE 4 RESULTS SUMMARY"
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
        results.append((row['loss_config'], float(row['best_val_dice'])))

results.sort(key=lambda x: x[1], reverse=True)
for i, (config, dice) in enumerate(results[:2], 1):
    print(f"  {i}. {config:20} → dice={dice:.4f} ✓ ADVANCE TO PHASE 5")

for i, (config, dice) in enumerate(results[2:], 3):
    print(f"  {i}. {config:20} → dice={dice:.4f} (did not advance)")
PYTHON_EOF

echo ""
echo "Full results:"
cat "$RESULTS_FILE"
echo ""
echo "=============================================================================="
echo "Next step: Review winners, update PHASE4_WINNERS.md, then run Phase 5"
echo "=============================================================================="
echo ""

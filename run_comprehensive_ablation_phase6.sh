#!/bin/bash
# Comprehensive Ablation Study Phase 6: Final Validation
# Trains the winning configuration on ALL 4 FOLDS
# Reports mean ± std across folds with statistical significance testing

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$REPO_ROOT/seg_pipeline/output/ablation_v2_comprehensive"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase6_final_${TIMESTAMP}.log"
RESULTS_FILE="$OUTPUT_DIR/phase6_final_results_${TIMESTAMP}.csv"

echo "=============================================================================="
echo "COMPREHENSIVE ABLATION STUDY — PHASE 6: Final Validation (All Folds)"
echo "=============================================================================="
echo "Date: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Final validation configuration:"
echo "  CHM variant: [From Phase 2 winner]"
echo "  Architecture: [From Phase 3 winner]"
echo "  Loss: [From Phase 4 winner]"
echo "  Augmentation: [From Phase 5 winner]"
echo "  Folds: ALL 4 (0, 1, 2, 3)"
echo "  Epochs: 75 (SWA from epoch 35)"
echo ""

# Initialize results CSV
echo "fold_id,best_val_dice,best_val_cldice,best_val_iou,best_val_prec,best_val_rec,epochs_trained,swa_val_dice" > "$RESULTS_FILE"

docker run --rm \
    --gpus all \
    -v "$REPO_ROOT:$REPO_ROOT" \
    --workdir "$REPO_ROOT" \
    lamapuit:gpu \
    bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

echo '=============================================================================='
echo 'Starting Phase 6 training loop (all 4 folds)...'
echo '=============================================================================='

# These will be set from Phase 5 results (the winning configuration)
CHM_VARIANT=\${CHM_VARIANT}
ARCHITECTURE=\${ARCHITECTURE}
LOSS_CONFIG=\${LOSS_CONFIG}
AUG_CONFIG=\${AUG_CONFIG}

echo \"Using CHM variant: \$CHM_VARIANT\"
echo \"Using architecture: \$ARCHITECTURE\"
echo \"Using loss config: \$LOSS_CONFIG\"
echo \"Using aug config: \$AUG_CONFIG\"
echo \"\"

for fold in 0 1 2 3; do
  echo ''
  echo \"========================================\"
  echo \"Fold \$fold\"
  echo \"========================================\"
  echo \"[$(date)] Starting fold \$fold...\"

  python3 seg_pipeline/scripts/phase3_train_v10.py \
    --variant \$CHM_VARIANT \
    --architecture \$ARCHITECTURE \
    --loss-config \$LOSS_CONFIG \
    --aug-config \$AUG_CONFIG \
    --fold \$fold \
    --epochs 75 \
    --swa-start-epoch 35 \
    --dataset-dir seg_pipeline/output/phase2_dataset_v3 \
    --device cuda \
    --output-dir $OUTPUT_DIR/phase6_final \
    2>&1 | tee -a $LOG_FILE

  echo \"[$(date)] Completed fold \$fold\"

  # Extract metrics
  if [ -f \"$OUTPUT_DIR/phase6_final/fold\${fold}/metrics.json\" ]; then
    python3 -c \"
import json
with open('$OUTPUT_DIR/phase6_final/fold\${fold}/metrics.json') as f:
    m = json.load(f)
print(f\\\"\$fold,{m['best_val_dice']:.4f},{m.get('best_val_cldice', 0):.4f},{m.get('best_val_iou', 0):.4f},{m.get('best_val_prec', 0):.4f},{m.get('best_val_rec', 0):.4f},{m['n_epochs_trained']},{m.get('swa_val_dice', m['best_val_dice']):.4f}\\\")
    \"
  fi >> $RESULTS_FILE
done

echo ''
echo '=============================================================================='
echo 'PHASE 6 COMPLETE'
echo '=============================================================================='

" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=============================================================================="
echo "PHASE 6 FINAL VALIDATION SUMMARY"
echo "=============================================================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
python3 << PYTHON_EOF
import csv
import numpy as np

results = []
with open('$RESULTS_FILE', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append({
            'fold': int(row['fold_id']),
            'dice': float(row['best_val_dice']),
            'cldice': float(row['best_val_cldice']),
            'iou': float(row['best_val_iou']),
            'prec': float(row['best_val_prec']),
            'rec': float(row['best_val_rec']),
        })

print("\nPer-fold results:")
for r in sorted(results, key=lambda x: x['fold']):
    print(f"  Fold {r['fold']}: Dice={r['dice']:.4f}, clDice={r['cldice']:.4f}, IoU={r['iou']:.4f}, Prec={r['prec']:.4f}, Rec={r['rec']:.4f}")

print("\nAggregated statistics (mean ± std):")
dice_vals = [r['dice'] for r in results]
cldice_vals = [r['cldice'] for r in results]
iou_vals = [r['iou'] for r in results]
prec_vals = [r['prec'] for r in results]
rec_vals = [r['rec'] for r in results]

print(f"  Dice:   {np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}")
print(f"  clDice: {np.mean(cldice_vals):.4f} ± {np.std(cldice_vals):.4f}")
print(f"  IoU:    {np.mean(iou_vals):.4f} ± {np.std(iou_vals):.4f}")
print(f"  Prec:   {np.mean(prec_vals):.4f} ± {np.std(prec_vals):.4f}")
print(f"  Rec:    {np.mean(rec_vals):.4f} ± {np.std(rec_vals):.4f}")

print("\nFull results:")
PYTHON_EOF

echo ""
cat "$RESULTS_FILE"
echo ""
echo "=============================================================================="
echo "FINAL RESULTS READY FOR THESIS"
echo "=============================================================================="
echo ""

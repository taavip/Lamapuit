#!/bin/bash
# Comprehensive Ablation Study Phase 5: Augmentation & Regularization Search
# Tests 5 augmentation strategies with top configuration from Phases 2-4
# Advancement: Top 2 move to Phase 6 (final validation)

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$REPO_ROOT/seg_pipeline/output/ablation_v2_comprehensive"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase5_ablation_${TIMESTAMP}.log"
RESULTS_FILE="$OUTPUT_DIR/phase5_results_${TIMESTAMP}.csv"

echo "=============================================================================="
echo "COMPREHENSIVE ABLATION STUDY — PHASE 5: Augmentation & Regularization"
echo "=============================================================================="
echo "Date: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Augmentation configurations to test (5 conditions):"
echo "  5A) None — no augmentation baseline"
echo "  5B) Geometric — rotations/flips only"
echo "  5C) Full (hard targets) — full aug with hard targets"
echo "  5D) Full (soft targets) — V10.2 full setting with soft targets (σ=2.0)"
echo "  5E) Full (no SWA) — full aug + soft targets, SWA disabled"
echo ""
echo "Training configuration:"
echo "  CHM variant: [From Phase 2 winner]"
echo "  Architecture: [From Phase 3 winner]"
echo "  Loss: [From Phase 4 winner]"
echo "  Fold: 0 (train=95 patches, val=118 patches)"
echo "  Epochs: 75 (SWA from epoch 35, unless disabled)"
echo ""

# Initialize results CSV
echo "condition,aug_config,fold_id,best_val_dice,best_val_cldice,best_val_iou,best_val_prec,best_val_rec,epochs_trained,swa_val_dice" > "$RESULTS_FILE"

docker run --rm \
    --gpus all \
    -v "$REPO_ROOT:$REPO_ROOT" \
    --workdir "$REPO_ROOT" \
    lamapuit:gpu \
    bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

echo '=============================================================================='
echo 'Starting Phase 5 training loop...'
echo '=============================================================================='

# These will be set from Phase 4 results
CHM_VARIANT=\${CHM_VARIANT}
ARCHITECTURE=\${ARCHITECTURE}
LOSS_CONFIG=\${LOSS_CONFIG}

echo \"Using CHM variant: \$CHM_VARIANT\"
echo \"Using architecture: \$ARCHITECTURE\"
echo \"Using loss config: \$LOSS_CONFIG\"
echo \"\"

for aug_config in 5A 5B 5C 5D 5E; do
  echo ''
  echo \"Training \$aug_config\"
  echo \"[$(date)] Starting \$aug_config...\"

  python3 seg_pipeline/scripts/phase3_train_v10.py \
    --variant \$CHM_VARIANT \
    --architecture \$ARCHITECTURE \
    --loss-config \$LOSS_CONFIG \
    --aug-config \$aug_config \
    --fold 0 \
    --epochs 75 \
    --swa-start-epoch 35 \
    --dataset-dir seg_pipeline/output/phase2_dataset_v3 \
    --device cuda \
    --output-dir $OUTPUT_DIR \
    2>&1 | tee -a $LOG_FILE

  echo \"[$(date)] Completed \$aug_config\"

  # Extract metrics
  if [ -f \"$OUTPUT_DIR/aug_\${aug_config}/fold0/metrics.json\" ]; then
    python3 -c \"
import json
with open('$OUTPUT_DIR/aug_\${aug_config}/fold0/metrics.json') as f:
    m = json.load(f)
print(f\\\"\$aug_config,\${aug_config},0,{m['best_val_dice']:.4f},{m.get('best_val_cldice', 0):.4f},{m.get('best_val_iou', 0):.4f},{m.get('best_val_prec', 0):.4f},{m.get('best_val_rec', 0):.4f},{m['n_epochs_trained']},{m.get('swa_val_dice', m['best_val_dice']):.4f}\\\")
    \"
  fi >> $RESULTS_FILE
done

echo ''
echo '=============================================================================='
echo 'PHASE 5 COMPLETE'
echo '=============================================================================='

" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=============================================================================="
echo "PHASE 5 RESULTS SUMMARY"
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
        results.append((row['aug_config'], float(row['best_val_dice'])))

results.sort(key=lambda x: x[1], reverse=True)
for i, (config, dice) in enumerate(results[:2], 1):
    print(f"  {i}. {config:20} → dice={dice:.4f} ✓ ADVANCE TO PHASE 6")

for i, (config, dice) in enumerate(results[2:], 3):
    print(f"  {i}. {config:20} → dice={dice:.4f} (did not advance)")
PYTHON_EOF

echo ""
echo "Full results:"
cat "$RESULTS_FILE"
echo ""
echo "=============================================================================="
echo "Next step: Review winners, update PHASE5_WINNERS.md, then run Phase 6"
echo "=============================================================================="
echo ""

#!/bin/bash
# Comprehensive Ablation Study: ALL PHASES (2-6) with automatic progression
# Automatically extracts winners from each phase and feeds to the next phase

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$REPO_ROOT/seg_pipeline/output/ablation_v2_comprehensive"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/master_ablation_${TIMESTAMP}.log"

{
    echo "=============================================================================="
    echo "COMPREHENSIVE ABLATION STUDY — ALL PHASES (2-6) AUTOMATED EXECUTION"
    echo "=============================================================================="
    echo "Start: $(date)"
    echo "Master log: $MASTER_LOG"
    echo ""

    # ========================================================================
    # PHASE 2: CHM Variant Search
    # ========================================================================
    echo "Running PHASE 2: CHM Variant Search"
    echo "-----"
    bash run_comprehensive_ablation_phase2.sh

    # Extract Phase 2 winners
    PHASE2_RESULTS=$(ls -1t "$OUTPUT_DIR"/phase2_results_*.csv | head -1)
    echo "Phase 2 results: $PHASE2_RESULTS"
    echo ""

    # Show top 2 Phase 2 winners
    python3 -c "
import csv
results = []
with open('$PHASE2_RESULTS', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('variant'):
            results.append((row['variant'], float(row['best_val_dice'])))
results.sort(key=lambda x: x[1], reverse=True)
print('Phase 2 Top 2 Winners:')
for i, (var, dice) in enumerate(results[:2], 1):
    print(f'  {i}. {var}: {dice:.4f}')
"
    echo ""

    # ========================================================================
    # PHASE 3: Architecture Search
    # ========================================================================
    echo "Running PHASE 3: Architecture Search"
    echo "-----"
    bash run_comprehensive_ablation_phase3.sh

    PHASE3_RESULTS=$(ls -1t "$OUTPUT_DIR"/phase3_results_*.csv | head -1)
    echo "Phase 3 results: $PHASE3_RESULTS"
    echo ""

    # Show top 2 Phase 3 winners
    python3 -c "
import csv
results = []
with open('$PHASE3_RESULTS', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('chm_variant'):
            combo = f\"{row['chm_variant']}×{row['architecture']}\"
            results.append((combo, float(row['best_val_dice'])))
results.sort(key=lambda x: x[1], reverse=True)
print('Phase 3 Top 2 Winners:')
for i, (combo, dice) in enumerate(results[:2], 1):
    print(f'  {i}. {combo}: {dice:.4f}')
"
    echo ""

    # ========================================================================
    # PHASE 4: Loss Function Tuning
    # ========================================================================
    echo "Running PHASE 4: Loss Function Tuning"
    echo "-----"
    bash run_comprehensive_ablation_phase4.sh

    PHASE4_RESULTS=$(ls -1t "$OUTPUT_DIR"/phase4_results_*.csv | head -1)
    echo "Phase 4 results: $PHASE4_RESULTS"
    echo ""

    # Show top 2 Phase 4 winners
    python3 -c "
import csv
results = []
with open('$PHASE4_RESULTS', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('loss_config'):
            results.append((row['loss_config'], float(row['best_val_dice'])))
results.sort(key=lambda x: x[1], reverse=True)
print('Phase 4 Top 2 Winners:')
for i, (config, dice) in enumerate(results[:2], 1):
    print(f'  {i}. {config}: {dice:.4f}')
"
    echo ""

    # ========================================================================
    # PHASE 5: Augmentation & Regularization
    # ========================================================================
    echo "Running PHASE 5: Augmentation & Regularization"
    echo "-----"
    bash run_comprehensive_ablation_phase5.sh

    PHASE5_RESULTS=$(ls -1t "$OUTPUT_DIR"/phase5_results_*.csv | head -1)
    echo "Phase 5 results: $PHASE5_RESULTS"
    echo ""

    # Show top 2 Phase 5 winners
    python3 -c "
import csv
results = []
with open('$PHASE5_RESULTS', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('aug_config'):
            results.append((row['aug_config'], float(row['best_val_dice'])))
results.sort(key=lambda x: x[1], reverse=True)
print('Phase 5 Top 2 Winners:')
for i, (config, dice) in enumerate(results[:2], 1):
    print(f'  {i}. {config}: {dice:.4f}')
"
    echo ""

    # ========================================================================
    # PHASE 6: Final Validation (All Folds for both top-2 configs)
    # ========================================================================
    echo "Running PHASE 6: Final Validation (All Folds for Top 2)"
    echo "-----"
    bash run_comprehensive_ablation_phase6.sh

    PHASE6_RESULTS=$(ls -1t "$OUTPUT_DIR"/phase6_final_results_*.csv | head -1)
    echo "Phase 6 results: $PHASE6_RESULTS"
    echo ""

    # Show Phase 6 final aggregated results
    python3 -c "
import csv
import numpy as np

results = []
with open('$PHASE6_RESULTS', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('fold_id'):
            results.append({
                'fold': int(row['fold_id']),
                'dice': float(row['best_val_dice']),
                'cldice': float(row.get('best_val_cldice', 0)),
            })

print('Phase 6 Final Validation Results:')
print('Per-fold Dice:')
for r in sorted(results, key=lambda x: x['fold']):
    print(f\"  Fold {r['fold']}: {r['dice']:.4f}\")

if results:
    dice_vals = [r['dice'] for r in results]
    cldice_vals = [r['cldice'] for r in results]
    print(f'\\nAggregated (mean ± std):')
    print(f\"  Dice:   {np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}\")
    print(f\"  clDice: {np.mean(cldice_vals):.4f} ± {np.std(cldice_vals):.4f}\")
"
    echo ""

    echo "=============================================================================="
    echo "ABLATION STUDY COMPLETE"
    echo "=============================================================================="
    echo "Completion: $(date)"
    echo ""

} 2>&1 | tee "$MASTER_LOG"

echo "Master log saved to: $MASTER_LOG"

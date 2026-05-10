#!/usr/bin/env bash
# V5 Instance Segmentation Pipeline
# Runs inside Docker container: lamapuit:gpu
# Total runtime: ~6-8 hours GPU

set -euo pipefail

SCRIPTS="/workspace/seg_pipeline/scripts"
LOG_DIR="/workspace/seg_pipeline/output/phase4_report_v5"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/v5_pipeline_run.log"

echo "=== V5 Instance Segmentation Pipeline ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Install additional dependencies
echo "--- Installing dependencies ---" | tee -a "$LOG"
pip install -q geopandas shapely Pillow pycocotools "transformers>=4.35,<5.0" 2>>"$LOG" || true

# Phase II: Build YOLO + COCO datasets (~10 min)
echo "" | tee -a "$LOG"
echo "--- Phase II: Dataset Preparation ---" | tee -a "$LOG"
python "$SCRIPTS/phase2_dataset_v5.py" --validate 2>&1 | tee -a "$LOG"
python "$SCRIPTS/phase2_dataset_v5.py" 2>&1 | tee -a "$LOG"

# Phase III-a: YOLO11m-seg (all 4 folds, ~2-3h)
echo "" | tee -a "$LOG"
echo "--- Phase III-a: YOLO11m-seg Training ---" | tee -a "$LOG"
python "$SCRIPTS/phase3_train_v5_yolo.py" --device cuda --batch 8 2>&1 | tee -a "$LOG"

# Phase III-b: Mask2Former (all 4 folds, ~4-6h)
echo "" | tee -a "$LOG"
echo "--- Phase III-b: Mask2Former Training ---" | tee -a "$LOG"
python "$SCRIPTS/phase3_train_v5_mask2former.py" --device cuda --batch-size 4 2>&1 | tee -a "$LOG"

# Phase IV: Instance evaluation + GeoPackage
echo "" | tee -a "$LOG"
echo "--- Phase IV: Evaluation ---" | tee -a "$LOG"
python "$SCRIPTS/phase4_evaluate_v5.py" --device cuda 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== COMPLETE ===" | tee -a "$LOG"
echo "End: $(date)" | tee -a "$LOG"
echo "Outputs: $LOG_DIR"

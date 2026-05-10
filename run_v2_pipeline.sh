#!/bin/bash
# V2 Segmentation Pipeline — Full multi-variant run
# Trains U-Net++ EfficientNet-B2 on all 5 CHM variants, 2 folds, 75 epochs each
set -e

cd /home/tpipar/project/Lamapuit

mkdir -p seg_pipeline/output/logs

LOG=seg_pipeline/output/logs/v2_pipeline_run.log

echo "=== V2 Pipeline Start: $(date) ===" | tee -a $LOG

docker run --rm --gpus all \
  -v "$(pwd):/workspace" -w /workspace \
  -e TORCH_HOME=/workspace/.docker_torch_cache \
  --name seg_v2_runner \
  lamapuit:gpu \
  bash -lc "
    source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect
    set -e

    echo '=== Phase II V2: Building datasets for all variants ===' && \
    for variant in baseline raw gauss masked composite; do
      echo \"--- Variant: \$variant ---\" && \
      python seg_pipeline/scripts/phase2_dataset_v2.py --chm-variant \$variant 2>&1
    done && \

    echo '=== Phase III V2: Training all variants (75 epochs, 2 folds) ===' && \
    for variant in baseline raw gauss masked composite; do
      echo \"--- Training variant: \$variant ---\" && \
      python seg_pipeline/scripts/phase3_train_v2.py \
        --chm-variant \$variant \
        --epochs 75 \
        --patience 12 \
        --device cuda \
        --num-workers 0 \
        2>&1
    done && \

    echo '=== ALL VARIANTS COMPLETE: $(date) ==='
  " 2>&1 | tee -a $LOG

echo "=== V2 Pipeline End: $(date) ===" | tee -a $LOG

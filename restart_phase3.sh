#!/bin/bash
# Restart Phase III skipping completed U-Net++ and using num_workers=0 for safety
set -e

cd /home/tpipar/project/Lamapuit

mkdir -p seg_pipeline/output/logs

echo "=== Restarting Phase III (DeepLabV3+ and SegFormer only) ===" | tee -a seg_pipeline/output/logs/pipeline_run.log

docker run --rm --gpus all \
  -v "$(pwd):/workspace" -w /workspace \
  -e TORCH_HOME=/workspace/.docker_torch_cache \
  --name seg_pipeline_runner \
  lamapuit:gpu \
  bash -lc "
    source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect
    echo '=== DeepLabV3+ Training ===' && \
    python seg_pipeline/scripts/phase3_train.py \
      --arch deeplabv3plus_r50 \
      --device cuda \
      --num-workers 0 \
      2>&1 && \
    echo '=== SegFormer Training ===' && \
    python seg_pipeline/scripts/phase3_train.py \
      --arch segformer_b2 \
      --device cuda \
      --num-workers 0 \
      2>&1 && \
    echo '=== Phase IV: Evaluation ===' && \
    python seg_pipeline/scripts/phase4_evaluate.py --device cuda 2>&1 && \
    echo 'ALL PHASES COMPLETE'
  " 2>&1 | tee -a seg_pipeline/output/logs/pipeline_run.log

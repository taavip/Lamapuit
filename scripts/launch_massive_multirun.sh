#!/bin/bash
# Launch massive multi-run training suite
# 60 runs × 4 configs = 240 total runs
# Estimated time: ~12 hours

set -e

echo "################################################################################"
echo "MASSIVE MULTI-RUN TRAINING SUITE"
echo "################################################################################"
echo ""
echo "Configuration:"
echo "  - Top 4 models from experiments"
echo "  - 60 runs per model"
echo "  - Total: 240 training runs"
echo "  - Estimated time: 12 hours @ 3 min/run"
echo ""
echo "Models:"
echo "  1. exp7_conservative (YOLO11s, conservative aug)"
echo "  2. exp4_low_lr (YOLO11s, low learning rate)"
echo "  3. exp3_small (YOLO11s, standard)"
echo "  4. exp2_medium (YOLO11m, standard)"
echo ""
echo "################################################################################"
echo ""

read -p "Start massive multi-run? This will take ~12 hours. (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Starting massive multi-run training..."
echo "Monitor progress: tail -f massive_multirun_*.json"
echo ""

docker run -it --rm --gpus all --shm-size=8g \
    -v "$PWD":/workspace -w /workspace lamapuit-dev bash -c \
    "source /opt/conda/etc/profile.d/conda.sh && \
     conda activate cwd-detect && \
     python scripts/run_massive_multirun.py \
       --config all \
       --runs 60 \
       --epochs 150 \
       --patience 40 \
       --dataset data/dataset_final/dataset_trainval.yaml \
       --device 0" 2>&1 | tee massive_multirun_log.txt

echo ""
echo "################################################################################"
echo "MASSIVE MULTI-RUN COMPLETE!"
echo "################################################################################"
echo ""
echo "Results:"
echo "  - massive_multirun_comparison.csv"
echo "  - massive_multirun_comparison.png"
echo "  - massive_multirun_*_statistics.csv (per config)"
echo "  - massive_multirun_*_plots.png (per config)"
echo ""

#!/bin/bash
set -e

DOCKER="/mnt/c/Program Files/Docker/Docker/resources/bin/docker"

"$DOCKER" run --gpus all -d \
  --ipc=host --shm-size=16g \
  --name ensemble_comparison_v2 \
  -v /home/tpipar/project/Lamapuit:/workspace \
  -v /home/tpipar/project/Lamapuit/.docker_torch_cache:/root/.cache/torch \
  -w /workspace \
  lamapuit:gpu \
  bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python scripts/ensemble_4th_model_comparison_v2.py"

echo "Container started: ensemble_comparison_v2"
echo "Monitor logs with: \"$DOCKER\" logs -f ensemble_comparison_v2"

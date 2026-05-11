#!/usr/bin/env bash
set -euo pipefail

# Run project Python commands through Docker + Conda (cwd-detect env).
# Usage:
#   bash scripts/run_python_in_docker_conda.sh -m py_compile scripts/foo.py
#   bash scripts/run_python_in_docker_conda.sh scripts/labelstudio/generate_labelstudio_tasks.py --limit 200

IMAGE_NAME="${LAMAPUIT_IMAGE:-lamapuit-dev}"
ENV_NAME="${LAMAPUIT_CONDA_ENV:-cwd-detect}"
GPU_FLAG=""
NETWORK_ARGS="--add-host=host.docker.internal:host-gateway"

if [[ "${LAMAPUIT_DOCKER_GPU:-0}" == "1" ]]; then
  GPU_FLAG="--gpus all"
fi

if [[ "${LAMAPUIT_DOCKER_NETWORK_HOST:-0}" == "1" ]]; then
  NETWORK_ARGS="--network host"
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <python-args...>" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found" >&2
  exit 1
fi

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Docker image '$IMAGE_NAME' not found. Build it first:" >&2
  echo "  docker build -t $IMAGE_NAME ." >&2
  exit 1
fi

PY_ARGS=""
for arg in "$@"; do
  printf -v _q '%q' "$arg"
  PY_ARGS+="$_q "
done

docker run --rm $GPU_FLAG $NETWORK_ARGS -v "$PWD":/workspace -w /workspace "$IMAGE_NAME" \
  bash -lc "conda run -n $ENV_NAME python $PY_ARGS"

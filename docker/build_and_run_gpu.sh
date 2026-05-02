#!/usr/bin/env bash
set -euo pipefail

IMAGE=lamapuit:gpu
DOCKERFILE=docker/Dockerfile.gpu
export DOCKER_BUILDKIT=1

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "Building GPU image $IMAGE..."
  docker build --pull -t "$IMAGE" -f "$DOCKERFILE" .
else
  echo "Skipping image build because SKIP_BUILD=1"
fi

echo "Validating CUDA runtime inside container..."
docker run --rm --gpus all "$IMAGE" bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
print('torch=', torch.__version__)
print('torch_cuda_build=', torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('CUDA runtime unavailable in container.')
PY"

echo "Running training in container (this may take hours)..."
docker run --gpus all --rm -it \
  --ipc=host \
  --shm-size=16g \
  -v "$PWD":/workspace \
  -w /workspace \
  "$IMAGE" \
  bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python -m py_compile scripts/model_search_v2/model_search_v2.py && python scripts/model_search_v2/model_search_v2.py --output output/model_search_v2 --n-models 12 --t-high 0.9995 --t-low 0.0698 --extra-test-fraction 0.1 --max-extra-test 500"

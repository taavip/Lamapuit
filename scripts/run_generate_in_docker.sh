#!/usr/bin/env bash
set -euo pipefail
# Helper: build image and run the generator inside Docker using the project's Dockerfile
IMAGE_NAME=lamapuit-generate
REPO_DIR="$(pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Please install Docker to use this helper." >&2
  exit 1
fi

if [ -f Dockerfile ]; then
  echo "Building Docker image ${IMAGE_NAME} from Dockerfile..."
  docker build -t ${IMAGE_NAME} .
else
  echo "No Dockerfile found in repo root. Please build an image that provides conda and the project files." >&2
  echo "You can also run the generator directly in a conda env:" >&2
  echo "  conda activate <env> && python scripts/generate_confusion_tiles.py" >&2
  exit 1
fi

mkdir -p tmp/sep
echo "Running generator inside container (mounting repository)..."
docker run --rm -v "${REPO_DIR}":/workspace -w /workspace ${IMAGE_NAME} \
  bash -lc "conda run -n cwd-detect python scripts/generate_confusion_tiles.py"

echo "If the container didn't have an environment named 'cwd-detect', create one with:"
echo "  conda env create -f environment.yml -n cwd-detect"

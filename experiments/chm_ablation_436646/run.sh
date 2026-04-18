#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ARGS="$(printf '%q ' "$@")"

# Runs CHM ablation in the project-standard docker + conda environment.
docker run --rm -v "${WORKSPACE_ROOT}":/workspace -w /workspace lamapuit-dev \
	bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && cd experiments/chm_ablation_436646 && python run_chm_ablation.py ${ARGS}"

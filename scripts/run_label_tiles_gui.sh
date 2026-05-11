#!/usr/bin/env bash
set -euo pipefail

# Launch the interactive CHM labeler in Docker with WSLg/X11 support.
# This wrapper is tuned for the common thesis screenshot workflow:
#   - open the GUI immediately
#   - focus the view near the requested map coordinate
#   - write outputs to output/tile_labels when possible

IMAGE_NAME="${LAMAPUIT_IMAGE:-lamapuit-dev}"
CONDA_ENV="${LAMAPUIT_CONDA_ENV:-cwd-detect}"
REPO_DIR="$(pwd)"

CHM_PATH="${1:-source/406455_2021_tava/chm_variants_reconstructed_original_20260510/baseline_chm_0p2m/406455_2021_tava_chm_max_hag_20cm.tif}"
FOCUS_X="${2:-455695.1617}"
FOCUS_Y="${3:-6406968.1024}"
OUTPUT_DIR="output/tile_labels"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Please install Docker Desktop / Docker CLI." >&2
  exit 1
fi

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Docker image '$IMAGE_NAME' not found. Build it first:" >&2
  echo "  docker build -t $IMAGE_NAME ." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if ! [ -w "$OUTPUT_DIR" ]; then
  if command -v sudo >/dev/null 2>&1; then
    sudo chown -R "$(id -u):$(id -g)" "$OUTPUT_DIR" || true
  fi
fi

if ! [ -w "$OUTPUT_DIR" ]; then
  echo "Warning: $OUTPUT_DIR is still not writable; using /tmp/tile_labels_gui instead." >&2
  OUTPUT_DIR="/tmp/tile_labels_gui"
  mkdir -p "$OUTPUT_DIR"
fi

PY_ARGS=(
  scripts/label_tiles.py
  --chm "$CHM_PATH"
  --output "$OUTPUT_DIR"
  --focus-x "$FOCUS_X"
  --focus-y "$FOCUS_Y"
  --no-finetune
)

INNER_CMD=$(
  cat <<'EOF'
source /opt/conda/etc/profile.d/conda.sh
conda activate __CONDA_ENV__
if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
have_tk = importlib.util.find_spec("tkinter") is not None
have_qt = any(
    importlib.util.find_spec(mod) is not None
    for mod in ("PyQt5", "PyQt6", "PySide2", "PySide6")
)
raise SystemExit(0 if (have_tk or have_qt) else 1)
PY
then
  (mamba install -y -c conda-forge pyqt tk) || (conda install -y -c conda-forge pyqt tk)
fi
PYTHONPATH=/workspace/src python __PY_CMD__
EOF
)

INNER_CMD="${INNER_CMD/__CONDA_ENV__/$CONDA_ENV}"

PY_CMD=""
for arg in "${PY_ARGS[@]}"; do
  printf -v q '%q' "$arg"
  PY_CMD+="$q "
done
INNER_CMD="${INNER_CMD/__PY_CMD__/$PY_CMD}"

docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -e DISPLAY="${DISPLAY:-}" \
  -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
  -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" \
  -e PULSE_SERVER="${PULSE_SERVER:-}" \
  -e QT_QPA_PLATFORM=xcb \
  -e HOME=/tmp \
  -e MPLCONFIGDIR=/tmp/mplconfig \
  -e XDG_CACHE_HOME=/tmp/fontconfig \
  -v /mnt/wslg:/mnt/wslg \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$REPO_DIR":/workspace \
  -w /workspace \
  "$IMAGE_NAME" \
  bash -lc "$INNER_CMD"

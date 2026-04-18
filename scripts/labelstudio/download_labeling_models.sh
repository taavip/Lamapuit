#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-output/labelstudio_pipeline/models}"
mkdir -p "$OUT_DIR"

YOLO26_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt"
YOLO11S_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt"
YOLO11N_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"

fetch_if_missing () {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "[skip] $out already exists"
    return
  fi
  echo "[download] $url -> $out"
  curl -L "$url" -o "$out"
}

fetch_if_missing "$YOLO26_URL" "$OUT_DIR/yolo26s-seg.pt" || true
fetch_if_missing "$YOLO11S_URL" "$OUT_DIR/yolo11s-seg.pt" || true
fetch_if_missing "$YOLO11N_URL" "$OUT_DIR/yolo11n-seg.pt" || true

echo "Model download step finished."

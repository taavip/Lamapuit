#!/usr/bin/env bash
set -euo pipefail

# Paths
SRC="experiments/dtm_hag_436646_research/results/chm/2018/2018_tin_linear_gauss_chm.tif"
EXP_DIR="experiments/dtm_hag_436646_reproduce_single_2018"
REF_DIR="$EXP_DIR/reference/chm/2018"
OUT_DIR="$EXP_DIR/results"

mkdir -p "$REF_DIR"
if [ -f "$SRC" ]; then
  cp -v "$SRC" "$REF_DIR/2018_tin_linear_gauss_chm.tif"
  echo "Copied research CHM to $REF_DIR/2018_tin_linear_gauss_chm.tif"
else
  echo "Warning: source CHM not found: $SRC" >&2
fi

# Run targeted reproduction (single-year). Adjust flags if needed.
python3 experiments/dtm_hag_436646_research/run_experiment.py \
  --years 2018 \
  --out-dir "$OUT_DIR" \
  --laz-dir data/lamapuit/laz \
  --labels-dir output/onboarding_labels_v2_drop13 \
  --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop \
  --gaussian-sigma 1.0 \
  --point-sample-rate 1.0 \
  --reuse-csf

# Keep only the target CHM file (if produced)
TARGET="$OUT_DIR/chm/2018/2018_tin_linear_gauss_chm.tif"
if [ -f "$TARGET" ]; then
  echo "Produced target CHM at $TARGET"
else
  echo "Target CHM not produced." >&2
fi

# Remove other CHM files from the year folder, leave only the target
if [ -d "$OUT_DIR/chm/2018" ]; then
  find "$OUT_DIR/chm/2018" -type f ! -name "2018_tin_linear_gauss_chm.tif" -delete || true
fi

echo "Done. Reference kept at $REF_DIR/2018_tin_linear_gauss_chm.tif; results (if produced) at $TARGET"

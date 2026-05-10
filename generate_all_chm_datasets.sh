#!/bin/bash
# Generate all CHM variant datasets for Phase 2 study
# Creates: baseline, raw, gauss, masked, composite patch indices and statistics

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$REPO_ROOT/seg_pipeline/output/phase2_dataset_v3"
MASK_TIF="$REPO_ROOT/source/406455_2021_tava/phase1_masks/406455_2021_tava_truemask.tif"

mkdir -p "$OUTPUT_DIR"

echo "=============================================================================="
echo "Generating CHM Variant Datasets for Phase 2 Ablation Study"
echo "=============================================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Mask TIF: $MASK_TIF"
echo ""

# Check that mask exists
if [ ! -f "$MASK_TIF" ]; then
    echo "❌ ERROR: Mask TIF not found at $MASK_TIF"
    exit 1
fi

echo "Generating variant datasets..."
echo ""

VARIANTS=("baseline" "raw" "gauss" "masked" "composite")

for variant in "${VARIANTS[@]}"; do
    echo "---"
    echo "Variant: $variant"

    # Check if dataset already exists
    patch_index="$OUTPUT_DIR/patch_index_${variant}.csv"
    band_stats="$OUTPUT_DIR/band_stats_${variant}.json"

    if [ -f "$patch_index" ] && [ -f "$band_stats" ]; then
        echo "  ✓ Already exists, skipping"
        continue
    fi

    echo "  Generating patch index and statistics..."
    docker run --rm \
        -v "$REPO_ROOT:$REPO_ROOT" \
        --workdir "$REPO_ROOT" \
        lamapuit:gpu \
        bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

export NO_ALBUMENTATIONS_UPDATE=1
python3 seg_pipeline/scripts/phase2_dataset_v3.py \
    --variant $variant \
    --mask-tif $MASK_TIF \
    --output-dir $OUTPUT_DIR
" 2>&1 | grep -v "UserWarning\|albumentations\|pip as the\|venv\|--root-user-action" || true

    if [ -f "$patch_index" ] && [ -f "$band_stats" ]; then
        patches=$(wc -l < "$patch_index" | tail -1)
        echo "  ✓ Generated ($patches patches)"
    else
        echo "  ❌ FAILED to generate dataset"
        exit 1
    fi
done

echo ""
echo "=============================================================================="
echo "All CHM variant datasets ready"
echo "=============================================================================="
echo ""
ls -lh "$OUTPUT_DIR"/patch_index_*.csv
echo ""

#!/bin/bash
###############################################################################
# Build LAZ → all CHM variants pipeline for tile 406455_2021_tava
#
# This script:
# 1. Downloads 406455_2021_tava LAZ file from Maa-amet geoportaal
# 2. Generates all CHM variants (baseline, raw, gaussian, composite, masked-raw)
# 3. Stores output in source/406455_2021_tava/ with reproducible structure
#
# Usage:
#   bash scripts/build_406455_2021_tava.sh                    # Full build
#   bash scripts/build_406455_2021_tava.sh --skip-download    # Skip download
#
# Output structure:
#   source/406455_2021_tava/
#   ├── laz_input/                    # Downloaded LAZ file
#   └── chm_variants/
#       ├── baseline_chm_0p2m/
#       ├── harmonized_raw_0p2m/
#       ├── harmonized_gauss_kernel0p8m_0p2m/
#       ├── composite_4band_raw_base_mask/
#       └── masked_raw_2band_0p2m/
###############################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_DIR="${PROJECT_ROOT}/source/406455_2021_tava"
LAZ_INPUT_DIR="${SOURCE_DIR}/laz_input"
CHM_OUTPUT_DIR="${SOURCE_DIR}/chm_variants"

# Parse arguments
SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --skip-download) SKIP_DOWNLOAD=1 ;;
  esac
done

echo "======================================================================="
echo "Build 406455_2021_tava: LAZ → CHM Variants"
echo "======================================================================="
echo "Project root: $PROJECT_ROOT"
echo "Source dir: $SOURCE_DIR"
echo ""

# Create directories
mkdir -p "$LAZ_INPUT_DIR" "$CHM_OUTPUT_DIR"

# Detect if Docker is needed
if ! command -v python &> /dev/null; then
  echo "Python not found locally, will use Docker..."
  USE_DOCKER=true
elif ! python -c "import requests, bs4" 2>/dev/null; then
  echo "Required packages not found, will use Docker..."
  USE_DOCKER=true
else
  USE_DOCKER=false
fi

if [ "$USE_DOCKER" = true ]; then
  echo "Executing in Docker (cdw-detect environment)..."
  echo ""

  # Build image if needed
  if ! docker image inspect cdw-detect:latest > /dev/null 2>&1; then
    echo "Building Docker image cdw-detect:latest..."
    docker build -t cdw-detect:latest -f Dockerfile "$PROJECT_ROOT"
    echo ""
  fi

  # Run inside Docker
  docker run --rm \
    --gpus all \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    cdw-detect:latest \
    bash -c "
      source /opt/conda/etc/profile.d/conda.sh
      conda activate cwd-detect
      set -e

      echo 'Step 1: Downloading LAZ file for kaardiruut 406455...'
      if [ $SKIP_DOWNLOAD -eq 0 ]; then
        python scripts/laz_mass_downloader.py \
          --ids 406455 \
          --out '$LAZ_INPUT_DIR' \
          --workers 4
        echo '✓ LAZ download complete'
      else
        echo '  (Skipping download, using existing LAZ files)'
      fi
      echo ''

      LAZ_COUNT=\$(find '$LAZ_INPUT_DIR' -name '*.laz' | wc -l)
      if [ \$LAZ_COUNT -eq 0 ]; then
        echo '✗ Error: No LAZ files found in $LAZ_INPUT_DIR'
        exit 1
      fi
      echo \"Found \$LAZ_COUNT LAZ file(s)\"
      echo ''

      echo 'Step 2: Generating all CHM variants...'
      python scripts/generate_all_chm_variants.py \
        --laz-dir '$LAZ_INPUT_DIR' \
        --output-dir '$CHM_OUTPUT_DIR' \
        --variants baseline,raw,gaussian,composite,masked-raw \
        --verbose
    " SKIP_DOWNLOAD="$SKIP_DOWNLOAD"
else
  echo "Executing locally with system Python..."
  echo ""

  echo "Step 1: Downloading LAZ file for kaardiruut 406455..."
  if [ $SKIP_DOWNLOAD -eq 0 ]; then
    python scripts/laz_mass_downloader.py \
      --ids 406455 \
      --out "$LAZ_INPUT_DIR" \
      --workers 4
    echo "✓ LAZ download complete"
  else
    echo "  (Skipping download, using existing LAZ files)"
  fi
  echo ""

  LAZ_COUNT=$(find "$LAZ_INPUT_DIR" -name "*.laz" | wc -l)
  if [ "$LAZ_COUNT" -eq 0 ]; then
    echo "✗ Error: No LAZ files found in $LAZ_INPUT_DIR"
    exit 1
  fi
  echo "Found $LAZ_COUNT LAZ file(s)"
  echo ""

  echo "Step 2: Generating all CHM variants..."
  python scripts/generate_all_chm_variants.py \
    --laz-dir "$LAZ_INPUT_DIR" \
    --output-dir "$CHM_OUTPUT_DIR" \
    --variants baseline,raw,gaussian,composite,masked-raw \
    --verbose
fi

echo ""
echo "======================================================================="
echo "✓ Build complete!"
echo "======================================================================="
echo ""
echo "Generated CHM variants:"
find "$CHM_OUTPUT_DIR" -maxdepth 1 -type d | sort | sed 's|^|  |'
echo ""
echo "Next steps:"
echo "  1. Review CHM outputs in $CHM_OUTPUT_DIR"
echo "  2. Commit to git:"
echo "     git add source/406455_2021_tava/"
echo "     git commit -m 'source(data): add 406455_2021_tava LAZ & CHM variants'"
echo "======================================================================="

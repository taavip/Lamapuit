#!/bin/bash
# Wrapper: install dependencies + run ensemble voting plan

source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

echo "Installing missing dependencies..."
pip install -q captum rasterio opencv-python scipy 2>/dev/null || true

echo "Verifying PyTorch & dependencies..."
python -c "
import torch
import captum
import cv2
import scipy
import rasterio
print('✓ PyTorch:', torch.__version__)
print('✓ captum: OK')
print('✓ cv2: OK')
print('✓ scipy: OK')
print('✓ rasterio: OK')
"

echo ""
echo "Dependencies ready. Running ensemble voting plan..."
echo ""

bash run_ensemble_voting_plan.sh

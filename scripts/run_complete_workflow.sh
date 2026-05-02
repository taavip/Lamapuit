#!/bin/bash
# Complete workflow: Test → Split → Train → Evaluate

set -e  # Exit on error

echo "============================================"
echo "LAMAPUIT COMPREHENSIVE WORKFLOW"
echo "============================================"
echo ""

# Step 1: Run tests
echo "Step 1: Running test suite..."
echo "--------------------------------------------"
docker run -it --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -c "
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate cwd-detect && \
    pytest tests/ -v --cov=src/cdw_detect --cov-report=term-missing
"

if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Fix issues before proceeding."
    exit 1
fi
echo "✓ All tests passed!"
echo ""

# Step 2: Create test split
echo "Step 2: Creating train/val/test split..."
echo "--------------------------------------------"
docker run -it --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -c "
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate cwd-detect && \
    python scripts/create_test_split.py \
        --source data/dataset_enhanced_robust \
        --output data/dataset_final \
        --train-ratio 0.70 \
        --val-ratio 0.15 \
        --test-ratio 0.15
"

if [ $? -ne 0 ]; then
    echo "❌ Dataset split failed!"
    exit 1
fi
echo "✓ Dataset split created!"
echo ""

# Step 3: Run comprehensive experiments
echo "Step 3: Running training experiments..."
echo "--------------------------------------------"
echo "This will take several hours. Monitor progress in runs/experiments/"
echo ""

docker run -it --rm --gpus all --shm-size=8g \
    -v "$PWD":/workspace -w /workspace lamapuit-dev bash -c "
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate cwd-detect && \
    python scripts/run_experiments.py --dataset data/dataset_final/dataset_trainval.yaml
"

if [ $? -ne 0 ]; then
    echo "⚠️  Some experiments may have failed. Check experiments_results.yaml"
else
    echo "✓ All experiments completed!"
fi
echo ""

# Step 4: Analyze results
echo "Step 4: Analyzing experiment results..."
echo "--------------------------------------------"
docker run -it --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -c "
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate cwd-detect && \
    python scripts/analyze_experiments.py
"

echo ""
echo "============================================"
echo "WORKFLOW COMPLETE!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Review results in: experiments_results.yaml"
echo "2. Check visualizations in: runs/experiments/"
echo "3. Run best model with multi-run: python scripts/train_multirun.py"
echo ""

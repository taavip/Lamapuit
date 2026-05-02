#!/bin/bash
# Simple wrapper to run CHM ablation experiment

set -e

echo "=========================================="
echo "CHM Ablation Experiment"
echo "=========================================="

python scripts/chm_ablation_train.py \
    --labels-dir output/model_search_v4/prepared/labels_main_budget \
    --output-dir output/model_search_chm_ablation_results \
    --chm-raw output/chm_dataset_harmonized_0p8m_raw_gauss_stable/chm_raw \
    --chm-gauss output/chm_dataset_harmonized_0p8m_raw_gauss_stable/chm_gauss \
    --chm-baseline chm_max_hag \
    --n-folds 3 \
    --test-fraction 0.10 \
    --epochs 30 \
    --batch-size 16 \
    --patience 5

echo ""
echo "=========================================="
echo "Training complete. Analyzing results..."
echo "=========================================="

python scripts/chm_ablation_analyze.py output/model_search_chm_ablation_results/results.json

echo ""
echo "=========================================="
echo "Generating final report..."
echo "=========================================="

python scripts/chm_ablation_final_report.py \
    output/model_search_chm_ablation_results/results.json \
    output/model_search_chm_ablation_results/FINAL_REPORT.txt

echo ""
echo "Results saved to output/model_search_chm_ablation_results/"
ls -lah output/model_search_chm_ablation_results/

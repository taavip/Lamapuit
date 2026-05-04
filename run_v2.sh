#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect
rm -rf experiments/dtm_hag_436646_research/results/res_*/scratch/*.laz
for res in 0.5 0.2 0.1; do
    echo "Running dem-resolution \$res"
    out_dir="experiments/dtm_hag_436646_research/results/res_\${res}"
    mkdir -p \$out_dir
    python experiments/dtm_hag_436646_research/run_experiment.py --tile-id 436646 --years 2018 --out-dir \$out_dir --dem-resolution \$res --epsg 3301 --point-sample-rate 1.0 --gaussian-sigma 0.0
done
python generate_comparison.py

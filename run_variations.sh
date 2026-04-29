#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

# We test three DEM resolutions: 0.5m, 0.2m, and 0.1m to see how it affects the smoothing.
# Higher resolution (smaller numbers) means less smoothing of the ground.
for res in 0.5 0.2 0.1; do
    echo "Running dem-resolution $res..."
    out_dir="experiments/dtm_hag_436646_research/results/res_${res}"
    mkdir -p $out_dir
    python experiments/dtm_hag_436646_research/run_experiment.py \
        --tile-id 436646 \
        --years 2018 \
        --out-dir $out_dir \
        --dem-resolution $res \
        --reuse-csf \
        --epsg 3301 \
        --point-sample-rate 1.0 \
        --gaussian-sigma 0.0
done

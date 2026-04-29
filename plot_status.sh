#!/bin/bash
while true; do
  if [ -f "experiments/dtm_hag_436646_research/results/res_0.1/chm/2018/2018_idw_k3_raw_chm.tif" ]; then
    docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python generate_comparison.py"
    echo "Plotting finished!"
    exit 0
  fi
  sleep 10
done

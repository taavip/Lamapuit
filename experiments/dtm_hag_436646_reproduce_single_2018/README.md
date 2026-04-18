This experiment folder holds a reference copy of the research 2018 CHM and a small wrapper to run a targeted reproduction.

Files:
- reference/chm/2018/2018_tin_linear_gauss_chm.tif  # copied from research outputs
- run_targeted_experiment.sh  # copies reference and runs the research experiment for year 2018 only

Usage:
- From the repo root run:
  ./experiments/dtm_hag_436646_reproduce_single_2018/run_targeted_experiment.sh

Notes:
- The script will attempt to run the research `run_experiment.py` for `--years 2018` and will keep only the produced
  `2018_tin_linear_gauss_chm.tif` under `results/chm/2018/` (deleting other CHM files for that year).
- Adjust `--laz-dir`, `--labels-dir`, or other flags inside the script if your environment differs.

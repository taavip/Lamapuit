# DTM/HAG Reproduce + Agile Experiment (2026-04-14)

## Goal
Reproduce the research reference style from:
- `experiments/dtm_hag_436646_research/results/chm/2018/2018_tin_linear_gauss_chm.tif`

Then improve it by:
- making DTM behavior more responsive on steep slopes (ditch-like terrain)
- allowing negative near-ground CHM values (instead of forcing all CHM >= 0)
- sweeping Gaussian smoothing values to improve detail-vs-smoothness balance

## Main Script
- `run_experiment.py`

Key additions for this dated experiment:
- `--chm-clip-min`: lower clip bound for CHM output (set negative values to preserve near-ground negatives)
- slope-adaptive DTM options reused from the hybrid workflow

## Suite Runner
- `run_reproduce_suite.sh`

Runs five variants:
1. research-like reproduction baseline (all returns, no class exclusion, gaussian=1.0)
2. agile slope-adaptive g=0.8 with negative CHM floor
3. agile slope-adaptive g=0.6 (more detail)
4. agile slope-adaptive g=0.4 (sharp detail)
5. agile slope-adaptive g=1.2 (smoother control)

## Summary Script
- `summarize_reproduce_suite.py`

Outputs:
- `results/suite_summary.csv`
- `results/suite_report.md`

Additional metrics include:
- `neg_pct_2018`: percent of valid CHM pixels below 0 m
- `min_2018`, `p05_2018`
- corrected false-high handling in score (`1 - false_high_rate`)

## Run
```bash
bash experiments/dtm_hag_436646_reproduce_2026-04-14/run_reproduce_suite.sh
```

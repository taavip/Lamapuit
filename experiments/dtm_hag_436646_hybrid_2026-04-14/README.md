# Hybrid Nuance-Slope Experiment (2026-04-14)

This dated experiment targets a practical hybrid objective:

- Keep slope stability from the SOTA followup workflow (SMRF + alignment + SOR + slope-adaptive harmonization).
- Move CHM appearance and subtle-object recoverability closer to the research-style nuance.

## Design

The suite runs multiple variants with controlled CHM-generation changes:

1. Return policy (`last`, `last2`, `all`)
2. Class exclusion policy (`6,9` vs `9`)
3. Soft HAG inclusion floor (`hag_min`), with output clipped to `[0, hag_max]`
4. Gaussian post-filter sigma sweep for nuance-vs-chunkiness tradeoff

Terrain harmonization and interpolation remain fixed to preserve slope robustness.

## Files

- [run_experiment.py](run_experiment.py): Hybrid-capable experiment script (adds `--hag-min`).
- [run_hybrid_suite.sh](run_hybrid_suite.sh): Executes multi-variant suite in Docker + conda.
- [summarize_hybrid_suite.py](summarize_hybrid_suite.py): Builds quantitative suite summary and markdown report.

## Run

```bash
cd /home/tpipar/project/Lamapuit
bash experiments/dtm_hag_436646_hybrid_2026-04-14/run_hybrid_suite.sh
```

## Outputs

- Variant runs: [results/runs](results/runs)
- Suite CSV summary: [results/suite_summary.csv](results/suite_summary.csv)
- Academic report: [results/suite_report.md](results/suite_report.md)

## Best-Practice Notes

- Reproducibility: fixed seeds, versioned dated directory, explicit run grid.
- Comparability: same tile, years, and label set for all variants.
- Stability-first: slope-adaptive terrain model held constant across variants.
- Nuance-aware assessment: combines detection metrics and CHM texture similarity to research reference.

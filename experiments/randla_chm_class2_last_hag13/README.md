# RandLA CHM Class2 Last HAG13 Experiment

This experiment is a focused pipeline for LAZ files in:

- output/laz_reclassified_randla

It is separated from older CDW experiments and enforces:

- DTM from class-2 ground points
- Return filtering set to last returns
- HAG drop mode to keep only 0.0 to 1.3 m
- Class exclusion for man-made objects (default: classes 6 and 17)

## Folder Layout

- description: experiment purpose and run notes
- results: generated CHM outputs
- analysis: JSON, CSV, and Markdown reports

## Run In Docker

```bash
cd /home/tpipar/project/Lamapuit

docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python experiments/randla_chm_class2_last_hag13/run_experiment.py --input-dir output/laz_reclassified_randla --pattern '*_reclassified_randla.laz' --results-dir experiments/randla_chm_class2_last_hag13/results --analysis-dir experiments/randla_chm_class2_last_hag13/analysis --exclude-classes 6,17 --return-mode last --hag-max 1.3 --overwrite"
```

## Notes

- If a file has zero class-2 points, the run fails by default (strict class-2 mode).
- To allow fallback behavior, add `--allow-ground-fallback`.
- To include additional classes for exclusion, pass for example `--exclude-classes 6,17,64`.

# LAZ to CHM Dataset (Harmonized Ground, 0.8m DEM)

This folder contains a fresh, reproducible pipeline for academic experiments.

Core design:
- CSF ground classification.
- Multi-year harmonized ground DEM at 0.8 m.
- CHM generation with no IDW/TIN/TPS interpolation.
- Two CHM outputs only: raw and gaussian.
- Label copies organized by tile and year.
- Detailed logging and error tracking.

## Output Structure

Given an output root passed to `--out-dir`, the script writes:

- `chm_raw/<tile>/<year>/*_raw_chm.tif`
- `chm_gauss/<tile>/<year>/*_gauss_chm.tif`
- `labels/<tile>/<year>/*_labels.csv` (if source labels exist)
- `reports/run.log`
- `reports/errors.jsonl`
- `reports/run_parameters.json`
- `reports/dataset_manifest.csv`
- `reports/dataset_summary.json`

No interpolation-based DTM methods are created.

## Fast Start (Docker, GPU)

```bash
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace lamapuit:gpu bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python experiments/laz_to_chm_harmonized_0p8m/build_dataset.py --laz-dir data/lamapuit/laz --labels-dir output/onboarding_labels_v2_drop13 --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop --out-dir output/chm_dataset_harmonized_0p8m_raw_gauss --dem-resolution 0.8 --hag-max 1.3 --chm-clip-min 0.0 --hag-upper-mode drop --gaussian-sigma 0.3 --return-mode last --reuse-csf --gpu-mode auto"
```

GPU backend selection in `--gpu-mode auto`:
- `cuda-cupy` if CuPy + CUDA is available.
- `cuda-torch` if torch CUDA is available.
- `cpu` if no GPU backend is available.

To force GPU usage for validation:

```bash
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace lamapuit:gpu bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python experiments/laz_to_chm_harmonized_0p8m/build_dataset.py --laz-dir data/lamapuit/laz --labels-dir output/onboarding_labels_v2_drop13 --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop --out-dir output/chm_dataset_harmonized_0p8m_raw_gauss --dem-resolution 0.8 --hag-max 1.3 --chm-clip-min 0.0 --hag-upper-mode drop --gaussian-sigma 0.3 --return-mode last --reuse-csf --gpu-mode force --tiles 436646"
```

To speed up full runs with existing CSF files:

```bash
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace lamapuit:gpu bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python experiments/laz_to_chm_harmonized_0p8m/build_dataset.py --laz-dir data/lamapuit/laz --labels-dir output/onboarding_labels_v2_drop13 --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop --out-dir output/chm_dataset_harmonized_0p8m_raw_gauss --dem-resolution 0.8 --hag-max 1.3 --chm-clip-min 0.0 --hag-upper-mode drop --gaussian-sigma 0.3 --return-mode last --reuse-csf --csf-cache-dir output/chm_dataset_lastreturns_hag0_1p3/scratch --gpu-mode auto"
```

For stability on full-dataset CPU-heavy runs, use explicit worker limits:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace lamapuit-dev bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && python experiments/laz_to_chm_harmonized_0p8m/build_dataset.py --laz-dir data/lamapuit/laz --labels-dir output/onboarding_labels_v2_drop13 --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop --out-dir output/chm_dataset_harmonized_0p8m_raw_gauss --dem-resolution 0.8 --hag-max 1.3 --chm-clip-min 0.0 --hag-upper-mode drop --gaussian-sigma 0.3 --return-mode last --reuse-csf --csf-cache-dir output/chm_dataset_lastreturns_hag0_1p3/scratch --gpu-mode off --workers 4 --max-safe-workers 4 --continue-on-error"
```

## Reliability Controls

- Default behavior is fail-fast (stops at first error).
- Use `--continue-on-error` to process remaining tiles even if some fail.
- Use `--keep-work` to keep temporary CSF and intermediate files.
- `reports/errors.jsonl` records machine-readable failures.
- `--max-safe-workers` (default `4`) auto-caps CPU parallelism to reduce abrupt Docker exits under heavy memory/IO pressure.

If a run stops abruptly and `reports/dataset_summary.json` is missing while `reports/run.log` ends mid-line,
the most common cause is worker oversubscription (too many concurrent tile jobs). Lower `--workers`.

## Docker Crash Checklist (Engine-Level)

If Docker itself crashes (container disappears with no Python traceback in `run.log`), check engine pressure first:

- Host Docker storage nearly full (common on Docker Desktop when `C:` is near 100%).
- Restart-loop containers generating continuous logs/restarts.
- Very large build cache and many stale images.

Recommended cleanup before long CHM runs:

```bash
docker image prune -f
docker builder prune -f
docker system df
```

If another container is flapping, stop it during the CHM run:

```bash
docker update --restart=no <container_name>
docker stop <container_name>
```

Then restart the CHM run with conservative workers (`--workers 4 --max-safe-workers 4`).

## Metadata Edge Cases (Handled)

The pipeline includes guardrails for malformed LAZ metadata that commonly appears in mixed archives:

- `readers.las Error) Could not create an SRS.`
	- The CSF PDAL reader runs with `nosrs=true` and `default_srs=EPSG:<epsg>`.
	- The writer sets `a_srs=EPSG:<epsg>` so output files still carry the expected CRS.
- `OverflowError: value 8 is greater than allowed (max: 7)` during sanitize fallback
	- Sanitization clamps `return_number` and `number_of_returns` to LAS point-format 3 limits (`1..7`) and enforces `return_number <= number_of_returns`.

If a tile already has stale intermediates from an older run, remove that tile folder under `_work/` and rerun the tile with `--tiles <tile_id>`.

## Useful Options

- `--tiles 436646,436647` for focused debugging.
- `--gpu-mode off|auto|force`.
- `--point-sample-rate <0..1>` for speed studies.
- `--fallback-grid-resolution 0.2` for LAZ files without baseline CHM geometry.

## Method Notes

Detailed methodology and equations are documented in [METHODOLOGY.md](METHODOLOGY.md).

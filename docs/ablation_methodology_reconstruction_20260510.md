# Ablation Methodology Reconstruction (2026-05-10)

## Scope
- Reconstructed CHM generation pipeline outputs from original harmonized methodology.
- Re-wired ablation pipeline to use those outputs in a reproducible, thesis-safe way.
- Removed silent fallbacks that mixed incompatible dataset assets.

## Key Validity Fixes
- `phase3_ablation_v10.py` now uses one dataset root (`--dataset-dir`) for both:
  - `patch_index_<variant>.csv`
  - `band_stats_<variant>.json`
- Removed composite fallback when a variant patch index is missing.
  - Before: variant could silently train on composite patch index.
  - Now: explicit error, forcing per-variant dataset preparation.
- Added `--condition-ids` filter for controlled pruning of weak ideas.

## Wrapper Improvements
- `run_full_ablation_automated_top2.sh` now supports:
  - `CHM_SOURCE_DIR`: relinks `seg_pipeline/input/*.tif` to a chosen CHM output set.
  - `DATASET_DIR`: explicit dataset asset directory.
  - `REBUILD_DATASET=true|false`: regenerate per-variant indices/stats before ablation.
  - `PHASE2_CONDITIONS`: comma-separated phase-2 condition IDs for faster/focused study.
- Preflight step added:
  - Relinks CHM inputs.
  - Rebuilds dataset assets per variant via `phase2_dataset_v3.py --cv-version 3`.

## Reconstructed CHM Source Used
- `source/406455_2021_tava/chm_variants_reconstructed_original_20260510`

## Recommended Repro Command
```bash
CHM_SOURCE_DIR=/home/tpipar/project/Lamapuit/source/406455_2021_tava/chm_variants_reconstructed_original_20260510 \
DATASET_DIR=/home/tpipar/project/Lamapuit/seg_pipeline/output/phase2_dataset_v10_reconstructed \
REBUILD_DATASET=true \
PHASE2_CONDITIONS=2A,2B,2C,2D,2E \
EPOCHS=100 \
bash run_full_ablation_automated_top2.sh
```

## Optional Faster Search
If phase-2 confirms weak variants again, restrict to stronger candidates:
```bash
PHASE2_CONDITIONS=2A,2B,2C
```

## Notes for Thesis Reporting
- Model selection remains validation-only during search phases.
- Held-out test stripe should be used only once after final configuration lock.
- Dataset assets are now variant-matched and reproducible from one directory.

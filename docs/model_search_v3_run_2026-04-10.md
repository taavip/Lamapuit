# Model Search V3 Run Log (2026-04-10)

## Objective

Implement and run a leakage-safe V3 experiment with:

- strict train/test anti-leak split by place (all years of same place on one side only)
- spatial test holdout blocks
- 1-tile spatial fence buffer (26m)
- spatial-block cross-validation with block-size probe
- model pool updated from V2 results (remove redundant/weak variants)
- controlled pseudo-label thresholding for Drop13 auto-labels
- nodata-aware augmentation policy for robustness

## User Requirements Mapped to Decisions

1. Stop active GPU calculations first
- Action: stopped active `lamapuit:gpu` container and verified no training processes remained.

2. Keep same place across years in only train or test
- Action: added `split_mode=spatial_blocks` in `model_search_v2.py`.
- Implementation: place-level grouping by parsed raster identity (`tile + site`), then test selection by spatial blocks.
- Validation metric: `place_overlap_train_vs_test = 0` in split metadata.

3. Use block-based split and 1 full tile buffer (~26m)
- Action: spatial-block test split + `--spatial-fence-m 26.0` train/test fence in `model_search.py`.

4. Train/test split approximately 20/80
- Action: `--test-fraction 0.2` (interpreted as ~20% test, ~80% train).

5. Use baseline labels + Drop13 labels
- Action: curated label build still merges baseline + Drop13; dedup conflict resolution preserved.

6. Revisit pseudo-label thresholds
- Action: changed V3 defaults to `t_high=0.98`, `t_low=0.05` in `model_search_v3.py`.
- Rationale: previous `0.9995/0.0698` was too extreme for the new Drop13 probability distribution.

7. Remove models from V3 priorities
- Removed: `convnext_tiny`
- Removed: `deep_cnn_attn_headwide`
- Retained compact pool:
  - `convnext_small`
  - `convnextv2_small`
  - `deep_cnn_attn_dropout_tuned`
  - `efficientnet_b2`
  - `maxvit_small`
  - `eva02_small`

8. Spatial Block Cross-Validation with block-size test
- Action: added CV block probe (`--auto-cv-block-size`, `--cv-block-candidates-m`).
- Current candidates: `26,39,52,78,104` meters.

9. Nodata augmentation constraints
- Action: added optional training augmentation knobs in `model_search.py`:
  - random nodata mask fraction (`--augment-random-nodata-frac`)
  - repeating pattern nodata fraction (`--augment-pattern-nodata-frac`)
- V3 defaults set to:
  - random nodata = `0.50`
  - repeating nodata pattern = `0.75`

## Files Changed

- `scripts/model_search_v2/model_search_v2.py`
- `scripts/model_search.py`
- `scripts/model_search_v3/model_search_v3.py`
- `docs/model_search_v3_priorities.md`

## Commands Executed (Docker + conda)

### Prepare-only validation

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD":/workspace -w /workspace lamapuit:gpu \
  bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && \
  python scripts/model_search_v3/model_search_v3.py \
    --output output/model_search_v3_academic_leakage26 \
    --prepare-only"
```

### Full run launch

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD":/workspace -w /workspace lamapuit:gpu \
  bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && \
  python scripts/model_search_v3/model_search_v3.py \
    --output output/model_search_v3_academic_leakage26"
```

### Full run relaunch with timm (required for ConvNeXtV2 + EVA02)

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD":/workspace -w /workspace lamapuit:gpu \
  bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && \
  python -m pip install --quiet 'timm>=0.9.16,<1.1' && \
  python scripts/model_search_v3/model_search_v3.py \
    --output output/model_search_v3_academic_leakage26"
```

Reason for relaunch:

- initial full run skipped `convnextv2_small` and `eva02_small` due missing `timm` in the runtime env
- relaunch with in-container `timm` install restored the full planned model pool

## Prepare-only Output Snapshot

- `curated_rows_after_dedup`: `40627`
- `drop_rows_after_dedup`: `18629`
- split mode: `spatial_blocks`
- target test fraction: `0.2`
- actual test fraction: `0.2021315873680065`
- places total/test/train: `30 / 5 / 25`
- blocks total/test: `21 / 4`
- same-place leakage check: `place_overlap_train_vs_test = 0`

## Live Training Log Snapshot (after full launch)

- `Records loaded | train=32415 test=8212`
- `CV block probe | selected_block_m=26.00 candidates=26.0,39.0,52.0,78.0,104.0`
- `Data analysis complete | n=32415, cdw=7262, no_cdw=25153, rasters=101, gini=0.515`
- model candidates after corrected relaunch:
  - `convnext_small`
  - `convnextv2_small`
  - `deep_cnn_attn_dropout_tuned`
  - `efficientnet_b2`
  - `maxvit_small`
  - `eva02_small`

## Artifacts and Logs

- Output root: `output/model_search_v3_academic_leakage26`
- Prepared summary: `output/model_search_v3_academic_leakage26/prepared/prepared_dataset_summary.json`
- Prepared counts: `output/model_search_v3_academic_leakage26/prepared/prepared_dataset_counts.csv`
- Test split keys: `output/model_search_v3_academic_leakage26/prepared/cnn_test_split_v2.json`
- CV block probe: `output/model_search_v3_academic_leakage26/cv_block_probe.json`
- Training log (stream + file): `output/model_search_v3_academic_leakage26/model_search.log`

## Notes

- The run is configured to prioritize trustworthiness and leakage control while limiting redundant model retests.
- Stage 2 remains reduced (`mixup_swa,tta`) to keep runtime practical.

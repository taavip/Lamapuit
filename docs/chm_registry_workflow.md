# CHM Registry Workflow

This workflow keeps existing data untouched and generates a unified registry for
all CHM variants (raw, gaussian, baseline) that share the same true label.

## Why this structure

- One sample key per LAZ: `mapsheet_year_campaign`.
- One label path reused across all CHM variants.
- Wide + long manifests for different training pipelines.
- SQLite export for robust filtering and split generation.

## Build the registry

From workspace root:

```bash
python3 scripts/build_chm_registry.py
```

## Outputs

The command writes:

- `data/lamapuit/laz/eligible_manifest.csv`
  - Updated to the unique local LAZ inputs currently present.
  - Keeps source metadata columns (`url`, `size_bytes`) when available.
  - If this path is read-only, a fallback file is written instead.
- `registry/chm_dataset_harmonized_0p8m_raw_gauss/eligible_manifest.local.csv`
  - Fallback manifest when source manifest location is not writable.
- `registry/chm_dataset_harmonized_0p8m_raw_gauss/ml_samples.csv`
  - One row per sample with paths to LAZ, label, raw/gauss/baseline TIFF.
- `registry/chm_dataset_harmonized_0p8m_raw_gauss/ml_variants.csv`
  - One row per sample+variant with `trainable` flag.
- `registry/chm_dataset_harmonized_0p8m_raw_gauss/laz_folder_inventory.csv`
  - LAZ folder inventory with `delete_candidate` flag for copy-like folders.
- `registry/chm_dataset_harmonized_0p8m_raw_gauss/ml_registry.sqlite`
  - SQLite database with tables:
    - `samples`
    - `variants`
    - `laz_folders`

If your `output/` tree is writable and you want registry files there, pass custom
arguments such as `--samples-csv` and `--registry-db`.

## Train/Val/Test split guidance

Use tile-level splits to avoid leakage across years of the same location.

- Prefer splitting on `mapsheet` instead of random per-row split.
- Filter trainable rows with:
  - `variants.trainable = 1`
- Choose variant with `variants.variant IN ('raw', 'gauss', 'baseline')`.

Example SQL for trainable gaussian samples:

```sql
SELECT *
FROM variants
WHERE variant = 'gauss' AND trainable = 1;
```

## Cleanup guidance

For LAZ cleanup planning, use `laz_folders.delete_candidate = 1` as candidates.
Review before deletion to avoid removing primary source LAZ files.

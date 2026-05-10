# Experiment Log — CWD Semantic Segmentation Pipeline

## Template for each run

```
### Run YYYY-MM-DD — [arch] [fold] [notes]
- Phase: I / II / III / IV
- Command: python seg_pipeline/scripts/phaseN_*.py --args ...
- Duration: X min
- GPU: utilization XX%, mem XX/XX MiB
- Key metric: val_dice=X.XXX, val_iou=X.XXX
- Notes: ...
```

---

## Run Log

<!-- Append new entries below as experiments are completed -->

### 2026-05-05 — Pipeline setup
- Created directory structure: `seg_pipeline/{doc,input,output,scripts/common}`
- Created input symlinks to CHM variants and labels
- Verified `cdw_labels_MP.gpkg` bounds overlap 406455 tile (250 MultiPolygon features)
- CHM tile: 5000×5000 px, EPSG:3301, 29.1% valid pixel coverage

---

## GPU Utilization Reference

GPU log is written every 10 s by the `gpu-monitor` sidecar to:
`seg_pipeline/output/phase4_report/gpu_log.csv`

Read with:
```python
import pandas as pd
df = pd.read_csv("seg_pipeline/output/phase4_report/gpu_log.csv")
print(df.describe())
```

---

## Phase I Notes

Expected runtime: 15–45 min (depends on GPU; 5000×5000 tile, 128×128 chunks, stride=64 → ~5776 chunks × 8 TTA × 4 models).

Smoke test (512×512 crop): `python seg_pipeline/scripts/phase1_mask_synthesis.py --smoke-test`

---

## Phase III Expected Runtimes

| Architecture      | Params  | Est. time/fold (GPU) |
|-------------------|---------|---------------------|
| unetpp_effb2      | ~8M     | 20–40 min            |
| deeplabv3plus_r50 | ~26M    | 30–50 min            |
| segformer_b2      | ~25M    | 25–45 min            |

Total (3 archs × 4 folds): ~5–8 hours GPU

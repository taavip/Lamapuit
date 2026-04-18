**Models Training — Inventory, Data Sources & Results**

**Summary (manual vs auto + models)**
- **Manual rows with model probability:** 8,386 (rows labeled `manual` that include `model_prob`).
- **Manual CDW / No-CDW split:** 3,526 CDW, 4,860 no_CDW.
- **Model-search run (overview):** comprehensive swept experiments and final retrain/ensembles; see `output/model_search` for artifacts and summary.

---

**Manual vs Auto: Counts & thresholds (manual-only subset)**
- **Total (manual w/prob):** 8,386
- **CDW:** 3,526 — model_prob: min=0.0708, median≈0.5638, max=0.9999
- **No-CDW:** 4,860 — model_prob: min=0.0315, median≈0.3976, max=0.9994

**Data-driven thresholds (using only manual labels)**
- **t_high (perfect precision for CDW on manual sample):** 0.9995
- **t_low  (perfect precision for No-CDW on manual sample):** 0.0698
- **t_high (≥5-sample conservative):** 0.9997

Notes: thresholds above guarantee no errors on the manual-labeled subset; they are conservative and may under-cover correct positives. Use for safe auto-accept decisions only, and validate on held-out data if you lower them.

---

**Model artifacts (workspace paths)**
- **Ensemble single-checkpoint:** [output/tile_labels/ensemble_model.pt](output/tile_labels/ensemble_model.pt)
- **Ensemble metadata:** [output/tile_labels/ensemble_meta.json](output/tile_labels/ensemble_meta.json)
- **CNN seeds:** [output/tile_labels/cnn_seed42.pt](output/tile_labels/cnn_seed42.pt), [output/tile_labels/cnn_seed43.pt](output/tile_labels/cnn_seed43.pt), [output/tile_labels/cnn_seed44.pt](output/tile_labels/cnn_seed44.pt)
- **EfficientNet-B2:** [output/tile_labels/effnet_b2.pt](output/tile_labels/effnet_b2.pt)
- **Model metadata & run artifacts:** [output/tile_labels/model_meta.json](output/tile_labels/model_meta.json), [output/tile_labels/finetune_done.json](output/tile_labels/finetune_done.json), [output/tile_labels/finetune.log](output/tile_labels/finetune.log)

**Key training & orchestration scripts**
- **Comprehensive model search:** [scripts/model_search.py](scripts/model_search.py)
- **Fine-tune single CNN and save checkpoint:** [scripts/fine_tune_cnn.py](scripts/fine_tune_cnn.py)
- **Train ensemble (3 CNN seeds + EfficientNet-B2):** [scripts/train_ensemble.py](scripts/train_ensemble.py)
- **Resave/clean .pt helper:** [scripts/_resave_pt.py](scripts/_resave_pt.py)
- **Labeler GUI (hot-reload + inference):** [scripts/label_tiles.py](scripts/label_tiles.py)
- **Per-raster label orchestration:** [scripts/label_all_rasters.py](scripts/label_all_rasters.py)

---

**Models Training — model_search run (what was run, what data, and results)**

Script: [scripts/model_search.py](scripts/model_search.py)
- Purpose: stage a broad, leakage-aware model search (Stage 1 baseline sweep across architectures; Stage 2 focused matrix on top models), retrain top-k on non-test data, evaluate final test set, and evaluate soft-vote / stacking ensembles.
- Typical reproduction command (uses defaults):

```bash
python scripts/model_search.py \
  --labels output/tile_labels \
  --chm-dir chm_max_hag \
  --output output/model_search
```

Data used by the run (artifact links):
- Labels directory used: [output/tile_labels/](output/tile_labels/) — label CSVs with `model_prob`, `source`, and provenance.
- CHM rasters directory (default): `chm_max_hag/` (project root) — per-raster names governing tiles; per-raster stats: [output/model_search/data_analysis/per_raster_stats.csv](output/model_search/data_analysis/per_raster_stats.csv).
- Run manifest (run parameters & summary): [output/model_search/run_manifest.json](output/model_search/run_manifest.json) — key fields: `n_records_train=19812`, `n_records_test=2186`, `n_experiments=35`, `selected_models` list, `device`.

Quick data summary (from the run):
- Training tiles: **19812** (CDW: 3053, No-CDW: 16759) — see [output/model_search/data_analysis/analysis_summary.json](output/model_search/data_analysis/analysis_summary.json).
- Number of rasters used: **11** (see per-raster counts at [output/model_search/data_analysis/per_raster_stats.csv](output/model_search/data_analysis/per_raster_stats.csv)).

Final results (top retrained models and ensembles): see [output/model_search/final_test_results.csv](output/model_search/final_test_results.csv) and [output/model_search/RESEARCH_REPORT.md](output/model_search/RESEARCH_REPORT.md).

Representative final-test rows (top results):
- `soft_vote_top5` (soft-vote ensemble): test_auc=0.99819, test_f1=0.96460, precision=0.96460, recall=0.96460, threshold≈0.596
- `stacking_top5` (stacking logistic-reg): test_auc=0.99824, test_f1=0.96716, precision=0.97885, recall=0.95575, threshold≈0.924
- Best single-model entries (examples) are in [output/model_search/final_test_results.csv](output/model_search/final_test_results.csv).

Where to inspect the full experiment artifacts
- Per-experiment summaries and OOFs: [output/model_search/experiments/](output/model_search/experiments/)
- CV experiment summary: [output/model_search/experiment_summary.csv](output/model_search/experiment_summary.csv)
- Final retrained models: [output/model_search/final_models/](output/model_search/final_models/)
- Training logs and run progress: [output/model_search/model_search.log](output/model_search/model_search.log) and [output/model_search/progress.json](output/model_search/progress.json)

Notes and recommendations
- The `model_search.py` pipeline explicitly uses grouped/spatial CV (raster + coarse block) to address spatial leakage; it writes `run_manifest.json` and a `RESEARCH_REPORT.md` summarizing top experiments and final test metrics — these are the authoritative artifacts for this run.
- Reproducing the run at scale requires a GPU and time (the recorded run elapsed ~139,557s on `cuda`). For quick checks use `--smoke-test`.

---

**Practical next steps (suggested)**
- If you want safe auto-labeling coverage numbers per threshold: run per-raster threshold coverage analysis (I can run this and output a CSV).
- If you want calibration or reliability diagrams for seeds vs ensemble, I can generate ECE plots from the `final_models` / OOF predictions.

**References / relevant files**
- Model search and artifacts: [output/model_search/RESEARCH_REPORT.md](output/model_search/RESEARCH_REPORT.md), [output/model_search/final_test_results.csv](output/model_search/final_test_results.csv), [output/model_search/run_manifest.json](output/model_search/run_manifest.json), [output/model_search/experiment_summary.csv](output/model_search/experiment_summary.csv)
- Label CSVs and queues: [output/onboarding_labels_v2_drop13/manual_review_queue_pre_split.csv](output/onboarding_labels_v2_drop13/manual_review_queue_pre_split.csv) and the per-raster CSVs under [output/tile_labels/](output/tile_labels/).
- Training & labeler scripts: [scripts/model_search.py](scripts/model_search.py), [scripts/fine_tune_cnn.py](scripts/fine_tune_cnn.py), [scripts/train_ensemble.py](scripts/train_ensemble.py), [scripts/label_tiles.py](scripts/label_tiles.py)

---

Generated by GitHub Copilot (GPT-5 mini).

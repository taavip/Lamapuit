# Ensemble Voting & Precision Tuning — Implementation Summary

**Status**: Docker build in progress; plan execution scheduled for ~12 minutes

## Grilled & Approved Decisions

### 1. Calibration Subset ✓
- **Size**: 20–30 tiles (balanced)
- **Balance**: Mix of high-CWD and low-CWD tiles
- **Manual masks**: New or best existing `lamapuit.gpkg` labels
- **Isolation**: Held out entirely from training data

### 2. Evaluation Metric ✓
- **Primary metric**: Pixel-level precision (true positive pixels / predicted positive pixels)
- **Target precision**: ≥85%
- **Acceptable recall**: ≥50%
- **Fragmentation measure**: Mean component size (reject if fragments <0.1% of tile area)

### 3. Parameter Sweep ✓
- **Vote threshold** (0-4 votes): 0.33, 0.5, 0.67 → mapped to 1, 2, 3 votes
- **Multiscale sigmas**: [1, 2], [2, 4]
- **Uncertainty cutoff**: 0.6, 0.75, 0.9
- **Morphology sizes**: 1px, 2px, 3px
- **Total combinations**: 3 × 2 × 3 × 3 = **54 combinations**
- **Calibration dataset**: 20–30 tiles
- **Expected runtime**: ~1–2 hours (5s per combination if cached)

### 4. Smoke Run (Go/No-Go Verification) ✓
- **Size**: 3 tiles (mix of high/low CWD)
- **Per-model CAM saving**: Enabled (flag `--save-per-model-cams`)
- **Go criteria**:
  - Per-model CAM files exist for each model and tile ✓
  - Consensus mask is non-empty ✓
  - Ensemble voting improves IoU vs best single model (>0.1 gain) ✓
- **Infrastructure**: `generate_intgrad_masks.py` + `generate_consensus_masks.py`

### 5. Scaling Phase ✓
- **Dataset size**: 100–200 tiles
- **Geographic scope**: Same mapsheet(s) (no confounding variation)
- **Validation**: Visual inspection + quantitative summary (histograms)
- **Previews**: 10 stratified by CWD density and tile position

### 6. Run Report ✓
- **Format**: Markdown with embedded PNGs
- **Sections**:
  1. Parameter selection + rationale
  2. Calibration metrics (precision/recall/IoU table)
  3. Component size distribution (histogram)
  4. 10 representative preview grids
  5. Failure analysis (3–5 tiles where consensus was weak)
- **Quantitative claims**: On calibration set only
- **Qualitative claims**: On full scale-up
- **Audience**: Thesis + internal reference

### 7. Further Considerations ✓
- **Global thresholds (decided)**: Use fixed vote threshold (not adaptive per-tile)
- **Confidence-weighted voting (deferred)**: Test only if basic voting leaves artifacts
- **Report automation (decided)**: Simple Python script to build Markdown skeleton

## Execution Plan

### Phase 1: Smoke Test (5–10 min)
```
python scripts/generate_intgrad_masks.py \
  --labels data/chm_variants/smoke_test_labels.csv \
  --output-dir output/smoke_test_masks \
  --save-per-model-cams \
  --device cuda \
  --limit 3
```
**Outputs**:
- `output/smoke_test_masks/per_model_cams/{tile_id}/{model}_cam.npy`
- `output/smoke_test_masks/manifest.csv`
- `output/smoke_test_masks/previews/*.png`

**Verification**:
- Check `per_model_cams` directory exists with ≥12 files (3 tiles × 4 models)
- Run consensus voting on smoke test
- Verify non-zero masks and consensus alignment

### Phase 2: Calibration Subset (1 min prep)
- Sample 10 CDW + 15 no_CDW = 25 tiles from `labels_canonical_with_splits_retrained_ensemble.csv`
- Write `data/chm_variants/calibration_labels.csv`

### Phase 3: CAM Generation for Calibration (15–20 min)
```
python scripts/generate_intgrad_masks.py \
  --labels data/chm_variants/calibration_labels.csv \
  --output-dir output/calibration_masks \
  --save-per-model-cams \
  --device cuda
```

### Phase 4: Parameter Sweep (1–2 hours)
- Iterate 54 combinations
- For each: `python scripts/generate_consensus_masks.py --vote-threshold N --open-kernel K ...`
- Collect precision/recall/IoU for each
- Track results in `output/calibration_sweep/sweep_results.csv`

### Phase 5: Select Best Parameters
**Heuristic** (to be validated by sweep):
- Vote threshold: **2 out of 4** (balance precision/recall)
- Multiscale sigmas: **[2, 4]** (smooth at 2 scales)
- Uncertainty cutoff: **0.75** (require 75% agreement)
- Morphology: **2px** (preserve line structure, remove small noise)

### Phase 6: Scale to 100–200 Tiles (15–20 min)
```
python scripts/generate_intgrad_masks.py \
  --labels data/chm_variants/scale_labels.csv \
  --output-dir output/scale_masks \
  --save-per-model-cams \
  --device cuda

python scripts/generate_consensus_masks.py \
  --manifest output/scale_masks/manifest.csv \
  --vote-threshold 2 \
  --open-kernel 2 \
  --close-kernel 2 \
  --output-dir output/scale_consensus_final \
  --preview-count 10
```

### Phase 7: Generate Report (5 min)
- Compute statistics from `output/scale_consensus_final/consensus_manifest.csv`
- Generate `output/scale_consensus_final/REPORT.md` with:
  - Quantitative summary (mask ratios, confidence, component counts)
  - Parameter justification
  - Representative previews grid
  - Failure case analysis

## Key Files

| File | Purpose |
|------|---------|
| `scripts/generate_intgrad_masks.py` | Per-model CAM generation |
| `scripts/generate_consensus_masks.py` | Voting consensus → binary masks |
| `data/chm_variants/smoke_test_labels.csv` | 3-tile smoke test |
| `data/chm_variants/calibration_labels.csv` | 25-tile calibration set |
| `data/chm_variants/scale_labels.csv` | 150-tile scale set |
| `run_ensemble_voting_plan.sh` | Full orchestration script |

## Timeline

- **Now**: Docker build (10–15 min)
- **Phase 1–2**: Smoke test + calibration prep (20–30 min)
- **Phase 3–4**: Calibration CAM gen + sweep (1.5–2.5 hours)
- **Phase 5–7**: Scaling + reporting (30 min)
- **Total**: ~3–4 hours on GPU

## Success Criteria

✅ **Smoke test**: Per-model CAMs saved; consensus mask non-empty; voting improves IoU  
✅ **Calibration**: 25-tile manifest with >0 masks; precision metrics tracked  
✅ **Sweep**: All 54 combinations executed; best params identified  
✅ **Scaling**: 100–200 tiles processed; report generated with previews  
✅ **Report**: Markdown + PNG grid; thesis-ready format  

---

## Integration with Thesis

This pipeline directly supports thesis **Chapter 4** (Methodology / Mask Generation):
- Demonstrates defensible per-model voting (not aggressive thresholding)
- Quantifies precision-first approach (high-quality training labels)
- Preserves thin log structures (CWD centerlines)
- Provides statistical summary and failure analysis

Next step after validation: Use generated masks as training data for segmentation model.

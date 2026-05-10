# V3 Ensemble Prediction Inspection Guide

## Summary

The **V3 ensemble segmentation model** has generated a full-tile probability raster predicting CWD locations. This document guides you through loading, inspecting, and using it for label refinement.

### Model Performance

- **V1 baseline** (previous): test_dice_tta = 0.1686
- **V3 ensemble**: test_dice_tta_best_thr = **0.1920** ✅ **+14% improvement**
- **V3 best fold (Fold 1)**: test_dice_tta_best_thr = **0.2241** ✅ **+33% improvement**

### What the Raster Contains

- **File**: `seg_pipeline/output/phase4_report_v3/pred_ensemble_tta1.tif`
- **Dimensions**: 5000 rows × 1000 columns (test stripe, westernmost 1000m)
- **Spatial ref**: EPSG:3301 (Estonian LCC)
- **Resolution**: 0.2 m/pixel
- **Data type**: float32 (probability estimates [0, 1])
- **Size**: 22 MB (full-tile, 8-fold TTA averaged)

### Value Distribution

```
Mean confidence:     0.00952 (sparse, as CWD is rare)
Median confidence:   0.00165 (highly right-skewed)
Std deviation:       0.04609
Max confidence:      0.955

Percentiles:
  10th:  0.0009
  25th:  0.0012
  50th:  0.0016
  75th:  0.0028
  90th:  0.0088
  95th:  0.0304
```

### Pixels by Confidence Threshold

| Threshold | Count | % of Raster | Interpretation |
|-----------|-------|-------------|-----------------|
| > 0.3     | 31,489 | 0.63% | Candidate CWD (moderate) |
| > 0.5     | 13,330 | 0.27% | Strong CWD predictions |
| > 0.65    | 5,560  | 0.11% | High-confidence predictions |
| > 0.75    | 2,668  | 0.05% | Gold-standard predictions |

## QGIS Setup

### 1. Open QGIS and Load the Prediction Raster

```
Layer → Add Raster Layer → Choose:
  /home/tpipar/project/Lamapuit/seg_pipeline/output/phase4_report_v3/pred_ensemble_tta1.tif
```

Alternatively, drag-drop the file into QGIS.

### 2. Symbolize with Colormap

1. Right-click the layer → **Properties**
2. Go to **Symbology** tab
3. Set:
   - **Render type**: Singleband pseudocolor
   - **Min value**: 0.0
   - **Max value**: 1.0
   - **Color ramp**: Viridis (dark blue = low prob, bright yellow = high prob)
   - **Mode**: Continuous

### 3. Adjust Transparency (Optional)

- **Transparency** tab → set global opacity to 70% to see layers beneath
- This helps overlay context while viewing predictions

### 4. Load Context Layers

Add for comparison:

1. **CHM (visual context)**:
   ```
   Layer → Add Raster Layer →
   seg_pipeline/input/composite_4band.tif
   ```
   - First three bands auto-load as RGB

2. **Existing Labels (your current annotation)**:
   ```
   Layer → Add Vector Layer →
   lamapuit.gpkg
   ```
   - Symbolize as transparent outline (no fill) so predictions show through

3. **Orthophoto (if available)**:
   - Maa-amet WMS or Copernicus Sentinel-2
   - Helps distinguish forest stands, water bodies, clearcuts

### 5. Pan & Zoom

The full raster is 5000 m north-south × 1000 m east-west. Pan to areas of interest and zoom in (±2–5m/pixel for detailed inspection).

## Label Correction Workflow

### Identify Three Categories

1. **High confidence outside labels** (0.6+ probability, no GPKG polygon)
   - **Action**: ✅ **Add new labels** — model found CWD your annotations missed
   - **Priority**: High — likely true positives with high model confidence

2. **Low confidence inside labels** (<0.2 probability, inside GPKG polygon)
   - **Action**: ⚠️ **Review and possibly remove** — label may be mislabeled or ambiguous
   - **Priority**: Medium — check if it's thin CWD (sub-pixel), noise, or error

3. **Medium confidence uncertain** (0.3–0.6 probability, mixed label status)
   - **Action**: 🔍 **Inspect and decide** — borderline cases
   - **Priority**: Low — least certain, can address later

### Practical Steps

1. **Identify high-confidence regions** (yellow patches in raster)
   - Zoom to pixel level to see CHM texture
   - Compare with orthophoto (green=forest, gray=open, brown=deadwood)
   - If aligned with realistic log patterns → add to labels

2. **Check predicted vs. labeled overlap**
   - Use QGIS selection tools (Query Builder, Bounding Box)
   - Identify mislabeled areas (prediction low but GPKG polygon exists)
   - Verify with orthophoto — if prediction is correct, remove or trim label

3. **Export corrected labels**
   - Edit → Toggle Edit Mode
   - Add/delete/modify geometries in GPKG
   - Save → File → Export as new version (e.g., `lamapuit_v3_refined.gpkg`)

## Advanced Inspection

### Threshold-Based Masks

To export binary CWD masks at specific thresholds:

```bash
gdal_calc.py -A pred_ensemble_tta1.tif \
  --outfile=pred_binary_thr50.tif \
  --calc="(A > 0.5) * 1" --type=Byte
```

Then overlay as second raster layer in QGIS to see binary predictions.

### Compare Individual Folds

Individual fold predictions are also available:
- `pred_composite_fold0_tta1.tif`
- `pred_composite_fold1_tta1.tif`
- `pred_composite_fold2_tta1.tif`
- `pred_composite_fold3_tta1.tif`

Load all in QGIS to see per-fold agreement/disagreement. High agreement (all 4 folds high confidence) = reliable prediction.

### Statistics by Region

Use QGIS Raster → Raster Calculator to compute:

```
Sum of ensemble predictions by area (compute variance across tile)
Identify "hotspots" of high model uncertainty
```

## Next Steps

Once you've refined labels via inspection:

1. **Export corrected GPKG** as `lamapuit_refined.gpkg`
2. **Re-run Phase II** with new labels to rebuild patch index
3. **Re-run Phase III** to train V4 with improved ground truth
4. **Compare V3 vs V4 test performance** to validate refinement value

## Files Reference

| File | Purpose |
|------|---------|
| `pred_ensemble_tta1.tif` | **Main**: 5-fold ensemble predictions (recommended for inspection) |
| `pred_composite_fold0_tta1.tif` | Fold 0 (stripe 1 validation, stripe 0 test) predictions |
| `pred_composite_fold1_tta1.tif` | Fold 1 (stripe 2 validation) predictions — **highest test Dice** |
| `pred_composite_fold2_tta1.tif` | Fold 2 predictions |
| `pred_composite_fold3_tta1.tif` | Fold 3 predictions — **highest validation Dice** |
| `overlay_ensemble.png` | Raster + label overlay visualization (quick preview) |
| `final_metrics_v3.json` | Detailed per-fold and ensemble metrics |
| `thesis_table_v3.csv` | Summary table for thesis figures |

## Troubleshooting

**Raster not loading?**
- Check file path spelling (case-sensitive on Linux)
- Verify file is not corrupted: `gdalinfo pred_ensemble_tta1.tif | head -20`
- Try drag-drop instead of File → Open

**Colormap not showing?**
- Make sure render type is **Singleband pseudocolor**, not "Single band gray"
- Verify min/max are 0.0–1.0 (not auto-scaled to data range)

**Predictions look too sparse?**
- This is normal — CWD is 0.6–1% of the raster
- Adjust color ramp to accentuate low values (e.g., Blues, Greens) if needed

**Want to compare to V2?**
- Ensemble predictions for V2 are in `phase4_report_v2_cols1000/pred_ensemble_tta1.tif`
- Load both side-by-side to see V2 vs V3 improvements

---

**Questions?** See `docs/V3_EXPERIMENT_ANALYSIS.md` for detailed experiment design, rationale, and findings.

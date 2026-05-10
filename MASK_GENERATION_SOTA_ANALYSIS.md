# SOTA Mask Generation for CWD Segmentation: Deep Analysis & Strategies

## 1. Current Problem Analysis

### CAM Quality Issue
```
CAM Statistics (test file):
- Max: 0.165 (VERY WEAK)
- Mean: 0.0008 (dominated by noise)
- Median: 0 (75% zero pixels)
- P99: 0.022 (top 1% barely above noise)
- Pixels above 0.05: 68 (0.42% of tile)
- Result: Otsu threshold = 0.031, all masks stripped to 0
```

### Root Cause
1. **Weak Integrated Gradients signal** — 8 IG steps insufficient for dense CAM
2. **Aggressive post-processing** — morphology + connected components removes line features
3. **Mismatch between signal and filter** — CWD are thin logs (sparse), not dense blobs
4. **Single-model per inference** — no confidence/agreement from ensemble

---

## 2. SOTA Approaches for CAM → Segmentation

### Approach A: Gradient-Based Ensemble Aggregation (Recommended)
**Key Idea:** Don't threshold CAM; instead use multi-model agreement + uncertainty

**Steps:**
1. Extract saliency/CAM from each ensemble model independently
2. Normalize CAM per model (robust to absolute values)
3. **Consensus Rule:** Pixel marked positive only if N/M models agree (e.g., 3/4)
4. Use model uncertainty as secondary mask (std across ensemble)
5. Optional: Gaussian blur for spatial smoothness

**Advantages:**
- Doesn't rely on weak absolute CAM values
- Robust to outlier models
- Captures line features naturally (agreement along log centerline)
- Handles missing/sparse pixels better than thresholding

**Implementation:**
```python
# For each model, extract IG attribution
# Ensemble voting: pixel positive if >2 models above their own p90
mask = (model_count_above_p90 >= 3).astype(float32)
```

---

### Approach B: Gradient Magnitude + Directional Flow (GradCAM++)
**Key Idea:** Use gradient magnitudes + spatial coherence to find CWD centerlines

**Steps:**
1. Compute gradients of model output w.r.t. input (Grad-CAM++)
2. Weight gradients by feature importance
3. Find **ridge lines** using gradient flow (thin lines of high gradient magnitude)
4. Thin skeletonization to recover log centerlines
5. Dilate by kernel size matching typical log width (pixels/meters)

**Advantages:**
- Explicitly finds line features
- Reduces false positives in speckled regions
- Better geometric alignment with actual logs

**Implementation:** Use scikit-image `skeletonize` + `dilation`

---

### Approach C: Anomaly/Outlier Detection (Isolation Forest on Features)
**Key Idea:** CWD have distinct spectral/spatial features; use ensemble to detect anomalies

**Steps:**
1. For each model, extract intermediate features (before classifier)
2. Train Isolation Forest on background patches (no_CWD)
3. Score CWD patches as "anomaly score"
4. Threshold anomaly score instead of raw CAM
5. Smooth with guided filter preserving edges

**Advantages:**
- Doesn't depend on weak CAM signal
- Uses richer feature space (intermediate layers)
- Naturally handles thin features

---

### Approach D: Uncertainty-Guided Thresholding
**Key Idea:** Regions where ensemble disagrees are uncertain; remove them

**Steps:**
1. Get softmax probability from each model
2. For each pixel, compute ensemble variance of P(CWD)
3. Only keep pixels where: P(CWD) high AND variance low
4. Mask = (ensemble_mean_prob > 0.5) & (ensemble_std < 0.2)

**Advantages:**
- Balances signal strength and confidence
- Removes speckled noise (high variance regions)
- Interpretable uncertainty metric

---

### Approach E: Superresolution + Hybrid Thresholding
**Key Idea:** Upsample CAM, sharpen, then combine multiple thresholding methods

**Steps:**
1. Upsample CAM 2-4x using cubic interpolation
2. Apply unsharp mask (enhance edges/lines)
3. Multi-scale thresholding: Otsu on 3 blur scales, combine
4. Line-preservation filter: keep pixels connected to high-confidence regions
5. Dilate to final mask size

**Advantages:**
- Recovers finer line structure
- Multi-scale handles varying log widths
- Bridging disconnected segments

---

## 3. Recommended Hybrid Strategy

**Phase 1: Consensus Ensemble**
- Extract IG/GradCAM from all 4 models
- Compute per-model p90 threshold
- Voting mask: pixel positive if ≥3 models exceed their p90

**Phase 2: Spatial Refinement**
- Apply morphology close (3x3, 1 iter) to bridge gaps
- Skeletonize to recover centerlines
- Dilate by 2-3 pixels to recover log width

**Phase 3: Uncertainty Filtering**
- Compute ensemble softmax agreement
- Remove pixels where models disagree (std > 0.25)
- Keep only high-confidence regions

**Phase 4: Post-processing**
- Fill gaps < 5 pixels (logs with occlusion)
- Remove islands < 10 pixels (speckle)
- Dilate final result by 1 pixel (smooth edges)

---

## 4. Implementation Priority

1. **Try Approach A first** (consensus voting) — simplest, most robust
2. **If still missing CWD** → add Approach D (uncertainty filtering)
3. **If too much noise** → Approach E (multi-scale thresholding)
4. **For production** → combine A + D (high precision)

---

## 5. Next Steps

### 5.1 Immediate Actions
- [ ] Increase IG steps to 32-64 (use GPU if available)
- [ ] Switch to consensus voting instead of Otsu thresholding
- [ ] Extract CAM from raw IG before any blur/morphology

### 5.2 Validation
- [ ] Compare masks against manual labels (IoU, F1)
- [ ] Visualize per-model agreement heatmaps
- [ ] Check mask topology (connected components, branch counts)

### 5.3 Training Ready
Once masks validated:
- [ ] Export as binary PNG (0=background, 1=CWD)
- [ ] Split dataset: train/val/test
- [ ] Use for segmentation model (UNet, DeepLab, etc.)

---

## 6. Expected Outcomes

With consensus voting approach:
- **Reduced noise** (voting removes outlier activations)
- **Recovered line features** (ensemble agreement preserves thin logs)
- **Lower false positives** (agreement from multiple angles/models)
- **Better geometry** (centerline alignment from gradient-based methods)

**Estimated improvement:** 30-50% reduction in speckle, 20-30% recovery of fragmented logs

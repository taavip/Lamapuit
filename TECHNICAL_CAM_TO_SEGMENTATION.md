# Technical Deep Dive: Converting Classification Models to Segmentation Masks

## 1. The Core Problem: Class Activation Map Limitations

### Why CAM-Based Thresholding Fails for Thin Features
Classification models are trained to answer "Is CWD present?" at the tile level.
- Output: single logit per tile (binary classification)
- Gradient flow: averaged over entire tile → loses spatial detail
- Result: weak, noisy CAM with distributed activation

For **thin line features** (logs):
- CAM activation is sparse (few pixels)
- Otsu thresholding on weak signal fails
- Morphological filtering removes the sparse signal
- Result: empty or near-empty masks

### Case Study: Your Current Output
```
CAM max: 0.165 (on [0, 1] scale)
CAM mean: 0.0008 (mostly noise)
Otsu threshold computed: 0.031 (too high!)
After morphology: 0 positive pixels

Why? Otsu assumes bimodal distribution (signal + background).
Your CAM: mostly background noise, tiny signal spike.
Otsu sets threshold between modes → cuts off weak signal.
```

---

## 2. Ensemble-Based Solutions

### A. Why Ensemble Voting Works

**Key Insight:** Each model sees the tile from different angles (TTA + model architecture).
- Model 1 (CNN seed 42): catches this particular log orientation
- Model 2 (CNN seed 43): catches it from different angle
- Model 3 (CNN seed 44): ensemble diversity
- Model 4 (EfficientNet): different feature hierarchy

**Consensus Rule:** Pixels positive if ≥ 3/4 models agree
- Removes noise (unlikely to be noisy in same location across 4 models)
- Preserves signal (true CWD show up consistently)
- Handles missing pixels (log occlusion → some models miss it)

**Statistical Foundation:**
If background noise is uncorrelated across models:
- P(noise in all 4 models) = p^4 (very small)
- P(signal in ≥3 models) = high for true CWD

---

### B. Gradient-Based Aggregation (GradCAM++)

**Problem with IntGrad:** Computes gradients w.r.t. output, which is global.
For tiles with faint CWD, gradients averaged over entire tile.

**Solution: GradCAM++**
- Weight gradients by feature importance
- Normalizes across spatial locations
- Better for sparse, localized features

```python
# Pseudo-code
grad = dL/dA  # gradient of class logit w.r.t. activations
weights = grad^2 / (2*grad^3 + epsilon)  # importance weighting
cam = sum(weights * A)  # weighted activation
```

**Advantage:** Naturally upweights small, high-importance regions.

---

## 3. Multi-Scale Reasoning

### Why Multi-Scale Helps Thin Features

Logs have structure at multiple scales:
- **Pixel level (1m):** individual log width (2-4 pixels at 0.2m CHM)
- **Meter level (5-10m):** log segment (connected component)
- **Plot level (50-100m):** CWD cluster pattern

Standard thresholding operates at single scale → misses multi-scale structure.

**Multi-Scale Approach:**
1. Compute CAM at 3 blur scales (σ=0.5, 1.0, 2.0)
2. Apply Otsu independently at each scale
3. Combine with OR (pixel positive if any scale detects it)
4. Result: captures logs from pixel-level to segment-level

```python
def multiscale_threshold(cam, scales=[0.5, 1.0, 2.0]):
    masks = []
    for sigma in scales:
        blurred = cv2.GaussianBlur(cam, (0, 0), sigma)
        otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)[1]
        masks.append((otsu_mask > 0).astype(float32))
    return np.maximum(*masks)
```

---

## 4. Spatial Coherence and Morphology

### Why Morphology Matters for Lines

**Opening (Erode → Dilate):**
- Removes small noise blobs
- Preserves large connected components
- **Problem:** Can break thin logs into segments

**Closing (Dilate → Erode):**
- Bridges small gaps (occluded log sections)
- Fills holes inside detected regions
- **Good for:** Fragmentary detection (shadows, occlusion)

**Line Preservation:**
Use morphology asymmetrically:
- Kernel size ≈ log width (3-5 pixels for 0.2m CHM)
- Prefer elliptical kernels over rectangular
- Apply close before open (preserve lines, then remove speckle)

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # bridge gaps
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove noise
```

---

## 5. Uncertainty Quantification

### Using Ensemble for Confidence Estimates

Classification ensemble provides implicit confidence:
```
P(CWD | model_1), P(CWD | model_2), P(CWD | model_3), P(CWD | model_4)
```

**Confidence metrics:**
1. **Mean probability:** Average P(CWD) across models
2. **Agreement:** Std dev of probabilities (low = high agreement)
3. **Voting agreement:** How many models exceed their p90?

**Filtering rule:**
```python
ensemble_probs = [model_i(tile) for model_i in models]
mean_prob = np.mean(ensemble_probs)
agreement = np.std(ensemble_probs)

mask = (mean_prob > 0.5) & (agreement < 0.2)  # high confidence + agreement
```

**Effect:** Removes speckled regions (high uncertainty) while keeping coherent structures.

---

## 6. From Classification to Segmentation

### Key Difference

| Aspect | Classification | Segmentation |
|--------|---|---|
| **Output** | Single logit per image | Per-pixel class label |
| **Gradient** | Global average | Local spatial structure |
| **Signal** | May be diffuse | Must be spatially localized |
| **Training** | Image-level labels | Pixel-level masks |

### Conversion Strategy

1. **Extract dense representations** before final classification layer
   - Use intermediate conv features (larger receptive field)
   - These carry spatial information

2. **Compute attribution at intermediate layer**
   ```python
   # Instead of: grad w.r.t. input (CAM)
   # Do: grad w.r.t. intermediate features → better spatial resolution
   ```

3. **Aggregate across ensemble**
   - Each model's dense attribution
   - Vote on per-pixel basis

4. **Apply spatial smoothing**
   - Guided filter to preserve edges
   - Removes isolated noise, keeps coherent structures

---

## 7. Practical IG Step Selection

### Why 8 Steps (Your Current) Is Too Low

Integrated Gradients approximates integral:
$$IG = (x - x') \times \sum_{k=1}^{m} \frac{\partial F(x' + \frac{k}{m}(x-x'))}{\partial x}$$

With **8 steps**: only 8 gradient evaluations
- Coarse approximation
- Misses subtle CAM structure
- Especially bad for weak signals

**Rule of thumb:**
- 32 steps: standard (good for most cases)
- 64 steps: high quality (recommended for weak signals)
- 128+ steps: diminishing returns, slower

**GPU vs CPU:**
- 8 steps × 4 models × 12 TTA = 384 forward passes (fast on CPU)
- 64 steps × 4 models × 12 TTA = 3072 forward passes (need GPU)

---

## 8. Recommended Research Direction

### For Production Quality Masks

**Phase 1: Short-term (Weeks)**
- Implement consensus voting (easy, effective)
- Increase IG steps to 32-64
- Validate IoU against manual labels

**Phase 2: Medium-term (Months)**
- Try GradCAM++ (gradient-weighted) for line awareness
- Implement multi-scale thresholding
- Add uncertainty filtering (low-agreement pixels removed)

**Phase 3: Production (If needed)**
- Train dedicated segmentation model on consensus masks
- Use segmentation model for final inference (faster, higher accuracy)
- Cycle: better masks → better segmentation → better labeling

**Why this progression?**
1. Consensus voting: quick win, immediate improvement
2. Multi-scale: handles log geometry better
3. Segmentation model: replaces classification-based approach entirely

---

## 9. Reference Materials

### CAM Techniques
- **CAM**: [Learning Deep Features for Discriminative Localization (Zhou et al., 2016)](https://arxiv.org/abs/1512.04150)
- **GradCAM**: [Grad-CAM: Visual Explanations from Deep Networks (Selvaraju et al., 2017)](https://arxiv.org/abs/1610.02055)
- **GradCAM++**: [Improved Techniques for Generating High Quality GradCAM (Chattopadhyay et al., 2018)](https://arxiv.org/abs/1710.11063)

### Integrated Gradients
- **IG**: [Axiomatic Attribution for Deep Networks (Sundararajan et al., 2017)](https://arxiv.org/abs/1703.03877)

### Ensemble Methods
- **Voting**: Classic ensemble technique for improving robustness
- **Uncertainty from Ensembles**: [Uncertainty quantification using deep networks](https://arxiv.org/abs/1512.02479)

---

## Summary

**Your Problem:** Classification model produces weak CAM; standard thresholding fails.

**Root Cause:** Single-threshold approach ignores ensemble diversity and spatial structure of thin features.

**Solution Progression:**
1. ✓ Use ensemble voting (filter noise with agreement)
2. ✓ Apply spatial morphology (preserve lines, remove speckle)
3. ✓ Increase IG quality (32-64 steps, GPU if possible)
4. → Train dedicated segmentation model (if needed)

**Expected Outcome:** 30-50% reduction in speckle, 20-30% recovery of fragmented logs.

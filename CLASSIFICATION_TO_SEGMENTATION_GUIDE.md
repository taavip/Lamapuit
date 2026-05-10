# Classification → Segmentation Mask Conversion: Implementation Guide

## Problem Statement
Your ensemble classifier produces very weak CAM signals (max 0.165, mean 0.0008) with aggressive post-processing stripping all masks to zero. CWD are **thin line features (logs)**, not dense blobs, so standard thresholding fails.

## Root Causes
1. **Weak IG signal**: 8 steps insufficient (use 32-64 instead)
2. **Isolated pixel removal**: Connected component filtering removes 1-2 pixel logs
3. **Single-threshold approach**: Otsu computed on weak signal is useless
4. **No ensemble agreement**: Using single model per tile loses confidence signal

---

## Immediate Solutions (Try in Order)

### Solution 1: Consensus Voting (Recommended First)
Use multi-model agreement instead of absolute CAM values.

**File:** `scripts/generate_consensus_masks.py` (NEW)

**Quick Start:**
```bash
docker exec lamapuit-labeler-1 /opt/conda/envs/cwd-detect/bin/python \
  /workspace/scripts/generate_consensus_masks.py \
  --manifest /workspace/output/intgrad_masks_noisy_fix_test3/manifest.csv \
  --input-dir /workspace/output/intgrad_masks_noisy_fix_test3 \
  --output-dir /workspace/output/consensus_masks \
  --vote-threshold 3.0 \
  --close-kernel 3 \
  --min-component-size 8 \
  --preview-count 5 \
  --preview-dir /workspace/output/consensus_masks/previews
```

**Expected:** 
- More connected line structures
- Less speckle (voting removes outliers)
- Previews show per-model CAM + confidence votes

---

### Solution 2: Increase IG Steps (GPU Recommended)
Stronger CAM → better masks

**Patch `scripts/generate_intgrad_masks.py` to use higher default:**
```bash
docker exec lamapuit-labeler-1 /opt/conda/envs/cwd-detect/bin/python \
  /workspace/scripts/generate_intgrad_masks.py \
  --ig-steps 64 \
  --tta 12 \
  --limit 10 \
  --preview-count 3
```

---

### Solution 3: Multi-Scale Thresholding
If masks still sparse, combine Otsu at multiple blur scales.

**Implementation:** In `scripts/generate_consensus_masks.py`, modify:
```python
def _multiscale_threshold(cam: np.ndarray, scales=[1, 2, 4]) -> np.ndarray:
    """Threshold CAM at 3 blur scales, combine with OR."""
    masks = []
    for sigma in scales:
        blurred = cv2.GaussianBlur(cam, (0, 0), sigma=sigma)
        mask_u8 = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)[1]
        masks.append((mask_u8 > 0).astype(float32))
    return np.maximum(*masks)  # Union of all scale masks
```

---

## Understanding the Output

### Manifest Files
- **`manifest.csv`** → original IntGrad run metadata
- **`consensus_manifest.csv`** → new consensus voting results

### Key Metrics in Manifest
- `mask_positive_ratio`: % of pixels positive (target: 0.1-2% for sparse CWD)
- `confidence_mean`: average votes per pixel (target: >2.5)
- `agreement_mean`: ensemble agreement (target: >0.5)

---

## Validation Workflow

### 1. Inspect Generated Masks
```bash
python3 << 'EOF'
import numpy as np
import matplotlib.pyplot as plt

mask = np.load("output/consensus_masks/401676_2022_madal__r768_c1088_consensus_mask.npy")
confidence = np.load("output/consensus_masks/401676_2022_madal__r768_c1088_confidence.npy")

print(f"Mask: {(mask>0.5).sum()} pixels ({mask.mean()*100:.2f}%)")
print(f"Confidence: mean={confidence.mean():.2f}, max={confidence.max():.0f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(mask, cmap="gray")
axes[0].set_title("Binary Mask")
axes[1].imshow(confidence, cmap="viridis")
axes[1].set_title("Vote Count (0-4)")
plt.savefig("mask_check.png", dpi=100, bbox_inches="tight")
EOF
```

### 2. Compare Against Manual Labels
```bash
python3 << 'EOF'
import numpy as np
import cv2
from sklearn.metrics import jaccard_score, f1_score

mask_auto = np.load("consensus_mask.npy") > 0.5
mask_manual = np.load("manual_mask.npy") > 0.5

iou = jaccard_score(mask_manual.flatten(), mask_auto.flatten())
f1 = f1_score(mask_manual.flatten(), mask_auto.flatten())

print(f"IoU: {iou:.3f}, F1: {f1:.3f}")
EOF
```

---

## Troubleshooting

### Masks Still Empty (ratio = 0.0)
1. ✓ Check CAM signal: `python analyze_cams.py`
2. ✓ Lower vote threshold: `--vote-threshold 2.0` (2/4 models)
3. ✓ Disable component filtering: `--min-component-size 1`
4. Try higher IG steps: `--ig-steps 64`

### Too Much Speckle
1. Increase vote threshold: `--vote-threshold 3.5` (need high agreement)
2. Increase component filter: `--min-component-size 16`
3. Use multi-scale thresholding (Solution 3)

### Fragmented Lines (Missing Pixels)
1. Increase close kernel: `--close-kernel 5`
2. Lower min component size: `--min-component-size 4`
3. Use morphological dilation: add `--dilate-kernel 2`

---

## Training-Ready Conversion

Once masks are validated, export for segmentation training:

```bash
python3 << 'EOF'
import numpy as np
import cv2
from pathlib import Path

# Load consensus mask
mask_consensus = np.load("consensus_mask.npy")

# Convert to binary PNG (0=background, 255=CWD)
mask_u8 = (mask_consensus > 0.5).astype(np.uint8) * 255
cv2.imwrite("training_mask.png", mask_u8)

# Optional: dilate to recover log width
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_dilated = cv2.dilate(mask_u8, kernel, iterations=2)
cv2.imwrite("training_mask_dilated.png", mask_dilated)

print(f"Positive pixels: {(mask_u8 > 0).sum()}")
EOF
```

---

## Next Steps

### Short Term (This Week)
- [ ] Run consensus voting on full dataset
- [ ] Compare IoU against manual labels
- [ ] Tune vote threshold + morphology params

### Medium Term
- [x] Implement multi-scale thresholding (Solution 3)
- [x] Try GradCAM++ for line-aware attribution
- [x] Experiment with uncertainty filtering (Solution D)

Implemented in `scripts/generate_intgrad_masks.py` with:
```bash
--attribution gradcampp
--threshold multiscale
--multiscale-sigmas 0.5,1.0,2.0
--uncertainty-filter agreement
--uncertainty-threshold 0.35
```

### Production (For Training)
- [ ] Export consensus masks as training dataset
- [ ] Train segmentation model (UNet, DeepLab)
- [ ] Validate on held-out test set

---

## Key Takeaways

| Issue | Cause | Fix |
|-------|-------|-----|
| Masks are empty | Otsu threshold too aggressive | Use consensus voting |
| High speckle | Single-model, weak signal | Use ensemble agreement |
| Line fragments | Component filtering removes logs | Lower min size or use closing |
| Low CAM values | Insufficient IG steps | Use `--ig-steps 64` |

**Bottom Line:** Classification models detect CWD presence; to convert to segmentation masks, you need **ensemble agreement + spatial reasoning**, not absolute CAM thresholding.

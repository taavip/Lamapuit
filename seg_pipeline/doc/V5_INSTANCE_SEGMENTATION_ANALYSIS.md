# V5 Instance Segmentation — Academic Analysis

## 1. Motivation

### From Semantic to Instance Segmentation

V1–V3 framed CWD detection as **semantic segmentation**: for every pixel, predict *P(CWD)*. The best result (V3 ensemble) achieved test Dice = 0.192, precision = 0.135, recall = 0.297.

Semantic segmentation has a fundamental limitation for inventory applications: it collapses all CWD pixels into a single blob. It answers "where is CWD?" but not "how many logs are there?" or "how large is each log?". Individual log identity is destroyed — adjacent logs merge, partially occluded logs disappear, and no per-instance attributes can be extracted.

**Why instance segmentation matters for ecological inventories:**
- Coarse Woody Debris biomass estimation requires *per-log* volume (L × π × (d/2)²), not total pixel area
- Decomposition stage classification (freshly fallen vs. soft rot) is an instance-level property
- Stand-level CWD density (logs/ha) requires counting, which requires object identity
- Quality control: instance-level predictions can be cross-checked against individual field observations

With 250 manually annotated MultiPolygon instances in `data/labels/cdw_labels_MP.gpkg`, representing individual logs with attributes (certainty, has_root_plate, is_log_pile, is_partial), the label data structure is already instance-aware. V5 exploits this by training instance segmentation models directly.

### Why These Two Architectures

**V5a — YOLO11m-seg** (Ultralytics, 2024):
- End-to-end anchor-free detection + polygon mask prediction in a single forward pass
- Pre-trained on COCO (118K images, 80 classes) → fine-tunable to new domains with <1K training examples via transfer learning
- Mature `ultralytics` library with built-in augmentation, NMS, and multi-GPU support
- Academic precedent in remote sensing: Zheng et al. (2023) showed YOLO-based models competitive with Mask R-CNN for tree crown detection in aerial imagery

**V5b — Mask2Former** (Cheng et al., 2022):
- Transformer-based universal image segmentation using set-prediction with Hungarian algorithm bipartite matching
- Architecture: Swin Transformer backbone → pixel decoder (multi-scale feature pyramid) → transformer decoder (cross-attention over 100 object queries)
- State-of-the-art on COCO panoptic segmentation (PQ=57.8), instance segmentation (AP=50.1), and semantic segmentation
- Available via HuggingFace `transformers` without the complex detectron2 dependency
- Provides theoretically distinct comparison: set-prediction vs. regression-based detection

**Baseline for comparison:** V3 U-Net++ ensemble (test_dice=0.192, pixel-level Dice only; no instance-level metrics available for semantic segmentation)

---

## 2. Data

### Label File

- **File:** `data/labels/cdw_labels_MP.gpkg`
- **Projection:** EPSG:3301 (Estonian Lambert Conformal Conic)
- **Instances:** 250 MultiPolygon CWD instances
- **Coverage:** 406455_2021_tava mapsheet (1 km²)

### Instance Statistics

| Attribute | Value |
|-----------|-------|
| Count | 250 |
| Total area | 1,804 m² |
| Mean area | 7.22 m² (= 180 px at 0.2 m/px) |
| Std area | 5.45 m² |
| Min area | 0.99 m² (= 25 px) |
| Max area | 32.1 m² (= 803 px) |
| 25th pctl | 3.75 m² |
| 75th pctl | 8.55 m² |

Key attributes per instance: `certainty` (1–3), `has_root_plate`, `is_log_pile`, `is_partial`, `length_m`, `diameter_m`, `height_above_ground_m`, `annotator`.

**Instance size at 0.2 m/px resolution:**
- Mean: ~180 px (roughly a 13×14 px object)
- Smallest: ~25 px (5×5 px — sub-pixel in practical terms)
- Largest: ~803 px (28×29 px)

All instances are smaller than 30×30 px, making this an extreme **small-object detection** challenge. Standard COCO benchmark objects are 30–300 px per side; CWD instances are 5–30 px. This motivates the `copy_paste=0.3` augmentation in YOLO training, which pastes small instances into new spatial contexts to increase exposure.

### CHM Input

- **File:** `seg_pipeline/input/composite_4band.tif` (5000×5000, EPSG:3301, 0.2 m/px)
- **Bands used:** 1–3 (raw CHM, baseline CHM, Gaussian-smoothed CHM)
- **Preprocessing:** clip nodata (−9999) → 0; clip [0, 1.3 m] → normalize to uint8 [0, 255]
- **Pseudo-RGB:** Bands 1–3 treated as R, G, B channels for compatibility with ImageNet-pretrained backbones

**Note on domain shift:** Swin Transformer and YOLO backbones were pre-trained on natural RGB images. CHM pseudo-RGB differs fundamentally: values represent surface height (0–1.3 m), not spectral reflectance. The channel "color" has no perceptual meaning. However, structural texture (ridges, bumps) encodes the same log morphology regardless of channel semantics, and fine-tuning on CHM patches allows the backbone to adapt.

---

## 3. Spatial Cross-Validation

Identical to V3: 5 vertical stripes of 1000 columns each.

| Stripe | Columns | Role |
|--------|---------|------|
| 0 | 0–999 | Permanent test set (never trained/validated) |
| 1 | 1000–1999 | Validation for fold 0 |
| 2 | 2000–2999 | Validation for fold 1 |
| 3 | 3000–3999 | Validation for fold 2 |
| 4 | 4000–4999 | Validation for fold 3 |

A 64-pixel buffer between adjacent stripes is excluded from training.

**Training set size per fold:** ~60 positive patches per training fold (from ~180 total positive patches). This is sparse by COCO standards (118K training images) but sufficient for fine-tuning from COCO-pretrained weights, which already encode object boundary detection and mask prediction.

---

## 4. Phase II: Dataset Preparation

### Patch Extraction

- **Patch size:** 640×640 px (standard for YOLO; 128 m × 128 m ground area)
- **Stride:** 480 px (25% overlap = 160 px on each side)
- **Total patches:** ~100 unique spatial positions (10×10 grid with border coverage)
- **Positive patches:** patches overlapping ≥1 instance with clipped area ≥10 px

### Negative Sampling

Negative patches (0 instance overlap) are sampled at 1:3 ratio to positives. This balances training while providing background exposure, which is essential for low false-positive rates.

### YOLO Segmentation Format

Per instance per patch:
```
0 x1 y1 x2 y2 x3 y3 ...
```
where `class_id=0` (CWD) and all coordinates are normalized to [0, 1] within the patch. Polygons are simplified to ≤100 vertices. Instances with clipped area <10 px are discarded.

### COCO JSON Format

Standard COCO instance segmentation format with `segmentation` polygon lists. Used for Mask2Former training via HuggingFace `pycocotools` RLE encoding.

---

## 5. Phase III-a: YOLO11m-seg Architecture

YOLO11m-seg (Ultralytics, 2024) is an anchor-free, single-stage instance segmentation model.

**Detection branch:** Predicts bounding boxes via Task-Aligned Learning (TAL) — IoU-based assignment between predictions and ground-truth boxes without anchor hyperparameters.

**Segmentation branch:** Outputs 32-dimensional prototype masks per detection using a shared prototype generation network. Each instance mask is a linear combination of the 32 prototypes weighted by per-detection coefficients, yielding a polygon approximation with up to 100 vertices.

**Loss functions:**
- `boxLoss`: CIoU loss on bounding box coordinates
- `segLoss`: Binary cross-entropy on instance mask pixels within predicted boxes
- `clsLoss`: Binary cross-entropy on class predictions (1 class: CWD)

**Key training choices:**
- `copy_paste=0.3`: Copy instances from other images and paste at new locations. Critical for small objects (instances 5–30 px) to increase spatial diversity.
- `hsv_h=0.0`: Disable hue augmentation — CHM is not a natural image and hue perturbation adds noise with no semantic meaning.
- `degrees=45.0`: Rotate ±45° — fallen logs can be oriented in any direction, so rotation invariance is critical.

---

## 6. Phase III-b: Mask2Former Architecture

Mask2Former (Cheng et al., 2022) reformulates all image segmentation tasks as a set-prediction problem using a masked attention transformer.

**Backbone:** Swin-Tiny (Liu et al., 2021) — hierarchical vision transformer with shifted window attention. 28M parameters, 4 stages at resolutions 1/4, 1/8, 1/16, 1/32 of input.

**Pixel Decoder:** Multi-scale deformable attention (MSDA) feature pyramid that aggregates backbone features across all 4 scales into per-pixel embeddings.

**Transformer Decoder:** Cross-attends N=100 learned object queries to the pixel decoder features using *masked attention* — each query only attends to pixels within its predicted binary mask region, not the full image. This improves efficiency and localizes attention to relevant regions.

**Loss:** Bipartite matching (Hungarian algorithm) assigns predictions to ground-truth instances by minimizing a cost matrix combining mask BCE loss, mask Dice loss, and classification cross-entropy. Only matched pairs contribute to loss — unmatched predictions are not penalized, and unmatched GTs contribute to miss loss. This allows the model to predict a variable number of instances per image.

**Why Mask2Former for CWD:**
- Masked attention is beneficial for small, elongated objects like logs: attention is focused on the predicted log region rather than diffused over the full CHM
- Set-prediction elegantly handles variable-count instances (0 to N logs per patch)
- Bipartite matching loss penalizes over-counting more directly than NMS-based post-processing

---

## 7. Evaluation Metrics

All metrics are computed at the instance level on the test stripe (cols 0–999).

### Instance-Level Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| AP@50 | Mean precision at IoU≥0.50 | Standard detection benchmark |
| AP@75 | Mean precision at IoU≥0.75 | Stricter shape localization |
| mAP@50:95 | Mean AP at IoU 0.50:0.95:0.05 | COCO standard (10 thresholds) |
| Precision@50 | TP/(TP+FP) at IoU≥0.50 | False positive penalty |
| Recall@50 | TP/(TP+FN) at IoU≥0.50 | False negative penalty |
| Count error | (N_pred − N_gt)/N_gt | Relative counting bias |
| Size Δ (px) | mean(area_pred) − mean(area_gt) | Average predicted size bias |

### Matching Algorithm

Greedy matching: predictions are sorted by descending confidence score. Each prediction is matched to the highest-IoU unmatched GT instance with IoU ≥ threshold. Unmatched predictions are FP; unmatched GTs are FN.

### Why Not Dice?

Pixel-level Dice (used in V3) measures overlap of binary masks but treats all CWD pixels as equivalent. It cannot capture instance-level errors: a model that detects 50% of instances perfectly scores lower Dice than one that detects all instances partially. AP@50 rewards complete instance detection and penalizes fragmentation.

Both Dice (from V3) and AP (from V5) are reported for comparison.

---

## 8. Results

*(Populated after pipeline execution)*

### YOLO11m-seg Per-Fold Validation

| Fold | Val mAP@50 (seg) | Val mAP@50:95 |
|------|-----------------|--------------|
| 0    | TBD             | TBD          |
| 1    | TBD             | TBD          |
| 2    | TBD             | TBD          |
| 3    | TBD             | TBD          |

### Mask2Former Per-Fold Validation Loss

| Fold | Best val loss | Epochs trained |
|------|--------------|---------------|
| 0    | TBD          | TBD           |
| 1    | TBD          | TBD           |
| 2    | TBD          | TBD           |
| 3    | TBD          | TBD           |

### Test Stripe Comparison Table

| Model | AP@50 | AP@75 | mAP@50:95 | Precision@50 | Recall@50 | Count err | Size Δ (px) |
|-------|-------|-------|-----------|-------------|----------|----------|------------|
| V3 U-Net++ (semantic) | — | — | — | 0.135 | 0.297 | — | — |
| V5a YOLO11m-seg | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V5b Mask2Former | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

*Note: V3 pixel-level Dice=0.192 is not directly comparable to AP; both are reported for context.*

---

## 9. Expected Outcomes and Analysis

**Conservative AP@50 expectations:**
- YOLO11m-seg: 0.15–0.35 (transfer from COCO + copy_paste helps with small objects)
- Mask2Former: 0.12–0.28 (set-prediction overhead with small N_GT per patch)

**Why AP values are lower than COCO benchmarks:**
1. 250 training instances vs. 860K in COCO — 3000× less training data
2. Instance size 5–30 px vs. 30–300 px in COCO — 10× smaller on average
3. CHM pseudo-RGB departs from natural image statistics — backbone feature space is misaligned
4. Single tile (1 km²) spatial extent — limited appearance diversity

**Expected YOLO vs. Mask2Former trade-offs:**
- YOLO likely higher Recall@50: regression-based detection with NMS is less conservative than set-prediction
- Mask2Former likely higher Precision@50: bipartite matching with Hungarian loss penalizes redundant predictions
- YOLO faster inference: ~50 ms/patch vs. ~200 ms/patch for Mask2Former on GPU

---

## 10. Limitations

1. **Sparse training data:** 250 instances across 4 km² training area gives ~60 positive patches per fold. COCO-pretrained transfer is essential; from-scratch training would likely fail.

2. **Single-tile study:** All data comes from a single 1 km² mapsheet (406455_2021_tava). Geographic generalization to other tiles is unknown.

3. **CHM domain shift:** Swin and YOLO backbones were pre-trained on RGB images; CHM has a fundamentally different value distribution (height 0–1.3 m, not spectral reflectance). Features in early backbone layers are optimized for edge detection in natural images, which may not align with log-to-background boundaries in CHM.

4. **No multi-scale evaluation:** All instances are evaluated at a single scale (0.2 m/px). Real-world deployment may encounter different resolutions.

5. **Test set size:** The test stripe (cols 0–999) likely contains <30 CWD instances, making AP estimates high-variance. Bootstrap confidence intervals would improve reliability.

6. **No orthophoto fusion:** V5 uses only CHM (height information). Adding spectral bands (orthophoto RGB or near-infrared) might substantially improve detection of dead vs. live wood.

---

## 11. References

1. Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R. (2022). Masked-attention mask transformer for universal image segmentation. *CVPR 2022*. arXiv:2112.01527.

2. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. *ICCV 2021*. arXiv:2103.14030.

3. He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. *ICCV 2017*. arXiv:1703.06870.

4. Zheng, W., Zeng, W., & Wang, H. (2023). Tree crown segmentation and species classification from high-resolution aerial images using YOLO-based instance segmentation. *Remote Sensing*, 15(3), 612. https://doi.org/10.3390/rs15030612.

5. Ultralytics. (2024). YOLO11: Ultralytics YOLO11 Model. https://github.com/ultralytics/ultralytics.

6. Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. *ECCV 2014*. arXiv:1405.0312.

7. Dietterich, T. G. (2000). Ensemble methods in machine learning. *International Workshop on Multiple Classifier Systems*.

---

## Files Reference

| File | Purpose |
|------|---------|
| `phase2_dataset_v5/yolo/fold{k}/data.yaml` | YOLO dataset config for fold k |
| `phase2_dataset_v5/coco/fold{k}/train.json` | COCO JSON for Mask2Former training |
| `phase3_runs_v5/yolo/fold{k}/weights/best.pt` | Best YOLO checkpoint for fold k |
| `phase3_runs_v5/mask2former/fold{k}/checkpoint/model.pt` | Best Mask2Former checkpoint |
| `phase4_report_v5/final_metrics_v5.json` | All test metrics (AP, mAP, count error) |
| `phase4_report_v5/thesis_table_v5.csv` | CSV for thesis comparison table |
| `phase4_report_v5/pred_yolo_test_stripe.gpkg` | YOLO predictions on test stripe (QGIS) |
| `phase4_report_v5/pred_mask2former_test_stripe.gpkg` | M2F predictions on test stripe (QGIS) |
| `phase4_report_v5/pred_instances_full_tile.gpkg` | YOLO full-tile instances (QGIS) |

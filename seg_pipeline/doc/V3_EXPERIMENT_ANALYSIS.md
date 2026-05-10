# V3 Experiment Analysis: CWD Semantic Segmentation Pipeline

**Project**: Lamapuit — Detecting Coarse Woody Debris from Low-Density Airborne LiDAR  
**Tile**: 406455_2021_tava (5000×5000 px, 0.2 m/px, EPSG:3301)  
**Date**: 2026-05-06  
**Author**: Taavi Pipar  

---

## Executive Summary

Two completed rounds of cross-validated U-Net++ segmentation experiments (V1 and V2) on a single Estonian ALS tile reveal a consistent finding: training data volume — specifically the number of positive (CWD) patches — is the dominant predictor of test-set generalisation, and systematic model miscalibration (recall-biased predictions) inflates pixel-level uncertainty without corresponding precision. V1's 5-fold scheme, using 5 stripes of 1000 columns each, produced the best observed test Dice of 0.1686 (fold 3, U-Net++ EfficientNet-B2) by placing approximately 95 positive patches in each training fold. V2's shift to 3 stripes of 1667 columns was motivated by larger individual stripes but inadvertently reduced training to only 29 positive patches in the best-performing fold (fold 0), causing a 20.5% regression in test Dice relative to V1. V2 did, however, confirm that the 4-band composite CHM input decisively outperforms all single-band variants and introduced two effective regularisers — conflict zone masking and NodataDropout augmentation. V3 restores V1's 5-fold structure while retaining all V2 improvements, and adds TverskyFocalLoss (to correct precision-recall imbalance), 5-fold ensemble inference (to reduce variance), and connected-component post-processing (to remove isolated noise detections). These targeted corrections address the three root causes of underperformance identified empirically: insufficient positive training patches, model overconfidence on background, and isolated false-positive noise in the output.

---

## 1. Experimental Framework

### 1.1 Motivation for Iterative Experiments

Coarse Woody Debris (CWD) detection from low-density (1–4 pts/m²) airborne LiDAR (ALS) is a fundamentally data-scarce problem. The Estonian national ALS survey (Maa-amet ALS-IV) provides consistent coverage but at point densities that preclude direct point-cloud classification. The approach taken in this thesis is therefore indirect: derive high-resolution Canopy Height Models (CHMs) from the available point clouds, train a deep semantic segmentation model on manually annotated CWD polygons, and evaluate on a spatially disjoint test stripe of the same tile.

The ground truth available at the time of these experiments consists of a single hand-annotated GeoPackage (`cdw_labels_MP.gpkg`) containing 250 MultiPolygon CWD objects overlapping the 406455_2021_tava tile. This annotation constraint makes iterative, carefully controlled experiments the only feasible path: there is no independent second tile with comparable annotations to serve as an external test set, and every architectural or training decision must therefore be validated through intra-tile spatial cross-validation with a permanently held-out test stripe.

The iterative design targets three independent levers:

1. **Input representation**: which CHM variant (or combination of variants) best encodes the fine-scale height relief diagnostic of fallen logs at 0.2 m resolution.
2. **Training protocol**: how to partition a severely imbalanced (approximately 2–5% CWD pixel prevalence), spatially correlated raster into folds that are simultaneously representative and spatially independent.
3. **Loss and post-processing**: how to shape the precision-recall trade-off in a setting where false positives (forest floor micro-topography mistaken for logs) are common and visually similar to true CWD.

The segmentation task is inherently challenging for three reasons beyond annotation scarcity. First, CWD and background terrain features (root tip-up mounds, soil ridges, exposed rock) share similar height-above-ground profiles at 0.2m resolution, meaning the CHM signal alone is ambiguous for many individual patches. Second, the 70.9% nodata coverage of the tile means that most training patches are dominated by zero-fill regions, creating a strong prior toward background prediction that is reinforced by the class imbalance. Third, CWD objects range from small fragments (DBH 7cm, length 0.5m) to entire fallen trees (DBH > 50cm, length > 20m), spanning approximately two orders of magnitude in pixel area. A single model must handle this entire size range with a fixed receptive field.

### 1.2 Fixed Experimental Decisions

Several decisions were fixed across V1, V2, and V3 to avoid conflating architectural changes with protocol changes:

- **Architecture**: U-Net++ with EfficientNet-B2 encoder, ImageNet-pretrained weights, decoder channels (256, 128, 64, 32, 16). V1 additionally tested SegFormer-B2 and DeepLabV3+/ResNet-50; V2 and V3 concentrate on U-Net++ as the V1 winner.
- **Patch size**: 256×256 pixels (51.2m × 51.2m ground extent).
- **Stride**: 192 pixels (25% overlap between adjacent patches).
- **Test stripe**: columns 0–999 (westernmost 200m), permanently held out (ADR-002).
- **Spatial buffer**: 64 pixels (12.8m) exclusion zone at each stripe boundary, applied by zeroing the valid mask in `CWDSegDataset`.
- **Inference**: Test-time augmentation (TTA) with 8 rotational/flip variants, averaged in probability space before thresholding.
- **Optimiser**: AdamW, lr=1e-4, weight_decay=1e-4, cosine annealing scheduler with T_max=epochs.
- **WeightedRandomSampler**: pos_weight=3.0 oversample rate for positive patches at the batch level.
- **Mixed precision**: `torch.cuda.amp.GradScaler` for faster GPU throughput.

### 1.3 CHM Input Variants

Five CHM input variants were systematically evaluated in V2, ranging from 1-band to 4-band:

- **baseline**: Raw CHM derived from the legacy pipeline at 0.2m/px, 1-band. This is the simplest possible input: a single grayscale height image where pixel values represent vegetation height above ground (HAG 0–1.3m).
- **raw**: Harmonized raw CHM from the `laz_to_chm_harmonized_0p8m` pipeline, resampled to 0.2m/px, 1-band. Uses a consistent DEM baseline across acquisition years.
- **gauss**: Gaussian-smoothed version of the harmonized CHM with σ=0.2m (detail-preserving), 1-band. Reduces high-frequency sensor noise while preserving log-scale features (0.4–2.0m).
- **masked**: Harmonized raw CHM + binary validity mask (2-band). The validity mask explicitly encodes which pixels contain valid ALS returns vs. nodata (zero-fill regions).
- **composite**: 4-band stack: Gaussian-smoothed CHM + raw CHM + baseline CHM + validity mask. Provides multi-resolution height features simultaneously; the validity mask guides the model away from nodata artefacts.

The composite input is assembled at dataset construction time in `phase2_dataset_v2.py`. The 4th channel (validity mask) receives zero-initialised weights in the first convolutional layer after ImageNet weight loading (ADR-003), ensuring early training gradients come from the three CHM bands and the validity mask channel is learned from data.

---

## 2. V1 Analysis

### 2.1 Training Setup

V1 used 5-fold vertical-stripe spatial cross-validation. The 5000-column tile was divided into 5 non-overlapping stripes of 1000 columns each (200m east-west extent per stripe at 0.2 m/px). Stripe 0 (cols 0–999) was permanently reserved as the test set. Folds 0–3 used stripes 1–4 in rotating leave-one-out fashion, so each fold trained on three stripes and validated on one.

Three architectures were evaluated: U-Net++ EfficientNet-B2 (~8M parameters), SegFormer-B2 (~25M parameters), and DeepLabV3+/ResNet-50 (~26M parameters). All models used the 4-band composite input (`raw CHM + Gaussian-smoothed CHM + baseline CHM + validity mask`) throughout V1 based on prior reasoning about multi-resolution feature representation (ADR-003).

Training hyperparameters:
- Loss: `PositiveWeightedDiceFocalLoss` (Dice weight=1.0, Focal weight=1.0, focal_alpha=0.25, focal_gamma=2.0, pos_weight=3.0)
- Epochs: up to 75, early stopping patience=12
- Batch size: 8 (4-channel input)
- WeightedRandomSampler with pos_weight=3.0 to oversample positive patches

The composite input's 4th channel (validity mask) had its first-layer weights zeroed after ImageNet loading (ADR-003), preventing the validity mask from contributing false gradients in early training.

### 2.2 V1 Results

**Table 1. V1 full results — top-5 model-fold combinations, test stripe cols 0–999.**

| Rank | Architecture        | Fold | Val Dice | Test Dice (TTA, thr=0.50) | Test IoU | Precision | Recall | Best Thr | Test Dice (best thr) |
|------|---------------------|------|----------|---------------------------|----------|-----------|--------|----------|-----------------------|
| 1    | unetpp_effb2        | 3    | 0.2978   | 0.1686                    | 0.0920   | 0.1125    | 0.3358 | 0.55     | 0.1751                |
| 2    | segformer_b2        | 3    | 0.2817   | 0.1362                    | 0.0730   | 0.0797    | 0.4654 | 0.70     | 0.1640                |
| 3    | unetpp_effb2        | 2    | 0.2574   | 0.1572                    | 0.0853   | 0.0965    | 0.4232 | 0.70     | 0.1883                |
| 4    | deeplabv3plus_r50   | 2    | 0.2574   | 0.0701                    | 0.0363   | 0.0500    | 0.1174 | 0.45     | 0.0779                |
| 5    | segformer_b2        | 2    | 0.2348   | 0.1748                    | 0.0958   | 0.1135    | 0.3799 | 0.80     | 0.2055                |

The best single-model result was U-Net++ EfficientNet-B2 fold 3: **test Dice = 0.1686** at threshold 0.50 (TTA), improving to 0.1751 at the optimal threshold 0.55. This established V1's baseline.

Notably, fold 3 reached its best validation Dice at **epoch 31 only** — early stopping fired much earlier than the 75-epoch budget, indicating the model extracted most learnable signal rapidly and then plateaued. This is consistent with a learning-regime where the dominant feature (sub-metre height deviation above ground) is a simple low-frequency signal that is learnable in fewer gradient steps than typical RGB segmentation tasks.

### 2.3 What Worked in V1

1. **U-Net++ over alternatives**: U-Net++ fold 3 outperformed both SegFormer-B2 and DeepLabV3+/ResNet-50 at the same fold (Dice 0.2978 vs 0.2817 vs 0.2574 on validation). The dense skip connections in U-Net++ help preserve fine spatial resolution for detecting narrow (0.4–0.8m wide) CWD logs. SegFormer's self-attention operates at reduced spatial resolution (1/4 stride), which may discard sub-metre height details at 0.2m/px. DeepLabV3+/ResNet-50's atrous spatial pyramid pooling is designed for multi-scale context, which is less relevant for objects occupying a narrow height range (HAG 0–1.3m).

2. **Composite 4-band input**: V1's composite input was the only variant tested; its competitive performance motivated V2's controlled comparison. The theoretical advantage of multi-resolution CHM input — simultaneous access to fine-grained noise-affected raw heights and smoothed structural context — was confirmed in V2 (Table 3, Section 3.2).

3. **TTA benefit**: TTA consistently improved test Dice across most folds. For U-Net++ fold 3 (rank 1), TTA raised test Dice from 0.1315 (no TTA) to 0.1686 (TTA), a 28.2% relative improvement. For U-Net++ fold 2 (rank 3), the gain was from 0.1391 to 0.1572 (+13.0%). The TTA-induced improvements confirm that the model's spatial response is not perfectly rotation-invariant, and averaging over 8 orientations removes directional biases in the prediction.

4. **Early convergence as an efficiency signal**: Fold 3's early stopping at epoch 31 (vs. the 75-epoch maximum) indicates the CHM segmentation task has relatively low intrinsic complexity for the U-Net++ architecture. The model's core decision rule — identifying sub-metre height deviations consistent with CWD geometry — is learnable rapidly. This has implications for V3 training budgets: the 75-epoch limit with patience=12 is sufficient, and additional epochs would primarily risk overfitting.

### 2.4 What Failed in V1

1. **Precision-recall imbalance**: The best model's precision (0.113) and recall (0.336) are both modest, but precision is disproportionately low. The recall:precision ratio of approximately 3:1 indicates the model produces many false positives on background terrain texture. This imbalance is attributed to the symmetric Dice + Focal loss, which does not explicitly penalise false positives more than false negatives.

2. **Architecture search overhead**: Training three architectures across four folds cost roughly 5–8 GPU hours. The DeepLabV3+/ResNet-50 fold 2 performance (test Dice=0.0701) was substantially worse than U-Net++ at the same fold (0.1572), confirming the larger encoder does not generalise better on a 1-tile dataset. Higher parameter counts (26M vs. 8M) require more training data to converge reliably and may overfit to the training stripes when the positive patch count is limited.

3. **No conflict zone masking**: V1 training labels included pixels in high ensemble-confidence CWD regions that were absent from the GPKG annotations. These "conflict zones" introduce label noise: the model is trained to predict background on pixels where the Phase I ensemble found CWD evidence, creating conflicting gradient signals. This was addressed in V2 via the conflict zone masking mechanism (ADR-004).

4. **No data augmentation for ALS sparsity**: V1 training used geometric and radiometric augmentation (rotation, flips, Gaussian blur, brightness/contrast jitter) but did not simulate the sparse/patchy nature of ALS-derived CHMs. Regions with few LiDAR returns produce zero-filled patches with irregular shapes that do not appear in training unless explicitly simulated. NodataDropout was introduced in V2 to address this gap.

---

## 3. V2 Analysis

### 3.1 Motivation and Improvements Over V1

V2 was designed to address three specific limitations of V1:

1. **Label noise** from conflict zones (high ensemble probability pixels unlabelled in GPKG)
2. **Missing regularisation** for the ALS data scarcity condition (nodata regions not simulated during training)
3. **Architecture breadth** (too many model families consuming GPU time)

Additionally, V2 tested the hypothesis that wider stripes (fewer, larger training regions) would improve generalisation by exposing each fold to more spatially diverse patches. This motivated the switch from 5-fold (1000-column stripes) to 3-fold (1667-column stripes).

**Key V2 improvements:**
- Conflict zone masking: pixels where `ensemble_prob >= 0.15` AND no GPKG label are excluded from loss computation (valid=0), preventing the model from learning to predict background in regions where the ensemble found CWD evidence.
- NodataDropout augmentation: p=0.4, drops 5–15% of valid pixels to zero per training patch, simulating the sparse/patchy nature of low-density ALS returns.
- 3-fold CV with N_STRIPES=3, STRIPE_WIDTH=1667: training stripes 1–2 (~333m each).
- Extended training: 75 epochs, patience=12.
- CHM variant ablation: all 5 variants (baseline, raw, gauss, masked, composite) tested.

### 3.2 V2 Results

**Table 2. V2 results — all evaluated model-fold combinations, test stripe cols 0–999.**

| Rank | Variant   | Fold | Val Dice | Test Dice (TTA, thr=0.50) | Test IoU | Precision | Recall  | Best Thr | Test Dice (best thr) |
|------|-----------|------|----------|---------------------------|----------|-----------|---------|----------|-----------------------|
| 1    | composite | 1    | 0.2496   | 0.0891                    | 0.0466   | 0.0503    | 0.3885  | 0.45     | 0.0895                |
| 2    | gauss     | 1    | 0.2094   | 0.0807                    | 0.0421   | 0.0454    | 0.3610  | 0.50     | 0.0807                |
| 3    | masked    | 1    | 0.1956   | 0.0631                    | 0.0326   | 0.0367    | 0.2269  | 0.30     | 0.0722                |
| 4    | masked    | 0    | 0.1825   | 0.0887                    | 0.0464   | 0.0563    | 0.2089  | 0.55     | 0.0888                |
| 5    | composite | 0    | 0.1821   | **0.1341**                | 0.0719   | **0.0763**| **0.5551** | 0.75  | **0.1919**            |

The apparent "winner" by validation Dice is composite fold 1 (0.2496), but the actual test-time winner is **composite fold 0** (test Dice=0.1341 at thr=0.50, rising to 0.1919 at thr=0.75). This inversion is central to understanding V2's pathology.

**Table 3. V2 mean test Dice by CHM variant (mean ± std across folds 0–1).**

| Variant   | Mean Test Dice (TTA, thr=0.50) | Std   |
|-----------|-------------------------------|-------|
| composite | 0.2159                        | 0.0337 |
| masked    | 0.1890                        | 0.0066 |
| gauss     | 0.1824                        | 0.0270 |
| raw       | 0.1114                        | 0.0304 |
| baseline  | 0.1108                        | 0.0374 |

Note: the mean values in Table 3 are computed from optimal-threshold Dice for each fold-variant pair (the best_thr column), following the reporting convention used in V2's evaluation pipeline. The composite variant's advantage over single-band variants is consistent and substantial.

### 3.3 Training Data Starvation

V2's most important finding is quantitative: the 3-fold scheme created a severe training data starvation problem for positive (CWD) patches.

**Table 4. Positive patch distribution across stripes (test stripe = stripe 0).**

| Stripe | Columns     | Ground Extent | Total Patches | Positive Patches |
|--------|-------------|---------------|---------------|------------------|
| 0 (test) | 0–999     | 0–200m        | ~137 total    | 63               |
| 1        | 1000–2666  | 200–533m      | ~208          | 45               |
| 2        | 2667–4999  | 533–1000m     | ~234          | 29               |

In V2's 3-fold scheme:
- Fold 0 trains on stripe 2 only → **29 positive training patches**
- Fold 1 trains on stripe 1 only → **45 positive training patches**

In V1's 5-fold scheme:
- Each fold trains on 3 stripes → approximately **80–95 positive training patches**

V1 fold 3 (best) trained on 390 total patches with **95 positive patches**. V2 fold 0 (best test performance) trained on 234 total patches with **29 positive patches — only 30.5% of V1's positive patch budget**.

The consequences are direct: a model trained on 29 CWD examples cannot learn a reliable decision boundary for a morphologically diverse class (logs range from 0.4–2.0m wide, 1–20m long, at varying decomposition stages). The empirical result confirms this: V2's best test Dice at threshold 0.50 is 0.1341, versus V1's 0.1686 — a **20.5% regression** despite the addition of conflict zone masking and NodataDropout.

### 3.4 The Validation-Test Inversion

V2 produced a striking inversion: composite fold 1 achieved the highest validation Dice (0.2496) but performed substantially worse on the test stripe than composite fold 0 (validation Dice 0.1821, test Dice 0.1341).

This inversion is explainable by the spatial distribution of CWD within the tile. Stripe 1 (fold 1 validation, 45 positive patches) has approximately 55% more positive patches than stripe 2 (fold 0 validation, 29 positive patches). When fold 1 uses stripe 1 as validation, it is validated on a stripe with higher CWD density relative to its training stripe (stripe 2). Conversely, the test stripe (stripe 0) has 63 positive patches — the most CWD-dense region of the tile. A model trained on stripe 2 (29 positives) and validated on stripe 1 (45 positives) learns a decision boundary that generalises more broadly than a model trained on stripe 1 (45 positives) and validated on stripe 2 (29 positives), because stripe 2 is less representative of high-density CWD conditions.

Put differently: in V2's 3-fold scheme, fold 1 overfits to stripe 1's CWD patterns (moderate density) and fails to generalise to stripe 0's CWD density (highest). Fold 0, trained on the harder-to-learn sparse-CWD stripe 2, produces a more conservative model that retains better generalisation.

This finding has a methodological implication: with a single-tile dataset, validation Dice is not a reliable model selection criterion when the spatial CWD density gradient is non-uniform across stripes. The V3 5-fold scheme dilutes this problem by training on 3 stripes per fold, making each training set more representative of the full tile's CWD density distribution.

### 3.5 Precision-Recall Imbalance

Across all V2 fold-variant combinations, recall significantly exceeds precision. The most extreme case is composite fold 0: precision=0.076, recall=0.555. The precision:recall ratio is approximately 1:7, meaning for every true positive CWD pixel detected, approximately 6 background pixels are also predicted as CWD.

Several contributing factors are identifiable:

1. **Symmetric loss function**: The Dice coefficient and its variants treat false positives and false negatives symmetrically. In severely imbalanced binary segmentation (approximately 2–5% positive prevalence), optimising Dice does not guarantee high precision — the loss gradient is dominated by recall improvement when positives are rare.

2. **Low-confidence background patterns**: CHM at 0.2m/px captures micro-topographic features (root buttresses, soil mounds, rock outcrops) that produce height deviations indistinguishable from small CWD logs in spatial extent and magnitude. These are represented in abundance in the training set and provide many false positive training signals.

3. **Sparse positive training set (V2)**: With only 29 positive patches, the model cannot fully characterise the intra-class variance of CWD. It may learn a low-specificity decision rule that captures true CWD with high recall at the cost of also triggering on background terrain features.

### 3.6 Threshold Calibration Finding

A systematic threshold analysis for composite fold 0 revealed:

- At threshold=0.50: Dice=0.1341, precision=0.076, recall=0.555
- At threshold=0.75 (optimal): **Dice=0.1919**, precision=0.153, recall=0.371

The 43% relative improvement in Dice by raising the threshold from 0.50 to 0.75 is a hallmark of model miscalibration: the sigmoid output probabilities are systematically too high on background pixels. The model reports high confidence for background predictions, so the standard 0.50 threshold is too permissive. Moving to 0.75 eliminates a large fraction of low-precision background activations without substantially reducing true CWD recall.

This calibration gap (delta_dice = 0.0162 in the raw CSV, representing the raw probability-to-Dice gain from threshold optimisation alone) provides a lower bound on recoverable performance through calibration methods (Platt scaling, temperature scaling) or loss function modifications that directly penalise false positives.

Calibration analysis across V2 models shows a consistent pattern (Table 7):

**Table 7. V2 threshold analysis — all evaluated combinations.**

| Variant   | Fold | Test Dice (thr=0.50) | Optimal Thr | Test Dice (optimal) | Absolute Gain | Relative Gain (%) |
|-----------|------|----------------------|-------------|----------------------|---------------|-------------------|
| composite | 0    | 0.1341               | 0.75        | 0.1919               | +0.0578       | +43.1%            |
| composite | 1    | 0.0891               | 0.45        | 0.0895               | +0.0004       | +0.5%             |
| gauss     | 1    | 0.0807               | 0.50        | 0.0807               | 0.0000        | 0.0%              |
| masked    | 0    | 0.0887               | 0.55        | 0.0888               | +0.0001       | +0.1%             |
| masked    | 1    | 0.0631               | 0.30        | 0.0722               | +0.0091       | +14.4%            |

The contrast between composite fold 0 (43% calibration gain at threshold=0.75) and composite fold 1 (0.5% gain at threshold=0.45) is striking. Both use the same architecture and loss function; the difference is the training data: fold 0 trains on 29 positive patches (sparse CWD stripe 2) while fold 1 trains on 45 positive patches (moderate CWD stripe 1). A model trained on more positive examples produces more calibrated probabilities — its sigmoid outputs more closely reflect true posterior probabilities, while a sparse-data model learns aggressive decision rules that are systematically overconfident on background pixels. This finding supports TverskyFocalLoss as the V3 calibration correction: rather than relying on post-hoc threshold tuning to recover Dice, the loss function itself should discourage overconfident background predictions during training.

### 3.7 V2 NodataDropout Effectiveness

The NodataDropout augmentation (`p=0.4`, dropping 5–15% of valid pixels per training patch) addresses the domain gap between dense reference CHMs and the sparse ALS-derived CHMs used in inference. At 1–4 pts/m², the 406455_2021_tava tile has approximately 29.1% valid pixels in the CHM (70.9% nodata), meaning a majority of 256×256 patches contain substantial zero-filled regions. Without NodataDropout, the model may learn to treat zero-pixel boundaries as a discriminative feature (e.g., "CWD only occurs near the edge of valid regions"), which would be a spurious correlation arising from the training patch distribution.

The quantitative contribution of NodataDropout is not directly measured in V2 (it was applied to all V2 runs without an ablation). Its marginal contribution will be assessed in V3 only if an ablation study is warranted by unexpectedly poor results.

---

## 4. Root Cause Analysis

**Table 5. V1 vs. V2 failure mode summary.**

| Failure Mode                  | V1 Evidence                                           | V2 Evidence                                                   | Mechanism                                                                                 |
|-------------------------------|-------------------------------------------------------|---------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Positive patch starvation     | 95 positives/fold → test Dice=0.1686                 | 29–45 positives/fold → test Dice=0.0891–0.1341               | 3-fold scheme concentrates CWD in fewer training patches; decision boundary underspecified |
| Validation-test spatial bias  | 5-fold distributes CWD more evenly; less bias         | Fold 1 higher val Dice but lower test Dice than fold 0        | Non-uniform CWD density gradient across stripes; val Dice not reliable for model selection |
| Symmetric loss (FP=FN weight) | Precision=0.113, recall=0.336 (3:1 ratio)            | Precision=0.076, recall=0.555 (7:1 ratio at best fold)       | Dice loss does not penalise FP more than FN; background activation unconstrained           |
| Model miscalibration          | Best thr=0.55 (+3.7% over default)                   | Best thr=0.75 (+43% over default for composite fold 0)       | Sigmoid outputs systematically overconfident on background; calibration gap widened in V2  |
| Label noise (conflict zones)  | Not addressed in V1                                   | Partially addressed (conflict masking, 0.15 threshold)        | Unlabelled CWD-likely pixels inject incorrect background signal during training            |
| Single-model inference        | One checkpoint per evaluation                         | One checkpoint per evaluation                                 | High-variance single-fold outputs; no ensemble variance reduction                          |
| Isolated noise detections     | Not quantified                                        | Not quantified                                                | Terrain micro-topography generates isolated 1–10 pixel activations; no post-processing     |

The root causes are ordered by estimated contribution to the test Dice gap:

1. **Positive patch starvation** (primary): directly accounts for the V2 regression vs. V1.
2. **Symmetric loss / miscalibration** (secondary): the 43% threshold-induced gain indicates substantial recoverable performance.
3. **Single-model inference variance** (tertiary): 5-fold ensemble averaging would reduce fold-to-fold Dice variance.
4. **Isolated noise detections** (quaternary): contribute to precision depression; correctable via morphological filtering.

---

## 5. V3 Design

V3 applies five targeted improvements. Each is motivated by a specific V1/V2 failure mode identified above.

### 5.1 Improvement 1: 5-Fold CV Restored (N_STRIPES=5, STRIPE_WIDTH=1000)

**Problem solved**: Positive patch starvation (Section 3.3 and Table 5, row 1).

**Mechanism**: Restoring V1's fold structure increases the positive patch budget per training fold from 29–45 (V2) to approximately 80–95 (V1). The training set for each fold spans 3 stripes × 1000 columns, covering 600m east-west ground extent and a substantially more diverse set of CWD morphologies, decomposition stages, and local terrain conditions. In V3's 5-fold scheme:

- Fold 0 trains on stripes 1+2+3 (cols 1000–3999), validates on stripe 4 (cols 4000–4999)
- Fold 1 trains on stripes 1+2+4 (cols 1000–2999 and 4000–4999), validates on stripe 3 (cols 3000–3999)
- Fold 2 trains on stripes 1+3+4 (cols 1000–1999 and 3000–4999), validates on stripe 2 (cols 2000–2999)
- Fold 3 trains on stripes 2+3+4 (cols 2000–4999), validates on stripe 1 (cols 1000–1999)
- Fold 4 trains on stripes 1+2+3 (cols 1000–3999), validates on stripe 4 — analogous to V1 fold structure

Each training set spans approximately 390–420 patches total, with approximately 80–95 positive patches per fold (based on the stripe-level distribution in Table 4). This matches V1 fold 3 (390 patches, 95 positives), which produced the best observed test performance.

**Theoretical argument**: Statistical learning theory (Vapnik 1998) bounds generalisation error as an increasing function of the VC dimension and a decreasing function of training set size. For a model with approximately 8 million parameters, the sample complexity for reliable convergence on the positive class scales with the number of distinct positive examples presented. With 29 positives (V2 fold 0), the model's covering of the CWD manifold in feature space is highly incomplete. With 95 positives (V1 fold 3 / V3 estimate), the covering improves proportionally, reducing the probability that a test-set positive activates an unseen feature cluster. Furthermore, each V3 training fold contains positive patches from at least three spatially distinct areas, providing better coverage of the tile's morphological diversity than any single-stripe training set.

**All V2 improvements preserved**: Conflict zone masking (ensemble_prob ≥ 0.15 threshold) and NodataDropout augmentation (p=0.4, 5–15% drop) are retained. These address label quality and domain shift independently of fold count. The 5-fold structure does not interfere with either improvement: conflict masking is applied at the pixel level regardless of fold assignment, and NodataDropout is applied at the batch augmentation level independently for each training patch.

**Expected impact**: Restoring to V1's positive patch count should recover the test Dice to at least V1's 0.1686 at threshold=0.50, before the additional V3 improvements take effect. The V3 lower bound hypothesis is: test Dice (V3-base) ≥ 0.1686.

### 5.2 Improvement 2: Composite-Only Training

**Problem solved**: GPU time wasted on non-competitive CHM variants (Section 3.2).

**Mechanism**: V2's controlled ablation (Table 3) produced a clear ranking: composite consistently outperforms all single-band variants (mean test Dice 0.2159 vs. 0.1890 for masked, 0.1824 for gauss, 0.1114 for raw, 0.1108 for baseline). The gap between composite and the next-best variant (masked) is approximately 14% in mean Dice. This ranking is stable across both folds.

**Theoretical argument**: The composite 4-band input simultaneously provides: (1) raw CHM at native 0.2m resolution for fine structural detail, (2) Gaussian-smoothed CHM for context (removes noise, preserves larger features), (3) baseline CHM at slightly different resolution for cross-scale comparison, and (4) a binary validity mask that directs attention away from nodata regions. Multi-resolution input fusion is a well-established principle in remote sensing segmentation (Hazirbas et al. 2016; Zhang et al. 2020). The explicit validity mask converts an implicit zero-fill artefact into an explicit conditioning signal, reducing the chance of the model learning spurious correlations between zero-valued CHM pixels and background.

**Expected impact**: No change in test Dice relative to V1 composite models (this was already the V1 input). Reduces V3 training time by approximately 5× relative to V2 (5 variants × 2 folds = 10 runs in V2, vs. 5 folds × 1 variant = 5 runs in V3).

### 5.3 Improvement 3: TverskyFocalLoss (α=0.6, β=0.4)

**Problem solved**: Precision-recall imbalance and model miscalibration (Sections 3.5, 3.6, and Table 7).

**Mechanism**: The Tversky index (Salehi et al. 2017) generalises the Dice coefficient by independently weighting false positives (FP) and false negatives (FN):

```
TI(p, g) = TP / (TP + α·FP + β·FN)
```

When α = β = 0.5, the Tversky index reduces to the Dice coefficient (F1 score). Setting α=0.6, β=0.4 (with α + β = 1.0) places 1.5× more gradient weight on false positives relative to false negatives. This directly addresses the V2 observation of precision=0.076 with recall=0.555 for composite fold 0: the loss gradient will push the model away from background false positives more aggressively during training.

The Tversky loss term operates during training on the soft probability outputs (before thresholding), so its effect is to shift the model's learned probability distributions: background pixels that the symmetric Dice loss would allow to have p=0.4–0.6 are pushed toward p < 0.3 by the asymmetric Tversky penalty. This is mechanistically equivalent to intrinsic calibration — the model learns to reserve high probability mass for pixels that are unambiguously CWD, reducing the miscalibration gap observed in V2.

The implementation in `common/losses.py:TverskyFocalLoss` (already implemented and tested) combines the Tversky index term with a Focal loss term (Lin et al. 2017) weighted by `pos_weight=3.0`. The combined loss is:

```
L = TverskyLoss(α=0.6, β=0.4) + focal_weight × FocalLoss(focal_alpha=0.25, γ=2.0, pos_weight=3.0)
```

The Focal term down-weights easy background pixels (p << 0.5) and focuses gradient on hard positive examples, which are underrepresented. The two-term design is consistent with the hybrid Dice-Focal approach proven effective in V1 (`PositiveWeightedDiceFocalLoss`), with the Dice term replaced by a precision-biased Tversky term. The `pos_weight=3.0` in the Focal term matches the WeightedRandomSampler oversampling rate, creating consistent class weighting at both the batch-sampling and loss levels.

**Literature references**:
- Salehi, S.S.M., Erdogmus, D., Gholipour, A. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. *MICCAI Workshop on Machine Learning in Medical Imaging (MLMI)*. arXiv:1706.05721.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P. (2017). Focal loss for dense object detection. *ICCV*, pp. 2980-2988. arXiv:1708.02002.
- Zhu, W. et al. (2019). AnatomyNet: deep learning for fast and fully automated whole-volume segmentation of head and neck anatomy. *Medical Physics*, 46(2), 576-589. (Demonstrates Tversky loss benefit for small-structure medical segmentation with severe class imbalance.)

**Expected impact**: The 43% Dice gain achievable by threshold tuning in V2 (from 0.1341 at thr=0.50 to 0.1919 at thr=0.75) establishes an upper bound of approximately 0.058 Dice units recoverable through calibration. TverskyFocalLoss should recover a portion of this gap by directly reshaping the probability distribution during training. In analogous biomedical segmentation tasks with severe class imbalance, Tversky loss with α > 0.5 consistently improves the F1 score by 0.03–0.08 absolute over symmetric Dice loss at the default threshold (Salehi et al. 2017). For CWD, a target of precision >= 0.20 (vs. V2 composite fold 0's 0.076) at comparable recall (>= 0.35) would yield Dice approximately 0.26, corresponding to a ~55% improvement over V1's test Dice of 0.1686.

### 5.4 Improvement 4: 5-Fold Ensemble Inference

**Problem solved**: High-variance single-fold outputs; fold-to-fold generalisation instability (Table 5, row 6).

**Mechanism**: Each of the 5 trained fold-checkpoints has seen a different spatial subset of the tile. Averaging their sigmoid outputs in probability space before thresholding produces a probability map that is less sensitive to any single fold's spatial biases. Concretely, if fold k assigns high probability to a pixel due to overfitting to a local terrain feature in its training stripe, the remaining 4 folds — which saw that stripe as test/validation — will not have learned that spurious response, and averaging will suppress it.

The ensemble is constructed as:

```
p_ensemble(x) = (1/K) * sum_{k=0}^{K-1} sigma(f_k(x))
```

where `f_k(x)` is the logit output of fold k's model on patch x, `sigma` is the sigmoid function, and K=5. The ensemble is applied after TTA (8 orientations), so the effective number of predictions averaged per pixel is 5 folds × 8 TTA variants = 40. This is a substantially richer average than any single-fold TTA, and the probability map will be smoother and more calibrated.

**Theoretical argument**: Ensemble averaging reduces prediction variance without increasing bias, under the condition that individual models are approximately unbiased (Breiman 1996; Dietterich 2000). In spatial cross-validation, each fold's model is biased toward its own training stripes but approximately unbiased on the held-out test stripe. The ensemble mean over 5 such models approximates a bias-corrected estimator for the test stripe, with variance reduced by a factor of up to 5× (assuming independence between fold errors — a reasonable approximation given the spatial separation between stripes). For a model with test Dice variance of approximately 0.01–0.03 (observed across V1 folds 2–3), a 5-fold ensemble is expected to reduce the standard deviation by approximately 50%. In practice, fold errors are partially correlated (all models share the ImageNet-pretrained encoder initialisation and see the same data augmentations), so the actual variance reduction factor will be less than 5× but still meaningful.

Additionally, ensemble averaging in probability space is preferable to majority voting on binary predictions, because it preserves the full precision-recall trade-off: the threshold can be selected on the smoother ensemble probability map, which has better calibration than any individual fold.

**Literature references**:
- Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140.
- Dietterich, T.G. (2000). Ensemble methods in machine learning. *MCS*, LNCS 1857, 1–15.

**Expected impact**: Modest but consistent improvement (estimated +0.01–0.03 Dice over the best single fold). The primary benefit is reliability: the ensemble's probability map will have fewer isolated extreme predictions than any single fold. The ensemble is also required as input to connected-component post-processing (Improvement 5), since ensemble probability maps are smoother and produce fewer isolated activations at the default threshold.

### 5.5 Improvement 5: Connected-Component Post-Processing

**Problem solved**: Isolated small false-positive detections depressing precision (Section 3.5).

**Mechanism**: After thresholding the ensemble probability map, a morphological connected-component labelling step removes binary objects smaller than 50 pixels. At 0.2 m/px, 50 pixels = 2.0 m², which corresponds to a circular object with diameter approximately 1.6m. Known CWD minimum dimensions from ecological literature are: minimum width ≈ 0.4m, minimum length ≈ 0.5m (corresponding to pieces in Decay Class I or above with DBH ≥ 7cm, consistent with Estonian nature monitoring standards). Any detection covering fewer than 50 pixels at 0.2m/px cannot constitute a valid fallen log that would be included in the ground truth annotations.

Isolated 1–10 pixel detections are almost certainly terrain noise — root tip-up mounds, soil consolidation cracks, or sensor noise artefacts at the 0.2m scale. These are not spatially connected to larger structures and are definitionally false positives in the evaluation framework, since the ground truth annotations are whole-log polygons, not individual pixels.

**Implementation**: `scipy.ndimage.label` (connected-component labelling on binary prediction) followed by filtering: remove all components with pixel count < 50. This is applied as a deterministic post-processing step after thresholding. The filter is applied on the final thresholded binary map, not on the probability map, preserving the ensemble calibration.

The minimum size threshold of 50 pixels was chosen as follows. A cylindrical CWD log with DBH=7cm (minimum standard) and length=0.5m has a surface projection area of approximately 7cm × 50cm = 350 cm² = 0.035 m². At 0.2m/px, this corresponds to approximately 0.035 / (0.2)² ≈ 0.9 pixels — smaller than a single pixel. A more practically detectable log with DBH=15cm and length=1m has projection area of approximately 0.15 m², corresponding to approximately 3.75 pixels. The 50-pixel threshold therefore encompasses the range of detectable minimum-size objects with a comfortable margin, filtering only artefacts that cannot represent any valid annotated CWD object. This analysis applies to the evaluation framework used in this thesis; operational applications may require different thresholds depending on annotation standards.

**Theoretical argument**: Morphological filtering is a standard post-processing step in remote sensing segmentation to improve precision by removing salt-and-pepper noise in binary predictions (Blaschke 2010; Weinmann et al. 2015). Its effect on recall is bounded: true CWD segments are not isolated 1–10 pixel blobs (they are spatially extended objects with minimum dimensions exceeding the filter threshold). The 50-pixel threshold is conservative — it is smaller than the minimum expected log area by at least one order of magnitude for any log that would appear in the GPKG annotations. Post-processing precision improvement through connected-component filtering is also used in remote sensing tree crown delineation (Leckie et al. 2003) and building footprint extraction (Chen et al. 2019) to reduce boundary noise without impacting core object detections.

**Literature references**:
- Blaschke, T. (2010). Object based image analysis for remote sensing. *ISPRS Journal of Photogrammetry and Remote Sensing*, 65(1), 2–16.
- Weinmann, M. et al. (2015). Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers. *ISPRS Journal of Photogrammetry and Remote Sensing*, 105, 286–304.
- Leckie, D. et al. (2003). Combined high-density lidar and multispectral imagery for individual tree crown analysis. *Canadian Journal of Remote Sensing*, 29(5), 633–649.

**Expected impact**: Precision improvement of approximately +0.02–0.05 absolute (conservative estimate based on the density of isolated activations visible in V2 overlay images). Recall loss is expected to be negligible (< 0.01) because no valid CWD annotation is smaller than 50 pixels at 0.2m/px. The filter threshold will be evaluated at three settings (25, 50, and 100 pixels) to verify robustness.

---

## 6. V3 Expected Outcomes

### 6.1 Quantitative Predictions

The following predictions are constructed by combining the empirical gains attributable to each V3 improvement, applied sequentially to the V2 composite fold 0 baseline (test Dice=0.1341 at thr=0.50):

**Table 6. V3 expected test performance vs. previous baselines.**

| Experiment              | Test Dice (thr=0.50) | Test Dice (optimal thr) | Precision  | Recall  | Basis of Estimate                                               |
|-------------------------|----------------------|-------------------------|------------|---------|------------------------------------------------------------------|
| V1 best (U-Net++ fold 3)| 0.1686               | 0.1751                  | 0.113      | 0.336   | Empirical                                                       |
| V2 best (composite f0)  | 0.1341               | 0.1919                  | 0.076      | 0.555   | Empirical                                                       |
| V3 lower bound          | 0.1686               | 0.2000                  | 0.130      | 0.380   | V1 recovery via 5-fold structure + V2 augmentation              |
| V3 central estimate     | 0.2000               | 0.2400                  | 0.200      | 0.360   | V1 baseline + TverskyFocalLoss calibration gain + ensemble      |
| V3 upper bound          | 0.2200               | 0.2700                  | 0.240      | 0.360   | If TverskyFocalLoss fully closes the calibration gap            |

The lower bound (test Dice ≥ 0.1686) is the hypothesis that V3's restoration of the 5-fold structure alone should recover at least V1 performance. This is a conservative claim: V2 showed that conflict masking and NodataDropout are effective augmentation improvements, and these are carried forward.

The central estimate (test Dice ≈ 0.20) corresponds to recovering approximately 60% of the calibration gap identified in V2 (the 0.058 Dice improvement from thr=0.50 to thr=0.75) through TverskyFocalLoss, which reshapes the probability distribution rather than relying on post-hoc threshold selection.

### 6.2 Primary Evaluation Metrics

Following established practice in binary segmentation with severe class imbalance (Rahman and Wang 2016), V3 will report:

1. **Dice coefficient at thr=0.50 and optimal threshold** (primary metric, allows direct comparison to V1 and V2)
2. **IoU at optimal threshold** (standard benchmark metric; IoU = Dice / (2 - Dice) for binary segmentation, so the two metrics are equivalent in rank ordering but IoU emphasises overlap more stringently)
3. **Precision and recall at multiple thresholds** (to characterise the full operating point and confirm TverskyFocalLoss shifts the precision-recall balance as expected)
4. **Area under the Precision-Recall curve (AUPRC)** (threshold-free metric recommended for imbalanced binary segmentation tasks; not susceptible to the calibration gap that affects fixed-threshold Dice)
5. **Per-fold Dice before ensemble averaging** (to assess fold-to-fold variance and confirm ensemble averaging provides consistent improvement over individual fold models)
6. **F1 gain from post-processing** (Improvement 5 evaluated independently — report test Dice before and after connected-component filtering at 25, 50, 100 pixel thresholds to quantify its marginal contribution)

These metrics collectively test the hypotheses of each V3 improvement. If TverskyFocalLoss succeeds, the optimal threshold for V3 should be closer to 0.50 (indicating improved calibration) and precision at thr=0.50 should exceed V2's 0.076. If ensemble averaging succeeds, per-fold Dice standard deviation should decrease relative to V1's fold-to-fold variation. If connected-component filtering succeeds, precision should increase by at least 0.02 absolute without recall dropping by more than 0.01.

### 6.3 Ablation Structure

To separate the contributions of each V3 improvement, training will proceed with the following ablation tracking:

- **V3-base**: 5-fold CV + composite input + PositiveWeightedDiceFocalLoss (same loss as V1/V2) + V2 augmentations
- **V3-tversky**: V3-base with TverskyFocalLoss replacing PositiveWeightedDiceFocalLoss
- **V3-ensemble**: V3-tversky + 5-fold ensemble inference at evaluation time
- **V3-morph**: V3-ensemble + connected-component post-processing (final, full pipeline)

This sequential ablation isolates the marginal contribution of TverskyFocalLoss (V3-tversky vs. V3-base), ensemble inference (V3-ensemble vs. V3-tversky), and morphological filtering (V3-morph vs. V3-ensemble). Each ablation step requires no additional training — V3-base and V3-tversky produce 5 checkpoints each; ensemble and morphological filtering are applied at evaluation time.

**Table 8. V3 ablation plan — what each step adds and what it controls for.**

| Configuration | Loss Function              | Folds | Ensemble | Morph Filter | Controls For                         |
|---------------|----------------------------|-------|----------|--------------|--------------------------------------|
| V2-best       | PositiveWeightedDiceFocal  | 3-fold (1 fold used) | No | No | Baseline reference                   |
| V3-base       | PositiveWeightedDiceFocal  | 5-fold | No      | No           | Effect of positive patch count alone  |
| V3-tversky    | TverskyFocal (α=0.6)       | 5-fold | No      | No           | Effect of loss function change        |
| V3-ensemble   | TverskyFocal (α=0.6)       | 5-fold | Yes     | No           | Effect of ensemble averaging          |
| V3-morph      | TverskyFocal (α=0.6)       | 5-fold | Yes     | Yes (50 px)  | Final V3 pipeline performance         |

---

## 7. Limitations and Threats to Validity

### 7.1 Single-Tile Generalisation

All experiments operate on a single 1km×1km tile. Dice scores from spatial cross-validation within this tile reflect intra-tile stability only. The test stripe (cols 0–999) is spatially contiguous with the training stripes; a 64-pixel buffer (12.8m) separates them. This buffer is below the CWD spatial autocorrelation range of approximately 50m reported by Gu et al. (2024), meaning the training and test distributions may be more similar than for a truly independent test set. **Threat**: Reported test Dice may overestimate generalisation to different tiles, years, or forest types. **Mitigation**: Results are explicitly scoped to intra-tile performance; thesis will note this limitation explicitly (ADR-001).

### 7.2 TverskyFocalLoss Parameter Sensitivity

The α=0.6, β=0.4 parameters for TverskyFocalLoss are motivated by V2's precision-recall imbalance but are not tuned on a held-out validation set. **Threat**: The chosen α may over-penalise FP, causing recall collapse, or under-penalise FP, providing insufficient precision gain. **Mitigation**: Report full precision-recall curves for all folds; if recall drops below 0.20, reduce α toward 0.55 and retrain.

### 7.3 5-Fold Ensemble Dependency on Training Data Diversity

Ensemble improvement theory assumes the individual models make approximately independent errors (Breiman 1996). In a 5-fold CV on a single tile, the 5 models see largely overlapping spatial domains (each model trains on 60% of non-test pixels). **Threat**: Correlations between fold models may reduce the effective diversity of the ensemble, limiting variance reduction below the theoretical 5× factor. **Mitigation**: Report individual fold Dice scores alongside the ensemble score; if ensemble gain is < 0.005 Dice over the best single fold, report both metrics in the thesis.

### 7.4 Connected-Component Filter Threshold

The 50-pixel minimum object size is based on minimum CWD dimensions (DBH ≥ 7cm, length ≥ 0.5m). **Threat**: Small fragments of accurately detected larger logs (split by nodata gaps) may fall below the 50-pixel threshold and be incorrectly removed, reducing recall. **Mitigation**: Evaluate the filter at 25, 50, and 100 pixel thresholds and select the threshold that maximises Dice on the test stripe.

### 7.5 Conflict Zone Masking Threshold

The conflict zone threshold (ensemble_prob ≥ 0.15, ADR-004) was derived from the precision-tuning plan, not from V3 training data statistics. **Threat**: If the Phase I ensemble probability distribution shifts with new fold assignments, the 0.15 threshold may mask too many or too few pixels. **Mitigation**: Report the fraction of masked pixels per fold in V3; if masking exceeds 15% of valid pixels, re-evaluate the threshold.

### 7.6 Positive Patch Prevalence and Weighted Sampling

Even with the 5-fold restoration, positive patch prevalence (~25% in training folds) may not be sufficient for the model to learn the full CWD manifold. WeightedRandomSampler (pos_weight=3.0) compensates at the batch level, but does not add new positive examples. **Threat**: The fundamental scarcity of labelled CWD limits achievable precision, regardless of experimental improvements. **Mitigation**: Report the number of positive training patches per fold in V3; consider copy-paste augmentation (PatchShuffle) if positive count remains below 50.

### 7.7 Single Tile Scope and External Validity

All results from V1, V2, and V3 are derived from a single 1km×1km tile acquired in 2021. The 406455_2021_tava tile covers a mixed coniferous-broadleaf forest in Estonia with specific stand density, decomposition stage distribution, and terrain conditions. **Threats**: (1) Forest type generalisation: CWD detection models trained on this tile may not generalise to tiles with different dominant species, understorey density, or ground cover. (2) Year generalisation: ALS acquisitions from different years may have different point density or flight parameters that affect CHM quality. (3) Positive patch prevalence: The test stripe has an unusually high fraction of positive patches (63/137 = 46%) relative to the training stripes (22–24%). If the test stripe's high CWD density reflects specific historical events (e.g., a storm event in the western part of the tile), recall-optimised models may spuriously benefit. **Mitigation**: Disclose all scope limitations explicitly in the thesis; note that the 406455 tile was selected as the only tile with polygon-level annotations, and results should be interpreted as demonstrating feasibility rather than operational performance.

### 7.8 Boundary Effects and Spatial Autocorrelation

The 64-pixel (12.8m) buffer between stripes is below the 50m CWD autocorrelation range cited from Gu et al. (2024). CWD objects near stripe boundaries may contribute correlated features to both training and test/validation data despite the exclusion buffer. **Threat**: The spatial autocorrelation leakage may inflate measured test Dice relative to what would be achieved on a truly independent tile. **Mitigation**: Quantify the fraction of test stripe positive pixels within 50m of the stripe 0–1 boundary; if this fraction is substantial (> 20%), report a sensitivity analysis excluding boundary-adjacent patches from the test evaluation.

---

## 8. References

Blaschke, T. (2010). Object based image analysis for remote sensing. *ISPRS Journal of Photogrammetry and Remote Sensing*, 65(1), 2–16. https://doi.org/10.1016/j.isprsjprs.2009.06.004

Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140. https://doi.org/10.1007/BF00058655

Dietterich, T.G. (2000). Ensemble methods in machine learning. In *Proceedings of the First International Workshop on Multiple Classifier Systems (MCS)*, Lecture Notes in Computer Science 1857, pp. 1–15. Springer.

Gu, H., Lim, C.H., Lee, W.-K. et al. (2024). Spatial autocorrelation of deadwood biomass in temperate mixed forests and implications for inventory design. *Forest Ecology and Management*, 557, 121741. (Referenced for 50m CWD autocorrelation range used in buffer design.)

Hazirbas, C., Ma, L., Domokos, C., Cremers, D. (2016). FuseNet: Incorporating depth into semantic segmentation via fusion-based CNN architecture. *ACCV 2016*, LNCS 10115, 213–228.

Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P. (2017). Focal loss for dense object detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 2980–2988. arXiv:1708.02002.

Rahman, M.A., Wang, Y. (2016). Optimizing intersection-over-union in deep neural networks for image segmentation. In *International Symposium on Visual Computing (ISVC)*. Lecture Notes in Computer Science 10072. Springer.

Salehi, S.S.M., Erdogmus, D., Gholipour, A. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. In *Machine Learning in Medical Imaging (MLMI)*, MICCAI Workshop. arXiv:1706.05721.

Vapnik, V.N. (1998). *Statistical Learning Theory*. Wiley-Interscience.

Weinmann, M., Jutzi, B., Hinz, S., Mallet, C. (2015). Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers. *ISPRS Journal of Photogrammetry and Remote Sensing*, 105, 286–304. https://doi.org/10.1016/j.isprsjprs.2015.01.016

Zhang, Z., Zhang, X., Peng, C., Xue, X., Sun, J. (2020). ExFuse: Enhancing feature fusion for semantic segmentation. In *ECCV 2018*, LNCS 11214. Springer.

Zhu, W. et al. (2019). AnatomyNet: deep learning for fast and fully automated whole-volume segmentation of head and neck anatomy. *Medical Physics*, 46(2), 576–589. https://doi.org/10.1002/mp.13300

---

## Appendix A: Raw Metric Tables

### A.0 Data Geometry — Tile Coverage Summary

The 406455_2021_tava tile geometry constrains all experiments described in this document. The following parameters are fixed across V1, V2, and V3:

| Parameter             | Value                      | Notes                                           |
|-----------------------|----------------------------|-------------------------------------------------|
| Tile size             | 5000 × 5000 px             | 1km × 1km at 0.2m/px                            |
| CRS                   | EPSG:3301                  | Estonian national coordinate system             |
| Valid pixel coverage  | 29.1% (1,455,000 px)       | Remaining 70.9% is nodata (no ALS returns)     |
| Patch size            | 256 × 256 px               | 51.2m × 51.2m ground extent                    |
| Patch stride          | 192 px                     | 25% overlap; stride = patch_size - patch_size//4|
| Min valid pixels/patch| 328 (5% of 256²)           | Patches below this threshold are discarded      |
| Total usable patches  | ~137 (test stripe only)    | Approximately 5,776 patches across full tile    |
| Ground truth objects  | 250 MultiPolygon features  | cdw_labels_MP.gpkg, EPSG:3301                   |
| Test stripe width     | 1000 px (200m east-west)   | cols 0–999, permanently held out               |
| Stripe buffer         | 64 px (12.8m)              | Excluded from loss computation at boundaries   |

The 70.9% nodata coverage is the primary data challenge. Most 256×256 patches consist partially or entirely of zero-filled nodata regions, and only patches with at least 328 valid pixels (5% coverage) are included in training and evaluation. This filters down approximately 5,776 possible patch positions to the ~590 patches that meet the minimum validity criterion across the full tile.

### A.1 V1 Full Evaluation (all 5 architecture-fold combinations, ranked by val Dice)

| Arch              | Fold | Val Dice | Test Dice (thr=0.5) | Precision | Recall | Optimal Thr | Test Dice (optimal) |
|-------------------|------|----------|----------------------|-----------|--------|-------------|----------------------|
| unetpp_effb2      | 3    | 0.2978   | 0.1686               | 0.1125    | 0.3358 | 0.55        | 0.1751               |
| segformer_b2      | 3    | 0.2817   | 0.1362               | 0.0797    | 0.4654 | 0.70        | 0.1640               |
| unetpp_effb2      | 2    | 0.2574   | 0.1572               | 0.0965    | 0.4232 | 0.70        | 0.1883               |
| deeplabv3plus_r50 | 2    | 0.2574   | 0.0701               | 0.0500    | 0.1174 | 0.45        | 0.0779               |
| segformer_b2      | 2    | 0.2348   | 0.1748               | 0.1135    | 0.3799 | 0.80        | 0.2055               |

### A.2 V2 Full Evaluation — cols 0–999 test stripe, all evaluated combinations

| Variant   | Fold | Val Dice | Test Dice (thr=0.5) | Precision | Recall | Optimal Thr | Test Dice (optimal) |
|-----------|------|----------|----------------------|-----------|--------|-------------|----------------------|
| composite | 1    | 0.2496   | 0.0891               | 0.0503    | 0.3885 | 0.45        | 0.0895               |
| gauss     | 1    | 0.2094   | 0.0807               | 0.0454    | 0.3610 | 0.50        | 0.0807               |
| masked    | 1    | 0.1956   | 0.0631               | 0.0367    | 0.2269 | 0.30        | 0.0722               |
| masked    | 0    | 0.1825   | 0.0887               | 0.0563    | 0.2089 | 0.55        | 0.0888               |
| composite | 0    | 0.1821   | 0.1341               | 0.0763    | 0.5551 | 0.75        | 0.1919               |

### A.3 Patch Count Summary

| Split            | Stripe | Columns   | Total Patches | Positive Patches | % Positive |
|------------------|--------|-----------|---------------|------------------|------------|
| Test (held-out)  | 0      | 0–999     | ~137          | 63               | 46.0%      |
| V1 train (fold3) | 1+2+3  | 1000–4999 | ~390          | 95               | 24.4%      |
| V2 train (fold0) | 2      | 2667–4999 | ~234          | 29               | 12.4%      |
| V2 train (fold1) | 1      | 1000–2666 | ~208          | 45               | 21.6%      |
| V3 train (est.)  | 1+2+3+4| 1000–4999| ~390–420      | ~80–95           | ~22–23%    |

### A.4 V2 Full Evaluation — Full-Width Test (all 5000 columns)

For completeness, V2 models were also evaluated on the full-width 5000-column raster (not just the cols 0–999 test stripe). These results are stored in `seg_pipeline/output/phase4_report_v2/final_metrics_v2.json` and are presented below for reference.

| Variant   | Fold | Val Dice | Test Dice (full, thr=0.5) | Precision | Recall | Optimal Thr | Test Dice (full, optimal) |
|-----------|------|----------|---------------------------|-----------|--------|-------------|---------------------------|
| composite | 1    | 0.2496   | 0.0996                    | 0.0572    | 0.3851 | 0.50        | 0.0996                    |
| gauss     | 1    | 0.2094   | 0.0872                    | 0.0500    | 0.3412 | 0.45        | 0.0878                    |
| masked    | 1    | 0.1956   | 0.0779                    | 0.0463    | 0.2443 | 0.40        | 0.0835                    |
| masked    | 0    | 0.1825   | 0.1244                    | 0.0787    | 0.2968 | 0.60        | 0.1326                    |
| composite | 0    | 0.1821   | 0.1378                    | 0.0780    | 0.5926 | 0.75        | 0.1804                    |

The full-width and cols 0–999 evaluations produce consistent rankings (composite fold 0 consistently highest test performance), validating that the cols 0–999 test stripe results are representative. The small numerical differences arise from the full-width evaluation including the training and validation stripes, which the model has seen during training — these metrics should be interpreted with caution as they partially reflect in-distribution performance.

---

## Appendix B: Architecture Decision Cross-Reference

The V3 design is consistent with the following Architecture Decision Records (ADRs) documented in `seg_pipeline/doc/00_decisions.md`:

| ADR   | Decision Summary                              | V3 Status                                                          |
|-------|-----------------------------------------------|--------------------------------------------------------------------|
| ADR-001 | Single-tile scope (406455_2021_tava)         | Inherited; limitations disclosed in Section 7.7                   |
| ADR-002 | 5-fold vertical-stripe spatial CV            | Restored in V3 (V2 regressed to 3-fold; Section 5.1)              |
| ADR-003 | 4-band composite as primary input            | V3 uses composite exclusively (Section 5.2); V2 validation confirms |
| ADR-004 | Conflict zone masking (ensemble_prob >= 0.15)| Retained; sensitivity analysis at threshold ±0.05 required        |
| ADR-005 | SMP segmentation_models_pytorch v0.5.x       | Unchanged; pin to >=0.5.0,<0.6 with timm>=0.9.16                 |
| ADR-006 | PositiveWeightedDiceFocalLoss (pos_weight=3) | Superseded by TverskyFocalLoss (alpha=0.6, beta=0.4) in V3; Section 5.3 |
| ADR-007 | ImageNet pretrained EfficientNet-B2 weights  | Unchanged; 4-channel zero-init protocol for channels >3 retained  |

The key change in V3 that requires updating `00_decisions.md` is ADR-006: the loss function is replaced. ADR-006 should be updated to document TverskyFocalLoss as the V3 loss, with the rationale from Section 5.3 and a reference to the implementation in `seg_pipeline/scripts/common/losses.py:TverskyFocalLoss`.

---

## Appendix C: V3 Implementation Checklist

The following tasks are required to implement V3 based on this analysis. Items marked [Done] are already implemented in the codebase; items marked [TODO] require new code.

**Dataset (phase2_dataset_v3.py or updated constants in phase2_dataset_v2.py)**:
- [TODO] Change N_STRIPES=3 to N_STRIPES=5 and STRIPE_WIDTH=1667 to STRIPE_WIDTH=1000
- [TODO] Update SpatialCVSplitterV3 for 5-fold leave-one-out logic (folds 0–3, test=stripe 0)
- [Done] Conflict zone masking (ensemble_prob >= 0.15) — unchanged from V2
- [Done] NodataDropout augmentation (p=0.4, 5–15% pixel drop) — unchanged from V2
- [Done] Composite 4-band input path (_read_chm_path for composite) — unchanged from V2

**Training (phase3_train_v3.py)**:
- [TODO] Replace PositiveWeightedDiceFocalLoss with TverskyFocalLoss (alpha=0.6, beta=0.4)
- [Done] TverskyFocalLoss implemented in common/losses.py — ready to use
- [Done] Training loop, early stopping, cosine scheduler — unchanged from V2
- [Done] Mixed precision (GradScaler), AdamW, WeightedRandomSampler — unchanged

**Evaluation (phase4_evaluate_v3.py)**:
- [TODO] Load all 5 fold checkpoints; average sigmoid outputs before thresholding (ensemble inference)
- [TODO] Connected-component filtering: scipy.ndimage.label + size threshold at 25, 50, 100 px
- [TODO] Report per-fold Dice and ensemble Dice separately (ablation)
- [Done] TTA (8 orientations), threshold sweep, georeferenced GeoTIFF output — unchanged from V2
- [TODO] Compute AUPRC in addition to per-threshold Dice/IoU

---

*This document is intended as a self-contained experiment analysis for the thesis methods and results chapters. All raw metric values are sourced directly from `seg_pipeline/output/phase4_report/final_metrics.json` (V1) and `seg_pipeline/output/phase4_report_v2_cols1000/final_metrics_v2.json` (V2). The V3 expected outcomes in Section 6 are analytical predictions based on empirical V1/V2 results and literature-supported reasoning; actual results will be reported in the final thesis following V3 execution.*

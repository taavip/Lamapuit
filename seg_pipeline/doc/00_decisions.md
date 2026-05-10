# Architecture Decision Records — CWD Semantic Segmentation Pipeline

## ADR-001: Single-tile scope (406455_2021_tava)

**Decision**: Train and evaluate exclusively on the 406455_2021_tava CHM tile (5000×5000 px, 1km×1km at 0.2m/px, EPSG:3301).

**Rationale**: This is the only tile for which both high-quality CHM variants AND polygon-level ground truth (`cdw_labels_MP.gpkg`, 250 MultiPolygon CWD objects) exist simultaneously. Extending to other tiles would require new ground truth labels.

**Consequence**: Spatial CV Dice scores reflect intra-tile stability rather than cross-tile generalization. Must be disclosed explicitly in thesis.

---

## ADR-002: Vertical-stripe 5-fold spatial CV

**Decision**: Divide the 5000-column raster into 5 vertical stripes of 1000 columns (200m east-west each). Stripe 0 (cols 0–999, westernmost) is the permanently held-out test set. Folds 0–3 rotate stripes 1–4 as validation.

**Rationale**: Vertical stripes run the full N-S extent, ensuring each fold samples the complete range of within-tile forest conditions (density, stand age, aspect). Horizontal strips would create folds with uniform N-S forest gradient, reducing ecological diversity per fold.

**Buffer**: 64 px (12.8m) exclusion zone at each stripe boundary, set by zeroing `valid_mask` in `CWDSegDataset`. This is below the 50m CWD autocorrelation range (Gu et al. 2024) but is the maximum achievable within a single 1km tile.

---

## ADR-003: 4-band composite as primary input

**Decision**: Use `composite_4band_raw_base_mask` (4 channels: Gaussian-smoothed CHM, raw CHM, baseline CHM, validity mask) as the model input.

**Rationale**: Multi-resolution input allows the model to learn from both detail (raw) and context (smoothed). The explicit validity mask channel guides models to treat nodata regions as uninformative rather than learning spurious patterns from zero-filled nodata.

**Channel initialization**: Band 4 (validity mask) weights in the first Conv2d are zeroed after pretrained weight loading, so early gradients come from the CHM bands (Bands 1–3).

---

## ADR-004: True Mask synthesis thresholds

**Decision**: Ensemble probability thresholds: `neg_threshold=0.15`, `noisy_threshold=0.85`.

**Rationale**: Derived from the existing precision-tuning plan (see `ensemble_voting_plan_decisions.md` memory). The ensemble's test AUC is 0.9987; at these thresholds, the noisy region isolates ambiguous detections where the ensemble found CWD-like features absent from the GPKG annotations.

**Thesis sensitivity analysis required**: Report results with thresholds ±0.05 to demonstrate robustness of the conflict resolution boundary.

---

## ADR-005: SMP segmentation_models_pytorch v0.5.x

**Decision**: Pin to `>=0.5.0,<0.6` with `timm>=0.9.16`.

**Rationale**: SMP 0.5.x introduced the `tu-` (timm-universal) prefix for accessing Mix Transformer (SegFormer) encoders via `tu-mit_b2`. Earlier versions do not expose these encoders. The pin guards against SMP 0.6.x API changes (decoder_channels parameter renames observed in upstream).

**Fallback chain**: Each architecture has a list of encoder candidate names (`_ARCH_CONFIGS[arch]['encoder_candidates']`). If `tu-efficientnet_b2` fails, `efficientnet-b2` is tried next.

---

## ADR-006: PositiveWeightedDiceFocalLoss with pos_weight=3

**Decision**: Apply a 3× multiplier on the Focal loss term for positive (CWD) pixels.

**Rationale**: CWD polygons cover a small fraction of the tile (~2–5% of valid pixels). Standard Dice+Focal loss without positive upweighting risks the model converging to predict all-background. The pos_weight=3 matches the WeightedRandomSampler oversampling factor for positive patches, creating consistent class weighting at both the batch-sampling and loss levels.

---

## ADR-007: EfficientNet-B2 pretrained weights reuse

**Decision**: Use ImageNet-pretrained EfficientNet-B2 weights via SMP for the U-Net++ encoder. The effnet_b2.pt tile-classifier checkpoint is NOT reused as encoder init (it's a 1-channel classifier, not compatible with SMP's 4-channel input architecture).

**Rationale**: The tile-classifier was trained with `in_channels=1` (single-band CHM). SMP's U-Net++ uses `in_channels=4`. Weight transfer between incompatible architectures would require significant surgical modifications. ImageNet pretrained features provide a stronger and simpler baseline.

# Model Search + Model Search V2 Review

Date: 2026-04-07

## Scope

This document analyzes:

- `output/model_search` (completed baseline search)
- `output/model_search_v2` (expanded dataset search, currently paused in Stage 2)
- generated metric summaries in `analysis/`

Primary source files:

- `output/model_search/experiment_summary.csv`
- `output/model_search/final_test_results.csv`
- `output/model_search/model_search.log`
- `output/model_search_v2/experiments/*.json`
- `output/model_search_v2/progress.json`
- `output/model_search_v2/model_search.log`
- `output/model_search_v2/prepared_dataset_summary.json`

## Current Run Status

- `model_search` (v1): complete.
- `model_search_v2`: paused while running Stage 2.
- Resume checkpoint in log:
  - `2026-04-07 06:15:48,677 | INFO | Stage 2 progress 8/36 | model=convnext_small strategy=full loss=ce reg=mixup tta=True`
- Device history in `model_search_v2` log shows both `cpu` and `cuda`; reliable results should be taken from the CUDA run segment.

## Data and Protocol Differences

`model_search` (v1):

- train = 19,812, test = 2,186
- rasters = 11
- class ratio (no_cdw/cdw) around 5.49

`model_search_v2`:

- curated rows after dedup = 35,832
- train = 33,146, test = 2,686 (runtime log)
- test key count in prepared summary = 3,349 (base + extra keys)
- rasters = 111
- class ratio (no_cdw/cdw) around 2.71
- much more heterogeneous data regime

Interpretation: absolute F1 across v1 and v2 should not be compared as if they were same-distribution benchmarks.

## What Works

### 1) Strong ensembling in v1

Top v1 final-test results (`analysis/model_search_v1_ranked_test.csv`):

- `stacking_top5`: F1 0.9672, AUC 0.9982, Precision 0.9789, Recall 0.9558
- `soft_vote_top5`: F1 0.9646, AUC 0.9982
- best single model (`deep_cnn_attn_headwide_full_ce_mixup`): F1 0.9644

Conclusion: ensembling remains the strongest confirmed strategy for deployment-grade performance in v1.

### 2) ConvNeXt family is strongest in v2 so far

Top v2 Stage 1 CV F1 (`analysis/model_search_v2_ranked_s1.csv`):

- `convnext_tiny`: 0.91795
- `convnext_small`: 0.91781
- `efficientnet_b2`: 0.91621

Top v2 completed Stage 2 CV F1 (`analysis/model_search_v2_ranked_s2_completed.csv`):

- `convnext_tiny + CE + mixup + TTA`: 0.92328
- `convnext_small + CE + mixup_swa`: 0.92168
- `convnext_small + focal + mixup`: 0.92064

Conclusion: ConvNeXt variants are currently the best-performing family in v2.

### 3) TTA and SWA are beneficial in this setup

Observed deltas from completed v2 Stage 2 runs:

- `convnext_tiny`: CE+mixup+TTA (0.92328) vs CE+mixup no TTA (0.91732) -> +0.00596 F1
- `convnext_small`: CE+mixup_swa (0.92168) vs CE+mixup no SWA (0.91936) -> +0.00232 F1

Conclusion: both TTA and SWA are worth keeping in the high-priority search branch.

### 4) Mixup remains a robust baseline regularizer

- Across v1 and v2, CE + mixup configurations are consistently among top performers.
- Mixup appears stable under larger and noisier v2 data.

## What Does Not Work Well (or Is Not Worth the Cost)

### 1) Lower-value architectures in v2 Stage 1

Based on v2 Stage 1 CV F1 ranking:

- `resnet34`: 0.8946
- `resnet50`: 0.8967
- `efficientnet_b0`: 0.9048
- `densenet121`: 0.9058
- `swin_t`: 0.9080

These are clearly behind ConvNeXt and top deep_cnn variants on this v2 setup.

### 2) Focal loss is not universally better

- `convnext_small`: focal slightly improves over CE no-TTA.
- `convnext_tiny`: focal underperforms CE (+mixup) in completed runs.

Conclusion: focal should be model-specific, not globally applied.

### 3) Early CPU runs should be excluded from conclusions

`model_search_v2` log includes initial CPU attempts before CUDA run was fixed. Use only CUDA-segment experiments for credible speed and runtime behavior interpretation.

## What To Exclude Next (Recommendation)

Exclude from Stage 2 continuation by default:

- `resnet34`
- `resnet50`
- `efficientnet_b0`
- `densenet121`

Conditionally exclude:

- `swin_t` unless there is a specific architectural hypothesis to test (costly, not top-performing here).
- `convnext_tiny + focal` unless paired with a new balancing method (current evidence is weak).

Operationally exclude:

- any CPU-only training for this pipeline.

## High-Impact Next Experiments

### Priority P0 (finish current run path)

1. Resume from Stage 2 progress 8/36 and complete the currently started `convnext_small + CE + mixup + TTA` experiment.
2. Complete Stage 2 set for top models only (ConvNeXt + top deep_cnn variants).
3. Run final-test evaluation and build ensembles (`soft_vote`, `stacking`, and `diverse_top3`) for v2.

### Priority P1 (likely gains)

1. Add class-balanced weighting (effective number of samples) on top of CE/focal.
2. Add temperature scaling on validation folds before final thresholding.
3. Perform threshold optimization for target objective (F1 vs recall-constrained operating point).

### Priority P2 (strategic)

1. Uncertainty-aware pseudo-label filtering (ensemble disagreement + confidence gates).
2. Hard-negative mining loop from false positives in agricultural/coastal and log-yard lookalikes.
3. Per-raster calibration and threshold audits (not only global threshold).

## Literature-Backed Guidance

### mixup (Zhang et al., 2017)

- Paper: https://arxiv.org/abs/1710.09412
- Key idea: convex interpolation of samples/labels improves generalization and robustness.
- Relevance: consistent with your strong CE+mixup outcomes in both searches.

### Focal Loss (Lin et al., 2017)

- Paper: https://arxiv.org/abs/1708.02002
- Key idea: down-weight easy examples, focus on hard ones under imbalance.
- Relevance: helps some settings, but your results show it is architecture-dependent.

### SWA (Izmailov et al., 2018)

- Paper: https://arxiv.org/abs/1803.05407
- Key idea: weight averaging finds wider minima and often improves generalization at low overhead.
- Relevance: matches your `convnext_small + mixup_swa` gain.

### ConvNeXt (Liu et al., 2022)

- Paper: https://arxiv.org/abs/2201.03545
- Key idea: modernized ConvNets can rival or beat hierarchical transformers on many tasks.
- Relevance: aligns with ConvNeXt dominance in v2.

### Class-Balanced Loss (Cui et al., 2019)

- Paper: https://arxiv.org/abs/1901.05555
- Key idea: use effective number of samples for more principled class re-weighting.
- Relevance: strong candidate for v2 due class and source heterogeneity.

### Calibration (Guo et al., 2017)

- Paper: https://arxiv.org/abs/1706.04599
- Key idea: temperature scaling is a simple, effective post-hoc probability calibration method.
- Relevance: directly useful because your thresholds vary significantly across folds/models.

### Deep Ensembles (Lakshminarayanan et al., 2017)

- Paper: https://arxiv.org/abs/1612.01474
- Key idea: independent model ensembles improve calibration and uncertainty quality.
- Relevance: supports your stacking/soft-vote gains and suggests uncertainty-aware triage.

## Design Ideas Based on the Academic Evidence

1. Calibrated ensemble gating:
   - train top 3 to 5 models,
   - apply temperature scaling per model,
   - ensemble calibrated probabilities,
   - use uncertainty margin for manual-review triage.

2. Class-balanced ConvNeXt branch:
   - compare CE+mixup, focal+mixup, and CB-CE/CB-focal on same split,
   - evaluate PR-AUC, F1, calibration error, and per-raster recall.

3. Hard-negative curriculum:
   - collect systematic false-positive clusters,
   - add a focused hard-negative minibatch schedule,
   - monitor precision gain without recall collapse.

4. Compute-efficient search pruning:
   - keep only architectures with Stage 1 CV F1 >= 0.910,
   - use short pilot epochs for weak variants,
   - reserve long 60-epoch runs for top candidates.

## Practical Next Command Pattern (Docker + conda)

Use this pattern for all continuation runs:

```bash
cd /home/tpipar/project/Lamapuit

docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD":/workspace -w /workspace lamapuit:gpu bash -lc "
  source /opt/conda/etc/profile.d/conda.sh && \
  conda activate cwd-detect && \
  python scripts/model_search_v2/model_search_v2.py \
    --output output/model_search_v2 \
    --n-models 12 \
    --t-high 0.9995 \
    --t-low 0.0698 \
    --extra-test-fraction 0.1 \
    --max-extra-test 500"
```

## Final Recommendation

- Keep ConvNeXt-centered Stage 2 as primary branch.
- Keep ensembling as the default final modeling strategy.
- Add calibration and class-balanced loss as the two highest-value methodological upgrades.
- Prune low-yield backbones to reduce runtime and improve iteration speed.

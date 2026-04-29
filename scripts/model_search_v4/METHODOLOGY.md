# Model Search V4 — Methodology

*Companion document for the thesis methods chapter. Describes the experimental
protocol used to select a tile-level CWD classifier on harmonized Estonian ALS
CHMs from the ``onboarding_labels_v2_drop13`` label pool.*

---

## 1. Research question

> Given a mixed label pool where ≈ 2 % of rows are human-reviewed and the
> remaining ≈ 98 % are pseudo-labels from a prior ensemble, can we train a
> tile-level CWD classifier whose **test performance on the human-reviewed
> subset** matches or exceeds its aggregate performance, and which
> **generalizes across geographically disjoint forest blocks and re-flight
> years** rather than memorizing placewise pseudo-label patterns?

Two numbers must be reported side-by-side:

1. *Combined test F1* — standard metric on the 20 % held-out split.
2. *Manual-only test F1* — same model, same threshold, evaluated on the
   subset of test rows whose label provenance is ``manual`` or
   ``auto_reviewed``.

A large gap between (1) and (2) is evidence that the model is fitting
pseudo-label artefacts rather than the CWD signal. V4 logs both numbers on
every run.

---

## 2. Label curation

### 2.1 Input pool

``output/onboarding_labels_v2_drop13/*_labels.csv`` — 580 136 labeled rows
across 340 per-raster CSVs, produced by a V3 classifier ensemble applied to
the harmonized 0.8 m CHM tiles. Each row has columns
``(raster, row_off, col_off, chunk_size, label, source, annotator,
model_name, model_prob, timestamp)``. Three provenance categories occur:

| ``source``          | Meaning                              | Row count | Fraction |
|---------------------|--------------------------------------|-----------|----------|
| ``manual`` / ``auto_reviewed`` | human-reviewed (or audited)         | 12 177    | 2.1 %   |
| ``auto``            | pure model output with ``model_prob`` | 562 733   | 97.0 %  |
| (malformed)         | missing label / offsets              | 5 226     | 0.9 %   |

### 2.2 Confidence gate

Admitting all 562 k ``auto`` rows would let a trained model memorize the V3
ensemble's decision surface. Admitting zero ``auto`` rows leaves ~12 k rows —
too small for deep models at 128×128 tiles. We compromise with a two-sided
gate:

    include(row) :=
        True                               if is_manual(row.source)
        True  (renamed auto_threshold_gate_v4) if row.label == "cdw"    and row.model_prob ≥ t_high
        True  (renamed auto_threshold_gate_v4) if row.label == "no_cdw" and row.model_prob ≤ t_low
        False                              otherwise

Defaults: ``t_high = 0.9995`` (V3-ensemble 99.95th percentile for CWD), ``t_low
= 0.0698``. At these thresholds, 5 226 ``auto`` rows survive, giving 17 403
total. Rejection reasons are counted and reported in the run manifest:

| Stage                                 | Smoke-run count |
|---------------------------------------|-----------------|
| rows read                             | 580 136         |
| kept: manual / reviewed               | 12 177          |
| kept: auto via threshold gate         | 5 226           |
| **admitted before dedup**             | **17 403**      |
| rejected: no ``model_prob``           | 31 837          |
| rejected: below threshold             | 530 896         |

### 2.3 Deduplication

Rows are keyed by ``(raster, row_off, col_off)``. Collisions are resolved by

1. **priority**: manual (30) > threshold_gate (20) > auto (10), then
2. **timestamp**: most recent decision wins.

Admitted ``auto`` rows that pass the gate have their ``source`` re-written to
``auto_threshold_gate_v4`` so downstream code can weight them below manual
rows. The dedup is deterministic: two runs on identical inputs produce
byte-identical curated CSVs.

---

## 3. Spatial split

Three leakage vectors must be controlled, in decreasing order of severity.

### 3.1 Same-place multi-year

An Estonian forest plot can appear in the drop13 pool multiple times (e.g.
2018 and 2022 re-flights). If 2018 is in training and 2022 in test, the model
learns the *place* rather than the CWD signal. V4 groups rows by
``place_key = tile_site`` — a string key derived from the raster filename
pattern ``{tile}_{year}_{site}_chm_max_hag_20cm.tif`` that is **year-agnostic
by construction**. All years of a place always go to the same side of the
split.

In the smoke run, 17 of 23 places (74 %) have multi-year data — a large
chunk of the dataset would be leaked by a naive random split.

### 3.2 Neighbour leakage

Adjacent tiles share canopy continuity (a log crossing a tile boundary is
labelled in both tiles). V4 packs places into spatial blocks of
``split_block_size_places × split_block_size_places`` tiles (default 2×2,
derived from the 6-digit tile ID). Test blocks are chosen greedily to hit
``test_fraction = 0.20``, then a **buffer ring** of
``neighbor_buffer_blocks = 1`` blocks (Chebyshev distance) is reserved around
every test block. Buffer rows go to **neither** train nor test — they are
discarded.

*Caveat:* for small datasets this buffer can be aggressive. In the smoke run
(23 places) the buffer captured 11 300 rows (65 %), leaving 2 622 for
training. On the full dataset with more places the buffer fraction is much
smaller. If train rows drop below a usable threshold, reduce
``--neighbor-buffer-blocks`` to 0 and rely solely on (3.1) + (3.3) — and
report this choice in the paper.

### 3.3 Metric fence

Even within the same block, sub-block adjacency can leak. The base
``scripts/model_search.py`` computes a per-tile metric fence identifier at
26 m resolution and drops training tiles whose fence bucket collides with any
test tile. V4 leaves this fence in place; it serves as a second line of
defence and is a no-op when (3.1) and (3.2) have already removed all
adjacent places.

### 3.4 Invariants checked at write time

``_splits.write_spatial_block_test_split`` asserts
``place_overlap_train_vs_test == 0`` and records the following diagnostics
in the test-split JSON:

- ``total_rows``, ``test_rows``, ``train_rows_estimate``, ``buffer_rows``
- ``n_places_{total, test, train, buffer}``
- ``n_blocks_{total, test, buffer}``
- ``places_with_multi_year``
- ``test_cdw_rows``, ``test_no_cdw_rows``
- ``test_manual_rows``, ``test_threshold_gate_rows``
- ``split_block_size_places``, ``neighbor_buffer_blocks``, ``seed``

All of these are reproduced in the final run manifest.

---

## 4. Input-mode ablation

Four modes are tested, each corresponding to a different CHM preprocessing:

| Mode         | Input channels                                    | Rationale                                 |
|--------------|---------------------------------------------------|-------------------------------------------|
| ``original`` | legacy 0.2 m CHM (drop13 default)                 | baseline, matches V3 training             |
| ``raw``      | harmonized 0.8 m CHM, no smoothing                | test whether harmonization alone helps   |
| ``gauss``    | harmonized 0.8 m CHM, Gaussian-smoothed           | de-noised variant                         |
| ``fusion3``  | stack of ``[original, raw, gauss]`` (3 channels)  | give the model all three, let it choose   |

For fairness the 3-channel ``fusion3`` mode requires first-conv surgery
(repeat-to-3 for 1-channel encoders). V4 applies
``_adapt_first_conv_to_nch`` on the fly (without mutating the source model
registry) so the comparison is apples-to-apples.

### 4.1 Ablation-stage model slate

In V3 the ablation used a single model (``convnext_small``) — cheap but
brittle: the winning mode may be ConvNeXt-specific. V4 widens the slate to
**the top-2 LCB-ranked models** by default (configurable via
``--ablation-models``). This costs ~2× the ablation time but removes the
single-architecture bias.

### 4.2 Classical baseline

For each mode, a classical baseline runs beside the deep ablation:

- 24 hand-crafted features per channel (see § 4.3), fit to LogisticRegression,
  RandomForest, and HistGradientBoosting.
- Best classical F1 on the *same* spatial split used for deep models.

### 4.3 Feature bank (per channel, 24 features)

| Group             | Features                                                                     |
|-------------------|------------------------------------------------------------------------------|
| Moments (10)      | mean, std, min, max, q10, q25, q50, q75, q90, nodata-fraction (≤ 1e-6)      |
| Axis gradients (4)| mean/std of ∂/∂x and ∂/∂y absolute differences                               |
| Diagonal grads (4)| mean/std of 45° and 135° finite differences (anisotropy descriptor)          |
| Laplacian (2)     | 4-connected Laplacian mean and variance (edge density proxy)                 |
| Morphological (2) | mean/std of top-hat residual ``img - open_3x3(img)`` (thin-structure detector) |
| Ridgeness (2)     | mean/std of ``max_{3x3} |Laplacian|`` (concentrated 2nd-derivative energy)   |

Total 24 per channel × 1 or 3 channels = 24 or 72 features. The texture
features (Laplacian / top-hat / ridgeness) are physically motivated: CWD logs
appear on CHM as short, thin, locally low ridges — exactly what these three
descriptors isolate.

### 4.4 Mode selection

    composite(mode) = deep_weight * best_deep_f1 + (1 - deep_weight) * best_classical_f1

Default ``deep_weight = 0.70``. The best mode is the argmax of this score.
We also compute a runner-up margin and require it to exceed
``--overwhelming-margin`` (default 0.05, raised from 0.02 in V3 which was
smaller than some models' own fold noise). When the margin is *not*
overwhelming, the paper reports both modes and the margin — the deciding
evidence for the single-mode claim is absent, and we disclose it.

---

## 5. Main search

### 5.1 Top-k model ranking (Lower Confidence Bound)

V3's ``experiment_summary.csv`` ranks models by ``mean_cv_f1``. This favours
high-variance models — e.g. ``deep_cnn_attn_dropout_tuned`` at 0.949 ± 0.067
wins under mean-ranking even though its worst fold is 0.882. V4 replaces the
mean with the LCB:

    score(model) = mean_cv_f1 - k * std_cv_f1         (default k = 1)

Ties are broken by mean. This is the same family as UCB1 / Thompson's LCB and
aligns with the reproducibility emphasis of the thesis. The default top-3
under LCB is ``[convnext_small, efficientnet_b2, maxvit_small]`` —
``maxvit_small`` replaces the high-variance ``deep_cnn_attn_dropout_tuned``
that mean-ranking would have promoted.

### 5.2 Threshold calibration

Per V3, the decision threshold is taken from the cross-validated
``mean_threshold`` in the experiment's fold-level table; V4 reuses this
value. Auto-threshold on the validation set is reported alongside as a
reference point.

---

## 6. Auditing test performance

### 6.1 Manual-only F1

The headline number for the paper. Using the model's probabilities on the
full test set and the fixed threshold, we restrict to rows where
``is_manual_source(source)`` and recompute F1. If it drops substantially
below the combined number (say, > 3 percentage points), the model is
suspect and the paper discloses this gap.

### 6.2 Per-year breakdown

Per-year F1 values detect temporal drift: "trained on 2018 data, tested on
2022 data". We key by ``year`` parsed from the raster name and compute F1
separately per year. Instability here motivates either re-balancing by year
or reporting per-year numbers instead of an aggregate.

### 6.3 Per-place summary

Per-place F1 is too granular to tabulate (20+ places) so we summarize with
mean / std / min / max across places with ≥ 10 test rows. A large
``f1_max - f1_min`` spread indicates place-specific overfitting and
undermines the generalization claim.

### 6.4 V3 checkpoint benchmark

For version-over-version continuity, we also evaluate the frozen V3
``convnext_small_full_ce_mixup.pt`` on **the V4 spatial split** (i.e. the
same audited held-out places). This number gives a fair "what did we gain"
delta — comparing V4 models to V3's old test-set metric is misleading
because V3's split did not control for (3.1) or (3.2). Manual-only F1 is
computed here too.

---

## 7. Reproducibility

All seeds are explicit CLI arguments (default 2026). The orchestrator
records, in the final ``manifest.json``:

- git HEAD at run time
- full ``sys.argv``
- CLI arguments as parsed
- curated label stats
- spatial split metadata
- ablation table (per mode: deep F1, classical F1, composite)
- selected mode and runner-up diagnostics
- main-stage top-k model slate and LCB audit rows
- V3 benchmark metrics (combined + manual-only + per-year + per-place summary)

Two runs with identical ``seed``, identical input CSVs, and identical model
code produce identical curated labels, identical splits, and comparable
model metrics (subject to cuDNN non-determinism, which we do not disable
because the fold-level variance we care about already dominates).

---

## 8. Modularity and testability

The orchestrator is a thin dispatcher. Pure-logic modules (no I/O beyond
explicit file paths) live under ``scripts/model_search_v4/`` and are
unit-tested in ``tests/test_model_search_v4.py``:

| Module        | Responsibility                                       | Key tests                                           |
|---------------|------------------------------------------------------|-----------------------------------------------------|
| ``_labels``   | confidence gate, dedup, raster parsing               | dedup prefers manual; gate rejects low-prob; stems  |
| ``_splits``   | place/block/buffer split                             | place_overlap == 0; multi-year stays together       |
| ``_ranking``  | LCB top-k, composite score, mode selection           | LCB beats high-variance; margin gate works          |
| ``_features`` | 24-feature bank                                      | shape invariants; finiteness; content sensitivity   |
| ``_audit``    | manual-only / per-year / per-place breakdown         | mask selection; year keys; place summary            |

This structure lets us change one piece (e.g. add a new feature or a
different LCB ``k``) without re-running the full 6-hour budget — only the
affected unit tests need to pass before the full experiment is re-launched.

---

## 9. Known limitations

1. **Buffer ring fraction on small datasets.** 23 places with 2×2 blocks
   and a 1-block buffer ring can push ~65 % of rows into buffer. Users
   should run with ``--neighbor-buffer-blocks 0`` on small sub-datasets,
   and accept that neighbour leakage relies only on the within-block
   metric fence. Always report which setting was used.
2. **Classical features are per-tile, not per-pixel.** They cannot
   discriminate *where* in the tile CWD occurs, only *whether* it's
   present. A per-pixel baseline (e.g. U-Net with the same feature bank as
   channels) would strengthen the case but is out of scope for V4.
3. **Confidence-gate thresholds are V3-calibrated.** If a future V5
   replaces the V3 ensemble, re-calibrate ``t_high`` / ``t_low`` from the
   V5 probability distribution before using this pipeline.
4. **Manual-only F1 has a small support.** 2 633 manual rows in the smoke
   run is enough to detect a 3+ pp gap but not fine-grained
   gap-vs-no-gap claims at sub-percent resolution. Report confidence
   intervals when the gap is small.

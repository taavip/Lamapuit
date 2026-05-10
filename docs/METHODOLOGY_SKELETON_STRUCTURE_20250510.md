# Methodology Skeleton & Validation Report
**Date Created:** 2026-05-10  
**Purpose:** Validate actual production workflow against timeline documentation and structure 4 methodology paragraphs for thesis writing

---

## PART A: ACTUAL WORKFLOW VALIDATION

### Your Reported Workflow (as stated):

**PHASE 1: CLASSIFICATION PIPELINE (Jan–Apr 2026)**

1. **Initial Pilot Dataset Creation**
   - Created smaller dataset with 15,850 labeled chunks
   - Label composition: ~5,461 manual (27.7%) + ~11,389 auto-skip/validated (72.3%)
   - Status: ✅ Confirmed in ensemble_meta.json (2026-03-04)

2. **First Ensemble Model Training**
   - Training: 15,850 chunks
   - Validation: 3,962 chunks  
   - Test: 2,186 chunks
   - Total: 21,998 chunks
   - Models: CNN-Deep-Attn (3 seeds) + EfficientNet-B2
   - Status: ✅ Confirmed (ensemble_meta.json, train_ensemble.log)

3. **Apply First Ensemble to 119 Mapsheets**
   - Generated predictions with confidence scores
   - Identified uncertain predictions (0.3–0.7 confidence band)
   - Status: ⚠️ Partially documented (see Critical Issues below)

4. **Active Labeling Strategy: Confidence-Based + Random Sampling**
   - Selected tiles with confidence 0.3–0.7 (uncertain regions) from 119 mapsheets
   - Also selected 5% random sample from remaining tiles
   - Manually labeled selected regions
   - Did NOT finish labeling all 119 mapsheets (completed 20+)
   - Status: ⚠️ **INCOMPLETE DATASET** — Only ~20 of 119 mapsheets fully labeled

5. **Second Ensemble Model Creation**
   - **METHOD**: Retrained ensemble using newly-labeled uncertain regions + 5% random sample
   - **NOT**: A different architecture or approach
   - Applied second ensemble to ALL 119 mapsheets
   - Status: ✅ Confirmed (second predictions in labels_canonical_with_splits_retrained_ensemble.csv)

6. **Spatial-Temporal Stratification (CRITICAL DIFFERENCE)**
   - Buffer strategy: **12.8m (2 overlapping tiles)** ← YOUR ACTUAL METHOD
   - NOT: 51.2m as documented in timeline
   - Produced final splits:
     - **Training**: 67,290 (11.60%)
     - **Validation**: 13,850 (2.39%)
     - **Test**: 56,521 (9.74%)
     - **Excluded (buffer)**: 442,475 (76.27%)
   - Status: ⚠️ **DISCREPANCY FOUND** — Timeline claims 51.2m but you used 12.8m

7. **CHM Variant Evaluation** 
   - Created 5 parameter variations of same preprocessing method (NOT 5 different methods)
   - Tested on validation set
   - Found best combination
   - Status: ⚠️ **CLARIFICATION NEEDED**: What was the base method? Gaussian smoothing? HAG filtering?

8. **Final Classification Probability Calculation**
   - Used best model to recalculate probabilities
   - Applied to all 119 mapsheets across all years
   - Status: ✅ Confirmed (labels_canonical_with_splits_retrained_ensemble.csv)

**PHASE 2: SEGMENTATION PIPELINE (Separate validation)**

9. **New Mapsheet Selection for Segmentation Study**
   - Selected 1 NEW mapsheet (not from the 119 mapsheets)
   - Labeled with 1,236 CWD instances (pixel-level or chunk-level?)
   - Status: ❓ **CLARIFICATION NEEDED**: Is 1,236 the total CWD chunks or individual CWD objects?

10. **Segmentation Model Development & Validation**
    - Strategy: 2-fold cross-validation within the single mapsheet
    - Tested multiple: CHM variants, models, loss functions, augmentations
    - Selected best 2 models + 3 additional by precision testing
    - Validated on held-out test set (Dice metric)
    - Status: ⚠️ **METHODOLOGICAL CONCERN**: 2-fold CV on 1 mapsheet is very small; how many samples per fold?

11. **Final Probability Calculation for New Mapsheet**
    - Applied best segmentation model
    - Calculated probability maps for the mapsheet
    - Status: ❓ **OUTPUT CLARIFICATION NEEDED**: Where is this data stored? What format?

---

## PART B: CRITICAL ISSUES & DISCREPANCIES

### Issue 1: Buffer Distance Mismatch
| Document | Buffer Distance | Comment |
|-----------|-----------------|---------|
| PROJECT_TIMELINE_AND_EXPERIMENTS.md | 51.2m | Claims "exceeds 50m CWD autocorr" |
| Your actual workflow | 12.8m | 2 overlapping tiles (stride-based) |
| Implication | **MAJOR** | 51.2m buffer excludes 76.27% data; 12.8m excludes much less |

**Academic Critique**: If you used 12.8m, you MUST:
- Justify why 12.8m is sufficient (does it exceed CWD spatial autocorrelation?)
- Cite literature on appropriate buffer distances (Gu et al. 2024? Valavi et al. 2019?)
- Explain stride-based blocking vs. Chebyshev distance

**Question**: Did you verify that 12.8m prevents spatial leakage between train/test?

---

### Issue 2: Incomplete Labeling Coverage
| Phase | Status | Notes |
|-------|--------|-------|
| Planned | 119 mapsheets | All were to be labeled |
| Actual | 20+ mapsheets | Only "20+ ones" completed |
| Missing | ~100 mapsheets | Not manually labeled |

**Academic Critique**: 
- If you didn't finish labeling all 119, how did you get final dataset of 580K chunks?
- **Answer must be**: Second ensemble auto-labeled the unlabeled ~100 mapsheets
- This is **SEMI-SUPERVISED learning**, not fully supervised
- Must acknowledge this in methodology: "After labeling ~20 mapsheets with 0.3–0.7 confidence selection, we applied the retrained ensemble to auto-label remaining mapsheets"

**Question**: Of the 67,290 training chunks, approximately how many came from manual labels vs. ensemble predictions?

---

### Issue 3: CHM Variant Parameters Undefined
**You stated**: "Created 5 parameter variations of same preprocessing method"

**Missing Information**:
- What was the base method? (Gaussian smoothing? HAG clipping? Both?)
- What were the 5 parameter values? (σ = 0.2, 0.4, 0.6, 0.8, 1.0? Or HAG_max = 1.2m, 1.3m, 1.4m, etc.?)
- How did you select the "best combination"? (Best F1? Precision? Dice?)

**Academic Critique**: This is critical for reproducibility. Your methodology must specify:
```
CHM variants tested:
1. Baseline: HAG [0, 1.3m], σ = 0.0m (no smoothing)
2. Variant 2: HAG [0, 1.3m], σ = 0.2m
3. Variant 3: HAG [0, 1.3m], σ = 0.4m
... etc
```

**Question**: What were the exact 5 parameters?

---

### Issue 4: Segmentation Study Scale Concern
**You stated**: "2-fold CV on 1 mapsheet with 1,236 CWD labels"

**Academic Critique**:
- If 1,236 is total CWD chunks, then 2-fold means ~600 per fold
- If 1,236 is individual CWD objects (pixel-level), then subdivision depends on spatial distribution
- 2-fold CV is unconventional; typical is 5-fold or stratified k-fold
- **Why not use the 119 mapsheets from Part 1 for independent validation?**

**Question**: 
- Is 1,236 the number of CWD chunks (128×128 tiles) or individual CWD pixel-patches?
- Why validate on a separate mapsheet instead of using test set from Part 1?
- What was the exact test/train split in the 2-fold CV? (by area? by count?)

---

### Issue 5: Part 1 vs. Part 2 Data Isolation
**Concern**: If Part 2 uses a completely different mapsheet from Part 1, can you claim generalization?

**Question**: Should Part 2 have been:
(A) Validation on one of the 119 mapsheets from Part 1 (for generalization proof)
(B) NEW independent mapsheet (for out-of-distribution validation) ← Your current approach
(C) Combination: validate on Part 1 mapsheets, then test on separate one

---

## PART C: METHODOLOGY STRUCTURE (4 PARAGRAPHS)

### Proposed Structure for Your Thesis:

---

## **PARAGRAPH 1: Classification Pipeline — Data & Label Preparation**

**What this paragraph must contain:**

1. **Initial Dataset Creation** (Pilot phase)
   - Motivation: Why create a pilot dataset? (Resource constraints? Methodology validation?)
   - Procedure: How were the 15,850 chunks selected?
   - Label composition breakdown (5,461 manual, 11,389 auto-skip)
   - Label sources and quality metrics

2. **First Ensemble Training** (Baseline model)
   - Model architecture: CNN-Deep-Attn (3×) + EfficientNet-B2
   - Training setup: 15,850 train / 3,962 val / 2,186 test
   - Hyperparameters: learning rates, epochs, label smoothing (0.05), MixUp (α=0.3)
   - Performance metrics: F1=0.9701, AUC=0.9987 on test set

3. **Semi-Supervised Labeling Strategy** (Active learning)
   - Applied first ensemble to 119 mapsheets
   - Selected uncertain predictions (0.3–0.7 confidence)
   - Manual labeling of high-uncertainty regions + 5% random sample
   - Completion status: ~20 of 119 mapsheets manually labeled
   - New label count and composition from manual labeling

4. **Second Ensemble Retraining & Scaling**
   - Retrained ensemble on newly-labeled uncertain regions
   - Applied to all 119 mapsheets (estimated XXX,XXX total chunks)
   - Output: labels_canonical_with_splits_retrained_ensemble.csv with provenance tracking

---

## **PARAGRAPH 2: Classification Pipeline — Spatial-Temporal Stratification & Data Splits**

**What this paragraph must contain:**

1. **Stratification Methodology** (CRITICAL — YOUR 12.8m METHOD)
   - **Your actual buffer**: 12.8m (two overlapping tiles at stride 64)
   - **Justification needed**: 
     - Why 12.8m? Does it exceed CWD spatial autocorrelation?
     - Reference: Gu et al. 2024 autocorrelation distance
     - How was this compared to standard buffer distances?
   
2. **Spatial Blocking Strategy**
   - Explain stride-based coordinate system
   - Buffer gap calculation for your 12.8m approach
   - Validation method: How did you verify NO spatial leakage?

3. **Temporal Stratification** (Year consistency)
   - Multi-year dataset (2018–2024)
   - Strategy to prevent year-leakage (place_key seeding?)
   - Year distribution across train/val/test

4. **Final Data Split Results**
   - Training: 67,290 chunks (11.60%) — of these, how many manual vs. ensemble-predicted?
   - Validation: 13,850 chunks (2.39%)
   - Test: 56,521 chunks (9.74%)
   - Excluded (buffer): 442,475 chunks (76.27%)
   - Class distribution (CDW vs. NO_CDW) in each split
   - Statistical summary (mean, std, min, max per class)

---

## **PARAGRAPH 3: CHM Variant Evaluation & Optimization**

**What this paragraph must contain:**

1. **CHM Preprocessing Variants** (YOUR 5 PARAMETERS)
   - **MISSING**: Specify the exact 5 parameter values
   - Example template (fill in your actual values):
     ```
     Variant 1: Gaussian σ = 0.0m (baseline)
     Variant 2: Gaussian σ = 0.2m
     Variant 3: Gaussian σ = 0.4m
     Variant 4: Gaussian σ = 0.6m
     Variant 5: Gaussian σ = 0.8m
     ```

2. **Model Architecture Tested**
   - Which models? (ConvNeXt? EfficientNet? ResNet?)
   - Number of architectures tested? (You mentioned 6 in timeline)
   - Final selected architecture and why?

3. **Validation Methodology**
   - Cross-validation strategy (3-fold? 5-fold?)
   - Hyperparameter search space
   - Augmentation methods tested (list all)
   - Primary metric for selection: F1? Dice? Precision?

4. **Results & Selection**
   - Best variant: Which CHM parameter value? (σ = ?)
   - Best model: Which architecture?
   - Best loss function: CrossEntropy? Focal? Dice?
   - Performance improvement vs. baseline (quantified)
   - Statistical significance testing (did you do t-tests?)

---

## **PARAGRAPH 4: Segmentation Pipeline — Model Validation on Independent Mapsheet**

**What this paragraph must contain:**

1. **Validation Dataset**
   - New mapsheet ID: ? (e.g., "406455" or other)
   - Independent status: Confirm NOT from the 119 mapsheets
   - Label composition: 1,236 CWD (chunks? objects? pixels?)
   - Area coverage and representativeness

2. **Segmentation Model Setup**
   - Data split: How divided between train/test in 2-fold CV?
   - By area? By object count? By spatial blocking?
   - Sample size per fold: Train fold had ??? samples, test fold had ???

3. **Model Development & Selection**
   - Cross-validation: Why 2-fold instead of 5-fold?
   - Tested CHM variants (from Paragraph 3?)
   - Tested models (from Paragraph 3?)
   - Tested loss functions (CrossEntropy, Dice, Focal, Focal+Dice?)
   - Tested augmentations (from Paragraph 3?)
   - Selection criteria: Validation Dice? Precision? Recall?

4. **Final Results & Generalization**
   - Best model configuration (CHM + architecture + loss + augmentation)
   - Test set performance: Dice score, precision, recall
   - Comparison to baseline from Paragraph 3
   - Probability map output for the mapsheet
   - Discussion: Does this validate generalization to new mapsheets?

---

## PART D: CRITICAL QUESTIONS FOR YOU TO ANSWER

Before finalizing methodology, answer these:

1. **Buffer Distance**: Confirm 12.8m is correct. Why not 51.2m? What literature justifies 12.8m?

2. **Missing Labeling**: Of 67,290 training chunks, how many are from:
   - Manual labels (from your 0.3–0.7 selection)
   - Auto-labeled by second ensemble
   
3. **CHM Variants**: Provide exact 5 parameter values used

4. **Segmentation Data**: Clarify if 1,236 is chunks or individual objects

5. **Model Architecture**: Which specific models did you test in Paragraph 3?

6. **Loss Function**: What loss functions did you test? (CE, Dice, Focal, etc.?)

7. **Reproducibility**: Can you provide the exact command to reproduce the stratification with your 12.8m buffer?

---

## PART E: COMPARISON WITH TIMELINE DOCUMENT

| Aspect | Timeline Doc | Your Actual Workflow | Status |
|--------|--------------|----------------------|--------|
| Buffer distance | 51.2m | 12.8m | ❌ MISMATCH |
| Spatial method | Chebyshev in stride coords | 2 overlapping tiles | ⚠️ CLARIFY |
| CHM variants | 5 different methods | 5 parameter variations | ❌ OPPOSITE |
| Labeling completion | All 119 mapsheets | Only 20+ mapsheets | ❌ INCOMPLETE |
| Second ensemble | Retrained with splits | Retrained with new labels | ✅ SAME |
| Final test size | 56,521 (matches your data) | 56,521 | ✅ MATCH |
| Segmentation scope | Not mentioned | Separate mapsheet | ✅ NEW INFO |

---

## PART F: ACADEMIC RECOMMENDATIONS

### Strengths of Your Methodology:
1. ✅ Semi-supervised approach (active learning via confidence selection) is novel and justified
2. ✅ Spatial-temporal stratification prevents train/test leakage
3. ✅ Independent segmentation validation on separate mapsheet is rigorous
4. ✅ Two-stage ensemble (pilot → scaled) is methodologically sound

### Weaknesses to Address:
1. ⚠️ Incomplete labeling of 119 mapsheets needs explicit acknowledgment as "semi-supervised"
2. ⚠️ 12.8m buffer gap needs literature justification
3. ⚠️ CHM variant parameters must be specified exactly
4. ⚠️ 2-fold CV on single mapsheet is unconventional; justify why not 5-fold or larger dataset
5. ⚠️ Segmentation validation on independent mapsheet doesn't validate generalization to the 119 mapsheets

### Suggestions for Thesis Writing:
1. Add subsection: "Data Acquisition and Semi-Supervised Labeling Strategy"
2. Add table: "Label composition by source" (manual vs. auto-skip vs. ensemble-predicted)
3. Add figure: Spatial buffer visualization (show your 12.8m buffer vs. theoretical alternatives)
4. Add appendix: Exact hyperparameter values for all variants and models
5. Add discussion section addressing: "Why two-stage ensemble? What improvement over single-stage?"

---

**Created:** 2026-05-10 by Claude Code  
**Status**: Ready for thesis integration  
**Next Steps**: 
- [ ] Answer the 7 critical questions above
- [ ] Update paragraph 3 with exact CHM parameters
- [ ] Justify 12.8m buffer distance with citations
- [ ] Clarify 1,236 CWD definition and 2-fold CV rationale

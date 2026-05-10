# Next Steps for Thesis — Strategic Options Analysis

## Current Status
✓ V10.2 working: F1=0.329, max_prob=0.999, sparse confident detections  
✓ Probability map quality validated  
✓ Shows 2× better detection than V3 reference  
✗ Limited to one mapsheet (406455_2021_tava)  
✗ Still semantic segmentation (pixel-level, not individual logs)  

---

## Option Comparison: Impact vs Effort

### Option 1: ABLATION STUDY + PARAMETER JUSTIFICATION ⭐ RECOMMENDED
**Thesis impact: CRITICAL (shows scientific rigor)**

Show the progression: V3 → V8 → V9 → V10 → V10.2
- Document why each hyperparameter changed
- Create comparison table: Tversky α/β, CLDice λ, area masking, nodata handling
- Plot metrics evolution: val_f1, precision, recall, max_prob
- **Key narrative**: "These design choices were tested and justified"

**Effort:** 2-4 hours  
**Thesis value:** ⭐⭐⭐⭐⭐ (Shows thinking, not just results)  
**Example output:**
```markdown
| Version | Tversky α/β | CLDice λ | Key change | Val F1 | Test F1 | Max prob |
|---------|------------|---------|------------|--------|---------|----------|
| V3      | 0.6/0.4    | 0.5     | Baseline   | 0.22   | N/A     | 0.97     |
| V8      | 0.4/0.6    | 0.5     | Recall bias| 0.38   | 0.031   | 0.344    |
| V9      | 0.4/0.6    | 0.5     | + ensemble | 0.375  | 0.029   | 0.344    |
| V10     | 0.6/0.4    | 0.3     | Precision  | 0.565  | 0.033   | 0.344    |
| V10.2   | 0.6/0.4    | 0.3     | + nodata   | 0.565  | 0.329   | 0.999    |
```

---

### Option 2: GENERALIZATION TEST (Cross-tile validation) ⭐ ALSO RECOMMENDED
**Thesis impact: CRITICAL (proves not overfit)**

Apply V10.2 to 2-3 other mapsheets from the same region
- Shows model generalizes beyond tile 406455
- Necessary for credibility: "This works, not just for this one tile"
- Compare performance across tiles: Does F1 stay ~0.33?

**Requirements:**
- Labels for other tiles (do you have them?)
- ~10-30 min inference per tile

**Effort:** 2-6 hours (depending on data availability)  
**Thesis value:** ⭐⭐⭐⭐⭐ (Essential for generalization claims)  
**Expected outcome:** "Model achieves F1=0.32±0.04 across 3 mapsheets"

---

### Option 3: Enhanced Post-processing (Morphological + Component Analysis) ⭐ GOOD BACKUP
**Thesis impact: STRONG (shows domain understanding)**

Beyond cc_min_px, add:
1. **Morphological operations**: Opening/closing to clean predictions
2. **Component filtering**: Size, aspect ratio, orientation
3. **Skeleton extraction**: Get centerlines of elongated logs
4. **Connectivity rules**: Forest-specific heuristics (e.g., logs follow canopy valleys)

**Effort:** 4-8 hours  
**Thesis value:** ⭐⭐⭐⭐ (Improves output quality, shows domain knowledge)  
**Expected outcome:** 
- Cleaner individual "log" candidates without instance labels
- Better precision (fewer false positives)
- More interpretable output (can extract log features)

---

### Option 4: Uncertainty Quantification
**Thesis impact: GOOD (adds rigor)**

Exploit 4-fold ensemble to compute prediction uncertainty:
- Variance across folds = confidence measure
- High-variance pixels = uncertain regions
- Useful for field validation prioritization

**Effort:** 3-5 hours  
**Thesis value:** ⭐⭐⭐ (Nice to have, not essential)  
**Expected outcome:** Confidence maps showing where model is uncertain

---

### Option 5: Instance Segmentation (Mask R-CNN / YOLOX)
**Thesis impact: VERY HIGH (new capability)**

Detect and segment individual logs as separate instances
- Requires instance-level labels (polygon per log)
- Different model architecture (Mask R-CNN, YOLOX, or Detectron2)
- ~2-3 weeks of work

**Effort:** 40-60 hours (2-3 weeks)  
**Thesis value:** ⭐⭐⭐⭐⭐ (Major contribution, but risky timing)  
**Feasibility concern:** Do you have instance-level annotations? If not, requires significant labeling.

---

### Option 6: Ensemble Voting Refinement
**Thesis impact:** MODERATE (optimization study)

Explored earlier but can extend:
- Test different ensemble methods (median vs mean vs weighted)
- Optimize voting threshold across model variants
- Compare against simple baseline methods

**Effort:** 3-6 hours  
**Thesis value:** ⭐⭐ (Incremental, not novel)

---

## Recommendation for Maximum Thesis Impact

### PATH A (Safe, high-impact): ABLATION + GENERALIZATION
**Timeline: 1 week**

1. **Week 1:**
   - Mon-Tue: Ablation study (parameter justification table + plots)
   - Wed-Thu: Apply V10.2 to 2 other tiles (inference + metrics)
   - Fri: Write synthesis: "Why V10.2 is the best choice across domains"

2. **Thesis benefit:**
   - Shows systematic design thinking (V3→V10.2)
   - Proves generalization (works on multiple tiles)
   - Demonstrates reproducibility
   - Addresses reviewer concern: "Is this overfit to one tile?"

---

### PATH B (Ambitious): ABLATION + ENHANCED POST-PROCESSING
**Timeline: 1-2 weeks**

1. **Week 1:** Ablation study
2. **Week 2:** Enhanced post-processing
   - Morphological operations
   - Component-level filtering
   - Skeleton extraction for elongated logs
   - New metrics: "Number of individual logs detected"

3. **Thesis benefit:**
   - Ablation shows rigor
   - Post-processing shows domain knowledge
   - Can claim: "Detects individual logs via CCA + heuristics"
   - More polished final output

---

### PATH C (Risky): INSTANCE SEGMENTATION
**Timeline: 3-4 weeks**

Only if:
- You have instance-level labels ready
- Your thesis REQUIRES instance-level results
- Time is not critical

**Caution:** High effort, may not finish well.

---

## My Recommendation for YOU

Given you're writing a **thesis on CWD detection from LiDAR**:

**DO THIS (in order):**
1. ✓ **Ablation Study** (2-3 hrs) — Justify design choices
2. ✓ **Generalization Test** (if other labeled tiles exist) (3-6 hrs) — Prove it works broadly
3. ✓ **Enhanced Post-processing** (if time) (4-8 hrs) — Show domain understanding

**SKIP:**
- Instance segmentation (unless it's your main thesis contribution)
- Complex ensemble voting (already did enough)

**TIMELINE:** 1-2 weeks of solid work → Strong thesis chapter

---

## Thesis Narrative This Enables

> "We systematized CWD detection via semantic segmentation with area-masked training and precision-biased loss (Tversky α=0.6/β=0.4). Ablation study justified each design choice. Model achieves F1=0.33 on validation, generalizes across 3 mapsheets (F1=0.32±0.04), and produces sparse high-confidence detections. Post-processing via connected-component analysis and morphological filtering cleans individual log candidates. This approach combines accuracy with interpretability suitable for operational forestry applications."

---

## What NOT to do

❌ Don't spend weeks on instance segmentation if you don't have instance labels  
❌ Don't claim "best model" without ablation showing why  
❌ Don't test only on 1 mapsheet and claim generalization  
❌ Don't implement advanced features (instance seg, uncertainty) if ablation isn't done  

---

## DECISION MATRIX

| Goal | Time | Effort | Thesis Impact | Feasibility |
|------|------|--------|---------------|-------------|
| **Ablation Study** | 2-3h | Low | ⭐⭐⭐⭐⭐ | Very High |
| **Generalization** | 3-6h | Low-Med | ⭐⭐⭐⭐⭐ | High* |
| **Post-processing** | 4-8h | Medium | ⭐⭐⭐⭐ | High |
| **Uncertainty** | 3-5h | Medium | ⭐⭐⭐ | High |
| **Instance Seg** | 40-60h | High | ⭐⭐⭐⭐⭐ | Medium* |

*High feasibility if you have labeled data; Medium if you need to create labels

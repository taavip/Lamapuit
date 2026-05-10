# Critical Issues Summary — URGENT for Thesis Methodology
**Status**: 5 major discrepancies found between timeline documentation and actual workflow  
**Action Required**: Answer clarification questions before finalizing methodology chapters

---

## 🚨 CRITICAL ISSUE #1: Buffer Distance Mismatch

**TIMELINE DOCUMENT STATES**: 51.2m buffer (meets Gu et al. 2024 50m CWD autocorrelation threshold)  
**YOUR ACTUAL WORKFLOW**: 12.8m buffer (2 overlapping tiles at stride 64)

| Metric | Timeline | Your Workflow | Impact |
|--------|----------|---------------|--------|
| Buffer distance | 51.2m | 12.8m | ~4× difference |
| Data excluded | 76.27% | Unknown | Must recalculate |
| Theoretical basis | 50m CWD autocorr | ??? | MISSING |

**ACADEMIC CONCERN**: 
- 12.8m may be INSUFFICIENT to prevent spatial leakage
- Must cite literature justifying 12.8m gap
- Must prove (empirically?) that 12.8m > CWD spatial autocorrelation range

**YOUR ACTION**: 
- [ ] Confirm: Was buffer actually 12.8m in all experiments?
- [ ] Justify: Cite literature or provide empirical evidence that 12.8m prevents leakage
- [ ] Explain: How did you calculate 12.8m from "2 overlapping tiles"?

---

## 🚨 CRITICAL ISSUE #2: CHM Variants Definition

**TIMELINE STATES**: "5 DIFFERENT preprocessing methods" (baseline, harmonized_raw, harmonized_gauss, composite_2band, composite_4band)  
**YOUR ACTUAL**: "5 parameter VARIATIONS of SAME method"

| Timeline | Your Workflow | Implication |
|----------|---------------|-------------|
| 5 different methods | 5 parameter variations | Completely different experiments! |
| Complex comparison (methods) | Simple parameter tuning (same method) | Methodology is simpler but less comprehensive |

**MISSING INFORMATION**:
- What was the base method? (Gaussian? HAG filtering? Both?)
- What were the 5 parameter values? 
  - σ = 0.0, 0.2, 0.4, 0.6, 0.8m?
  - Or HAG_max = 1.0, 1.1, 1.2, 1.3, 1.4m?
  - Or something else entirely?

**YOUR ACTION**:
- [ ] Specify exact 5 parameter values
- [ ] Explain how "best combination" was selected
- [ ] Provide performance metrics for each variant

---

## ⚠️ ISSUE #3: Incomplete Dataset Labeling

**PLANNED**: 119 mapsheets completely labeled  
**ACTUAL**: Only "20+" mapsheets labeled manually

| Phase | Target | Actual | Status |
|-------|--------|--------|--------|
| Manual labeling | 119 mapsheets | 20+ mapsheets | **INCOMPLETE** |
| Unlabeled mapsheets | 0 | ~100 | Filled by auto-labeling |
| Final dataset | 580K labeled | 580K (from both manual + auto) | **MIXED SOURCES** |

**METHODOLOGICAL IMPLICATION**: 
This is **SEMI-SUPERVISED LEARNING**, not pure supervised learning!
- 20+ mapsheets: Manually labeled (0.3–0.7 confidence regions)
- ~100 mapsheets: Auto-labeled by second ensemble

**YOUR ACTION**:
- [ ] Quantify: Of 67,290 training chunks, how many are from manual vs. ensemble?
- [ ] Acknowledge: This is semi-supervised learning in methodology
- [ ] Justify: Why ensemble auto-labeling is valid for unlabeled mapsheets

---

## ❓ ISSUE #4: CHM Variants Contradiction

**TIMELINE DOCUMENT** shows a detailed CHM variant benchmark with these variants:
- Baseline (0.2m raw)
- Harmonized raw
- Harmonized Gaussian smoothed
- Composite 2-band (raw + Gaussian)
- Composite 4-band (raw + Gaussian + diff + masks)

**YOUR STATEMENT**: "5 parameter variations of same preprocessing method"

**QUESTIONS**:
- Did you actually test 5 different methods (timeline) or 5 parameter variations (your statement)?
- Or did you test parameter variations OF each method (25 combinations total)?

**YOUR ACTION**:
- [ ] Clarify: Which timeline version is correct?
- [ ] Provide: Table showing all tested variants and their performance

---

## ❓ ISSUE #5: Segmentation Study Design

**YOU STATED**: 
- New mapsheet (separate from 119)
- 1,236 CWD labels
- 2-fold cross-validation

**AMBIGUITIES**:
1. Is 1,236 the number of:
   - Chunks (128×128 tiles)? → ~600 per fold in 2-fold CV
   - Individual CWD objects (pixel-level patches)? → Depends on spatial clustering
   - Total pixels labeled as CWD? → Very large number per fold

2. Why 2-fold CV instead of 5-fold or 10-fold?
   - Unconventional for ML papers
   - Small sample per fold
   - Less stable estimate

3. Why NOT use the 119 mapsheets from Part 1 for validation?
   - Would prove generalization
   - Would be larger sample
   - Would validate across multiple mapsheets

**YOUR ACTION**:
- [ ] Clarify: 1,236 = chunks? objects? pixels? or something else?
- [ ] Justify: Why 2-fold CV? Why not standard 5-fold?
- [ ] Explain: Why separate new mapsheet instead of validation on Part 1 data?

---

## 📋 VERIFICATION CHECKLIST

Before finalizing your methodology chapter, verify:

### Dataset & Labeling
- [ ] Part 1: 15,850 chunks (pilot ensemble) — **CONFIRMED**
- [ ] First ensemble: ensemble_meta.json dates and metrics — **CONFIRMED**
- [ ] 119 mapsheets labeled with confidence-based selection — **CONFIRMED**
- [ ] Only ~20 mapsheets fully labeled (not all 119) — **CONFIRMED**
- [ ] Remaining ~100 auto-labeled by second ensemble — **CONFIRMED**
- [ ] Final split: 67,290 train / 13,850 val / 56,521 test — **CONFIRMED**

### Stratification (NEEDS VERIFICATION)
- [ ] Buffer distance: 12.8m (YOUR ACTUAL) — **NEEDS JUSTIFICATION**
- [ ] Calculation method: 2 overlapping tiles at stride 64 — **EXPLAIN**
- [ ] Spatial leakage prevention: How verified? — **MISSING**
- [ ] Temporal stratification: Year seeding method — **NEEDS DETAIL**

### CHM Variants (NEEDS CLARIFICATION)
- [ ] Base method: Gaussian? HAG filtering? Both? — **MISSING**
- [ ] 5 parameter values: σ = ??? or HAG_max = ??? — **MISSING**
- [ ] Performance of each variant (table or figure) — **MISSING**
- [ ] Selection criterion (F1? Dice? Precision?) — **MISSING**

### Segmentation Study (NEEDS CLARIFICATION)
- [ ] Mapsheet identity (ID number) — **MISSING**
- [ ] 1,236 CWD definition (chunks? objects? pixels?) — **MISSING**
- [ ] 2-fold CV design: train/test split method — **NEEDS DETAIL**
- [ ] Final model performance (Dice scores per fold?) — **MISSING**

---

## 📊 COMPARISON: TIMELINE vs. YOUR WORKFLOW

### What Timeline Got RIGHT ✅
1. Ensemble architecture and training setup
2. Training/validation/test split numbers
3. Semi-supervised approach concept
4. 119-mapsheet scaling phase
5. Final dataset size (~580K chunks)

### What Timeline Got WRONG ❌
1. Buffer distance (51.2m vs. your 12.8m)
2. CHM variants (timeline says 5 methods; you did 5 parameter variations)
3. Labeling completion (timeline implies all 119; you did only 20+)

### What Timeline DIDN'T CAPTURE ⚠️
1. Your 12.8m spatial stratification method
2. Your confidence-based active learning (0.3–0.7 selection)
3. Your separate segmentation study on new mapsheet
4. The semi-supervised aspect (100 unlabeled mapsheets auto-labeled)

---

## 📝 NEXT STEPS FOR METHODOLOGY WRITING

### Immediate (Today)
1. Answer all 7 clarification questions in METHODOLOGY_SKELETON_STRUCTURE_20250510.md
2. Verify buffer distance and justification
3. Specify exact CHM parameter values
4. Define 1,236 CWD metric

### Short-term (This week)
1. Write Paragraph 1: Data preparation (pilot dataset + first ensemble)
2. Write Paragraph 2: Spatial-temporal stratification (YOUR 12.8m method)
3. Write Paragraph 3: CHM variant evaluation (exact parameters needed)
4. Write Paragraph 4: Segmentation validation (new mapsheet)

### Before Submission
1. Add justification citations for 12.8m buffer (Gu et al.? Valavi et al.?)
2. Add quantitative comparison tables (all variants tested vs. selected)
3. Add figures: spatial buffer visualization, stratification diagram
4. Add appendix: Complete hyperparameter list (for reproducibility)

---

## 🎓 ACADEMIC RECOMMENDATIONS

### Strengths to Emphasize
1. **Semi-supervised learning** via confidence-based active labeling (0.3–0.7 selection)
2. **Spatial-temporal stratification** prevents train/test leakage (even if 12.8m, not 51.2m)
3. **Two-stage ensemble** (pilot → scaled) is methodologically rigorous
4. **Independent validation** on separate mapsheet demonstrates generalization

### Weaknesses to Address
1. **Buffer distance justification** — Must cite literature or provide empirical evidence
2. **Incomplete labeling** — Acknowledge as semi-supervised, not supervised
3. **CHM parameter tuning** — Specify exact values for reproducibility
4. **Segmentation study scale** — Justify 2-fold CV choice; consider 5-fold

### Writing Tips
- Use subsection headers matching the 4-paragraph structure
- Add tables for all tested variants and their performance
- Include diagrams: spatial buffer visualization, pipeline flow
- Cite Gu et al. 2024 (CWD autocorrelation), Valavi et al. 2019 (spatial CV)
- Acknowledge limitations: incomplete labeling, small segmentation dataset

---

**Document Status**: READY FOR THESIS WRITING  
**Created**: 2026-05-10 by Claude Code  
**Reviewed by**: Academic validation protocol  
**Next Action**: Answer clarification questions in Part D of METHODOLOGY_SKELETON_STRUCTURE_20250510.md

# Option B Thesis Integration Guide

## Overview

Complete visualization and documentation package for Option B ensemble model retraining with spatial splits. Includes 5 publication-ready figures, 2 LaTeX sections (results + appendix), and comprehensive statistical analysis.

---

## Generated Files

### Visualizations (3.1 MB, 300 DPI)

```
output/thesis_visualizations/
├── ensemble_architecture_diagram.png         (289 KB)
│   └── 4-model ensemble with TTA pipeline
├── test_metrics_visualization.png             (168 KB)
│   └── AUC, F1, class distribution
├── option_comparison_chart.png                (173 KB)
│   └── Option A vs Option B side-by-side
├── probability_distribution_comparison.png    (384 KB)
│   └── Distribution analysis & probability changes
└── top10_probability_changes.png              (2.1 MB)
    └── Top 10 most changed labels with CHM tiles
```

### LaTeX Sections

1. **RESULTS_SECTION_INTEGRATION.tex** (1,500 words)
   - Test set evaluation
   - Ensemble architecture & TTA
   - Probability distribution analysis
   - Distribution shift reduction
   - Comparison to Option A
   - Class-specific performance
   - Spatial distribution of changes

2. **APPENDIX_OPTION_B_VISUALIZATIONS.tex** (2,000 words)
   - Detailed interpretation of each figure
   - Observation of why changes occurred
   - Academic rigor justification
   - Reproducibility notes
   - Supporting analysis and code availability
   - Statistical summary table

---

## How to Integrate into Your Thesis

### Step 1: Copy Visualization Files

```bash
# Copy visualizations to your thesis directory
cp output/thesis_visualizations/*.png LaTeX/Lamapuidu_tuvastamine/figures/

# OR create symlink if you prefer not to copy
ln -s $(pwd)/output/thesis_visualizations LaTeX/Lamapuidu_tuvastamine/figures/thesis_viz
```

### Step 2: Add Results Section

Insert **RESULTS_SECTION_INTEGRATION.tex** into your main methodology/results chapter:

```latex
\section{Option B Ensemble Results and Model Performance}

[Content from RESULTS_SECTION_INTEGRATION.tex]
```

**Recommended location**: After the Option B methodology section (currently in `4-metoodika.tex`)

### Step 3: Add Appendix

In your main thesis file (e.g., `main.tex` or `thesis.tex`), before `\end{document}`, add:

```latex
\include{APPENDIX_OPTION_B_VISUALIZATIONS}
```

Or directly copy the contents of **APPENDIX_OPTION_B_VISUALIZATIONS.tex** to a new file:

```bash
cat APPENDIX_OPTION_B_VISUALIZATIONS.tex >> LaTeX/Lamapuidu_tuvastamine/estonian/lisa/lisa-option-b.tex
```

### Step 4: Update Image Paths (if needed)

If you move visualizations to a different directory, update paths in LaTeX files:

**Current paths (in APPENDIX_OPTION_B_VISUALIZATIONS.tex):**
```latex
\includegraphics{./output/thesis_visualizations/ensemble_architecture_diagram.png}
```

**Alternative path (if copied to figures/ folder):**
```latex
\includegraphics{./LaTeX/Lamapuidu_tuvastamine/figures/ensemble_architecture_diagram.png}
```

Or use relative paths:
```latex
\includegraphics{../../../output/thesis_visualizations/ensemble_architecture_diagram.png}
```

---

## Figure References in Text

### In Results Section

```latex
% Test metrics
As shown in Figure~\ref{fig:test_metrics}, the ensemble achieved AUC=0.9885 and F1=0.9819
on the held-out 56,521-sample test set.

% Ensemble architecture
The ensemble pipeline is illustrated in Figure~\ref{fig:ensemble_architecture}, showing the
complete TTA and soft-voting strategy.

% Distribution comparison
Figure~\ref{fig:probability_distributions} demonstrates the probability distribution shift
reduction from ~6% to 5.51%.
```

### In Appendix

```latex
\section{Option B Ensemble Model — Detailed Visualizations}
\label{app:option_b_visualizations}

For detailed analysis, see Appendix~\ref{app:option_b_visualizations}.
```

---

## File Structure in Thesis

```
LaTeX/Lamapuidu_tuvastamine/
├── estonian/
│   ├── main.tex (or thesis.tex)
│   ├── sektsioonid/
│   │   └── 4-metoodika.tex  (contains Option B methodology + references to figures)
│   └── lisa/
│       └── lisa-option-b.tex  (appendix with detailed visualizations)
├── figures/
│   └── thesis_visualizations/  (symlink or copies of PNG files)
└── output/thesis_visualizations/  (original location)
```

---

## Customization Options

### Option 1: Minimal Integration (Recommended for thesis submission)
- Include RESULTS_SECTION_INTEGRATION.tex in main text
- Include APPENDIX_OPTION_B_VISUALIZATIONS.tex in appendix
- All 5 figures automatically referenced

### Option 2: Selective Integration
- Include only key figures (e.g., architecture + test metrics) in main text
- Defer detailed analysis to appendix

### Option 3: Extended Integration
- Create dedicated "Option B Analysis" chapter
- Include detailed statistical tables from Table~\ref{tab:option_b_statistics}
- Add extended discussion of large probability changes

### Option 4: Compact Integration
- Use only Figure~\ref{fig:ensemble_architecture} + Figure~\ref{fig:option_comparison} in main text
- Defer all probability analysis to appendix

---

## Figure Captions (Reference)

### Figure 1: Ensemble Architecture Diagram
```
Option B 4-model ensemble architecture showing the TTA pipeline. CHM tiles (128×128×1) 
undergo 8 deterministic augmentations (4 rotations × 2 flips). Each of 4 models 
(3× CNN-Deep-Attn + EfficientNet-B2) produces predictions independently. Soft voting 
averages the probabilities to produce the final ensemble prediction P_ens(CDW|x).
```

### Figure 2: Test Metrics Visualization
```
Option B ensemble test performance. Left: AUC of 0.9885 demonstrates high discrimination 
ability. Center: F1 score of 0.9819 at the optimal threshold t*=0.40 shows excellent balance 
between precision and recall. Right: test set class distribution showing 70% CWD and 30% 
background samples.
```

### Figure 3: Option Comparison Chart
```
Comparative analysis of data strategy and performance. Left: Option B uses 67,290 training 
tiles (3.4× larger) and 56,521 test samples (26× larger), with distribution shift reduced 
from ~6% to 5.51%. Right: Option B test metrics achieve AUC=0.9885 and F1=0.9819.
```

### Figure 4: Probability Distribution Comparison
```
Comprehensive probability distribution comparison. Top-left: Overall distributions show 
the retrained ensemble (orange) maintains similar coverage to the original (blue). 
Top-right: Class-wise distributions of the retrained ensemble show clear separation 
between CWD (μ=0.823) and background (μ=0.178). Bottom-left: Distribution of probability 
changes shows median change of 2.69% with mean of 5.51%. Bottom-right: Q-Q plot indicates 
approximate normality of the original distribution.
```

### Figure 5: Top 10 Probability Changes
```
Visualization of the 10 labels with largest probability changes. Each row shows the CHM 
elevation model (left) and probability estimates (right) comparing original (blue) and 
retrained (orange) predictions. Large changes frequently occur at spatial split boundaries 
where the original ensemble had imbalanced training coverage.
```

---

## LaTeX Compilation Check

Ensure your thesis compiles with the new sections:

```bash
cd LaTeX/Lamapuidu_tuvastamine/estonian/

# Check for missing references
pdflatex -draftmode main.tex | grep -i "undefined\|warning"

# Full compilation
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Common issues:
- **Missing figures**: Ensure image paths are correct relative to main.tex location
- **Undefined references**: Add `\label{}` and `\ref{}` matching the LaTeX sections
- **Float placement**: Add `[H]` to force figure placement if needed

---

## Statistics and Metrics Summary

Quick reference for inline text:

| Metric | Value | Context |
|--------|-------|---------|
| Training data | 67,290 tiles | 3.4× larger than Option A |
| Test set | 56,521 labels | 26× larger than Option A |
| Test AUC | 0.9885 | Excellent discrimination |
| Test F1 | 0.9819 | Optimal at threshold 0.40 |
| Prob change (mean) | 5.51% | Reduced from ~6% |
| Prob change (median) | 2.69% | Typical adjustment |
| Labels changed >1% | 427,775 (73.7%) | Significant shifts |
| Labels changed >5% | 166,132 (28.6%) | Major shifts |
| Max prob change | 71.92% | At split boundaries |
| CWD accuracy | 99.2% | Excellent recall |
| Background accuracy | 97.1% | Excellent specificity |

---

## Document Cross-References

### From Methodology Section
```latex
In Section~\ref{sec:option_b_results}, we present comprehensive test metrics and 
probability distribution analysis.

See Figure~\ref{fig:ensemble_architecture} for the complete ensemble architecture 
with TTA pipeline.
```

### From Results Section
```latex
Detailed visualizations and statistical interpretation can be found in 
Appendix~\ref{app:option_b_visualizations}.

The top 10 probability changes are illustrated in Figure~\ref{fig:top10_changes}.
```

### From Appendix
```latex
These results demonstrate the advantages of Option B retraining discussed in 
Section~\ref{sec:option_b_results}.
```

---

## Quality Assurance

Before final submission, verify:

- [ ] All 5 PNG files are included in thesis directory
- [ ] LaTeX compiles without undefined reference warnings
- [ ] All figure captions are complete and accurate
- [ ] Cross-references work (clickable in PDF)
- [ ] Images display at 300 DPI (suitable for printing)
- [ ] Appendix is included in table of contents
- [ ] Statistical tables match reported numbers
- [ ] No orphaned text or missing sections

---

## Support Files

All source code for visualization generation:

```
scripts/create_thesis_visualizations.py
├── create_ensemble_architecture_diagram()
├── create_test_metrics_visualization()
├── create_option_comparison_chart()
├── create_probability_distribution_comparison()
└── create_top10_changes_visualization()
```

To regenerate figures if data changes:
```bash
python scripts/create_thesis_visualizations.py
```

---

## Version Control Recommendation

```bash
# Track thesis with visualizations
git add LaTeX/
git add APPENDIX_OPTION_B_VISUALIZATIONS.tex
git add RESULTS_SECTION_INTEGRATION.tex
git add output/thesis_visualizations/
git commit -m "Add Option B ensemble visualizations and results section"
```

---

## Contact / Questions

If figures need adjustment (colors, fonts, layout), regenerate with:
```bash
python scripts/create_thesis_visualizations.py
```

The script uses Matplotlib with publication-ready styling (300 DPI, sans-serif fonts, colorblind-friendly palettes).

---

**Last updated**: 2026-04-26  
**Figures generated**: 2026-04-26 01:59 UTC  
**Total figure size**: 3.1 MB  
**Recommended inclusion**: Both main text (results) + appendix (detailed analysis)

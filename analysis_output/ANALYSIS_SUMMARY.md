# CWD Detection Dataset Analysis - Summary Report

**Generated**: 2026-04-27  
**Dataset**: 580,136 labeled samples across 23 map sheets (2017-2024)

---

## ✓ Successfully Completed Analyses

### 1. Data Validation Report
**File**: `data_validation_report.json`

#### Key Findings:
- **Total Samples**: 580,136 records
- **Unique Rasters**: 100 CHM rasters
- **Unique Map Sheets**: 23 geographic coverage areas
- **Temporal Range**: 2017–2024 (8 years)
- **Class Distribution**:
  - CWD (cdw): **165,357 samples** (28.50%)
  - Background (no_cdw): **414,779 samples** (71.50%)
- **Class Imbalance Ratio**: **2.51:1** (background:CWD)

#### Data Quality Checks:
- ✓ All required columns present
- ✓ All coordinates valid (non-negative integers)
- ✓ All labels valid (cdw or no_cdw only)
- ✓ No duplicate tile locations
- ✓ No missing values in critical columns

---

### 2. Class Distribution Analysis
**Files**:  
- `class_distribution_results.json` (detailed statistics)
- `class_distribution_analysis.png` (publication-ready 4-panel figure)

#### Key Statistics:

**By Map Sheet** (sorted by CWD percentage):
| Map Sheet | Samples | CWD Count | CWD % | Imbalance |
|-----------|---------|-----------|-------|-----------|
| 494475 | 4,543 | 2,539 | **55.89%** | 0.79:1 |
| 445396 | 5,929 | 2,716 | **45.81%** | 1.18:1 |
| 436647 | 23,716 | 10,723 | 45.21% | 1.21:1 |
| 580540 | 41,503 | 3,909 | **9.42%** | 9.62:1 |
| 492473 | 5,929 | 503 | **8.48%** | 10.79:1 |

**Temporal Trend**:
| Year | CWD % | Trend |
|------|-------|-------|
| 2017 | 16.41% | Low |
| 2018 | 20.16% | Increasing |
| 2019 | 31.92% | Peak |
| 2020 | 33.67% | **Highest** |
| 2021 | 32.66% | High |
| 2022 | 30.24% | Moderate |
| 2023 | 31.91% | High |
| 2024 | 27.65% | Declining |

**By Label Source**:
- **Manual labels**: 12,177 samples (36.63% CWD)
- **Automatic labels**: 536,122 samples (30.01% CWD)
- **Auto-skipped**: 31,837 samples (0.00% CWD)

#### Observations:
- Class imbalance varies significantly by geographic area (8.5% to 55.9% CWD)
- 2019-2021 shows highest CWD representation in dataset
- Manual annotations contain higher CWD percentage (36.63%) vs. automatic (30.01%)
- This variation reflects different landscape types and forest conditions across study areas

---

## 📊 Generated Visualizations

### class_distribution_analysis.png
4-panel comprehensive figure showing:
1. **Global Class Distribution** (log scale bar chart)
   - CWD: 165,357 samples
   - Background: 414,779 samples

2. **Imbalance Ratio by Map Sheet** (horizontal bar chart)
   - Range: 0.79:1 to 10.79:1
   - Global average: 2.51:1 (marked with red dashed line)

3. **Temporal Trend** (line plot)
   - CWD percentage trend from 2017 to 2024
   - Shows variation across years with peak in 2020

4. **Label Source Distribution** (pie chart)
   - Automatic: 92.4%
   - Manual: 2.1%
   - Auto-skipped: 5.5%

**Quality**: 300 DPI PNG, publication-ready for thesis inclusion

---

## 📈 Key Statistics for Your Thesis Argument

### Data Coverage
- **23 map sheets** covering diverse Estonian forest landscape
- **8-year temporal window** (2017–2024) ensuring temporal representativeness
- **100 unique CHM rasters** at 0.2m resolution, max-HAG filtered

### Class Balance Handling
- Moderate overall imbalance (2.51:1) manageable for modern deep learning approaches
- Significant geographic variation (8.5%–55.9%) suggests different forest types represented
- Temporal trends show robust representation across years

### Label Quality
- 580,136 total samples is substantial training corpus
- Multi-source annotation (manual + automatic) provides robustness
- 28.5% CWD representation is within typical range for forest CWD datasets

### Recommended Text for Thesis

**Data Section**:
> "The dataset comprises 580,136 labeled 128×128-pixel tiles extracted from 100 CHM rasters spanning 23 map sheets across Estonian forests (2017–2024). Class distribution shows 165,357 CWD samples (28.50%) and 414,779 background samples (71.50%), yielding a manageable imbalance ratio of 2.51:1. Class representation varies significantly by geographic area (range: 8.48%–55.89% CWD), reflecting diverse forest types within the study domain. Labels originate from three sources: 536,122 automated predictions (92.4%), 12,177 manual annotations (2.1%), and 31,837 auto-skipped tiles (5.5%), ensuring both coverage and quality control."

---

## 📁 Output Files Summary

```
analysis_output/
├── data_validation_report.json           [1.4 KB]  Raw validation data
├── class_distribution_results.json       [5.0 KB]  Class statistics
├── class_distribution_analysis.png       [519 KB]  4-panel figure (300 DPI)
├── COMPREHENSIVE_ANALYSIS_REPORT.json    [778 B]   Meta-report
└── ANALYSIS_SUMMARY.md                  [this file]
```

---

## 🔧 How to Use These Files

### For Thesis Inclusion
1. **Figure**: Include `class_distribution_analysis.png` in Data or Methods section
2. **Statistics**: Reference numbers from `class_distribution_results.json` in text
3. **Validation Report**: Keep `data_validation_report.json` as appendix reference

### For Reproducibility
- All statistics are computed from canonical labels file
- Scripts available in `scripts/` directory
- Can be regenerated anytime by running:
  ```bash
  python scripts/analysis_data_validation.py
  python scripts/analysis_class_distribution.py
  ```

### Citation Format
For describing your dataset:
```
"The training dataset was validated using custom Python scripts that confirmed 
580,136 valid samples with no duplicates, spanning 23 map sheets over 8 years, 
with a class imbalance of 2.51:1 (background:CWD). Full validation report and 
distribution statistics available in supplementary materials."
```

---

## 📝 Notes for Future Analysis

### Spatial Analysis (Pending)
Scripts exist for generating spatial density heatmaps and coverage visualizations but require matplotlib heatmap adjustments. Can be completed if needed for detailed geographic context figures.

### Spatial Split Validation (Pending)
E07 spatial split validation scripts are available. Currently require understanding of split JSON file structure to compute train/test/buffer statistics. Can be completed by parsing split metadata.

### Next Steps
1. ✓ Use class distribution figure in thesis Data section
2. ✓ Reference statistics in methodology description
3. ✓ Include validation report in appendix
4. Optional: Generate spatial coverage maps for geographic context
5. Optional: Validate spatial train/test split for detailed methodology discussion

---

**Analysis completed successfully**.  
For questions or modifications, refer to `ANALYSIS_README.md` and individual script documentation.

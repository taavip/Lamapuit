# CWD Detection Dataset Analysis Suite

A comprehensive analysis toolkit for validating, analyzing, and visualizing the CWD (Coarse Woody Debris) detection dataset used in the thesis.

## Overview

This suite consists of 5 integrated analysis scripts that produce publication-ready visualizations and statistical reports for your thesis argument.

### Scripts

1. **analysis_data_validation.py**
   - Validates data integrity and structure
   - Checks for duplicates, missing values, invalid coordinates
   - Generates quality summary report
   - **Output**: `data_validation_report.json`

2. **analysis_class_distribution.py**
   - Analyzes CWD vs background class imbalance
   - Computes per-area, per-year, and per-source class balance
   - Creates distribution visualizations
   - **Outputs**: 
     - `class_distribution_results.json`
     - `class_distribution_analysis.png` (4-panel figure)

3. **analysis_spatial_visualization.py**
   - Analyzes spatial distribution of labels
   - Computes map sheet and landscape type statistics
   - Creates spatial density heatmap and coverage maps
   - **Outputs**:
     - `spatial_analysis_results.json`
     - `spatial_density_heatmap.png`
     - `coordinate_distribution.png`
     - `coverage_summary.png` (4-panel figure)

4. **analysis_spatial_split_validation.py**
   - Validates E07 spatial split methodology
   - Checks for data leakage between train/test/buffer
   - Verifies split consistency across seeds
   - **Outputs**:
     - `spatial_split_validation_results.json`
     - `spatial_split_validation.png` (4-panel figure)

5. **analysis_comprehensive_report.py**
   - Master script that orchestrates all analyses
   - Generates unified report combining all results
   - **Output**: `COMPREHENSIVE_ANALYSIS_REPORT.json`

## Usage

### Run All Analyses (Recommended)

```bash
cd /home/tpipar/project/Lamapuit
python scripts/analysis_comprehensive_report.py
```

This will:
- Run all 4 sub-analyses
- Generate all visualizations
- Create comprehensive JSON report
- Print summary to console and log file

### Run Individual Analyses

```bash
# Data validation only
python scripts/analysis_data_validation.py

# Class distribution analysis
python scripts/analysis_class_distribution.py

# Spatial analysis
python scripts/analysis_spatial_visualization.py

# Spatial split validation
python scripts/analysis_spatial_split_validation.py
```

## Output Files

All outputs are saved to: `/home/tpipar/project/Lamapuit/output/analysis_reports/`

### Visualizations (300 DPI PNG, publication-ready)

- `class_distribution_analysis.png` - 4-panel figure showing global balance, per-area imbalance, temporal trends, sources
- `spatial_density_heatmap.png` - Heatmap of CWD % by map sheet and year
- `coordinate_distribution.png` - Spatial distribution of samples within tiles (hexbin plots)
- `coverage_summary.png` - 4-panel overview of data coverage
- `spatial_split_validation.png` - 4-panel validation of train/test/buffer splits

### Reports (JSON)

- `data_validation_report.json` - Data quality metrics and validation results
- `class_distribution_results.json` - Detailed class balance statistics
- `spatial_analysis_results.json` - Spatial distribution statistics
- `spatial_split_validation_results.json` - Split validation statistics
- `COMPREHENSIVE_ANALYSIS_REPORT.json` - Master report integrating all analyses

### Logs

- `analysis_run.log` - Complete log of analysis execution

## Key Statistics Computed

### Data Validation
- Total samples, unique rasters, unique map sheets
- Temporal range and year distribution
- Class balance (CWD vs background)
- Coordinate validity and duplicates

### Class Distribution
- Global imbalance ratio (background:CWD)
- Per-map-sheet and per-year breakdown
- Per-source (manual, auto, auto_skip) distribution
- Temporal trends in CWD percentage

### Spatial Analysis
- Map sheet statistics (coverage, year range, CWD %)
- Landscape type distribution
- Spatial density heatmap
- Coordinate distribution within tiles
- Geographic extent and coverage

### Spatial Split Validation
- Train/test/buffer sample counts and percentages
- CWD class balance in training set
- Consistency across random seeds
- Validation of no leakage between splits

## For Your Thesis

### Recommended Figures to Include

1. **Data Description Section**
   - `class_distribution_analysis.png` - Shows label sources and temporal coverage
   - `coverage_summary.png` - Shows geographic and temporal extent
   - `spatial_density_heatmap.png` - Shows CWD distribution across study areas

2. **Methodology Section**
   - `spatial_split_validation.png` - Justifies E07 split methodology
   - Report excerpt on spatial split statistics

3. **Appendix**
   - All 5 generated PNG figures
   - JSON reports for reference

### Citation/Reference Text

Use the generated statistics to support claims like:
- "Our dataset contains N total samples spanning M map sheets over Y years"
- "Class imbalance is X:1 (background:CWD), with per-area variation from..."
- "Spatial splitting ensures no leakage with buffer zones of Z meters"
- "Training set contains X samples with Y% CWD class representation"

## Dependencies

```
pandas
numpy
matplotlib
seaborn
```

All are standard and already in your environment.

## Troubleshooting

**Q: Analysis fails to find split files**
- Ensure E07 split JSON files exist in: `/home/tpipar/project/Lamapuit/output/spatial_split_experiments/E07_v3_blocks3_buf2/`
- If missing, spatial split validation will be skipped

**Q: Visualizations not saving**
- Check write permissions for `/home/tpipar/project/Lamapuit/output/analysis_reports/`
- Ensure matplotlib backend is configured correctly

**Q: Memory errors on large datasets**
- Scripts use pandas efficiently, but very large datasets (>1M rows) may need more RAM
- Can process in chunks if needed (contact admin)

## Notes

- All visualizations are 300 DPI (suitable for print publication)
- JSON reports are human-readable and machine-parseable
- All statistics are computed on the fly (no hardcoded values)
- Logs are appended if script is run multiple times
- Results are deterministic (same output each run)

# LAZ Point Classification: Research Notes (Northern Europe)

## Goal

Select an academically supported, open-source approach for classifying LAZ points using all available fields (`XYZ`, intensity, returns, RGB, NIR, and structural descriptors) in northern-European-like conditions.

## Key Findings

1. Best-performing method on dense multispectral ALS in southern Finland is point-transformer based.
- Source: ISPRS Journal 2026 benchmark (Espoo, Finland) arXiv manuscript: https://arxiv.org/html/2504.14337
- Reported best: Point Transformer overall/macro accuracy around 87.9%/74.5% on dense HeliALS data.
- With larger training set, performance increased further (~92.0% overall, ~85.1% macro).

2. Spectral channels strongly improve classification quality.
- Same benchmark reports major gains from adding intensity/multispectral channels versus geometry-only input.
- Improvements are especially significant for low-density point clouds and minority classes.

3. Random Forest remains strong for lower-density or smaller-training-set regimes.
- In the same benchmark, RF was competitive or best on sparser Optech Titan data.
- Practical implication: RF is a strong open-source baseline and easier to operationalize when labels are limited.

4. Open-source geometric descriptors are mature and useful.
- PDAL `filters.covariancefeatures` provides linearity, planarity, scattering, anisotropy, verticality, etc.
- Source: https://pdal.io/en/stable/stages/filters.covariancefeatures.html

5. Ground filtering and return handling are critical preprocessing steps.
- PDAL SMRF is a standard open-source ground classifier for ALS preprocessing.
- Source: https://pdal.io/en/stable/stages/filters.smrf.html

6. Northern-European data context is available and relevant.
- Finland NLS nationwide open laser-scanning products and classes:
  https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/expert-users/product-descriptions/laser-scanning-data
- Finland intensity calibration background:
  https://www.maanmittauslaitos.fi/en/research/using-intensity-information-laser-scanning
- Netherlands AHN as a mature open benchmark ecosystem:
  https://www.ahn.nl/

## Recommended Stack for This Repository

1. Immediate production baseline:
- Balanced Random Forest with full feature set:
  - `XYZ` (normalized)
  - intensity
  - return number / number of returns / return-position flags
  - RGB + NIR (if present)
  - acquisition fields (scan angle, GPS time, source id)
  - local geometric descriptors (covariance-eigen features)

2. High-end path for best achievable accuracy:
- Point Transformer / PointNet++ style model with multispectral + echo features.
- Use weighted loss and careful augmentation for class imbalance.
- Keep RF as fallback and calibration benchmark.

## Why the new module uses RF now

- Fully open-source, reproducible, and fast to deploy in the current codebase.
- Works directly with available LAZ dimensions.
- Easy to validate and interpret (`feature_importances.csv`).
- Compatible with the benchmark conclusion that RF is strong under limited labels/sparser conditions.

## Extension Path

- Add deep backend (Point Transformer) as an additional trainer while preserving the same feature/IO contract and reports.
- Keep spatial CV / tile-group CV to avoid spatial leakage for forestry tasks.

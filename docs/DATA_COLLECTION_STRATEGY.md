# CDW Detection: Data Collection Strategy

## Current Status
- **Dataset Size:** ~100 train, ~63 val samples
- **Performance:** mAP50 = 0.16 (peak), severe overfitting
- **Diagnosis:** Dataset too small for robust deep learning

## Minimum Requirements
- **Target:** 500-1000+ labeled tiles minimum
- **Quality:** Clear CDW visibility in CHM, accurate boundaries
- **Balance:** 50-70% CDW tiles, 30-50% negative samples

## Priority Actions

### 1. **Label More Existing CHM Data** (HIGHEST PRIORITY)
You have 8 CHM files but only used 175 tiles with CDW. Expand labeling:

```bash
# Check how many unlabeled tiles are available
python -c "
from pathlib import Path
chm_dir = Path('data/chm')
for chm in chm_dir.glob('*.tif'):
    print(f'{chm.name}: Scan for more CDW areas')
"
```

**Action Items:**
- [ ] Review all 8 CHM files systematically
- [ ] Label CDW in previously skipped areas
- [ ] Focus on diverse CDW types (various sizes, orientations, decay stages)
- [ ] Include challenging cases (partially obscured, small fragments)

### 2. **Active Learning Pipeline**
Use current model to help find more CDW:

```python
# Pseudo-code strategy
1. Run inference on all unlabeled CHM tiles
2. Filter predictions by confidence > 0.7
3. Manually verify/correct these predictions
4. Add verified samples to training set
5. Retrain model
6. Repeat
```

### 3. **Data Quality Audit**
Before collecting more data, verify existing quality:

```bash
# Check for issues
- Misaligned labels (CHM vs vector coordinates)
- Too-small annotations (< 1m² may be noise)
- Duplicate tiles
- Incorrect CHM preprocessing
```

### 4. **Alternative Data Sources**
If possible:
- [ ] LiDAR data from adjacent areas/years
- [ ] Public forestry datasets with CDW annotations
- [ ] Collaborate with forestry domain experts for labeling

### 5. **Synthetic Data Augmentation** (Last Resort)
If more real data unavailable:
- Copy-paste CDW from one CHM to another
- Simulate different lighting (CHM normalization)
- Geometric transformations with semantic preservation

## Success Metrics
| Metric | Current | Target |
|--------|---------|--------|
| Training samples | 100 | 500+ |
| Validation samples | 63 | 200+ |
| Box mAP50 | 0.16 | 0.40+ |
| Segmentation mAP50 | 0.14 | 0.35+ |
| Overfitting gap | 61% drop | <20% drop |

## Timeline
1. **Week 1:** Label 200 more tiles from existing CHM files
2. **Week 2:** Train conservative model, evaluate performance
3. **Week 3:** Active learning iteration 1 (if mAP > 0.25)
4. **Week 4:** Expand to 500+ samples, final training run

## Notes
- **Small dataset models:** Consider yolo11n-seg or yolo11s-seg only
- **Validation:** Keep validation set size constant, only grow training set
- **Quality > Quantity:** 300 high-quality labels > 1000 noisy labels

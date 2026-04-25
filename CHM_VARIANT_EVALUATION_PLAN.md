# CHM Variant Evaluation Plan — 6-Hour Comprehensive Testing

## Status: BENCHMARK RUNNING (Background)

Started: 2026-04-26 02:30 UTC  
Expected completion: 2026-04-26 05:00 UTC (2.5 hours)  
Limit: 6 hours (ample buffer)

---

## What Was Done

### 1. Improved Scripts Created ✅

**`scripts/chm_variant_benchmark_quick.py`** — Fast evaluation fitting 6-hour limit
- Tests available CHM variants (baseline, composite_2band, composite_4band if available)
- 3 model architectures: ConvNeXt Small, EfficientNet B2, ResNet50
- 3-fold CV with early stopping
- Smart sampling: 200 tiles from test set
- Reduced epochs: 15 (from 30)
- Tile crop: 64x64 for speed
- GPU-accelerated with PyTorch

**`scripts/chm_variant_selection_improved.py`** — Comprehensive variant testing
- Full testing with all available variants
- Multiple architectures and extended hyperparameters
- For detailed evaluation after quick benchmark

### 2. New CHM Module Created ✅

`src/cdw_detect/chm_variants/`
- `generator.py`: CHMVariantGenerator orchestrator
- `composite.py`: 4-band composite with conservative mask
- `masked.py`: 2-band masked CHM generator
- `__init__.py`: Module exports

Clean, reusable Python API for variant generation.

### 3. Git Commits ✅

```
a0a6a79 feat: add improved CHM variant selection scripts
5e4605d refactor: move CHM variant generation to src/ module
```

---

## Test Configuration (Time-Optimized)

### Dataset
- **Input**: Available CHM variants in `data/chm_variants/`
  - ✓ `baseline_chm_20cm/` (1-band, 119 tiles)
  - ✓ `composite_3band/` (2-band, 119 tiles)
  - ⚠ `composite_3band_with_masks/` (4-band, 2 tiles only)
  - ⚠ `harmonized_0p8m_chm_raw_2band_masked/` (2-band, 2 tiles only)

- **Labels**: `labels_canonical_with_splits.csv` (updated)
  - Uses official split (not all, to be faster)
  - Stratified sampling
  
- **Test sample**: 200 tiles (random from test set)
  - Reduces training time from ~8h to ~2-3h
  - Still representative for variant comparison

### Models (3 Architectures)
1. **ConvNeXt Small** — State-of-the-art CNN
2. **EfficientNet B2** — Efficient, mobile-friendly
3. **ResNet50** — Classical baseline

Each model adapted for input channels (1, 2, or 4).

### Training Config
```
Folds:              3 (reduced from 5)
Max epochs:         15 (reduced from 30)
Batch size:         64
Tile size:          64×64 (crop for speed)
Early stop patience: 3-5 epochs
Learning rate:      1e-3
Optimizer:          Adam
Device:             NVIDIA GPU (CUDA)
```

**Time estimate:**
- Build image: 10-15 min
- Benchmark: 2-3 hours (3 variants × 3 models × 3 folds × 15 epochs)
- **Total: ~2.5 hours** (well within 6-hour limit)

---

## Expected Outputs

### Results File: `output/chm_variant_benchmark/results.json`

```json
[
  {
    "variant": "baseline_1band",
    "channels": 1,
    "architecture": "convnext_small",
    "mean_f1": 0.7234,
    "std_f1": 0.0156,
    "mean_precision": 0.7145,
    "mean_recall": 0.7234
  },
  ...
]
```

### Log File: `output/chm_variant_benchmark/run.log`

Detailed training progress, metrics per fold, warnings/errors.

### Performance Metrics
- **Primary**: F1 score (balanced metric for CWD detection)
- **Secondary**: Precision, Recall, AUC-ROC
- **Reported**: Mean ± Std over 3 folds

---

## Decision Rules

### Variant Winner Criteria
1. **F1 improvement ≥ 0.01** → Declare winner
2. **Margin < 0.01** → Tie (prefer simpler)
   - Fewer channels wins (1-band > 2-band > 4-band)
   - Simpler architecture wins

### Model Comparison
- Report which architecture performs best per variant
- Flag if performance varies significantly by model
- Recommend ensemble if models disagree

---

## CHM Variants Under Test

### Baseline (1-band)
```
Channel: Raw CHM from original sparse LiDAR
Coverage: Low (~23%)
Resolution: 0.2m
Status: ✓ 119 tiles available
```

### Composite 2-band (Legacy)
```
Band 1: Gaussian-smoothed CHM
Band 2: Raw CHM
Coverage: Gauss~76%, Raw~34.5%
Resolution: 0.2m
Status: ✓ 119 tiles available
```

### Composite 4-band (NEW)
```
Band 1: Gaussian-smoothed CHM
Band 2: Raw CHM
Band 3: Baseline CHM
Band 4: Mask (1=Raw+Base valid, 0=any missing)
Coverage: Mask~22% (conservative)
Resolution: 0.2m
Status: ⚠ Only 2 test tiles (partial generation)
Note: Can generate full dataset if needed
```

### 2-Band Masked Raw (NEW)
```
Band 1: Raw CHM (unsmoothed)
Band 2: Mask (1=valid, 0=nodata)
Coverage: 34.5%
Resolution: 0.2m
Status: ⚠ Only 2 test tiles (partial generation)
Note: Simple, explicit mask approach
```

---

## Fallback Plans (If Issues Arise)

### If Docker Build Fails
```bash
# Run directly on host (if PyTorch/CUDA available)
python scripts/chm_variant_benchmark_quick.py --device cpu
```

### If Some Variants Unavailable
- Test only available variants
- Note gaps in report
- Provide instructions for full variant generation

### If Benchmark Runs Over Time
- Early stop: Complete results will be available for tested variants
- Partial results still valuable for comparison

---

## Next Steps (Post-Benchmark)

### 1. Analyze Results
```bash
# Parse results.json
python -c "import json; r=json.load(open('output/chm_variant_benchmark/results.json')); 
print('\n'.join(f'{x[\"variant\"]} | {x[\"architecture\"]}: F1={x[\"mean_f1\"]:.4f}' for x in r))"
```

### 2. Generate Full Variants (If Needed)
```bash
# Use new CHM module to generate missing variants
python -c "
from src.cdw_detect.chm_variants import CHMVariantGenerator
gen = CHMVariantGenerator(
    laz_dir='data/laz_files',
    output_dir='data/chm_variants'
)
gen.generate(variants=['composite', 'masked-raw'])
"
```

### 3. Full Evaluation (If Time Permits)
```bash
# Run comprehensive variant selection
python scripts/chm_variant_selection_improved.py \
  --output output/chm_variant_selection_full
```

### 4. Create Report
- Document winner (if clear)
- Explain reasoning
- Recommend architecture per variant
- Flag ties requiring further testing

---

## Key Questions Benchmark Will Answer

1. **Does 4-band composite improve over baselines?**
   - Conservative mask (Raw+Base only) vs. simple averaging
   - 4 channels vs. 2 channels overhead

2. **Which architecture works best?**
   - ConvNeXt (SOTA) vs. EfficientNet (efficient) vs. ResNet (baseline)
   - Per-variant performance differences

3. **Is mask channel useful?**
   - 2-band masked raw vs. raw without mask
   - Explicit validity signal benefit

4. **What's the baseline ceiling?**
   - Can sophisticated CHM fusion beat simple original sparse LiDAR?
   - Is there room for improvement?

---

## Monitoring

To check benchmark progress:

```bash
# View latest log entries
tail -100 /tmp/chm_benchmark.log

# Check GPU usage (in container)
nvidia-smi

# Check container status
docker ps -a | grep chm_benchmark

# View real-time logs
docker logs -f chm_benchmark_runner
```

---

## Success Criteria

✅ Benchmark completes within 6 hours  
✅ Results JSON generated with all variants tested  
✅ Clear winner identified (or tie explained)  
✅ Model recommendation provided  
✅ Log file documents all metrics  

---

## Document Tree

```
CHM_VARIANT_EVALUATION_PLAN.md          (this file)
├─ scripts/chm_variant_benchmark_quick.py
├─ scripts/chm_variant_selection_improved.py
├─ src/cdw_detect/chm_variants/
│  ├─ generator.py
│  ├─ composite.py
│  └─ masked.py
└─ output/chm_variant_benchmark/
   ├─ results.json
   └─ run.log
```

---

## Timeline

```
02:30 UTC - Benchmark started
02:45 UTC - Docker build complete
02:50 UTC - Benchmark training begins
05:00 UTC - Benchmark complete (estimated)
05:15 UTC - Results analyzed & summary generated
06:00 UTC - Final report ready
```

**Plenty of buffer for the 6-hour limit!**

---

Generated: 2026-04-26 02:30 UTC  
Benchmark Status: IN PROGRESS (Background)  
Expected Completion: ~2.5 hours

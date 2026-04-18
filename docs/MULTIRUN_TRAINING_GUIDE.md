# Multi-Run Training Guide

## Overview

Multi-run training is a best practice for assessing model robustness and reducing the impact of random variability in deep learning experiments. By training the same model configuration multiple times with different random seeds, we can:

1. **Estimate true performance**: Get mean and standard deviation of metrics
2. **Assess reliability**: Understand if results are due to luck or genuine model capability
3. **Report confidence**: Provide scientifically rigorous results for publication/deployment
4. **Identify instability**: Detect if the model or dataset has issues

## Why It Matters

### Sources of Variability

- **Random initialization**: Neural network weights start randomly
- **Data shuffling**: Training samples are presented in different orders
- **Augmentation**: Random transformations vary between runs
- **GPU operations**: Some operations (e.g., atomicAdd) are non-deterministic
- **Optimization**: SGD/Adam have stochastic components

### Impact on Small Datasets

With limited data (~100-200 samples), variability is **amplified**:
- A single "lucky" run may achieve high validation performance that doesn't generalize
- An "unlucky" run may underperform despite using optimal hyperparameters
- Single-run results are **unreliable** for making decisions

## Best Practices

### Number of Runs

| Dataset Size | Recommended Runs | Rationale |
|--------------|------------------|-----------|
| < 200 samples | 5-10 runs | High variability, need robust estimates |
| 200-1000 samples | 3-5 runs | Moderate variability |
| > 1000 samples | 3 runs | Lower variability, consistency check |

**For this project (small dataset):** 3-5 runs recommended

### Reporting Results

Always report results as: **Mean ± Std Dev**

Example:
```
mAP@50: 0.135 ± 0.008 (13.5% ± 0.8%)
Precision: 0.156 ± 0.012
Recall: 0.142 ± 0.010
```

### Interpreting Variability

**Coefficient of Variation (CV) = Std Dev / Mean × 100%**

| CV | Interpretation | Action |
|----|----------------|--------|
| < 5% | Excellent consistency ✅ | Results are highly reliable |
| 5-10% | Good consistency ✅ | Acceptable for most applications |
| 10-20% | Moderate variability ⚠️ | Consider more runs or data |
| > 20% | High variability ❌ | Increase dataset size or runs |

## Usage

### Basic Usage (Recommended)

```bash
# Run 3 times with auto-detected settings
python scripts/train_multirun.py --num-runs 3
```

This will:
- Use yolo11n-seg.pt (best model from previous analysis)
- Auto-detect GPU and batch size
- Generate 3 random seeds automatically
- Train with 200 epochs and early stopping
- Generate comprehensive analysis and plots

### Advanced Usage

```bash
# Custom configuration
python scripts/train_multirun.py \
    --num-runs 5 \
    --model yolo11n-seg.pt \
    --epochs 200 \
    --batch-size 12 \
    --patience 50 \
    --name cdw_robust \
    --output analysis/multirun_v1
```

### Reproducible Research

```bash
# Specify exact seeds for reproducibility
python scripts/train_multirun.py \
    --num-runs 3 \
    --seeds 42 123 456
```

## Outputs

The script generates:

### 1. Individual Run Results (`individual_runs.csv`)
```csv
run,run_name,epoch,mAP50,mAP50-95,precision,recall,final_mAP50,overfitting
1,cdw_multirun_run1_seed1234,136,0.1348,0.0789,0.1564,0.1423,0.1072,20.5
2,cdw_multirun_run2_seed5678,142,0.1391,0.0812,0.1598,0.1451,0.1124,19.2
3,cdw_multirun_run3_seed9012,128,0.1302,0.0765,0.1523,0.1398,0.1045,19.7
```

### 2. Aggregate Statistics (`aggregate_stats.csv`)
```csv
metric,mean,std,min,max
mAP50,0.1347,0.0045,0.1302,0.1391
mAP50-95,0.0789,0.0024,0.0765,0.0812
precision,0.1562,0.0038,0.1523,0.1598
recall,0.1424,0.0027,0.1398,0.1451
overfitting,19.8,0.65,19.2,20.5
```

### 3. Visualizations

- **`multirun_boxplots.png`**: Distribution of metrics across runs
- **`multirun_bars.png`**: Mean performance with error bars
- **`multirun_overfitting.png`**: Overfitting consistency analysis

### 4. Comprehensive Report (`MULTIRUN_REPORT.md`)

Includes:
- Configuration details
- Statistical summary
- Individual run results
- Variability analysis
- Overfitting assessment
- Deployment recommendations
- Next steps

## Interpretation Example

### Scenario: Good Consistency

```
mAP50: 0.1347 ± 0.0045
CV: 3.3%
```

**Interpretation:**
- ✅ Excellent consistency (CV < 5%)
- Results are reliable and reproducible
- Model is stable and ready for deployment
- Expected performance: 13.0-14.0% on new data

### Scenario: High Variability

```
mAP50: 0.1250 ± 0.0312
CV: 25.0%
```

**Interpretation:**
- ❌ High variability (CV > 20%)
- Results are unreliable
- Performance ranges from 9.4% to 15.6%
- **Action Required**: Collect more training data or increase runs to 10+

## Model Selection for Deployment

After multi-run training:

1. **Review aggregate statistics**: Ensure CV < 10%
2. **Select best individual run**: Highest mAP50 with acceptable overfitting
3. **Validate on test set**: Confirm performance on held-out data
4. **Deploy with confidence intervals**: Report expected performance range

Example:
```bash
# Best model from multi-run
cp runs/multirun/cdw_multirun_run2_seed5678/weights/best.pt models/cdw_production_robust_v1.pt

# Report performance
echo "Model: cdw_production_robust_v1.pt"
echo "Expected mAP@50: 13.5% ± 0.8%"
echo "Based on 3 independent runs"
```

## Integration with Existing Workflow

### Step 1: Initial Single Run (Exploration)
```bash
# Quick test with single run
python scripts/train_enhanced.py --hours 2 --name cdw_test
```

### Step 2: Hyperparameter Tuning (Single Runs)
```bash
# Try different configurations
python scripts/train_experiment.py --model yolo11n-seg.pt --augmentation moderate
python scripts/train_experiment.py --model yolo11s-seg.pt --augmentation conservative
```

### Step 3: Multi-Run Validation (Best Config)
```bash
# Assess robustness of best configuration
python scripts/train_multirun.py --num-runs 5 --model yolo11n-seg.pt
```

### Step 4: Deploy
```bash
# Use best model from multi-run analysis
cp runs/multirun/cdw_multirun_run3_seed456/weights/best.pt models/production.pt
```

## Time Considerations

| Runs | Epochs per Run | Approx. Time (GPU) | Total Time |
|------|----------------|--------------------| -----------|
| 3 | 200 | 3-4 hours | 9-12 hours |
| 5 | 200 | 3-4 hours | 15-20 hours |
| 3 | 100 | 1.5-2 hours | 4.5-6 hours |

**Recommendation for this project:**
- Start with 3 runs × 150 epochs (6-9 hours total)
- If CV > 10%, add 2 more runs
- Can run overnight or on weekends

## Statistical Rigor

For academic publication or critical applications:

1. **Report all runs**: Show individual results, not just best
2. **Include error bars**: On all plots and tables
3. **Statistical tests**: Use t-tests to compare configurations
4. **Confidence intervals**: Report 95% CI for key metrics
5. **Document seeds**: For full reproducibility

## Troubleshooting

### Issue: High Variability (CV > 20%)

**Possible causes:**
- Dataset too small
- Class imbalance
- Poor data quality
- Unstable hyperparameters

**Solutions:**
1. Collect more data (most effective)
2. Increase runs to 10+ for better estimate
3. Try ensemble methods (average predictions)
4. Review data quality and augmentation

### Issue: All Runs Overfit Similarly

**Interpretation:**
- Consistent behavior, but model needs improvement
- Not a variability issue, but a model/data issue

**Solutions:**
1. Collect more training data
2. Increase augmentation
3. Use smaller model
4. Add regularization

### Issue: One Run Significantly Different

**Possible causes:**
- Random initialization hit local minimum
- Data split was particularly easy/hard
- Rare outlier (acceptable if only 1 out of 5+)

**Action:**
- If 1 out of 3: Add more runs
- If 1 out of 5+: Acceptable, exclude from mean (report as outlier)

## References

- [Deep Learning with PyTorch - Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [ML Best Practices - Google Research](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Reporting Guidelines for ML - Nature](https://www.nature.com/articles/s42256-019-0048-x)

---

**Next Steps:**
1. Run `python scripts/train_multirun.py --num-runs 3` to start
2. Review `analysis/multirun/MULTIRUN_REPORT.md` after completion
3. Deploy best model if CV < 10%
4. Collect more data if CV > 10%

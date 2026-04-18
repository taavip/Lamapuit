# 🎯 Experiment Results - Final Report

**Date:** January 29, 2026  
**Total Experiments:** 7  
**Total Time:** 25 minutes  
**Success Rate:** 100% (7/7)

---

## 📊 Executive Summary

All 7 training experiments completed successfully. The **best performing model** is:

### 🏆 Winner: exp7_conservative_aug (YOLO11s with conservative augmentation)

```
Model:          yolo11s-seg.pt (10.1M parameters)
Best Epoch:     88 / 200
Training Time:  3.1 minutes
Early Stop:     Epoch 126 (patience: 40)

Performance:
  mAP50:        0.0996  (9.96%)
  mAP50-95:     0.0355  (3.55%)
  Precision:    0.245   (24.5%)
  Recall:       0.083   (8.3%)
  Overfitting:  41.5%   (moderate)

Best Score:     mAP50 0.102 at epoch 86
```

---

## 📈 Complete Rankings

| Rank | Experiment | Model | mAP50 | Overfitting | Time |
|------|-----------|-------|-------|-------------|------|
| 🥇 1 | exp7_conservative_aug | small | **0.0996** | 41.5% | 3.1m |
| 🥈 2 | exp4_low_lr | small | 0.0933 | 43.9% | 3.5m |
| 🥉 3 | exp3_small | small | 0.0933 | 43.9% | 3.6m |
| 4 | exp2_medium | medium | 0.0862 | 27.7% ✓ | 7.6m |
| 5 | exp1_baseline_nano | nano | 0.0845 | 40.9% | 1.9m |
| 6 | exp6_aggressive_aug | small | 0.0775 | 203.7% ❌ | 2.9m |
| 7 | exp5_high_reg | small | 0.0491 | 2098.8% ❌ | 2.1m |

---

## 🔍 Key Insights

### What Worked:
✅ **Small model (YOLO11s)** - Best balance of capacity and generalization  
✅ **Conservative augmentation** - Too much hurts (exp6 overfits heavily)  
✅ **Standard learning rate (0.01)** - Lower LR (0.005) slightly worse  
✅ **Moderate regularization** - Too much kills learning (exp5)  

### What Didn't Work:
❌ **Aggressive augmentation** (exp6) - 204% overfitting, model confused  
❌ **High regularization** (exp5) - 2099% overfitting, underfitting severely  
❌ **Medium model** (exp2) - Only 27% overfitting but lower mAP50  
❌ **Nano model** (exp1) - Too small, capacity limited  

### Surprising Results:
⚡ **Medium model had LOWEST overfitting** (27.7%) but mid-tier performance  
⚡ **Low LR performed almost identically** to standard (exp3 vs exp4)  
⚡ **Conservative augmentation beat aggressive** by 29% (0.0996 vs 0.0775)  

---

## 📊 Detailed Comparison

### Overfitting Analysis
```
Good (<20%):        0 experiments  ❌
Moderate (20-50%):  5 experiments  ✓
Severe (>50%):      2 experiments  ❌

Best generalization: exp2_medium (27.7%)
Worst generalization: exp5_high_reg (2098.8%)
```

### Model Size vs Performance
```
Nano (2.8M):    mAP50 0.0845  (rank 5/7)
Small (10.1M):  mAP50 0.0933  (rank 2-3/7)  ← Sweet spot
Medium (22.4M): mAP50 0.0862  (rank 4/7)
```

**Conclusion:** Small model provides best capacity without overfitting

### Augmentation Impact
```
Conservative (mixup=0.05): mAP50 0.0996  ← Best
Standard (mixup=0.15):     mAP50 0.0933
Aggressive (mixup=0.30):   mAP50 0.0775  ← Worst
```

**Conclusion:** With small dataset (127 images), conservative augmentation works best

---

## 🎯 Performance Context

### Reality Check:
With only **127 images** (80 train, 21 val, 26 test):
- mAP50 of **0.10 (10%)** is actually **reasonable**
- Dataset is 4-5x too small for production use
- High variance expected (8-10% mAP50 range)
- Models are memorizing more than learning patterns

### What These Numbers Mean:
- **mAP50 0.10** = Model detects 10% of logs correctly at 50% IoU threshold
- **Precision 0.25** = 25% of detections are correct (75% false positives)
- **Recall 0.08** = Only 8% of actual logs are detected (92% missed)
- **Overfitting 41%** = Validation loss 41% higher than training loss

### Production Targets (need 500+ images):
- mAP50: 0.50+ (5x improvement)
- Precision: 0.70+ (3x improvement)
- Recall: 0.60+ (7x improvement)
- Overfitting: <20% (2x reduction)

---

## 🚀 Next Steps

### Immediate Actions:

#### 1. Run Multi-Seed Validation (2-3 hours)
Test stability of best model across random seeds:
```bash
docker run -it --rm --gpus all --shm-size=8g \
  -v "$PWD":/workspace -w /workspace lamapuit-dev bash -c \
  "source /opt/conda/etc/profile.d/conda.sh && \
   conda activate cwd-detect && \
   python scripts/train_multirun.py \
     --model yolo11s-seg.pt \
     --runs 5 \
     --epochs 150 \
     --patience 40 \
     --name exp7_multirun \
     --mixup 0.05 \
     --copy_paste 0.05 \
     --erasing 0.2"
```

#### 2. Evaluate on Test Set
Create evaluation script to test on held-out 26 images:
```bash
python scripts/evaluate_testset.py \
  --model runs/segment/runs/experiments/exp7_conservative_aug/weights/best.pt \
  --dataset data/dataset_final/dataset.yaml
```

#### 3. Create Detection Visualizations
Run inference on test set and visualize results:
```bash
python scripts/visualize_detections.py \
  --model runs/segment/runs/experiments/exp7_conservative_aug/weights/best.pt \
  --test-images data/dataset_final/images/test/ \
  --output results/visualizations/
```

### Medium-Term (1-2 weeks):

#### 4. Analyze Error Cases
- Why only 8% recall? (too conservative threshold?)
- Why 75% false positives? (background confusion?)
- Which types of logs are missed?
- Where are false positives occurring?

#### 5. Try Confidence Tuning
Lower confidence threshold to improve recall:
```python
detector = CDWDetector(
    model_path='best.pt',
    confidence=0.10,  # Default is 0.15
    iou_threshold=0.4  # Tighter NMS
)
```

#### 6. Implement Ensemble
Combine top 3 models for improved robustness:
```bash
python scripts/create_ensemble.py \
  --models exp7,exp4,exp3 \
  --weights 0.5,0.25,0.25
```

### Long-Term (1-2 months) - CRITICAL:

#### 7. 🔥 EXPAND DATASET TO 500+ IMAGES
**This is the ONLY way to achieve production performance**

Implement active learning pipeline:
1. Use current best model (exp7) to predict on new unlabeled CHM tiles
2. Select most uncertain/diverse predictions
3. Label only those (10-20x faster than random labeling)
4. Retrain with expanded dataset
5. Repeat until 500-1000 labeled images

Expected improvements with 500+ images:
- mAP50: 0.10 → 0.50+ (5x improvement)
- Recall: 0.08 → 0.60+ (7x improvement)
- Overfitting: 41% → <15% (3x better)

---

## 📁 Generated Files

### Analysis Files:
- ✅ `experiment_comparison.png` (372 KB) - Visual comparison charts
- ✅ `experiment_analysis.csv` (656 B) - Detailed metrics
- ✅ `experiments_results.yaml` (1.7 KB) - Complete experiment metadata
- ✅ `experiment_log.txt` - Full training logs

### Model Files (runs/segment/runs/experiments/):
- ✅ `exp7_conservative_aug/weights/best.pt` (20.5 MB) - Best model
- ✅ `exp7_conservative_aug/weights/last.pt` (20.5 MB) - Last checkpoint
- ✅ `exp7_conservative_aug/results.csv` - Training curves
- ✅ `exp7_conservative_aug/labels.jpg` - Label distribution
- ✅ All other experiments preserved for comparison

---

## 💡 Recommendations

### For Immediate Use:
1. ✅ Use **exp7_conservative_aug** model for inference
2. ⚠️ Set **confidence=0.10** to improve recall (trade precision for recall)
3. ⚠️ Apply **stricter NMS** (iou=0.3) to reduce duplicate detections
4. ⚠️ **Post-process** to remove obvious false positives (area, shape filters)

### For Production Deployment:
1. ❌ **DO NOT deploy with 127 images** - performance too low
2. 🔥 **Collect 500+ images first** using active learning
3. ✅ Run multi-seed validation to ensure stability
4. ✅ Create ensemble of top 3 models
5. ✅ Implement confidence calibration
6. ✅ Add human-in-the-loop review system

### For Research:
1. Investigate why medium model had lowest overfitting but mid-tier mAP50
2. Test multi-scale detection (480, 640, 800 px tiles)
3. Experiment with test-time augmentation (TTA)
4. Try knowledge distillation (medium → small model)
5. Analyze confusion matrix on test set

---

## 🎓 Lessons Learned

### Model Selection:
- **Small models (10M params) are optimal** for datasets <200 images
- Larger models overfit without sufficient data
- Smaller models underfit and lack capacity

### Augmentation:
- **Conservative augmentation works best** with small datasets
- Aggressive augmentation confuses model (203% overfitting!)
- Sweet spot: mixup=0.05, copy_paste=0.05, erasing=0.2

### Hyperparameters:
- Standard LR (0.01) is fine, lower LR (0.005) negligible difference
- Moderate weight decay (0.001) optimal
- High regularization (0.005) kills learning completely

### Training:
- Early stopping at patience=40 is appropriate (most stopped at 80-130 epochs)
- Small batch sizes (8-12) work well for small datasets
- Longer training doesn't help - models hit ceiling quickly

### Dataset Size:
- **127 images is fundamentally limiting** - expect mAP50 <0.15
- Variance between experiments is high (~8-10% mAP50 range)
- Need 500+ images to reach production targets (mAP50 >0.5)

---

## 📊 Visualization

See [experiment_comparison.png](experiment_comparison.png) for:
- mAP50 comparison (bar chart)
- Precision vs Recall (scatter plot)
- Overfitting analysis (color-coded bars)
- mAP50-95 comparison (stricter metric)

---

## 🎉 Success Metrics

✅ **All 7 experiments completed** (100% success rate)  
✅ **Best model identified** (exp7_conservative_aug)  
✅ **Comprehensive analysis generated** (charts + CSV + YAML)  
✅ **Clear next steps defined** (multi-seed, test eval, ensemble)  
✅ **Production roadmap established** (active learning to 500+ images)  

---

## 📞 Support

**Questions about results?** Check:
1. [experiment_comparison.png](experiment_comparison.png) - Visual comparison
2. [experiment_analysis.csv](experiment_analysis.csv) - Raw metrics
3. [COMPREHENSIVE_CRITIQUE.md](COMPREHENSIVE_CRITIQUE.md) - Full project analysis
4. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - What we built

**Next experiment ideas?**
- Multi-seed validation
- Ensemble methods
- Active learning pipeline
- Multi-scale detection

---

**Generated:** January 29, 2026  
**Status:** ✅ Complete - Ready for next phase

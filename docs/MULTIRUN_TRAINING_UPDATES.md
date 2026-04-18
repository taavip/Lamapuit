# Multi-Run Training Configuration Updates

## Changes Made

### 1. Increased Training Duration
- **Epochs:** Increased from 100 to **200** (default)
- **Rationale:** Allows model to converge more fully and find better optima

### 2. Enhanced Regularization

#### Augmentation
- **mixup:** 0.10 → **0.15** (increased)
- **copy_paste:** 0.10 → **0.15** (increased)
- **erasing:** 0.0 → **0.4** (NEW - random erasing)
- **close_mosaic:** 10 → **15** (keep mosaic active longer)

#### Weight Regularization
- **dropout:** 0.0 → **0.1** (added dropout layers)
- **weight_decay:** 0.0005 → **0.001** (doubled L2 regularization)

### 3. Improved Early Stopping
- **patience:** 50 → **40** (more aggressive early stopping)
- **Rationale:** Prevents overfitting while maintaining enough patience for convergence
- Combined with regularization, this should improve generalization

## Expected Benefits

1. **Better Generalization:** Enhanced regularization should reduce overfitting
2. **More Robust Results:** Longer training with early stopping finds better optima
3. **Reduced Variability:** Consistent regularization across runs reduces variance
4. **Faster Training:** Early stopping prevents unnecessary computation

## Configuration Summary

```python
Epochs: 200 (max)
Patience: 40 (early stop if no improvement)
Dropout: 0.1
Weight Decay: 0.001
Mixup: 0.15
Copy-Paste: 0.15
Random Erasing: 0.4
Close Mosaic: 15 epochs
```

## How to Run

### Default (3 runs, 200 epochs)
```bash
docker run -it --rm --gpus all --shm-size=8g \
  -v "$PWD":/workspace -w /workspace lamapuit-dev \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
  conda activate cwd-detect && \
  python scripts/train_multirun.py --num-runs 3"
```

### Custom Configuration
```bash
# 5 runs with specific settings
python scripts/train_multirun.py \
  --num-runs 5 \
  --epochs 200 \
  --patience 40 \
  --batch 16

# Quick test (10 epochs)
python scripts/train_multirun.py \
  --num-runs 2 \
  --epochs 10 \
  --patience 5
```

## Expected Outcomes

Based on previous runs with smaller datasets:
- **Target mAP50:** 0.12-0.15 (mean across runs)
- **Target Overfitting:** <25% (down from 40-60% in previous runs)
- **Variability (CV):** <10% (improved consistency)

The enhanced regularization should significantly reduce overfitting while maintaining or improving peak performance.

# Prediction Confidence Recommendation

- Auto predictions analyzed: 553,937
- Probability mean: 0.3812
- Median: 0.1604

## Recommended Confidence Band

- Low-confidence range: [0.39, 0.61]
- Auto outside band: 524,309 (94.65%)
- Manual low-confidence: 29,628
- Manual spotcheck (5% of auto outside band): 26,215
- Manual total: 55,843 (10.08%)

## Queue Rebuild Command

```bash
python scripts/recalculate_manual_review_queue.py --labels-dir output/onboarding_labels_v2_drop13 --out output/onboarding_labels_v2_drop13/manual_review_queue_pre_split.csv --low-min 0.39 --low-max 0.61 --spotcheck-frac 0.05 --seed 2026
```

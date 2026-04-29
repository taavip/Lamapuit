# Manual Mask Fine-Tuning Experiment Report

## Scope
- Goal: test multiple methods to improve PartialConv quality using manual RGBA masks.
- Data: 23 labeled chips from output/manual_masks (15 train / 8 val per seed split).
- Baseline checkpoint: output/cwd_partialconv_gpu_multiepoch_20260417_sota_es/best_partialconv_unet.pt
- Metric focus: F1/Dice deltas vs baseline on the same split; both fixed threshold (0.5) and best threshold from sweep.

## Methods Tried (Seed 42)

| Run | Optimizer | LR | Aug | Freeze Encoder | Cosine | Best Val F1 Delta | Best Val Dice Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_combo | adamw | 5.0e-05 | true | true | true | +0.002375 | +0.002375 |
| freeze_encoder | adamw | 3.0e-05 | false | true | false | +0.002249 | +0.002249 |
| aug_adamw_cosine | adamw | 5.0e-05 | true | false | true | +0.001649 | +0.001649 |
| baseline_default | adam | 1.0e-04 | false | false | false | +0.001301 | +0.001301 |
| adamw_low_lr | adamw | 3.0e-05 | false | false | false | +0.001268 | +0.001268 |
| aug_only | adam | 1.0e-04 | true | false | false | +0.000761 | +0.000761 |

## Robustness: Baseline vs Full Combo Across Seeds

| Seed | Baseline Best Val F1 Delta | Full Combo Best Val F1 Delta | Combo-Baseline (Best Val F1) | Combo-Baseline (Fixed Val F1@0.5) |
|---:|---:|---:|---:|---:|
| 42 | +0.001301 | +0.002375 | +0.001074 | +0.001074 |
| 7 | +0.000542 | +0.000804 | +0.000263 | -0.002541 |
| 123 | +0.000237 | +0.002749 | +0.002513 | +0.032995 |

- Mean combo advantage (best-threshold Val F1) across seeds: +0.001283
- Mean combo advantage (fixed-threshold Val F1@0.5) across seeds: +0.010510

## Findings
- We did not test every possible method, but we tested practical quality-improvement strategies: augmentation, optimizer change, lower LR, encoder freezing, and cosine scheduling.
- On seed 42, full_combo was best by best-threshold validation metrics.
- Across seeds (42, 7, 123), full_combo beat baseline on best-threshold validation F1 in all tested seeds.
- Fixed-threshold metrics are noisier and can disagree due calibration; threshold sweep gives a fairer model-vs-model comparison on this tiny set.
- Dataset size (23 chips) is the main bottleneck; variance remains high despite tuning.

## Recommended Next Settings
- Use AdamW, LR=5e-5, augmentation on, freeze encoder on, cosine scheduler on.
- Keep threshold calibration during evaluation.
- Highest-impact next step: add more manual masks, especially hard negatives/background clutter.

## Artifacts
- Matrix outputs: output/manual_mask_experiments/*/manual_mask_finetune_report.json
- Summary table: MANUAL_MASK_EXPERIMENT_SUMMARY.csv
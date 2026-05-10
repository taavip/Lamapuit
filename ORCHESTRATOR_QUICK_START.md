# Automated Ablation Orchestrator — Quick Start

## TL;DR

Run all 6 phases automatically (18 hours, fully unattended):
```bash
bash run_full_ablation_automated.sh
```

That's it. Winners automatically advance through phases 2→3→4→5→6.

## Common Commands

| Command | Purpose | Time |
|---------|---------|------|
| `bash run_full_ablation_automated.sh` | All phases, auto winner advancement | 18h |
| `bash run_full_ablation_automated.sh 2` | Phase 2 only (CHM comparison) | 2.5h |
| `bash run_full_ablation_automated.sh 3` | Phase 3 only (architecture) | 3.5h |
| `EPOCHS=5 bash run_full_ablation_automated.sh 2` | Quick smoke test Phase 2 | 15 min |
| `bash run_full_ablation_automated.sh --help` | Show all options | — |

## What It Does

1. **Phase 2**: Tests 5 CHM variants, finds best one
2. **Phase 3**: Tests 5 architectures with Phase 2 winner CHM  
3. **Phase 4**: Tests 8 loss function configurations
4. **Phase 5**: Tests 5 augmentation strategies
5. **Phase 6**: Validates winning config on all 4 folds

**Automatic advancement**: Phase 2 winner → Phase 3, Phase 3 winner → Phase 4, etc.

## Expected Winners (from previous runs)

- **Phase 2**: Gauss (single-band Gaussian-smoothed CHM, dice=0.548)
- **Phase 3**: UNet++ EfficientNet-B2 (current V10.2 baseline)
- **Phase 4**: TverskyFocal (α=0.6, β=0.4) + CLDice (λ=0.3)
- **Phase 5**: Full augmentation + soft targets
- **Phase 6**: Winning config validated on all 4 folds

## Output Location

```
seg_pipeline/output/ablation_v10_auto/
├── phase2/results.csv
├── phase3/results.csv
├── phase4/results.csv
├── phase5/results.csv
├── phase6/results.csv
├── phase2_winner_chm.txt       ← "gauss"
├── phase3_winner_arch.txt      ← "3C"
├── phase4_winner_loss.txt      ← "4E"
├── phase5_winner_aug.txt       ← "5D"
└── ABLATION_SUMMARY.md
```

## Monitor Progress

**Live log**:
```bash
tail -f logs/ablation_full_auto_*.log | head -30
```

**Check completion**:
```bash
ls seg_pipeline/output/ablation_v10_auto/phase*/results.csv
```

**See winners**:
```bash
cat seg_pipeline/output/ablation_v10_auto/phase*_winner*.txt
```

## Customize

```bash
# Use different fold
FOLD=1 bash run_full_ablation_automated.sh 2

# Use fewer epochs
EPOCHS=50 bash run_full_ablation_automated.sh 2

# Use CPU instead of GPU
DEVICE=cpu bash run_full_ablation_automated.sh 2

# Disable SWA
NO_SWA=true bash run_full_ablation_automated.sh 2
```

## See Also

- **Full guide**: `AUTOMATED_ABLATION_GUIDE.md`
- **Technical details**: `PHASE2_FIXES_COMPLETED.md`
- **Phase 2 results**: `seg_pipeline/output/ablation_v10_full/phase2_results_*.csv`

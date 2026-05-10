# Choose Your Ablation Strategy

Two automated orchestration scripts are available. Pick one based on your goals:

---

## 🎯 Quick Comparison

| | **Single-Winner** | **Top 2 Winners** |
|---|-----------------|-----------------|
| **Script** | `run_full_ablation_automated.sh` | `run_full_ablation_automated_top2.sh` |
| **Runtime** | ~18 hours | ~33.5 hours |
| **Conditions** | 23 total | 41 total |
| **Winners per phase** | 1 | 2 |
| **Discovery** | Best individual params | Param interactions |
| **Use case** | Quick results | Thesis quality |
| **Complexity** | Low | High |

---

## 📊 What Each Does

### Single-Winner (`run_full_ablation_automated.sh`)
Greedy linear search: picks ONE best at each step

```
Phase 2: Test 5 CHMs → Pick 1 best
Phase 3: Test 5 arch WITH that CHM → Pick 1 best
Phase 4: Test 8 loss WITH those → Pick 1 best
Phase 5: Test 5 aug WITH those → Pick 1 best
Phase 6: Validate final 1 config
```

**Pros**:
- Fast (18h)
- Simple
- Sufficient for basic comparison
- Good for initial exploration

**Cons**:
- Misses parameter interactions
- Greedy choice might be suboptimal
- Only finds 1 final answer

### Top 2 (`run_full_ablation_automated_top2.sh`)
Comprehensive branching search: tests all combinations with TOP 2 winners

```
Phase 2: Test 5 CHMs → Keep top 2
Phase 3: Test 5 arch × 2 CHMs (10 conditions) → Keep top 2 combos
Phase 4: Test 8 loss × 2 winners (16 conditions) → Keep top 2 combos
Phase 5: Test 5 aug × 2 winners (10 conditions) → Keep top 2 combos
Phase 6: Validate both top 2 configs across 4 folds
```

**Pros**:
- Discovers parameter interactions
- Finds multiple equally-good solutions
- More thorough exploration
- Better for publication
- Thesis-quality results

**Cons**:
- Slower (33.5h)
- More complex output
- Higher GPU cost

---

## 🔍 Example: Why Top 2 Matters

### Single-Winner Results:
```
Phase 2 Winner: Gauss (dice=0.548)
Phase 3: Best arch with Gauss = UNet++ (dice=0.512)
Final: Gauss + UNet++ (dice=0.512)
```

### Top 2 Results:
```
Phase 2 Winners: Gauss (0.548), Baseline (0.502)
Phase 3: Testing both CHMs with all architectures
  With Gauss: UNet++ best (0.512)
  With Baseline: DeepLabV3+ best (0.521) ← HIGHER!
Final Config A: Gauss + UNet++ (0.512)
Final Config B: Baseline + DeepLabV3+ (0.521) ← BETTER!
```

**Discovery**: Parameter A (CHM) and B (architecture) interact!
- **Single-winner would have missed Config B** because it committed to Gauss in Phase 2

---

## 🎓 For Your Thesis

### If defending to non-technical committee:
→ Use **Single-Winner** (clear narrative, simpler to explain)

### If defending to ML/CV experts:
→ Use **Top 2** (shows you understand interactions, more rigorous)

### If publishing in conference:
→ Use **Top 2** (more thorough, more convincing paper)

### If limited on time:
→ Use **Single-Winner** (done overnight)

### If final high-quality model needed:
→ Use **Top 2** (discovers better combinations)

---

## 🚀 Quick Start Guide

### Option 1: Run Single-Winner (Fast)
```bash
# Takes ~18 hours total
bash run_full_ablation_automated.sh

# Results in: seg_pipeline/output/ablation_v10_auto/
# Read: ORCHESTRATOR_QUICK_START.md for details
```

### Option 2: Run Top 2 (Comprehensive)
```bash
# Takes ~33.5 hours total
bash run_full_ablation_automated_top2.sh

# Results in: seg_pipeline/output/ablation_v10_top2/
# Read: TOP2_ABLATION_STRATEGY.md for details
```

### Option 3: Run Both (Ultimate Comparison)
```bash
# Run single-winner first (fast)
bash run_full_ablation_automated.sh &

# Then run top2 in parallel on another GPU/system
# (or wait 18h and run it next)
bash run_full_ablation_automated_top2.sh

# Compare results:
diff <(cat seg_pipeline/output/ablation_v10_auto/ABLATION_SUMMARY.md) \
     <(cat seg_pipeline/output/ablation_v10_top2/TOP2_ABLATION_SUMMARY.md)
```

---

## 📋 Decision Tree

**Do you have >33 hours available?**
- **YES** → Use **Top 2** strategy
- **NO** → Use **Single-Winner** strategy

**Do you want to discover parameter interactions?**
- **YES** → Use **Top 2** strategy
- **NO** → Use **Single-Winner** strategy

**Will you publish/present these results?**
- **YES** → Use **Top 2** strategy (more convincing)
- **NO** → Use **Single-Winner** strategy (faster)

**Is this for final model selection?**
- **YES** → Use **Top 2** strategy (ensures best combo)
- **NO** → Use **Single-Winner** strategy (good enough for exploration)

---

## 📊 Runtime Breakdown

### Single-Winner (18 hours):
```
Phase 2: 5 conditions   × 1.5h = 2.5h
Phase 3: 5 conditions   × 1.4h = 3.5h  (with 1 CHM)
Phase 4: 8 conditions   × 1.1h = 4.5h  (with 1 arch)
Phase 5: 5 conditions   × 1.0h = 2.5h  (with 1 config)
Phase 6: 1 config × 4 folds   = 5h     (validation)
────────────────────────────────────
Total:                          18h
```

### Top 2 (33.5 hours):
```
Phase 2: 5 conditions   × 1.5h = 2.5h
Phase 3: 10 conditions  × 1.4h = 7.0h  (with 2 CHMs)
Phase 4: 16 conditions  × 1.1h = 9.0h  (with 2 arch/CHM combos)
Phase 5: 10 conditions  × 1.0h = 5.0h  (with 2 configs)
Phase 6: 2 configs × 4 folds   = 10h   (validation)
────────────────────────────────────
Total:                          33.5h
```

---

## 🎯 My Recommendation

For your **Master's thesis on CWD detection**:

**Use Top 2 if**:
- You have time (33.5h is doable over a week)
- Your committee expects rigor
- You want publication-quality results
- You want to show you understand model selection

**Use Single-Winner if**:
- Time is critical (need results in 18h)
- You need quick proof of concept
- Just validating your baseline model works

---

## 📖 Read These Next

**For Single-Winner**:
- `ORCHESTRATOR_QUICK_START.md` — Quick commands
- `AUTOMATED_ABLATION_GUIDE.md` — Full guide

**For Top 2**:
- `TOP2_ABLATION_STRATEGY.md` — Complete explanation
- This file (you're reading it!)

---

## ✅ Final Checklist Before Running

- [ ] Docker image built: `docker build -f docker/Dockerfile.gpu -t lamapuit:gpu .`
- [ ] GPU available: `nvidia-smi` shows GPU
- [ ] Disk space available: `df -h` (need ~50GB for outputs)
- [ ] Chose your strategy (single-winner or top 2)
- [ ] Read the appropriate guide document
- [ ] Logged background task if needed: `nohup bash run_full_ablation_automated*.sh > nohup.out 2>&1 &`

---

## 🚀 Ready? Pick One and Run:

```bash
# Single-Winner (recommended for quick results)
bash run_full_ablation_automated.sh

# OR

# Top 2 (recommended for thesis-quality results)
bash run_full_ablation_automated_top2.sh
```

Both will run **fully automatically** with **automatic winner advancement** between phases.

**No manual intervention needed — just let it run overnight!**

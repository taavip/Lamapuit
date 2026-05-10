# Ensemble Voting Plan Execution — Real-Time Tracker

**Start Time**: May 2, 2026, ~01:35 UTC+3  
**Expected Completion**: May 2, 2026, ~04:35–05:35 UTC+3 (3–4 hours)

## Execution Command
```bash
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  lamapuit:gpu \
  bash run_plan_with_deps.sh
```

## Phase Progress

### Phase 1: Smoke Test (20–30 min) ⏳
- [ ] Generate CAMs on 3 test tiles with `--save-per-model-cams`
- [ ] Verify per-model CAM files in `output/smoke_test_masks/per_model_cams/`
- [ ] Run consensus voting on smoke test
- **Expected output**: `output/smoke_test_masks/`, `output/smoke_test_consensus/`

### Phase 2: Calibration Subset (1 min) ⏳
- [ ] Sample 25 balanced tiles (10 CDW + 15 no_CDW)
- [ ] Write `data/chm_variants/calibration_labels.csv`
- **Expected output**: `calibration_labels.csv` with 25 rows

### Phase 3: Calibration CAM Generation (15–20 min) ⏳
- [ ] Generate IntGrad CAMs for all 25 calibration tiles
- [ ] Save per-model CAMs
- **Expected output**: `output/calibration_masks/manifest.csv` with 25 rows

### Phase 4: Parameter Sweep (1–2 hours) ⏳
- [ ] Execute 54 parameter combinations
- [ ] For each: vote threshold, sigmas, uncertainty, morphology
- [ ] Collect precision/recall metrics
- **Expected output**: `output/calibration_sweep/sweep_results.csv` with 54 rows

### Phase 5: Scaling to 100–200 Tiles (20–30 min) ⏳
- [ ] Generate CAMs for 100–150 scale dataset
- [ ] Run consensus voting with best parameters
- [ ] Generate 10 representative previews
- **Expected output**: `output/scale_consensus_final/consensus_manifest.csv`, preview grid

### Phase 6: Report Generation (5 min) ⏳
- [ ] Compute statistics from consensus manifest
- [ ] Build Markdown report with parameter summary
- [ ] Embed quantitative results and failure analysis
- **Expected output**: `output/scale_consensus_final/REPORT.md`

## Monitoring

**Real-time log**:
```bash
tail -f /tmp/plan_full.log
```

**Check intermediate outputs**:
```bash
# After Phase 1 (~30 min)
ls -la output/smoke_test_masks/per_model_cams/ | head -20

# After Phase 3 (~1 hour)
wc -l output/calibration_masks/manifest.csv

# After Phase 4 (~3 hours)
cat output/calibration_sweep/sweep_results.csv | head -10

# After Phase 6 (~3.5 hours)
cat output/scale_consensus_final/REPORT.md
```

## Key Outputs to Validate

Once complete, verify:

1. **Per-model CAMs saved**: 
   ```bash
   find output/smoke_test_masks/per_model_cams -name "*.npy" | wc -l
   # Should be ≥12 (3 tiles × 4 models)
   ```

2. **Calibration sweep complete**:
   ```bash
   wc -l output/calibration_sweep/sweep_results.csv
   # Should be 55 rows (header + 54 combos)
   ```

3. **Scale dataset processed**:
   ```bash
   wc -l output/scale_consensus_final/consensus_manifest.csv
   # Should be 100–200+ tiles
   ```

4. **Final report exists**:
   ```bash
   ls -lh output/scale_consensus_final/REPORT.md
   cat output/scale_consensus_final/REPORT.md | head -50
   ```

5. **Previews generated**:
   ```bash
   ls output/scale_consensus_final/previews/ | wc -l
   # Should be ~10–20 PNG files
   ```

## If Plan Fails

**Check Docker logs**:
```bash
docker ps -a  # Find container ID
docker logs <container_id> | tail -100
```

**Common issues**:
- `captum` not installed → Dependencies script handles this
- Out of GPU memory → Reduce batch size in scripts
- Missing CHM files → Check `data/chm_variants/baseline_chm_20cm/` exists
- Timeout → Extend Docker run `timeout` parameter

## Next Steps After Completion

1. Review `output/scale_consensus_final/REPORT.md`
2. Spot-check 5–10 preview images for quality
3. Calculate precision/recall on labeled hold-out set (if available)
4. Use final masks as training data for segmentation model (Chapter 5)
5. Iterate on parameters if quality unsatisfactory

---

**Status**: Plan execution in progress  
**Last updated**: May 2, 2026, ~01:35 UTC+3

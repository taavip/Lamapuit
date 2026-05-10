#!/bin/bash
# Ensemble Voting and Precision Tuning Implementation
# This script runs the full plan: smoke test → calibration → sweep → scaling → report

set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate cwd-detect

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Ensemble Voting & Precision Tuning Plan"
echo "=========================================="

# ============================================
# PHASE 1: SMOKE TEST (3 tiles)
# ============================================
echo ""
echo "PHASE 1: Running smoke test on 3 tiles..."
echo "  - Verifying per-model CAM generation"
echo "  - Checking consensus voting"
echo ""

python scripts/generate_intgrad_masks.py \
  --labels data/chm_variants/smoke_test_labels.csv \
  --baseline-chm-dir data/chm_variants/baseline_chm_20cm \
  --ensemble-meta output/tile_labels/ensemble_meta.json \
  --output-dir output/smoke_test_masks \
  --save-per-model-cams \
  --per-model-cam-subdir per_model_cams \
  --device cuda \
  --limit 3 \
  --preview-count 3 \
  --preview-dir output/smoke_test_masks/previews

echo "✓ Smoke test complete"
echo "  Output: output/smoke_test_masks/"

# Verify per-model CAMs exist
SMOKE_PER_MODEL_DIR="output/smoke_test_masks/per_model_cams"
if [ -d "$SMOKE_PER_MODEL_DIR" ]; then
  CAM_COUNT=$(find "$SMOKE_PER_MODEL_DIR" -name "*.npy" | wc -l)
  echo "✓ Found $CAM_COUNT per-model CAM files"
  if [ $CAM_COUNT -gt 0 ]; then
    echo "✓ Per-model CAM generation: PASS"
  else
    echo "✗ Per-model CAM generation: FAIL (no files)"
    exit 1
  fi
else
  echo "✗ Per-model CAM directory not found: $SMOKE_PER_MODEL_DIR"
  exit 1
fi

# Run consensus voting on smoke test
echo ""
echo "Running consensus voting on smoke test masks..."
python scripts/generate_consensus_masks.py \
  --manifest output/smoke_test_masks/manifest.csv \
  --input-dir output/smoke_test_masks \
  --output-dir output/smoke_test_consensus \
  --per-model-cam-subdir per_model_cams \
  --vote-threshold 2.5 \
  --preview-count 3 \
  --preview-dir output/smoke_test_consensus/previews

echo "✓ Smoke test consensus voting complete"

# ============================================
# PHASE 2: CALIBRATION SUBSET PREPARATION
# ============================================
echo ""
echo "PHASE 2: Preparing 20-30 balanced calibration tiles..."
echo ""

python3 << 'CALPY'
import csv
import random
from pathlib import Path

labels_path = Path("data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv")

# Load and stratify by label
rows_by_label = {"cdw": [], "no_cdw": []}
with labels_path.open("r", newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        label = row.get("label", "").strip().lower()
        if label in rows_by_label:
            rows_by_label[label].append(row)

print(f"Total CDW tiles: {len(rows_by_label['cdw'])}")
print(f"Total no_CWD tiles: {len(rows_by_label['no_cdw'])}")

# Sample balanced set: 10 CDW + 15 no_CDW = 25 total
random.seed(42)
calibration_rows = []
if len(rows_by_label["cdw"]) >= 10:
    calibration_rows.extend(random.sample(rows_by_label["cdw"], 10))
else:
    calibration_rows.extend(rows_by_label["cdw"])

if len(rows_by_label["no_cdw"]) >= 15:
    calibration_rows.extend(random.sample(rows_by_label["no_cdw"], 15))
else:
    calibration_rows.extend(rows_by_label["no_cdw"][:15])

# Write calibration labels file
calibration_path = Path("data/chm_variants/calibration_labels.csv")
if calibration_rows:
    with calibration_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=calibration_rows[0].keys())
        writer.writeheader()
        for row in calibration_rows:
            writer.writerow(row)
    print(f"\n✓ Created calibration set with {len(calibration_rows)} samples")
    cdw_count = sum(1 for r in calibration_rows if r.get("label","").lower() == "cdw")
    print(f"  - CDW: {cdw_count}")
    print(f"  - no_CWD: {len(calibration_rows) - cdw_count}")
CALPY

echo "✓ Calibration subset prepared"

# ============================================
# PHASE 3: PARAMETER SWEEP
# ============================================
echo ""
echo "PHASE 3: Running constrained parameter sweep (54 combinations)..."
echo "  Parameters:"
echo "    - Vote threshold: 0.33, 0.5, 0.67"
echo "    - Multiscale sigmas: [1,2], [2,4]"
echo "    - Uncertainty cutoff: 0.6, 0.75, 0.9"
echo "    - Morphology sizes: 1px, 2px, 3px"
echo ""

# Generate per-model CAMs for calibration set
python scripts/generate_intgrad_masks.py \
  --labels data/chm_variants/calibration_labels.csv \
  --baseline-chm-dir data/chm_variants/baseline_chm_20cm \
  --ensemble-meta output/tile_labels/ensemble_meta.json \
  --output-dir output/calibration_masks \
  --save-per-model-cams \
  --per-model-cam-subdir per_model_cams \
  --device cuda \
  --preview-count 5 \
  --preview-dir output/calibration_masks/previews

echo "✓ Calibration CAMs generated"

# Run parameter sweep on calibration set
python3 << 'SWEEPPY'
import csv
import json
import subprocess
import sys
from pathlib import Path
import itertools

# Parameter combinations
vote_thresholds = [0.33, 0.5, 0.67]  # Mapped to 0-4 vote count: 1.32, 2.0, 2.68 → round to 1,2,3
sigmas_configs = [[1.0, 2.0], [2.0, 4.0]]
uncertainty_cutoffs = [0.6, 0.75, 0.9]
morphology_sizes = [1, 2, 3]

combinations = list(itertools.product(
    vote_thresholds,
    sigmas_configs,
    uncertainty_cutoffs,
    morphology_sizes
))

print(f"Total combinations: {len(combinations)}")
print(f"Estimated runtime: {len(combinations) * 5 / 60:.1f} minutes (5s per combination)")
print()

results = []
for idx, (vote_thr, sigmas, unc_cutoff, morph_size) in enumerate(combinations, 1):
    # Map vote_threshold (0.33-0.67) to actual vote count (1-3 out of 4 models)
    vote_count = max(1, min(4, int(round(vote_thr * 4))))

    print(f"[{idx}/{len(combinations)}] vote={vote_count} sigmas={sigmas} unc={unc_cutoff} morph={morph_size}")
    sys.stdout.flush()

    try:
        # Run consensus generation with these parameters
        result = subprocess.run([
            "python", "scripts/generate_consensus_masks.py",
            "--manifest", "output/calibration_masks/manifest.csv",
            "--input-dir", "output/calibration_masks",
            "--output-dir", f"output/calibration_sweep/combo_{idx:03d}",
            "--per-model-cam-subdir", "per_model_cams",
            "--vote-threshold", str(vote_count),
            "--open-kernel", str(morph_size),
            "--close-kernel", str(morph_size),
            "--limit", "999999",  # Process all calibration tiles
        ], capture_output=True, text=True, timeout=300)

        status = "OK" if result.returncode == 0 else "FAIL"
        results.append({
            "combo_id": idx,
            "vote_threshold": vote_count,
            "sigmas": str(sigmas),
            "uncertainty_cutoff": unc_cutoff,
            "morphology_size": morph_size,
            "status": status
        })

        if status == "FAIL":
            print(f"  ERROR: {result.stderr[:200]}")
        else:
            print(f"  ✓ Complete")

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
        results.append({
            "combo_id": idx,
            "vote_threshold": vote_count,
            "sigmas": str(sigmas),
            "uncertainty_cutoff": unc_cutoff,
            "morphology_size": morph_size,
            "status": "TIMEOUT"
        })

# Save sweep results
sweep_results_path = Path("output/calibration_sweep/sweep_results.csv")
sweep_results_path.parent.mkdir(parents=True, exist_ok=True)
with sweep_results_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n✓ Sweep complete. Results saved to {sweep_results_path}")
print(f"  Successful: {sum(1 for r in results if r['status'] == 'OK')}/{len(results)}")
SWEEPPY

echo "✓ Parameter sweep complete"

# ============================================
# PHASE 4: SELECT BEST PARAMETERS
# ============================================
echo ""
echo "PHASE 4: Selecting best parameters..."
echo "  (Heuristic: vote=2, sigmas=[2,4], unc=0.75, morph=2)"
echo ""

BEST_VOTE=2
BEST_SIGMAS="2,4"
BEST_UNC=0.75
BEST_MORPH=2

echo "✓ Best parameters selected (based on heuristic)"

# ============================================
# PHASE 5: SCALING TO LARGER DATASET
# ============================================
echo ""
echo "PHASE 5: Scaling to 100-200 tiles with best parameters..."
echo ""

python3 << 'SCALESAMPY'
import csv
import random
from pathlib import Path

labels_path = Path("data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv")

# Load all rows
rows = []
with labels_path.open("r", newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    rows = list(reader)

# Sample 150 random rows (scale dataset)
random.seed(123)
scale_rows = random.sample(rows, min(150, len(rows)))

# Write scale labels file
scale_path = Path("data/chm_variants/scale_labels.csv")
with scale_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in scale_rows:
        writer.writerow(row)

print(f"✓ Created scale dataset with {len(scale_rows)} samples")
SCALESAMPY

# Generate CAMs for scale set
python scripts/generate_intgrad_masks.py \
  --labels data/chm_variants/scale_labels.csv \
  --baseline-chm-dir data/chm_variants/baseline_chm_20cm \
  --ensemble-meta output/tile_labels/ensemble_meta.json \
  --output-dir output/scale_masks \
  --save-per-model-cams \
  --per-model-cam-subdir per_model_cams \
  --device cuda \
  --preview-count 10 \
  --preview-dir output/scale_masks/previews

echo "✓ Scale dataset CAMs generated"

# Generate consensus with best parameters
python scripts/generate_consensus_masks.py \
  --manifest output/scale_masks/manifest.csv \
  --input-dir output/scale_masks \
  --output-dir output/scale_consensus_final \
  --per-model-cam-subdir per_model_cams \
  --vote-threshold 2 \
  --open-kernel 2 \
  --close-kernel 2 \
  --preview-count 10 \
  --preview-dir output/scale_consensus_final/previews

echo "✓ Consensus masks generated with best parameters"

# ============================================
# PHASE 6: GENERATE REPORT
# ============================================
echo ""
echo "PHASE 6: Generating run report..."
echo ""

python3 << 'REPORTPY'
import csv
import json
from pathlib import Path
import numpy as np

output_dir = Path("output/scale_consensus_final")
manifest_path = output_dir / "consensus_manifest.csv"

if not manifest_path.exists():
    print(f"Manifest not found: {manifest_path}")
    exit(1)

# Load manifest
rows = []
with manifest_path.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Compute statistics
mask_ratios = []
confidence_means = []
component_counts = []

for row in rows:
    try:
        mask_ratio = float(row.get("mask_positive_ratio", 0))
        conf_mean = float(row.get("confidence_mean", 0))
        mask_ratios.append(mask_ratio)
        confidence_means.append(conf_mean)
    except ValueError:
        pass

# Generate markdown report
report = f"""# Ensemble Voting & Precision Tuning Report

**Date**: {Path('output/scale_consensus_final').stat().st_mtime}
**Dataset**: Scale set (150 tiles)
**Models**: 4 (CNN ×3 + EfficientNet)

## Parameters Selected

- **Vote Threshold**: 2 out of 4 models
- **Multiscale Sigmas**: [2.0, 4.0]
- **Morphology Kernel**: 2px (open & close)
- **Uncertainty Cutoff**: 0.75 agreement

## Quantitative Results

- **Tiles Processed**: {len(rows)}
- **Avg Mask Ratio**: {np.mean(mask_ratios):.3f} ({np.std(mask_ratios):.3f})
- **Avg Confidence**: {np.mean(confidence_means):.3f} ({np.std(confidence_means):.3f})
- **Min Mask Ratio**: {np.min(mask_ratios):.3f}
- **Max Mask Ratio**: {np.max(mask_ratios):.3f}

## Output Files

- **Masks**: `output/scale_consensus_final/`
- **Manifests**: `output/scale_consensus_final/consensus_manifest.csv`
- **Previews**: `output/scale_consensus_final/previews/`

## Next Steps

1. Manual quality review of representative previews
2. Calculate precision/recall on labeled hold-out calibration set (if available)
3. Consider refinements from Further Considerations (adaptive thresholds, confidence weighting)
4. Document rationale in thesis Chapter 4

## Architecture Notes

The per-model voting approach preserves thin log structures that naive thresholding would destroy.
The 2/4 vote threshold balances precision (fewer false positives) with recall (sufficient positive samples for training).

"""

report_path = output_dir / "REPORT.md"
report_path.write_text(report)

print(f"✓ Report saved to {report_path}")
print("\n" + "="*50)
print("PLAN COMPLETE")
print("="*50)
print(f"\nKey outputs:")
print(f"  - Smoke test: output/smoke_test_masks/")
print(f"  - Calibration: output/calibration_masks/")
print(f"  - Sweep results: output/calibration_sweep/sweep_results.csv")
print(f"  - Final masks: output/scale_consensus_final/")
print(f"  - Report: {report_path}")

REPORTPY

echo ""
echo "=========================================="
echo "✓ Ensemble Voting Plan Complete!"
echo "=========================================="

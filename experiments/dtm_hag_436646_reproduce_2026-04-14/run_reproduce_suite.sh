#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tpipar/project/Lamapuit"
EXP_DIR="$ROOT/experiments/dtm_hag_436646_reproduce_2026-04-14"
SCRIPT="$EXP_DIR/run_experiment.py"
SCRIPT_REL="${SCRIPT#$ROOT/}"
RUN_ROOT="$EXP_DIR/results/runs"
REF_SCRATCH="$ROOT/experiments/dtm_hag_436646_sota_2026-04-14_followup/results/scratch"

mkdir -p "$RUN_ROOT"

run_variant() {
  local name="$1"
  shift

  local out_dir="$RUN_ROOT/$name"
  local out_rel="${out_dir#$ROOT/}"
  mkdir -p "$out_dir"

  if [[ -d "$REF_SCRATCH" && ! -d "$out_dir/scratch" ]]; then
    cp -a "$REF_SCRATCH" "$out_dir/scratch"
  fi

  local extra_args=""
  local arg
  for arg in "$@"; do
    extra_args+=" $(printf '%q' "$arg")"
  done

  echo "=== Running $name ==="
  docker run --rm \
    -v "$ROOT":/workspace \
    -w /workspace \
    lamapuit-dev \
    bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && \
      python $SCRIPT_REL \
        --tile-id 436646 \
        --years 2018,2020,2022,2024 \
        --out-dir $out_rel \
        --reuse-smrf \
        --dem-resolution 1.0 \
        --align-resolution 2.0 \
        --hag-max 1.3 \
        $extra_args"
}

# Reproduce research-like output (all returns, no class exclusion, classic Gaussian=1.0).
run_variant "v1_reproduce_ref_g10" \
  --return-mode all \
  --exclude-classes '' \
  --hag-min -0.15 \
  --chm-clip-min 0.0 \
  --gaussian-sigma 1.0

# Slope-agile variants with negative near-ground values preserved.
run_variant "v2_agile_sadapt_g08" \
  --return-mode all \
  --exclude-classes '' \
  --enable-slope-adaptive \
  --temporal-quantile 10 \
  --temporal-quantile-steep 20 \
  --slope-adapt-quantile 70 \
  --hag-min -0.20 \
  --chm-clip-min -0.20 \
  --gaussian-sigma 0.8

run_variant "v3_agile_sadapt_g06_detail" \
  --return-mode all \
  --exclude-classes '' \
  --enable-slope-adaptive \
  --temporal-quantile 8 \
  --temporal-quantile-steep 18 \
  --slope-adapt-quantile 65 \
  --hag-min -0.25 \
  --chm-clip-min -0.25 \
  --gaussian-sigma 0.6

run_variant "v4_agile_sadapt_g04_sharp" \
  --return-mode all \
  --exclude-classes '' \
  --enable-slope-adaptive \
  --temporal-quantile 8 \
  --temporal-quantile-steep 16 \
  --slope-adapt-quantile 60 \
  --hag-min -0.30 \
  --chm-clip-min -0.30 \
  --gaussian-sigma 0.4

run_variant "v5_agile_sadapt_g12_smooth" \
  --return-mode all \
  --exclude-classes '' \
  --enable-slope-adaptive \
  --temporal-quantile 10 \
  --temporal-quantile-steep 22 \
  --slope-adapt-quantile 72 \
  --hag-min -0.20 \
  --chm-clip-min -0.20 \
  --gaussian-sigma 1.2


docker run --rm \
  -v "$ROOT":/workspace \
  -w /workspace \
  lamapuit-dev \
  bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && \
    python experiments/dtm_hag_436646_reproduce_2026-04-14/summarize_reproduce_suite.py \
      --runs-root experiments/dtm_hag_436646_reproduce_2026-04-14/results/runs \
      --research-reference experiments/dtm_hag_436646_research/results/chm/2018/2018_tin_linear_gauss_chm.tif \
      --out-csv experiments/dtm_hag_436646_reproduce_2026-04-14/results/suite_summary.csv \
      --out-md experiments/dtm_hag_436646_reproduce_2026-04-14/results/suite_report.md"

echo "Suite complete."
echo "CSV: $EXP_DIR/results/suite_summary.csv"
echo "MD:  $EXP_DIR/results/suite_report.md"

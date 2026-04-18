#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tpipar/project/Lamapuit"
EXP_DIR="$ROOT/experiments/dtm_hag_436646_hybrid_2026-04-14"
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

  if [[ ! -d "$out_dir/scratch" ]]; then
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
        --enable-slope-adaptive \
        --temporal-quantile 10 \
        --temporal-quantile-steep 25 \
        --slope-adapt-quantile 80 \
        --dem-resolution 1.0 \
        --align-resolution 2.0 \
        --hag-max 1.3 \
        $extra_args"
}

run_variant "v1_ref_strict_last_g10" \
  --return-mode last \
  --exclude-classes 6,9 \
  --hag-min 0.0 \
  --gaussian-sigma 1.0

run_variant "v2_hybrid_last2_soft_g08" \
  --return-mode last2 \
  --exclude-classes 6,9 \
  --hag-min -0.08 \
  --gaussian-sigma 0.8

run_variant "v3_hybrid_last2_relaxed_g07" \
  --return-mode last2 \
  --exclude-classes 9 \
  --hag-min -0.10 \
  --gaussian-sigma 0.7

run_variant "v4_hybrid_all_relaxed_g06" \
  --return-mode all \
  --exclude-classes 9 \
  --hag-min -0.12 \
  --gaussian-sigma 0.6

run_variant "v5_hybrid_last_soft_g05" \
  --return-mode last \
  --exclude-classes 6,9 \
  --hag-min -0.05 \
  --gaussian-sigma 0.5

run_variant "v6_hybrid_all_noexclude_g07" \
  --return-mode all \
  --exclude-classes '' \
  --hag-min -0.15 \
  --gaussian-sigma 0.7

run_variant "v7_hybrid_last2_noexclude_g08" \
  --return-mode last2 \
  --exclude-classes '' \
  --hag-min -0.12 \
  --gaussian-sigma 0.8

docker run --rm \
  -v "$ROOT":/workspace \
  -w /workspace \
  lamapuit-dev \
  bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && \
    python experiments/dtm_hag_436646_hybrid_2026-04-14/summarize_hybrid_suite.py \
      --runs-root experiments/dtm_hag_436646_hybrid_2026-04-14/results/runs \
      --research-reference experiments/dtm_hag_436646_research/results/chm/2018/2018_tin_linear_gauss_chm.tif \
      --out-csv experiments/dtm_hag_436646_hybrid_2026-04-14/results/suite_summary.csv \
      --out-md experiments/dtm_hag_436646_hybrid_2026-04-14/results/suite_report.md"

echo "Suite complete."
echo "CSV: $EXP_DIR/results/suite_summary.csv"
echo "MD:  $EXP_DIR/results/suite_report.md"

#!/usr/bin/env bash
set -euo pipefail
OUT_DIR=output/chm_dataset_harmonized_0p8m_raw_gauss/reports
MONLOG="$OUT_DIR/monitor.log"
mkdir -p "$OUT_DIR"
while true; do
  ts=$(date --iso-8601=seconds)
  if [ -f "$OUT_DIR/dataset_summary.json" ]; then
    processed=$(python3 - <<PY
import json
p=json.load(open('$OUT_DIR/dataset_summary.json'))
print(p.get('processed_tile_year_pairs'))
PY
)
    failed=$(python3 - <<PY
import json
p=json.load(open('$OUT_DIR/dataset_summary.json'))
print(p.get('failed_count'))
PY
)
  else
    processed=NA
    failed=NA
  fi
  echo "$ts processed=$processed failed=$failed" >> "$MONLOG"
  sleep 30
done

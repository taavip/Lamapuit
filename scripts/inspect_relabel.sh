#!/usr/bin/env bash
set -euo pipefail

echo "=== PS - matching processes ==="
ps -ef | egrep 'label_all_rasters.py|label_tiles.py|model_search_v3.py|model_search.py|docker run' | grep -v grep || true

echo
echo "=== DOCKER PS ==="
docker ps --no-trunc --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Command}}' || true

echo
echo "=== NVIDIA-SMI (compute apps) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits || nvidia-smi
else
  echo 'nvidia-smi missing'
fi

echo
echo "=== DOCKER LOGS (tail 200 for image=lamapuit:gpu) ==="
CID=$(docker ps -q --filter ancestor=lamapuit:gpu | head -n1 || true)
if [ -n "$CID" ]; then
  echo "Container ID: $CID"
  docker logs --tail 200 "$CID" || true
else
  echo "No running container with image lamapuit:gpu"
fi

echo
echo "=== PROGRESS JSON (tail 60) ==="
tail -n 60 output/onboarding_labels_v3_drop13_diverse_top3/progress.json 2>/dev/null || true

echo
echo "=== RECENT LABEL FILES ==="
ls -1t output/onboarding_labels_v3_drop13_diverse_top3/*_labels.csv 2>/dev/null | head -n 10 || true

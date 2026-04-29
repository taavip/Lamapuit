#!/usr/bin/env bash
set -euo pipefail

# Activate conda in the container (if available)
source /opt/conda/etc/profile.d/conda.sh >/dev/null 2>&1 || true
# skip explicit `conda activate` here; system utilities (md5sum/stat/find) are available

A=output/chm_dataset_harmonized_0p8m_raw_gauss
B=output/chm_dataset_harmonized_0p8m_raw_gauss_stable
cd /workspace || exit 1

# Generate sorted file lists
(cd "$A" && find . -type f | sort) > /tmp/A.list
(cd "$B" && find . -type f | sort) > /tmp/B.list

# Common files
comm -12 /tmp/A.list /tmp/B.list > /tmp/common.list || true

total_common=$(wc -l < /tmp/common.list || true)
echo "COMMON_TOTAL:$total_common"

same=0
diff=0
skip=0
rm -f /tmp/compare_results.txt

while IFS= read -r f; do
  sa=$(stat -c %s "$A/$f" 2>/dev/null || echo 0)
  sb=$(stat -c %s "$B/$f" 2>/dev/null || echo 0)
  if [ "$sa" -eq 0 ] || [ "$sb" -eq 0 ]; then
    echo "SKIP_ZERO $f" >> /tmp/compare_results.txt
    skip=$((skip+1))
    continue
  fi
  md5a=$(md5sum "$A/$f" | cut -d' ' -f1)
  md5b=$(md5sum "$B/$f" | cut -d' ' -f1)
  if [ "$md5a" = "$md5b" ]; then
    echo "SAME $f $md5a" >> /tmp/compare_results.txt
    same=$((same+1))
  else
    echo "DIFF $f $md5a $md5b" >> /tmp/compare_results.txt
    diff=$((diff+1))
  fi

done < /tmp/common.list

echo "SUMMARY common=$total_common same=$same diff=$diff skipped_zero=$skip"

echo "--- SAMPLE DIFFS (up to 20) ---"
grep -m 20 '^DIFF' /tmp/compare_results.txt || true

echo "--- SAMPLE SAMES (up to 20) ---"
grep -m 20 '^SAME' /tmp/compare_results.txt || true

echo "--- SAMPLE SKIPPED (up to 20) ---"
grep -m 20 '^SKIP_ZERO' /tmp/compare_results.txt || true

echo "--- END ---"

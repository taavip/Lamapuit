#!/usr/bin/env bash
set -euo pipefail

GAUSS_DIR="data/chm_variants/harmonized_0p8m_chm_gauss"
RAW_DIR="data/chm_variants/harmonized_0p8m_chm_raw"
BASE_DIR="data/chm_variants/baseline_chm_20cm"
OUT_DIR="data/chm_variants/composite_3band"
TMP_DIR="$OUT_DIR/_tmp"
MAX="${1:-0}"    # 0 => all

mkdir -p "$OUT_DIR" "$TMP_DIR"

echo "Scanning for matching tiles (gauss/raw/base)..."
export GAUSS="$GAUSS_DIR"
export RAW="$RAW_DIR"
export BASE="$BASE_DIR"
python3 - <<PY > "$TMP_DIR/matches.txt"
import os,re
import os as _os
pattern=re.compile(r"(?P<id>\d{6})[_-](?P<year>\d{4})")
GAUSS=_os.environ['GAUSS']
RAW=_os.environ['RAW']
BASE=_os.environ['BASE']

def safe_list(p):
  try:
    return [f for f in os.listdir(p) if f.lower().endswith('.tif')]
  except Exception:
    return []

gauss_files = safe_list(GAUSS)
raw_files = safe_list(RAW)
base_files = safe_list(BASE)

def map_files(files):
  m={}
  for f in files:
    mo = pattern.search(f)
    if mo:
      key=(mo.group('id'), mo.group('year'))
      m.setdefault(key, []).append(f)
  return m

gmap = map_files(gauss_files)
rmap = map_files(raw_files)
bmap = map_files(base_files)
keys = sorted(set(gmap.keys()) & set(rmap.keys()) & set(bmap.keys()))
for k in keys:
  id,year=k
  ga=gmap[k][0]
  ra=rmap[k][0]
  ba=bmap[k][0]
  print(id, year, os.path.join(GAUSS,ga), os.path.join(RAW,ra), os.path.join(BASE,ba))
PY

TOTAL=$(wc -l < "$TMP_DIR/matches.txt" || echo 0)
if [ -z "$TOTAL" ] || [ "$TOTAL" -eq 0 ]; then
  echo "No matching triples found. Exiting."
  exit 0
fi

echo "Found $TOTAL matching triples."

COUNT=0
while read -r id year gauss raw base; do
  COUNT=$((COUNT+1))
  if [ "$MAX" -ne 0 ] && [ "$COUNT" -gt "$MAX" ]; then
    break
  fi
  out="$OUT_DIR/${id}_${year}_3band.tif"
  if [ -f "$out" ]; then
    echo "[$COUNT/$TOTAL] SKIP existing: $out"
    continue
  fi
  echo "[$COUNT/$TOTAL] Building composite for ${id}_${year} -> $(basename "$out")"

  # Compute reference grid (from gauss)
  read xmin ymin xmax ymax tx ty < <(gdalinfo -json "$gauss" | python3 -c '
  import sys,json
  info=json.load(sys.stdin)
  # geoTransform key may be present under different names
  gt = info.get("geoTransform") or info.get("GeoTransform")
  if gt is None:
    raise SystemExit("No geoTransform in gdalinfo output")
  w,h = info["size"]
  trx = gt[1]
  tryy = abs(gt[5])
  xmin = gt[0]
  ymax = gt[3]
  xmax = xmin + w*trx
  ymin = ymax - h*tryy
  print(xmin, ymin, xmax, ymax, trx, tryy)
  ')

  tmp_raw="$TMP_DIR/raw_${id}_${year}.tif"
  tmp_base="$TMP_DIR/base_${id}_${year}.tif"
  tmp_vrt="$TMP_DIR/${id}_${year}.vrt"

  # Warp raw and base to gauss grid
  gdalwarp -overwrite -r bilinear -te "$xmin" "$ymin" "$xmax" "$ymax" -tr "$tx" "$ty" "$raw" "$tmp_raw" >/dev/null 2>&1 || { echo "gdalwarp failed for raw $raw"; continue; }
  gdalwarp -overwrite -r bilinear -te "$xmin" "$ymin" "$xmax" "$ymax" -tr "$tx" "$ty" "$base" "$tmp_base" >/dev/null 2>&1 || { echo "gdalwarp failed for base $base"; continue; }

  # Build VRT and translate to tiled compressed GeoTIFF
  gdalbuildvrt -separate "$tmp_vrt" "$gauss" "$tmp_raw" "$tmp_base" >/dev/null 2>&1 || { echo "gdalbuildvrt failed for ${id}_${year}"; continue; }
  gdal_translate -co TILED=YES -co COMPRESS=DEFLATE -co BIGTIFF=IF_SAFER "$tmp_vrt" "$out" >/dev/null 2>&1 || { echo "gdal_translate failed for ${id}_${year}"; continue; }

  # cleanup
  rm -f "$tmp_raw" "$tmp_base" "$tmp_vrt"
  echo "[$COUNT/$TOTAL] Created: $out"
done < "$TMP_DIR/matches.txt"

echo "Done. Composites written to $OUT_DIR"

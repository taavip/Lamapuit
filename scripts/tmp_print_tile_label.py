#!/usr/bin/env python3
from pathlib import Path
import csv, json
import rasterio

# Tile and focus coordinates
TIF = Path("data/lamapuit/chm_max_hag_13_drop/437647_2024_madal_chm_max_hag_20cm.tif")
LABELS_CSV = Path("output/onboarding_labels_v2_drop13/437647_2024_madal_chm_max_hag_20cm_labels.csv")
X = 647460.80
Y = 6437347.19

with rasterio.open(str(TIF)) as src:
    px_row, px_col = src.index(X, Y)

STEP = 64
ROW_OFF = (px_row // STEP) * STEP
COL_OFF = (px_col // STEP) * STEP

print(f"ROW_OFF={ROW_OFF},COL_OFF={COL_OFF}")

found = []
with open(LABELS_CSV, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            if r['raster'] == TIF.name and int(r['row_off']) == ROW_OFF and int(r['col_off']) == COL_OFF:
                found.append(r)
        except Exception:
            continue

if not found:
    print('NO_ROWS')
else:
    latest = sorted(found, key=lambda x: x['timestamp'])[-1]
    print('LATEST_RECORD')
    print(json.dumps(latest))

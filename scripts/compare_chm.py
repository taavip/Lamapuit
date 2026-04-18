#!/usr/bin/env python3
from pathlib import Path
import sys
try:
    import rasterio
    import numpy as np
except Exception as e:
    print('PY_RASTERIO_IMPORT_ERROR', e, file=sys.stderr)
    sys.exit(0)

files = {
 'backup': Path('experiments/dtm_hag_436646_research/backup/2018_tin_linear_gauss_chm_before_run.tif'),
 'new': Path('experiments/dtm_hag_436646_research/results/chm/2018/2018_tin_linear_gauss_chm.tif'),
 'reproduce': Path('experiments/dtm_hag_436646_reproduce_single_2018/produced/chm/2018/2018_tin_linear_gauss_chm.tif'),
}

for k,p in files.items():
    print('\n---', k, '---')
    print('path', p)
    print('exists', p.exists())
    if not p.exists():
        continue
    with rasterio.open(p) as src:
        arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        finite = np.isfinite(arr)
        print('shape', arr.shape)
        print('valid_count', int(finite.sum()))
        if finite.sum():
            print('min', float(np.nanmin(arr)), 'max', float(np.nanmax(arr)), 'mean', float(np.nanmean(arr)))
        else:
            print('no finite values')


def compare(p1, p2, name1, name2):
    if not (p1.exists() and p2.exists()):
        print('compare skipped', name1, name2)
        return
    with rasterio.open(p1) as a, rasterio.open(p2) as b:
        A = a.read(1, masked=True).filled(np.nan).astype(np.float32)
        B = b.read(1, masked=True).filled(np.nan).astype(np.float32)
    if A.shape != B.shape:
        print('SHAPE_MISMATCH', name1, name2, A.shape, B.shape)
        return
    mask = ~(np.isnan(A) & np.isnan(B))
    comp_pixels = int(mask.sum())
    diff = np.abs(np.nan_to_num(A, nan=0.0)[mask] - np.nan_to_num(B, nan=0.0)[mask])
    diff_count = int((diff > 1e-6).sum())
    max_diff = float(diff.max()) if diff.size else 0.0
    mean_diff = float(diff.mean()) if diff.size else 0.0
    print('COMPARE', name1, 'vs', name2, 'pixels_compared', comp_pixels, 'diff_pixels>1e-6', diff_count, 'max_diff', max_diff, 'mean_diff', mean_diff)


compare(files['backup'], files['new'], 'backup', 'new')
compare(files['new'], files['reproduce'], 'new', 'reproduce')
compare(files['backup'], files['reproduce'], 'backup', 'reproduce')

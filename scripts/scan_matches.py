#!/usr/bin/env python3
import os, re, sys
pattern = re.compile(r"(?P<id>\d{6})[_-](?P<year>\d{4})")
GAUSS = "data/chm_variants/harmonized_0p8m_chm_gauss"
RAW = "data/chm_variants/harmonized_0p8m_chm_raw"
BASE = "data/chm_variants/baseline_chm_20cm"

def safe_list(p):
    try:
        return [f for f in os.listdir(p) if f.lower().endswith('.tif')]
    except Exception:
        return []

gauss_files = safe_list(GAUSS)
raw_files = safe_list(RAW)
base_files = safe_list(BASE)

def map_files(files):
    m = {}
    for f in files:
        mo = pattern.search(f)
        if mo:
            key = (mo.group('id'), mo.group('year'))
            m.setdefault(key, []).append(f)
    return m

gmap = map_files(gauss_files)
rmap = map_files(raw_files)
bmap = map_files(base_files)
keys = sorted(set(gmap.keys()) & set(rmap.keys()) & set(bmap.keys()))
print(len(keys))
for k in keys[:50]:
    print(k[0], k[1], gmap[k][0], rmap[k][0], bmap[k][0])

# exit code 0

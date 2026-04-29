#!/usr/bin/env python3
"""Regenerate SLD PNGs with filenames including T_ and P_ flags, no caption text.

Clears `tmp/sep`, selects up to 5 tiles for each confusion class (TP/FP/FN/TN)
and 5 problem examples, writes SLD-colored PNGs named like:
  FN_T0_P1_<raster>_<row>_<col>_sld.png
"""

from pathlib import Path
import sys
import glob
import random

try:
    import pandas as pd
except Exception:
    print("pandas required", file=sys.stderr)
    raise

try:
    import rasterio
    import numpy as np
except Exception:
    rasterio = None
    np = None

try:
    from label_tiles import _apply_sld
except Exception as e:
    print("Could not import _apply_sld from label_tiles:", e, file=sys.stderr)
    raise

import csv

OUT = Path("tmp/sep")
OUT.mkdir(parents=True, exist_ok=True)
CHM_DIR = Path("chm_max_hag")


def clear_out():
    for p in list(OUT.glob("*")):
        try:
            p.unlink()
        except Exception:
            pass


def find_predictions_csv():
    pref = Path("output/metrics_and_mislabels_test/top_mispreds.csv")
    if pref.exists():
        return pref
    c = glob.glob("**/*pred*csv", recursive=True)
    return Path(c[0]) if c else None


def pick_cols(df):
    m = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "raster":
            m["raster"] = c
        if lc == "row_off":
            m["row_off"] = c
        if lc == "col_off":
            m["col_off"] = c
        if "chunk" in lc and "size" in lc:
            m["chunk_size"] = c
        if lc in ("label", "true_label", "gt") and "true" not in m:
            m["true"] = c
        if lc in ("predicted", "pred", "y_pred", "prediction") and "pred" not in m:
            m["pred"] = c
        if lc in ("model_prob", "prob", "confidence") and "prob" not in m:
            m["prob"] = c
        if any(k in lc for k in ("path", "tile", "image", "filepath")) and "path" not in m:
            m["path"] = c
    return m


def confusion(target, t, p):
    if pd.isna(t) or pd.isna(p):
        return None
    if t == target and p == target:
        return "TP"
    if t != target and p != target:
        return "TN"
    if t != target and p == target:
        return "FP"
    if t == target and p != target:
        return "FN"
    return None


def read_chm(raster_name, row_off, col_off, cs=128):
    if rasterio is None:
        raise RuntimeError("rasterio required")
    matches = list(CHM_DIR.glob(f"{Path(raster_name).stem}*.tif"))
    if not matches:
        raise FileNotFoundError(raster_name)
    with rasterio.open(matches[0]) as src:
        arr = src.read(
            1,
            window=rasterio.windows.Window(col_off, row_off, cs, cs),
            boundless=True,
            fill_value=0,
        ).astype("float32")
    return arr


def load_label_records(labels_dir=Path("output/tile_labels")):
    recs = []
    base = Path(labels_dir)
    if not base.exists():
        return recs
    for p in sorted(base.glob("*_labels.csv")):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                lbl = r.get("label")
                if lbl is None:
                    continue
                recs.append(
                    {
                        "raster": r["raster"],
                        "row_off": int(r["row_off"]),
                        "col_off": int(r["col_off"]),
                        "chunk_size": int(r.get("chunk_size", 128)),
                        "label": r.get("label"),
                    }
                )
    return recs


def save_sld(tile, out_path: Path):
    rgb = _apply_sld(tile)
    from imageio import imwrite

    imwrite(str(out_path), rgb)


def main():
    csvp = find_predictions_csv()
    if not csvp:
        print("No predictions CSV found; aborting", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(csvp)
    cols = pick_cols(df)
    true_col = cols.get("true") or ("label" if "label" in df.columns else None)
    pred_col = cols.get("pred") or ("predicted" if "predicted" in df.columns else None)
    if not true_col or not pred_col:
        print("Could not identify true/pred columns; aborting", file=sys.stderr)
        sys.exit(2)

    classes = list(pd.unique(df[true_col].dropna()))
    target = next((c for c in classes if str(c).lower() == "cdw"), classes[0])

    df = df.copy()
    df["conf"] = df.apply(lambda r: confusion(target, r[true_col], r[pred_col]), axis=1)

    clear_out()

    # collect for TP/FP/FN/TN
    for cls in ("TP", "FP", "FN", "TN"):
        subset = df[df["conf"] == cls]
        if subset.empty:
            # try to fall back to label files to produce TP/TN examples
            label_recs = load_label_records()
            if not label_recs:
                continue
            if cls == "TP":
                # find labeled-cdw examples
                cand = [r for r in label_recs if str(r["label"]).lower() == str(target).lower()]
                chosen = cand[:5]
            elif cls == "TN":
                cand = [r for r in label_recs if str(r["label"]).lower() != str(target).lower()]
                chosen = cand[:5]
            else:
                # FP/FN won't be available in label-only set; skip
                continue
            # turn chosen into a DataFrame-like iterable of dicts
            for idx, r in enumerate(chosen, start=1):
                T = 1 if str(r["label"]).lower() == str(target).lower() else 0
                P = T
                try:
                    tile = read_chm(
                        r["raster"],
                        int(r["row_off"]),
                        int(r["col_off"]),
                        int(r.get("chunk_size", 128)),
                    )
                except Exception as e:
                    print("skip label fallback read error", e)
                    continue
                fname = f"{cls}_T{T}_P{P}_{Path(r['raster']).stem}_{int(r['row_off'])}_{int(r['col_off'])}_sld.png"
                save_sld(tile, OUT / fname)
            continue
        chosen = subset.sample(min(5, len(subset)), random_state=42)
        for idx, (_, r) in enumerate(chosen.iterrows(), start=1):
            T = 1 if r[true_col] == target else 0
            P = 1 if r[pred_col] == target else 0
            if "raster" in cols:
                cs = (
                    int(r.get(cols.get("chunk_size"), 128))
                    if cols.get("chunk_size") in r.index
                    else 128
                )
                try:
                    tile = read_chm(
                        r[cols["raster"]], int(r[cols["row_off"]]), int(r[cols["col_off"]]), cs
                    )
                except Exception as e:
                    print("Skip read error", e)
                    continue
                fname = f"{cls}_T{T}_P{P}_{Path(r[cols['raster']]).stem}_{int(r[cols['row_off']])}_{int(r[cols['col_off']])}_sld.png"
                save_sld(tile, OUT / fname)
            else:
                p = Path(r[cols["path"]])
                if not p.exists():
                    alt = Path.cwd() / str(r[cols["path"]])
                    if alt.exists():
                        p = alt
                    else:
                        print("Missing image", r[cols["path"]])
                        continue
                # load original image and convert to SLD-like via simple greyscale then apply _apply_sld on placeholder
                # prefer to use original RGB directly — just save as-is with new name
                fname = f"{cls}_T{T}_P{P}_{p.stem}_sld.png"
                from PIL import Image

                img = Image.open(p).convert("RGB")
                img.save(OUT / fname)

    # problems (5)
    prefs = df[df["conf"].isin(["FP", "FN"])]
    if len(prefs) >= 5:
        problems = prefs.sample(5, random_state=1)
    else:
        problems = df.sample(5, random_state=1)

    for idx, (_, r) in enumerate(problems.iterrows(), start=1):
        T = 1 if r[true_col] == target else 0
        P = 1 if r[pred_col] == target else 0
        if "raster" in cols:
            try:
                cs = (
                    int(r.get(cols.get("chunk_size"), 128))
                    if cols.get("chunk_size") in r.index
                    else 128
                )
                tile = read_chm(
                    r[cols["raster"]], int(r[cols["row_off"]]), int(r[cols["col_off"]]), cs
                )
            except Exception as e:
                print("Skip problem read", e)
                continue
            fname = f"PROB_{idx}_T{T}_P{P}_{Path(r[cols['raster']]).stem}_{int(r[cols['row_off']])}_{int(r[cols['col_off']])}_sld.png"
            save_sld(tile, OUT / fname)
        else:
            p = Path(r[cols["path"]])
            if not p.exists():
                alt = Path.cwd() / str(r[cols["path"]])
                if alt.exists():
                    p = alt
                else:
                    print("Missing problem image", r[cols["path"]])
                    continue
            fname = f"PROB_{idx}_T{T}_P{P}_{p.stem}_sld.png"
            from PIL import Image

            Image.open(p).convert("RGB").save(OUT / fname)

    print("Done — regenerated SLD files without captions in", OUT)


if __name__ == "__main__":
    import pandas as pd

    main()

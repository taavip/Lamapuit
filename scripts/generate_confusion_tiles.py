#!/usr/bin/env python3
"""Generate confusion-matrix tiles and problem examples.

Supports two input modes:
- CSV listing image file paths + true/pred columns
- CSV listing CHM coordinates (raster, row_off, col_off) + true/pred columns

Writes PNGs + small caption files to tmp/sep/.
"""
from pathlib import Path
import sys

            pass
    return Image.open(path).convert('RGB')

def embed_caption(img: Image.Image, caption: str, height: int = 80):
    w,h = img.size
    out = Image.new('RGB', (w, h+height), (255,255,255))
    out.paste(img, (0,0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    lines = textwrap.wrap(caption, width=80)
    y = h + 6
    for line in lines:
        draw.text((6,y), line, fill=(0,0,0), font=font)
        y += 12
    return out

def read_tile(chm_dir: Path, raster_name: str, row_off: int, col_off: int, cs: int = 128):
    if rasterio is None:
        raise RuntimeError("rasterio is required for CHM tile mode")
    matches = list(chm_dir.glob(f"{Path(raster_name).stem}*.tif"))
    if not matches:
        raise FileNotFoundError(raster_name)
    with rasterio.open(matches[0]) as src:
        arr = src.read(1, window=rasterio.windows.Window(col_off, row_off, cs, cs), boundless=True, fill_value=0).astype('float32')
    return arr

def save_rgb_from_chm(tile, out_path: Path, caption: str | None = None):
    # simple clip & greyscale + optional caption
    import numpy as _np
    t = _np.clip(tile, 0.0, 20.0) / 20.0
    img8 = (_np.clip(t,0,1)*255).astype('uint8')
    rgb = _np.stack([img8, img8, img8], axis=2)
    im = Image.fromarray(rgb)
    if caption:
        im = embed_caption(im, caption)
    im.save(out_path)

def main():
    csvp = find_predictions_csv()
    if not csvp:
        print("No predictions CSV found.")
        sys.exit(2)
    print(f"Using predictions CSV: {csvp}")
    df = load_df(csvp)
    colmap = pick_columns(df)
    # determine mode
    path_mode = 'path' in colmap
    chm_mode = colmap.get('mode') == 'chm' or ('raster' in colmap and 'row_off' in colmap and 'col_off' in colmap)

    # need true/pred cols (try fallbacks)
    true_col = colmap.get('true') or ( 'label' if 'label' in df.columns else None )
    pred_col = colmap.get('pred') or ( 'predicted' if 'predicted' in df.columns else None )
    if true_col is None or pred_col is None:
        print("Could not identify true/pred columns. Columns:", list(df.columns))
        sys.exit(2)

    # pick target class prefer 'cdw'
    classes = list(pd.unique(df[true_col].dropna()))
    if not classes:
        print("No classes found in true column.")
        sys.exit(2)
    target = next((c for c in classes if str(c).lower()=='cdw'), classes[0])
    print(f"Target class: {target}")

    def confusion_row(r):
        t = r[true_col]
        p = r[pred_col]
        if pd.isna(t) or pd.isna(p):
            return None
        if t == target and p == target:
            return 'TP'
        if t != target and p != target:
            return 'TN'
        if t != target and p == target:
            return 'FP'
        if t == target and p != target:
            return 'FN'
        return None

    df = df.copy()
    df['confusion'] = df.apply(confusion_row, axis=1)

    examples = {k: [] for k in ('TP','FP','FN','TN')}

    if path_mode:
        pcol = colmap['path']
        for k in examples:
            subset = df[df['confusion']==k]
            if subset.empty:
                continue
            chosen = subset.sample(min(5, len(subset)), random_state=42)
            for _, r in chosen.iterrows():
                examples[k].append({'path': r[pcol], 'true': r[true_col], 'pred': r[pred_col]})
    elif chm_mode:
        rcol = colmap['raster']
        rowc = colmap['row_off']
        colc = colmap['col_off']
        chunkc = colmap.get('chunk_size')
        for k in examples:
            subset = df[df['confusion']==k]
            if subset.empty:
                continue
            chosen = subset.sample(min(5, len(subset)), random_state=42)
            for _, r in chosen.iterrows():
                cs = int(r[chunkc]) if chunkc and not pd.isna(r.get(chunkc, None)) else 128
                examples[k].append({'raster': r[rcol], 'row_off': int(r[rowc]), 'col_off': int(r[colc]), 'chunk_size': cs, 'true': r[true_col], 'pred': r[pred_col]})
    else:
        print("No path or CHM coordinate columns; cannot extract tiles from CSV.")
        sys.exit(2)

    # save images
    for k, items in examples.items():
        for i, itm in enumerate(items, start=1):
            if 'path' in itm:
                p = Path(itm['path'])
                if not p.exists():
                    alt = Path.cwd()/itm['path']
                    if alt.exists():
                        p = alt
                    else:
                        print(f"Warning: image not found: {itm['path']}")
                        continue
                try:
                    img = imread(p)
                except Exception as e:
                    print(f"Failed to open {p}: {e}")
                    continue
                caption = f"Confusion={k} True={itm.get('true')} Pred={itm.get('pred')}"
                out = embed_caption(img, caption)
                out_path = OUT / f"{k}_{i}_{p.stem}.png"
                out.save(out_path)
                (OUT / f"{out_path.stem}.txt").write_text(caption)
            else:
                try:
                    tile = read_tile(CHM_DIR, itm['raster'], itm['row_off'], itm['col_off'], itm['chunk_size'])
                except Exception as e:
                    print(f"Failed to read CHM tile {itm}: {e}")
                    continue
                caption = f"Confusion={k} True={itm.get('true')} Pred={itm.get('pred')}"
                fname = f"{k}_{i}_{Path(itm['raster']).stem}_{itm['row_off']}_{itm['col_off']}.png"
                save_rgb_from_chm(tile, OUT / fname, caption)
                (OUT / f"{Path(fname).stem}.txt").write_text(caption)

    # problem examples: prefer FP/FN else any
    prefs = df[df['confusion'].isin(['FP','FN'])]
    if len(prefs) >= 5:
        problems = prefs.sample(5, random_state=1)
    else:
        problems = df.sample(5, random_state=1)

    problem_captions = [
        "Shadow/lighting artifacts mimic CDW reflectance — likely false positive.",
        "Very small, scattered debris that blends with background vegetation — low SNR.",
        "Mixed materials (rocks + debris) causing ambiguous signatures.",
        "Tile resolution blurs object edges; classifier misses small patches.",
        "Sensor noise or compression artifacts create spurious patterns interpreted as CDW."
    ]

    for idx, (_, row) in enumerate(problems.iterrows(), start=1):
        cap_text = problem_captions[(idx-1) % len(problem_captions)]
        if path_mode:
            p = Path(row[colmap['path']])
            if not p.exists():
                alt = Path.cwd()/row[colmap['path']]
                if alt.exists():
                    p = alt
                else:
                    print(f"Warning: problem image not found: {row[colmap['path']]}")
                    continue
            try:
                img = imread(p)
            except Exception as e:
                print(f"Failed to open problem image {p}: {e}")
                continue
            caption = f"Problem {idx}: {cap_text} (True={row[true_col]} Pred={row[pred_col]})"
            out = embed_caption(img, caption)
            out_path = OUT / f"PROB_{idx}_{p.stem}.png"
            out.save(out_path)
            (OUT / f"{out_path.stem}.txt").write_text(caption)
        else:
            try:
                cs = int(row.get(colmap.get('chunk_size'), 128)) if colmap.get('chunk_size') in row.index else 128
                tile = read_tile(CHM_DIR, row[colmap['raster']], int(row[colmap['row_off']]), int(row[colmap['col_off']]), cs)
            except Exception as e:
                print(f"Failed to read problem CHM tile: {e}")
                continue
            caption = f"Problem {idx}: {cap_text} (True={row[true_col]} Pred={row[pred_col]})"
            out_name = f"PROB_{idx}_{Path(row[colmap['raster']]).stem}_{int(row[colmap['row_off']])}_{int(row[colmap['col_off']])}.png"
            save_rgb_from_chm(tile, OUT / out_name, caption)
            (OUT / f"{Path(out_name).stem}.txt").write_text(caption)

    print(f"Done. Saved outputs to {OUT}")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Generate confusion-matrix tiles and example problem images.

Finds a predictions CSV (heuristic), chooses a target class (prefers 'cdw'),
selects up to 5 tiles for each confusion category (TP/FP/FN/TN), and saves
PNG images and caption text files into `tmp/sep/`.

This script is designed to be run inside the project's environment (conda)
or inside Docker. It is defensive about file/column names but expects a
predictions CSV with columns that include a path to the tile image and the
true/predicted labels.
"""
import os
import sys
import glob
import random
from pathlib import Path
import textwrap

try:
    import pandas as pd
except Exception:
    print("pandas is required. Please install dependencies (see README).", file=sys.stderr)
    raise

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    print("Pillow is required. Please install dependencies (see README).", file=sys.stderr)
    raise

try:
    import rasterio
    import numpy as np
except Exception:
    # rasterio is optional; we'll try PIL open as fallback
    rasterio = None
    np = None


OUT_DIR = Path("tmp/sep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _chm_read_tile(chm_dir: Path, raster_name: str, row_off: int, col_off: int, cs: int):
    # lightweight CHM tile reader used by the CSV-driven path
    matches = list(Path(chm_dir).glob(f"{Path(raster_name).stem}*.tif"))
    if not matches:
        raise FileNotFoundError(raster_name)
    try:
        import rasterio

        with rasterio.open(matches[0]) as src:
            arr = src.read(1, window=rasterio.windows.Window(col_off, row_off, cs, cs), boundless=True, fill_value=0).astype('float32')
        return arr
    except Exception as exc:
        raise


def _chm_save_tile(tile: 'np.ndarray', out_path: Path, caption: str | None = None):
    # convert heights to 0-255 uint8 (clip to 0-20m like training pipeline)
    try:
        import numpy as _np
        from PIL import Image
    except Exception:
        raise
    t = _np.clip(tile, 0.0, 20.0) / 20.0
    img8 = (t * 255.0).astype('uint8')
    rgb = _np.stack([img8, img8, img8], axis=2)
    im = Image.fromarray(rgb)
    if caption:
        im = embed_caption(im, caption)
    im.save(out_path)


def find_predictions_csv():
    # common paths
    candidates = glob.glob("**/*pred*csv", recursive=True)
    if not candidates:
        return None
    # prefer files in runs/ or output/
    candidates = sorted(candidates, key=lambda p: ("runs/" in p or "output/" in p, os.path.getmtime(p)), reverse=True)
    return candidates[0]


def load_df(path):
    df = pd.read_csv(path)
    return df


def pick_columns(df):
    # heuristics for column names
    col_map = {}
    cols = [c.lower() for c in df.columns]
    for candidate in ("tile_path", "tile", "image", "img", "filepath", "path"):
        for c in df.columns:
            if c.lower() == candidate or candidate in c.lower():
                col_map['path'] = c
                break
        if 'path' in col_map:
            break
    for candidate in ("true_label", "label", "gt", "y_true", "target"):
        for c in df.columns:
            if c.lower() == candidate or candidate in c.lower():
                col_map['true'] = c
                break
        if 'true' in col_map:
            break
    for candidate in ("pred_label", "pred", "y_pred", "prediction"):
        for c in df.columns:
            if c.lower() == candidate or candidate in c.lower():
                col_map['pred'] = c
                break
        if 'pred' in col_map:
            break
    # fallback: detect CHM-style coordinate columns (raster, row_off, col_off)
    if 'pred' not in col_map or 'true' not in col_map or 'path' not in col_map:
        has_raster = any(c.lower() == 'raster' for c in df.columns)
        has_row = any(c.lower() == 'row_off' for c in df.columns)
        has_col = any(c.lower() == 'col_off' for c in df.columns)
        if has_raster and has_row and has_col:
            for c in df.columns:
                lc = c.lower()
                if lc == 'raster':
                    col_map['raster'] = c
                if lc == 'row_off':
                    col_map['row_off'] = c
                if lc == 'col_off':
                    col_map['col_off'] = c
                if lc == 'chunk_size':
                    col_map['chunk_size'] = c
            col_map['mode'] = 'chm'
    return col_map


def imread(path):
    # Try rasterio first for geotiffs, else PIL
    if rasterio and str(path).lower().endswith(('.tif', '.tiff')):
        try:
            with rasterio.open(path) as src:
                arr = src.read()
                # convert to HxWxC
                if arr.ndim == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                # normalize if floats
                if arr.dtype.kind == 'f':
                    arr = (255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)).astype('uint8')
                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                return Image.fromarray(arr)
        except Exception:
            pass
    try:
        return Image.open(path).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Failed to open image {path}: {e}")


def embed_caption(img, caption, height=80):
    # add a caption box at bottom
    w, h = img.size
    new_h = h + height
    out = Image.new('RGB', (w, new_h), (255, 255, 255))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    margin = 6
    lines = textwrap.wrap(caption, width=80)
    y = h + margin
    for line in lines:
        draw.text((margin, y), line, fill=(0, 0, 0), font=font)
        y += 12
    return out


def main():
    csv_path = find_predictions_csv()
    if not csv_path:
        print("No predictions CSV found in repository (searched for '*pred*csv').", file=sys.stderr)
        sys.exit(2)
    print(f"Using predictions CSV: {csv_path}")
    df = load_df(csv_path)
    col_map = pick_columns(df)
    # require true and pred at minimum
    if 'true' not in col_map or 'pred' not in col_map:
        print("Could not identify true/pred columns in the predictions CSV. Columns found:", df.columns.tolist(), file=sys.stderr)
        sys.exit(2)

    true_col = col_map['true']
    pred_col = col_map['pred']

    # choose a target class: prefer 'cdw' if available
    classes = list(pd.unique(df[true_col].dropna()))
    target = None
    for c in classes:
        if str(c).lower() == 'cdw':
            target = c
            break
    if target is None:
        target = classes[0]
    print(f"Target class for confusion matrix selection: {target}")

    # compute confusion type
    def confusion_row(r):
        t = r[true_col]
        p = r[pred_col]
        if pd.isna(t) or pd.isna(p):
            return None
        if t == target and p == target:
            return 'TP'
        if t != target and p != target:
            return 'TN'
        if t != target and p == target:
            return 'FP'
        if t == target and p != target:
            return 'FN'
        return None

    df = df.copy()
    df['confusion'] = df.apply(confusion_row, axis=1)

    examples = {}
    path_col = col_map.get('path')
    if path_col:
        for ctype in ('TP', 'FP', 'FN', 'TN'):
            subset = df[df['confusion'] == ctype]
            if subset.empty:
                examples[ctype] = []
                continue
            chosen = subset.sample(min(5, len(subset)), random_state=42)
            examples[ctype] = [{'path': p} for p in chosen[path_col].tolist()]
    elif col_map.get('mode') == 'chm':
        # CHM coordinate mode: store raster/row/col tuples
        rcol = col_map['raster']
        rowc = col_map['row_off']
        colc = col_map['col_off']
        chunkc = col_map.get('chunk_size')
        for ctype in ('TP', 'FP', 'FN', 'TN'):
            subset = df[df['confusion'] == ctype]
            if subset.empty:
                examples[ctype] = []
                continue
            chosen = subset.sample(min(5, len(subset)), random_state=42)
            rows = []
            for _, r in chosen.iterrows():
                cs = int(r[chunkc]) if chunkc and not pd.isna(r.get(chunkc, None)) else 128
                rows.append({'raster': r[rcol], 'row_off': int(r[rowc]), 'col_off': int(r[colc]), 'chunk_size': cs})
            examples[ctype] = rows
    else:
        print("No usable image path or CHM coordinate columns found; cannot extract tiles.", file=sys.stderr)
        sys.exit(2)

    # save images
    for ctype, items in examples.items():
        for i, itm in enumerate(items):
            if 'path' in itm:
                p = itm['path']
                src = Path(p)
                if not src.exists():
                    alt = Path.cwd() / p
                    if alt.exists():
                        src = alt
                    else:
                        print(f"Warning: image not found: {p}")
                        continue
                try:
                    img = imread(src)
                except Exception as e:
                    print(f"Failed to read {src}: {e}")
                    continue
                caption = f"Confusion: {ctype}. True={df.loc[df[path_col]==p, true_col].iloc[0]} Pred={df.loc[df[path_col]==p, pred_col].iloc[0]}"
                out_img = embed_caption(img, caption)
                out_path = OUT_DIR / f"{ctype}_{i+1}_{src.stem}.png"
                out_img.save(out_path)
                (OUT_DIR / f"{out_path.stem}.txt").write_text(caption)
            else:
                # CHM tile extraction
                try:
                    tile = read_tile(CHM_DIR, itm['raster'], itm['row_off'], itm['col_off'], itm['chunk_size'])
                except Exception as e:
                    print(f"Failed to read CHM tile {itm}: {e}")
                    continue
                fname = f"{ctype}_{i+1}_{Path(itm['raster']).stem}_{itm['row_off']}_{itm['col_off']}.png"
                # compute a simple caption from df match
                # find matching row
                match = df[(df[col_map.get('raster', 'raster')] == itm['raster']) & (df[col_map.get('row_off', 'row_off')] == itm['row_off']) & (df[col_map.get('col_off', 'col_off')] == itm['col_off'])]
                true_v = match[true_col].iloc[0] if not match.empty else ''
                pred_v = match[pred_col].iloc[0] if not match.empty else ''
                caption = f"Confusion: {ctype}. True={true_v} Pred={pred_v}"
                save_rgb(tile, OUT_DIR / fname, caption)
                (OUT_DIR / f"{Path(fname).stem}.txt").write_text(caption)

    # Create five additional problematic examples with distinct argument captions.
    problem_captions = [
        "Shadow/lighting artifacts mimic CDW reflectance — likely false positive.",
        "Very small, scattered debris that blends with background vegetation — low SNR for detection.",
        "Mixed materials (rocks + debris) causing ambiguous signatures leading to misclassification.",
        "Tile resolution and downsampling blur object edges; classifier misses small CDW patches.",
        "Sensor noise or compression artifacts create spurious high-frequency patterns interpreted as CDW."
    ]

    # select 5 distinct rows to illustrate problems (prefer FP/FN but fallback to any)
    preferred = df[df['confusion'].isin(['FP', 'FN'])]
    if len(preferred) >= 5:
        chosen_rows = preferred.sample(5, random_state=1)
    else:
        chosen_rows = df.sample(5, random_state=1)

    for idx, (irow, row) in enumerate(chosen_rows.iterrows(), start=1):
        p = row[path_col]
        src = Path(p)
        if not src.exists():
            alt = Path.cwd() / p
            if alt.exists():
                src = alt
            else:
                print(f"Warning: image not found for problem example: {p}")
                continue
        try:
            img = imread(src)
        except Exception as e:
            print(f"Failed to read {src}: {e}")
            continue
        caption = problem_captions[(idx-1) % len(problem_captions)]
        caption = f"Problem example {idx}: {caption} (True={row[true_col]} Pred={row[pred_col]})"
        out_img = embed_caption(img, caption)
        out_path = OUT_DIR / f"PROB_{idx}_{src.stem}.png"
        out_img.save(out_path)
        (OUT_DIR / f"{out_path.stem}.txt").write_text(caption)

    print("Done. Saved images and captions to:", OUT_DIR)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Generate PNG tiles for confusion matrix classes from the best model.

Saves 5 tiles for each of TP/FP/FN/TN plus 5 additional problematic examples
with captions into `tmp/sep/`.
"""
from pathlib import Path
import json
import random
import math

import cv2
import numpy as np
import rasterio
import torch

from label_tiles import _apply_sld, _build_deep_cnn_attn_net


OUT = Path("tmp/sep")
OUT.mkdir(parents=True, exist_ok=True)

# best model from model_search final_models (single-model CNN-Deep-Attn)
MODEL = Path("output/model_search/final_models/deep_cnn_attn_full_ce_mixup.pt")
LABEL_DIR = Path("output/tile_labels")
CHM_DIR = Path("chm_max_hag")


def load_labels(labels_dir: Path):
    recs = []
    for p in sorted(labels_dir.glob("*_labels.csv")):
        with open(p, newline="") as f:
            import csv

            for r in csv.DictReader(f):
                lbl = r.get("label", "")
                if lbl not in ("cdw", "no_cdw"):
                    continue
                recs.append({
                    "raster": r["raster"],
                    "row_off": int(r["row_off"]),
                    "col_off": int(r["col_off"]),
                    "chunk_size": int(r.get("chunk_size", 128)),
                    "label": 1 if lbl == "cdw" else 0,
                })
    return recs


def load_model(pt_path: Path):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(pt_path, map_location=dev)
    build_fn_name = ckpt.get("build_fn_name", "_build_deep_cnn_attn")
    # label_tiles provides the build fn for the CNN-Deep-Attn
    net = _build_deep_cnn_attn_net()
    net.load_state_dict(ckpt["state_dict"])
    net.to(dev).eval()
    meta = ckpt.get("meta", {})
    thresh = float(meta.get("best_thresh", 0.5))
    return net, dev, thresh


def read_tile(chm_dir: Path, raster_name: str, row_off: int, col_off: int, cs: int):
    # find raster file by stem
    matches = list(chm_dir.glob(f"{Path(raster_name).stem}*.tif"))
    if not matches:
        raise FileNotFoundError(raster_name)
    with rasterio.open(matches[0]) as src:
        arr = src.read(1, window=rasterio.windows.Window(col_off, row_off, cs, cs), boundless=True, fill_value=0).astype(np.float32)
    return arr


def predict_prob(net, dev, tile: np.ndarray):
    # normalise as label_tiles.CNNPredictor: clip [0,20]/20
    t = np.clip(tile, 0.0, 20.0) / 20.0
    x = torch.tensor(t[np.newaxis, np.newaxis], dtype=torch.float32).to(dev)
    with torch.no_grad():
        out = torch.softmax(net(x), dim=1)[0, 1].cpu().item()
    return float(out)


def save_rgb(tile: np.ndarray, path: Path, overlay: str | None = None):
    rgb = _apply_sld(tile)
    if overlay:
        # put small caption at top-left
        cv2.putText(rgb, overlay, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), rgb[:, :, ::-1])


def main():
    recs = load_labels(LABEL_DIR)
    print(f"Loaded {len(recs)} labelled tiles")
    net, dev, thresh = load_model(MODEL)
    print(f"Loaded model {MODEL}  thresh={thresh}")

    # compute probs and confusion class
    results = []
    for i, r in enumerate(recs):
        try:
            tile = read_tile(CHM_DIR, r["raster"], r["row_off"], r["col_off"], r["chunk_size"])
        except Exception:
            continue
        p = predict_prob(net, dev, tile)
        pred = 1 if p >= thresh else 0
        if r["label"] == 1 and pred == 1:
            cls = "TP"
        elif r["label"] == 0 and pred == 1:
            cls = "FP"
        elif r["label"] == 1 and pred == 0:
            cls = "FN"
        else:
            cls = "TN"
        results.append({**r, "prob": p, "pred": pred, "cls": cls})

    bycls = {k: [x for x in results if x["cls"] == k] for k in ("TP", "FP", "FN", "TN")}
    for k in bycls:
        random.shuffle(bycls[k])

    # pick up to 5 each
    picks = []
    for k in ("TP", "FP", "FN", "TN"):
        picks += bycls[k][:5]

    # save images
    for idx, r in enumerate(picks):
        tile = read_tile(CHM_DIR, r["raster"], r["row_off"], r["col_off"], r["chunk_size"])
        fname = f"{idx+1:02d}_{r['cls']}_{r['raster'].replace('/', '_')}_{r['row_off']}_{r['col_off']}.png"
        overlay = f"{r['cls']} p={r['prob']:.2f}"
        save_rgb(tile, OUT / fname, overlay)

    # create 5 additional problematic examples with distinct captions
    problems = []
    # 1: low max-height CDW missed (FN with low max)
    fn_sorted = sorted(bycls["FN"], key=lambda x: float(np.nanmax(read_tile(CHM_DIR, x['raster'], x['row_off'], x['col_off'], x['chunk_size']))))
    if fn_sorted:
        problems.append((fn_sorted[0], "Low maximum height — small pile likely missed"))
    # 2: edge tile with half-pile (ambiguous)
    if bycls["FP"]:
        problems.append((bycls["FP"][0], "Edge tile — CDW partly outside tile (boundary case)"))
    # 3: high-noise / low-contrast region
    all_sorted = sorted(results, key=lambda x: float(np.nanstd(read_tile(CHM_DIR, x['raster'], x['row_off'], x['col_off'], x['chunk_size']))))
    problems.append((all_sorted[0], "Very low contrast — likely lidar noise / flat ground"))
    # 4: tall tree crown mistaken for CDW
    tp_sorted = sorted(bycls['TP'], key=lambda x: -float(np.nanmax(read_tile(CHM_DIR, x['raster'], x['row_off'], x['col_off'], x['chunk_size']))))
    if tp_sorted:
        problems.append((tp_sorted[0], "High returned height — could be tree crown vs pile"))
    # 5: label disagreement candidate (model prob far from label expectation)
    disagree = sorted([r for r in results if abs(r['prob'] - r['label']) > 0.6], key=lambda x: -abs(x['prob'] - x['label']))
    if disagree:
        problems.append((disagree[0], "Strong disagreement between model and label — review candidate"))

    captions = []
    for i, (r, caption) in enumerate(problems[:5], start=1):
        tile = read_tile(CHM_DIR, r['raster'], r['row_off'], r['col_off'], r['chunk_size'])
        fname = f"problem_{i:02d}_{r['cls']}_{r['raster'].replace('/', '_')}_{r['row_off']}_{r['col_off']}.png"
        overlay = f"{r['cls']} p={r['prob']:.2f}"
        save_rgb(tile, OUT / fname, overlay)
        captions.append({"file": fname, "caption": caption, "prob": r['prob'], "cls": r['cls']})

    # write captions
    (OUT / "captions.json").write_text(json.dumps(captions, indent=2))
    print(f"Saved {len(picks)} confusion tiles and {len(captions)} problem examples to {OUT}")


if __name__ == "__main__":
    main()

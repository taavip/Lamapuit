#!/usr/bin/env python3
"""Clear tmp/sep and regenerate SLD-styled PNGs with embedded True/Pred/Prob captions.

Selects up to 5 tiles for each confusion class (TP/FP/FN/TN) using the
predictions CSV heuristics and produces 5 problem examples. Writes only PNG
files (no separate .txt). Uses `label_tiles._apply_sld` for coloring.
"""

from pathlib import Path
import sys
import glob
import random
import textwrap

try:
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    print("pandas + pillow required", file=sys.stderr)
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
    print("label_tiles._apply_sld import failed:", e, file=sys.stderr)
    raise

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
    preferred = Path("output/metrics_and_mislabels_test/top_mispreds.csv")
    if preferred.exists():
        return preferred
    c = glob.glob("**/*pred*csv", recursive=True)
    return Path(c[0]) if c else None


def load_df(p):
    return pd.read_csv(p)


def pick_cols(df):
    m = {}
    for c in df.columns:
        lc = c.lower()
        if "raster" == lc:
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
    return m


def confusion_fn(target, true_v, pred_v):
    if pd.isna(true_v) or pd.isna(pred_v):
        return None
    if true_v == target and pred_v == target:
        return "TP"
    if true_v != target and pred_v != target:
        return "TN"
    if true_v != target and pred_v == target:
        return "FP"
    if true_v == target and pred_v != target:
        return "FN"
    return None


def read_chm_tile(raster_name, row_off, col_off, cs=128):
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


def sld_to_pil(tile):
    rgb = _apply_sld(tile)  # expected HxWx3 uint8 RGB
    return Image.fromarray(rgb)


def embed_caption_pil(img, caption, height=80):
    w, h = img.size
    out = Image.new("RGB", (w, h + height), (255, 255, 255))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    lines = textwrap.wrap(caption, width=80)
    y = h + 6
    for line in lines:
        draw.text((6, y), line, fill=(0, 0, 0), font=font)
        y += 12
    return out


def main():
    csvp = find_predictions_csv()
    if not csvp:
        print("No predictions CSV found; aborting", file=sys.stderr)
        sys.exit(2)
    df = load_df(csvp)
    cols = pick_cols(df)
    true_col = cols.get("true") or ("label" if "label" in df.columns else None)
    pred_col = cols.get("pred") or ("predicted" if "predicted" in df.columns else None)
    prob_col = cols.get("prob")
    if not true_col or not pred_col:
        print("Could not find true/pred columns; aborting", file=sys.stderr)
        sys.exit(2)

    # choose target
    classes = list(pd.unique(df[true_col].dropna()))
    target = next((c for c in classes if str(c).lower() == "cdw"), classes[0])

    df = df.copy()
    df["confusion"] = df.apply(lambda r: confusion_fn(target, r[true_col], r[pred_col]), axis=1)

    clear_out()

    # pick up to 5 from each class
    picks = []
    for c in ("TP", "FP", "FN", "TN"):
        subset = df[df["confusion"] == c]
        if subset.empty:
            continue
        chosen = subset.sample(min(5, len(subset)), random_state=42)
        picks.append((c, chosen))

    # write confusion tiles SLD + caption
    for c, subset in picks:
        for idx, (_, r) in enumerate(subset.iterrows(), start=1):
            if "raster" in cols:
                cs = (
                    int(r.get(cols.get("chunk_size"), 128))
                    if cols.get("chunk_size") in r.index
                    else 128
                )
                try:
                    tile = read_chm_tile(
                        r[cols["raster"]], int(r[cols["row_off"]]), int(r[cols["col_off"]]), cs
                    )
                except Exception as e:
                    print("Skipping tile read error", e)
                    continue
                img = sld_to_pil(tile)
                prob = float(r[prob_col]) if prob_col and pd.notna(r.get(prob_col)) else None
                caption = f"Confusion={c}  True={r[true_col]}  Pred={r[pred_col]}"
                if prob is not None:
                    caption += f"  Prob={prob:.2f}"
                outname = f"{c}_{idx}_{Path(r[cols['raster']]).stem}_{int(r[cols['row_off']])}_{int(r[cols['col_off']])}_sld.png"
                outp = OUT / outname
                out_img = embed_caption_pil(img, caption)
                out_img.save(outp)
            else:
                # path mode
                p = Path(r[cols["path"]])
                if not p.exists():
                    alt = Path.cwd() / str(r[cols["path"]])
                    if alt.exists():
                        p = alt
                    else:
                        print("Missing image", r[cols["path"]])
                        continue
                img = Image.open(p).convert("RGB")
                prob = float(r[prob_col]) if prob_col and pd.notna(r.get(prob_col)) else None
                caption = f"Confusion={c}  True={r[true_col]}  Pred={r[pred_col]}"
                if prob is not None:
                    caption += f"  Prob={prob:.2f}"
                outp = OUT / f"{c}_{idx}_{p.stem}_sld.png"
                out_img = embed_caption_pil(img, caption)
                out_img.save(outp)

    # problems: prefer FP/FN else any
    prefs = df[df["confusion"].isin(["FP", "FN"])]
    if len(prefs) >= 5:
        problems = prefs.sample(5, random_state=1)
    else:
        problems = df.sample(5, random_state=1)

    for idx, (_, r) in enumerate(problems.iterrows(), start=1):
        if "raster" in cols:
            try:
                cs = (
                    int(r.get(cols.get("chunk_size"), 128))
                    if cols.get("chunk_size") in r.index
                    else 128
                )
                tile = read_chm_tile(
                    r[cols["raster"]], int(r[cols["row_off"]]), int(r[cols["col_off"]]), cs
                )
            except Exception as e:
                print("Skipping problem tile read error", e)
                continue
            img = sld_to_pil(tile)
        else:
            p = Path(r[cols["path"]])
            if not p.exists():
                alt = Path.cwd() / str(r[cols["path"]])
                if alt.exists():
                    p = alt
                else:
                    print("Missing problem image", r[cols["path"]])
                    continue
            img = Image.open(p).convert("RGB")
        prob = float(r[prob_col]) if prob_col and pd.notna(r.get(prob_col)) else None
        caption = f"Problem {idx}  True={r[true_col]}  Pred={r[pred_col]}"
        if prob is not None:
            caption += f"  Prob={prob:.2f}"
        outp = (
            OUT
            / f"PROB_{idx}_{idx}_{Path(r[cols['raster']]).stem if 'raster' in cols else Path(r[cols['path']]).stem}_sld.png"
        )
        out_img = embed_caption_pil(img, caption)
        out_img.save(outp)

    print("Regeneration complete; outputs in", OUT)


if __name__ == "__main__":
    main()

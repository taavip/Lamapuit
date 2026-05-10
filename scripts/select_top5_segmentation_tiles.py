"""
Select top 5 CHM TIF tiles for full-scale segmentation annotation.

Methodology:
  1. Data cleaning: filter tiles with < 5% valid pixels or mean_height < 0.05m
     Note: CHM is HAG-filtered 0-1.3m, so max height is always 1.3m by construction
     and NoData (=-9999) marks non-forest areas — both are expected and not disqualifying.
  2. CSV-based scoring: CWD density, uncertainty, from ensemble classifier
  3. CHM raster scoring: global_std, valid_frac, mean_height
  4. Diversity selection: pick best representative per category
  5. Document ranking rationale in docs/top5_segmentation_tiles.md
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

TIF_DIR = Path("/home/tpipar/project/Lamapuit/data/lamapuit/chm_max_hag_13_drop")
CSV_PATH = Path(
    "/home/tpipar/project/Lamapuit/data/chm_variants/"
    "labels_canonical_with_splits_retrained_ensemble.csv"
)
OUTPUT_MD = Path("/home/tpipar/project/Lamapuit/docs/top5_segmentation_tiles.md")

# CHM is HAG 0-1.3m: nodata=-9999 marks non-forest; max height always 1.3m by cap.
# Valid filters for this dataset type:
MIN_VALID_FRAC = 0.05     # at least 5% non-NoData pixels (has forest)
MIN_MEAN_HEIGHT = 0.05    # mean of valid pixels > 5 cm (not just noise floor)
TARGET_CWD_LOW = 0.10     # ideal CWD window ratio lower bound
TARGET_CWD_HIGH = 0.40    # ideal CWD window ratio upper bound
UNCERTAINTY_LOW = 0.40
UNCERTAINTY_HIGH = 0.60


def compute_raster_stats(tif_path: Path) -> dict:
    """Return valid_frac, mean_height, global_std for a HAG-filtered CWD CHM TIF."""
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        nodata_val = src.nodata if src.nodata is not None else -9999.0
        total_pixels = data.size
    valid = data[(data != nodata_val) & (data > -9000)]
    if len(valid) == 0:
        return {"valid_frac": 0.0, "mean_height": 0.0, "global_std": 0.0}
    return {
        "valid_frac": float(len(valid) / total_pixels),
        "mean_height": float(valid.mean()),
        "global_std": float(valid.std()),
    }


def compute_csv_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-raster stats from the ensemble labels CSV."""
    rows = []
    for raster, grp in df.groupby("raster"):
        total = len(grp)
        cdw_count = (grp["label"] == "cdw").sum()
        uncertain_count = (
            (grp["model_prob"] >= UNCERTAINTY_LOW) &
            (grp["model_prob"] <= UNCERTAINTY_HIGH)
        ).sum()
        cwd_ratio = cdw_count / total if total > 0 else 0.0
        uncertain_ratio = uncertain_count / total if total > 0 else 0.0
        cdw_probs = grp.loc[grp["label"] == "cdw", "model_prob"]
        mean_cdw_prob = float(cdw_probs.mean()) if len(cdw_probs) > 0 else 0.0
        rows.append({
            "raster": raster,
            "total_chunks": total,
            "cdw_count": int(cdw_count),
            "cwd_ratio": float(cwd_ratio),
            "uncertain_count": int(uncertain_count),
            "uncertain_ratio": float(uncertain_ratio),
            "mean_cdw_prob": mean_cdw_prob,
            "map_sheet": grp["map_sheet"].iloc[0],
            "year": grp["year"].iloc[0],
        })
    return pd.DataFrame(rows)


def load_raster_stats(rasters: list) -> pd.DataFrame:
    rows = []
    for fname in rasters:
        path = TIF_DIR / fname
        if not path.exists():
            print(f"  WARNING: {fname} not found", file=sys.stderr)
            continue
        stats = compute_raster_stats(path)
        stats["raster"] = fname
        rows.append(stats)
    print(f"  Loaded raster stats for {len(rows)} TIF files")
    return pd.DataFrame(rows)


def score_and_rank(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()

    # S_density: proximity to ideal CWD ratio [10%-40%]
    center = (TARGET_CWD_LOW + TARGET_CWD_HIGH) / 2
    half_width = (TARGET_CWD_HIGH - TARGET_CWD_LOW) / 2
    df["S_density"] = np.maximum(
        0.0,
        1.0 - abs(df["cwd_ratio"] - center) / half_width
    )

    # S_uncertainty: fraction of uncertain windows (higher = more informative)
    df["S_uncertainty"] = df["uncertain_ratio"]

    # S_complexity: high std = complex landscape
    std_max = df["global_std"].max()
    df["S_complexity"] = df["global_std"] / std_max if std_max > 0 else 0.0

    df["score_composite"] = (
        0.40 * df["S_density"] +
        0.35 * df["S_uncertainty"] +
        0.25 * df["S_complexity"]
    )

    # Thresholds calibrated from actual data distribution:
    # - uncertain_ratio max is 14.2% (ensemble is confident), so use 80th pct (~7.5%)
    # - cwd_ratio < 10%: 4 tiles; > 40%: 16 tiles
    uncertain_hi = df["uncertain_ratio"].quantile(0.80)
    std_hi = df["global_std"].quantile(0.75)
    std_mid = df["global_std"].quantile(0.60)

    def assign_category(row):
        if row["cwd_ratio"] > 0.40:
            return "Segane sasi (tormimurd)"
        if row["uncertain_ratio"] >= uncertain_hi:
            return "Maaraamatu (piiripealne)"
        if row["global_std"] >= std_hi and row["cwd_ratio"] < 0.15:
            return "Maastiku serv (kraavid/nolvad)"
        if row["cwd_ratio"] < 0.10 and row["global_std"] >= std_mid:
            return "Raske lank (oksavallid)"
        if 0.12 <= row["cwd_ratio"] <= 0.35:
            return "Kuldne kesktee (tyypiline mets)"
        return "Ulejaanud"

    df["category"] = df.apply(assign_category, axis=1)
    return df.sort_values("score_composite", ascending=False).reset_index(drop=True)


def select_diverse_top5(df: pd.DataFrame) -> list:
    priority_order = [
        "Kuldne kesktee (tyypiline mets)",
        "Raske lank (oksavallid)",
        "Segane sasi (tormimurd)",
        "Maaraamatu (piiripealne)",
        "Maastiku serv (kraavid/nolvad)",
        "Ulejaanud",
    ]
    selected = []
    used_map_sheets = set()
    df_sorted = df.sort_values("score_composite", ascending=False)

    for cat in priority_order:
        if len(selected) >= 5:
            break
        candidates = df_sorted[df_sorted["category"] == cat]
        for _, row in candidates.iterrows():
            ms = str(row["map_sheet"])[:4]
            if ms not in used_map_sheets:
                selected.append(row.to_dict())
                used_map_sheets.add(ms)
                break
        else:
            if len(candidates) > 0 and len(selected) < 5:
                row = candidates.iloc[0]
                if row["raster"] not in [s["raster"] for s in selected]:
                    selected.append(row.to_dict())

    for _, row in df_sorted.iterrows():
        if len(selected) >= 5:
            break
        if row["raster"] not in [s["raster"] for s in selected]:
            selected.append(row.to_dict())

    return selected[:5]


def write_report(df_all: pd.DataFrame, df_scored: pd.DataFrame, top5: list) -> None:
    filtered = df_scored  # already filtered and scored

    lines = [
        "# Top 5 CHM TIF faili segmenteerimise annoteerimiseks",
        "",
        "**Genereeritud:** 2026-05-04",
        "**Skript:** `scripts/select_top5_segmentation_tiles.py`",
        "**Sisend-CSV:** `data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv`",
        "",
        "## Metodoloogia",
        "",
        "### 1. Andmete taustainfo",
        "",
        "CHM-failid on HAG-filtreeritud vahemikus 0-1.3m (lamapuidu korgusruum).",
        "Seega on koigi failide max korgus tapselt 1.3m (cap) ning NoData (-9999)",
        "tahistab metsa-valist ala (pollumaa, soo) - see on oodatav, mitte viga.",
        "",
        "### 2. Filtreerimiskriteeriumid",
        "",
        "| Filter | Vaartus | Pohjus |",
        "|--------|---------|--------|",
        f"| `valid_frac >= {MIN_VALID_FRAC:.0%}` | Vahem kui {MIN_VALID_FRAC:.0%} piksleid on NoData | Tagab, et ruudul on piisavalt metsa |",
        f"| `mean_height >= {MIN_MEAN_HEIGHT:.2f} m` | Kesk-korgus alla {MIN_MEAN_HEIGHT:.2f} m | Valistab murapohja-ruudud |",
        "",
        "### 3. Moodikute arvutamine",
        "",
        "| Moodik | Allikas | Kirjeldus |",
        "|--------|---------|-----------|",
        "| `cwd_ratio` | CSV `label` | CWD-positiivsete akende osakaal koigist 128x128 aknast |",
        "| `uncertain_ratio` | CSV `model_prob in [0.40, 0.60]` | Osakaal aknaid, kus mudel kahtleb |",
        "| `global_std` | CHM TIF pikslid | Korguse standardhalve kogu 1km ruudul (valid pikslid) |",
        "| `valid_frac` | CHM TIF | Metsa-ala osakaal (mitte-NoData pikslid) |",
        "| `mean_height` | CHM TIF | Kesk-korgus valid pikslitel |",
        "",
        "### 4. Skoorimisvalem",
        "",
        "```",
        "S_density     = max(0, 1 - |cwd_ratio - 0.25| / 0.15)   # ideaal 10-40%",
        "S_uncertainty = uncertain_ratio                            # rohkem = informatiivsem",
        "S_complexity  = global_std / max(global_std)              # normaliseeritud",
        "score = 0.40 x S_density + 0.35 x S_uncertainty + 0.25 x S_complexity",
        "```",
        "",
        "### 5. Valiku strateegia (mitmekesisus)",
        "",
        "Igast kategooriast valiti parim esindaja, eelistades geograafilist mitmekesisust:",
        "",
        "| # | Kategooria | Miks vajalik |",
        "|---|-----------|--------------|",
        "| 1 | Kuldne kesktee | Mudeli baasteadmised puhtas metsas |",
        "| 2 | Raske lank | Et mudel ei peaks oksavalle palgiks |",
        "| 3 | Segane sasi | Uksteise peal asuvate palkide eristamine |",
        "| 4 | Maaraamatu | Mudeli oppimiskorvera sisendalad |",
        "| 5 | Maastiku serv | Kraaviserva ei pea palgiks (hard negative) |",
        "",
        "## Tulemused",
        "",
        f"- Kandidaate kokku: **{len(df_all)}** TIF-faili",
        f"- Parast filtreid: **{len(filtered)}** TIF-faili",
        "",
        "### Top 5 valitud failid",
        "",
        "| Jarg | TIF fail | Kategooria | cwd_ratio | uncertain_ratio | global_std | valid_frac | Skoor |",
        "|------|----------|-----------|-----------|-----------------|------------|------------|-------|",
    ]

    for i, t in enumerate(top5, 1):
        lines.append(
            f"| {i} | `{t['raster']}` | {t['category']} "
            f"| {t['cwd_ratio']:.1%} | {t['uncertain_ratio']:.1%} "
            f"| {t['global_std']:.3f} m | {t['valid_frac']:.1%} "
            f"| {t['score_composite']:.3f} |"
        )

    lines += ["", "### Detailne pohjendusdus", ""]
    for i, t in enumerate(top5, 1):
        lines += [
            f"#### {i}. `{t['raster']}` -- {t['category']}",
            f"- **map_sheet:** {t['map_sheet']} | **aasta:** {t['year']}",
            f"- **CWD aknad:** {t['cdw_count']} / {t['total_chunks']} ({t['cwd_ratio']:.1%})",
            f"- **Ebakindlad aknad:** {t['uncertain_count']} ({t['uncertain_ratio']:.1%})",
            f"- **CHM std:** {t['global_std']:.3f} m | **valid_frac:** {t['valid_frac']:.1%}"
            f" | **mean_height:** {t['mean_height']:.3f} m",
            f"- **Skoor:** S_density={t['S_density']:.3f},"
            f" S_uncertainty={t['S_uncertainty']:.3f},"
            f" S_complexity={t['S_complexity']:.3f} -> composite={t['score_composite']:.3f}",
            "",
        ]

    lines += [
        "### Kogu pingerivi (top 20 parast filtreid)",
        "",
        "| Jarg | TIF | Kategooria | cwd_ratio | uncertain_ratio | global_std | Skoor |",
        "|------|-----|-----------|-----------|-----------------|------------|-------|",
    ]
    for i, (_, row) in enumerate(
        filtered.sort_values("score_composite", ascending=False).head(20).iterrows(), 1
    ):
        lines.append(
            f"| {i} | `{row['raster']}` | {row['category']} "
            f"| {row['cwd_ratio']:.1%} | {row['uncertain_ratio']:.1%} "
            f"| {row['global_std']:.3f} | {row['score_composite']:.3f} |"
        )

    lines += [
        "",
        "---",
        "*Koik TIF-failid asuvad: `data/lamapuit/chm_max_hag_13_drop/`*",
        "*Enne sildistamist ava QGIS-is ja kontrolli visuaalselt metsatyypi.*",
    ]

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines))
    print(f"\n  Report saved: {OUTPUT_MD}")


def main():
    print("Loading CSV...")
    df_csv = pd.read_csv(CSV_PATH)
    print(f"  {len(df_csv)} rows, {df_csv['raster'].nunique()} unique rasters")

    print("Computing CSV stats per raster...")
    csv_stats = compute_csv_stats(df_csv)

    print("Loading CHM raster stats...")
    raster_stats = load_raster_stats(csv_stats["raster"].tolist())

    print("Merging...")
    merged = csv_stats.merge(raster_stats, on="raster", how="inner")
    print(f"  Merged: {len(merged)} rasters")

    print("Applying filters...")
    before = len(merged)
    filtered = merged[
        (merged["valid_frac"] >= MIN_VALID_FRAC) &
        (merged["mean_height"] >= MIN_MEAN_HEIGHT)
    ].copy()
    print(f"  After filters: {len(filtered)} / {before} rasters remain")

    print("Scoring and categorising...")
    scored = score_and_rank(filtered)

    print("\nCategory distribution:")
    print(scored["category"].value_counts().to_string())

    print("\nTop 10 by score:")
    cols = ["raster", "category", "cwd_ratio", "uncertain_ratio", "global_std", "score_composite"]
    print(scored[cols].head(10).to_string(index=False))

    print("\nSelecting diverse top 5...")
    top5 = select_diverse_top5(scored)

    print("\n=== TOP 5 SELECTED TILES ===")
    for i, t in enumerate(top5, 1):
        print(
            f"{i}. {t['raster']}\n"
            f"   Category: {t['category']}\n"
            f"   cwd_ratio={t['cwd_ratio']:.1%}  uncertain_ratio={t['uncertain_ratio']:.1%}"
            f"  global_std={t['global_std']:.3f}m  valid_frac={t['valid_frac']:.1%}\n"
            f"   Score: {t['score_composite']:.4f}\n"
        )

    write_report(merged, scored, top5)


if __name__ == "__main__":
    main()

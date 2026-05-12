#!/usr/bin/env python3
"""Create final-method focused support material for segmentation methodology."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "LaTeX/Lamapuidu_tuvastamine/estonian/segmenteerimise_metoodika_alusinfo.md"
FIG_DIR = ROOT / "LaTeX/Lamapuidu_tuvastamine/estonian/joonised/segmenteerimine_alusinfo"

MASK_TIF = ROOT / "seg_pipeline/output/phase1_masks/406455_2021_tava_truemask.tif"
MASK_META = ROOT / "seg_pipeline/output/phase1_masks/406455_2021_tava_truemask_meta.json"
PATCH_INDEX = ROOT / "seg_pipeline/output/phase2_dataset_v10_reconstructed/patch_index_composite.csv"
LABEL_GPKG = ROOT / "data/labels/cdw_labels_MP.gpkg"
AREA_GPKG = ROOT / "data/labels/valid_area.gpkg"

TILE_WIDTH = 5000
TILE_HEIGHT = 5000
PIXEL_SIZE_M = 0.2
STRIPE_WIDTH = 1000
PATCH_SIZE = 256
STRIDE = 192
BUFFER_PX = 64


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def md_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def read_mask_arrays() -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(MASK_TIF) as src:
        target = src.read(1).astype(bool)
        valid = src.read(2).astype(bool)
    return target, valid


def stripe_pixel_stats(target: np.ndarray, valid: np.ndarray) -> pd.DataFrame:
    rows = []
    for stripe in range(5):
        c0 = stripe * STRIPE_WIDTH
        c1 = (stripe + 1) * STRIPE_WIDTH
        v = valid[:, c0:c1]
        t = target[:, c0:c1] & v
        pos = int(t.sum())
        neg = int((v & ~t).sum())
        ignored = int((~v).sum())
        total = int(v.size)
        rows.append(
            {
                "stripe": stripe,
                "cols": f"{c0}-{c1 - 1}",
                "role": "test" if stripe == 0 else ("fold0 val / fold1 train" if stripe == 1 else "fold0 train / fold1 val"),
                "positive_px": pos,
                "negative_px": neg,
                "ignored_px": ignored,
                "valid_px": pos + neg,
                "total_px": total,
                "positive_valid_pct": 100 * pos / (pos + neg) if pos + neg else 0.0,
                "valid_total_pct": 100 * (pos + neg) / total,
            }
        )
    return pd.DataFrame(rows)


def role_pixel_stats(stripes: pd.DataFrame) -> pd.DataFrame:
    roles = [
        ("Püsiv testala", [0]),
        ("Fold 0 treening", [2, 3, 4]),
        ("Fold 0 valideerimine", [1]),
        ("Fold 1 treening", [1]),
        ("Fold 1 valideerimine", [2, 3, 4]),
    ]
    rows = []
    for role, stripe_ids in roles:
        subset = stripes[stripes["stripe"].isin(stripe_ids)]
        pos = int(subset["positive_px"].sum())
        neg = int(subset["negative_px"].sum())
        ignored = int(subset["ignored_px"].sum())
        total = int(subset["total_px"].sum())
        valid = pos + neg
        rows.append(
            {
                "role": role,
                "stripes": ", ".join(map(str, stripe_ids)),
                "positive_px": pos,
                "negative_px": neg,
                "ignored_px": ignored,
                "valid_px": valid,
                "total_px": total,
                "positive_valid_pct": 100 * pos / valid if valid else 0.0,
                "valid_total_pct": 100 * valid / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def patch_role_stats() -> pd.DataFrame:
    idx = pd.read_csv(PATCH_INDEX)
    idx["positive_patch"] = idx["n_positive"] > 0
    roles = [
        ("Püsiv testala", idx["stripe_id"].eq(0)),
        ("Fold 0 treening", idx["stripe_id"].isin([2, 3, 4])),
        ("Fold 0 valideerimine", idx["stripe_id"].eq(1)),
        ("Fold 1 treening", idx["stripe_id"].eq(1)),
        ("Fold 1 valideerimine", idx["stripe_id"].isin([2, 3, 4])),
    ]
    rows = []
    for role, mask in roles:
        subset = idx[mask]
        rows.append(
            {
                "role": role,
                "patches": len(subset),
                "positive_patches": int(subset["positive_patch"].sum()),
                "positive_patch_pct": 100 * subset["positive_patch"].mean() if len(subset) else 0.0,
                "positive_px_in_patches": int(subset["n_positive"].sum()),
                "valid_px_in_patches": int(subset["n_valid"].sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_mask_workflow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_axis_off()

    boxes = [
        (0.03, 0.68, 0.23, 0.18, "cdw_labels_MP.gpkg\n1236 lamapuidu polügooni\nEPSG:3301"),
        (0.03, 0.40, 0.23, 0.18, "valid_area.gpkg\n1 kontrollitud ala polügoon\nmäärab, kus taust on usaldatav"),
        (0.03, 0.12, 0.23, 0.18, "baseline_chm.tif\n5000×5000 pikslit\n0,2 m rastervõrk"),
        (0.38, 0.52, 0.25, 0.22, "Rasteriseerimine\nsama transform, ulatus ja CRS\nall_touched=True"),
        (0.72, 0.52, 0.24, 0.22, "3-kanaliline mask TIF\nB1 target: 1/0\nB2 valid_mask: 1/0\nB3 ensemble_stub: 0"),
        (0.72, 0.18, 0.24, 0.20, "Mudeli treeningpaan\nimage: CHM kanalid\ntarget: lamapuit/taust\nvalid: loss'i mask"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#F5F7FA", edgecolor="#2F3A45", linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.26, 0.77), (0.38, 0.64)),
        ((0.26, 0.49), (0.38, 0.60)),
        ((0.26, 0.21), (0.38, 0.56)),
        ((0.63, 0.63), (0.72, 0.63)),
        ((0.84, 0.52), (0.84, 0.38)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color="#2F3A45"))

    ax.text(0.50, 0.91, "GPKG polügoonidest pikslipõhiseks juhendmaskiks", ha="center", fontsize=15, weight="bold")
    ax.text(0.50, 0.04, "Oluline metoodiline valik: väljaspool kontrollitud ala piksleid ei käsitleta taustana, vaid ignoreeritakse loss'i arvutamisel.", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_spatial_split(path: Path, stripes: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.4), sharex=True)
    scenarios = [
        ("Püsiv testala", {0: "#D95F02", 1: "#EAEAEA", 2: "#EAEAEA", 3: "#EAEAEA", 4: "#EAEAEA"}),
        ("Fold 0: treening triibud 2-4, valideerimine triip 1", {0: "#D95F02", 1: "#7570B3", 2: "#1B9E77", 3: "#1B9E77", 4: "#1B9E77"}),
        ("Fold 1: treening triip 1, valideerimine triibud 2-4", {0: "#D95F02", 1: "#1B9E77", 2: "#7570B3", 3: "#7570B3", 4: "#7570B3"}),
    ]
    for ax, (title, colors) in zip(axes, scenarios):
        ax.set_xlim(0, 5000)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=11)
        for stripe in range(5):
            row = stripes[stripes["stripe"] == stripe].iloc[0]
            x0 = stripe * STRIPE_WIDTH
            ax.add_patch(Rectangle((x0, 0.12), STRIPE_WIDTH, 0.62, facecolor=colors[stripe], edgecolor="white", linewidth=1.6))
            label = f"triip {stripe}\n{row['cols']}\nLP {row['positive_valid_pct']:.2f}% valid"
            ax.text(x0 + STRIPE_WIDTH / 2, 0.43, label, ha="center", va="center", fontsize=9, color="white" if stripe in (0, 1, 2, 3, 4) and colors[stripe] != "#EAEAEA" else "#222")
        for boundary in [1000, 2000, 3000, 4000]:
            ax.axvspan(boundary - BUFFER_PX, boundary + BUFFER_PX, ymin=0.08, ymax=0.78, color="black", alpha=0.08)
    axes[-1].set_xlabel("Rasteri veerg (0,2 m piksel; 1000 veergu = 200 m)")
    fig.suptitle("Lõplik testala ja kahe foldi ruumiline jaotus", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pixel_distribution(path: Path, roles: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    labels = roles["role"].tolist()
    pos = roles["positive_px"].to_numpy()
    neg = roles["negative_px"].to_numpy()
    ignored = roles["ignored_px"].to_numpy()
    totals = roles["total_px"].to_numpy()
    pos_pct = pos / totals * 100
    neg_pct = neg / totals * 100
    ignored_pct = ignored / totals * 100

    x = np.arange(len(labels))
    ax.bar(x, pos_pct, label="lamapuit", color="#1B9E77")
    ax.bar(x, neg_pct, bottom=pos_pct, label="taust kehtival alal", color="#A6CEE3")
    ax.bar(x, ignored_pct, bottom=pos_pct + neg_pct, label="ignoreeritud", color="#D9D9D9")
    for i, row in roles.iterrows():
        ax.text(i, min(98, pos_pct[i] + neg_pct[i] + 2), f"LP valid alal\n{row['positive_valid_pct']:.2f}%", ha="center", fontsize=8)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Osakaal rolli kogupikslitest (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Lamapuidu, tausta ja ignoreeritud pikslite jaotus testalal ning foldides")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_patch_extraction(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 520)
    ax.set_aspect("equal")
    ax.set_axis_off()

    for i, (x, y) in enumerate([(80, 160), (272, 160), (464, 160), (80, 352), (272, 352)]):
        ax.add_patch(Rectangle((x, y), PATCH_SIZE, PATCH_SIZE, facecolor="#EDF3F8", edgecolor="#4C78A8", linewidth=1.4, alpha=0.75))
        ax.text(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, "256×256 px\n51,2×51,2 m", ha="center", va="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((80, 125), (272, 125), arrowstyle="<->", mutation_scale=12, linewidth=1.2))
    ax.text(176, 105, "stride 192 px = 38,4 m", ha="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((272, 160), (336, 160), arrowstyle="<->", mutation_scale=12, linewidth=1.2, color="#D95F02"))
    ax.text(304, 143, "ülekate\n64 px", ha="center", fontsize=8, color="#D95F02")
    ax.text(350, 485, "Paanide lõikamine CHM-ist ja maskist", ha="center", fontsize=15, weight="bold")
    ax.text(350, 55, "Iga paan sisaldab CHM sisendkanaleid ning sama akna target/valid maske. Paan jäetakse välja, kui kehtivaid piksleid on vähem kui 328.", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ablation_workflow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_axis_off()

    phases = [
        ("Eeltöötlus", "GeoPackage'i polügoonid\nrastermaskiks\nviie CHM-variandi\npaanideks jagamine\nja statistika"),
        ("Faas 2", "Sisendandmete valik\n2A baasmudel\n2B toor-CHM\n2C Gaussi silumine\n2D maskiga toor-CHM\n2E komposiit-CHM"),
        ("Faas 3", "Mudeliarhitektuuri valik\nkahe parima CHM-i põhjal\n3B U-Net++ EfficientNet-B0\n3C U-Net++ EfficientNet-B2\n3E DeepLabV3+\nEfficientNet-B2"),
        ("Faas 4", "Kaofunktsiooni valik\nkahe parima\nkonfiguratsiooni põhjal\n4A DiceFocal\n4D Tversky 0,5/0,5\n4F Tversky 0,7/0,3\n4H Tversky + clDice"),
        ("Faas 5", "Andmerikastamise valik\nkahe parima\nkonfiguratsiooni põhjal\n5A andmerikastamiseta\n5D täisandmerikastus,\npehmed sihtmaskid ja SWA\n5E sama ilma SWA-ta"),
        ("Faas 6", "Lõplik testhindamine\nkaks parimat konfiguratsiooni\nja varasem parim võrdlusmudel\ntreening kogu andmestikul\njättes välja testandmestiku\nhindamine testandmestikul"),
    ]

    positions = [
        (0.05, 0.58),
        (0.38, 0.58),
        (0.71, 0.58),
        (0.05, 0.18),
        (0.38, 0.18),
        (0.71, 0.18),
    ]
    box_w = 0.24
    box_h = 0.28
    for i, ((title, text), (x, y)) in enumerate(zip(phases, positions)):
        color = "#F5F7FA" if i not in (1, 5) else ("#E8F4EF" if i == 1 else "#FCEFE8")
        ax.add_patch(Rectangle((x, y), box_w, box_h, facecolor=color, edgecolor="#2F3A45", linewidth=1.2))
        ax.text(x + box_w / 2, y + box_h - 0.055, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + box_w / 2, y + box_h / 2 - 0.035, text, ha="center", va="center", fontsize=8.3, linespacing=1.18)

    arrows = [
        (positions[0], positions[1]),
        (positions[1], positions[2]),
        (positions[2], positions[3]),
        (positions[3], positions[4]),
        (positions[4], positions[5]),
    ]
    for idx, ((x0, y0), (x1, y1)) in enumerate(arrows):
        if y0 == y1:
            start = (x0 + box_w, y0 + box_h / 2)
            end = (x1, y1 + box_h / 2)
        else:
            start = (x0 + box_w / 2, y0)
            end = (x1 + box_w / 2, y1 + box_h)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.3, color="#2F3A45"))
        if idx != 0:
            label_x = (start[0] + end[0]) / 2
            label_y = (start[1] + end[1]) / 2 + 0.018
            ax.text(label_x, label_y, "2 parimat", ha="center", fontsize=9, color="#2F3A45")

    ax.text(0.50, 0.93, "Automaatne kahe parima konfiguratsiooni edasikandmisega metoodika valiku töövoog", ha="center", fontsize=14, weight="bold")
    ax.text(
        0.50,
        0.06,
        "Faasides 2-5 põhineb valik ainult valideerimistulemustel: keskmine valideerimis-clDice → keskmine valideerimis-F1 → väiksem valideerimis-clDice'i standardhälve.\n"
        "Püsivat testandmestikku kasutatakse hinnanguks ainult faasis 6.",
        ha="center",
        fontsize=8.8,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "font.size": 9})

    label_gdf = gpd.read_file(LABEL_GPKG).to_crs(epsg=3301)
    area_gdf = gpd.read_file(AREA_GPKG).to_crs(epsg=3301)
    label_area = label_gdf.geometry.area
    mask_meta = json.loads(MASK_META.read_text())
    target, valid = read_mask_arrays()

    stripes = stripe_pixel_stats(target, valid)
    roles = role_pixel_stats(stripes)
    patches = patch_role_stats()

    stripes.to_csv(FIG_DIR / "seg_final_stripe_pixel_stats.csv", index=False)
    roles.to_csv(FIG_DIR / "seg_final_role_pixel_stats.csv", index=False)
    patches.to_csv(FIG_DIR / "seg_final_role_patch_stats.csv", index=False)

    plot_mask_workflow(FIG_DIR / "seg_gpkg_mask_workflow.png")
    plot_spatial_split(FIG_DIR / "seg_final_spatial_split.png", stripes)
    plot_pixel_distribution(FIG_DIR / "seg_final_pixel_distribution.png", roles)
    plot_patch_extraction(FIG_DIR / "seg_patch_extraction_scheme.png")
    plot_ablation_workflow(FIG_DIR / "seg_top2_ablation_workflow.png")

    stripes_md = stripes.copy()
    for col in ["positive_valid_pct", "valid_total_pct"]:
        stripes_md[col] = stripes_md[col].map(lambda v: fmt(v))
    for col in ["positive_px", "negative_px", "ignored_px", "valid_px", "total_px"]:
        stripes_md[col] = stripes_md[col].map(lambda v: f"{int(v):,}".replace(",", " "))

    roles_md = roles.copy()
    for col in ["positive_valid_pct", "valid_total_pct"]:
        roles_md[col] = roles_md[col].map(lambda v: fmt(v))
    for col in ["positive_px", "negative_px", "ignored_px", "valid_px", "total_px"]:
        roles_md[col] = roles_md[col].map(lambda v: f"{int(v):,}".replace(",", " "))

    patches_md = patches.copy()
    patches_md["positive_patch_pct"] = patches_md["positive_patch_pct"].map(lambda v: fmt(v))
    for col in ["patches", "positive_patches", "positive_px_in_patches", "valid_px_in_patches"]:
        patches_md[col] = patches_md[col].map(lambda v: f"{int(v):,}".replace(",", " "))

    valid_pct = mask_meta["n_valid"] / (TILE_WIDTH * TILE_HEIGHT) * 100
    positive_valid_pct = mask_meta["n_positive"] / mask_meta["n_valid"] * 100
    label_total_area = float(label_area.sum())
    label_mean_area = float(label_area.mean())
    label_median_area = float(label_area.median())
    area_polygon_area = float(area_gdf.geometry.area.sum())

    md = f"""# Segmenteerimise metoodika alusinfo

See dokument koondab lõpliku segmenteerimise metoodika kirjutamiseks vajaliku tehnilise alusinfo. Fookus on viimasel kasutatud töövool: GeoPackage'i polügoonide rasteriseerimine, kehtiva analüüsiala mask, mudelile antavate paanide koostamine ning püsiva testala ja kahe foldi jaotus.

## 1. Lõpliku metoodika eesmärk

Segmenteerimise eesmärk oli muuta lamapuidu tuvastamine paanipõhisest klassifitseerimisest pikslipõhiseks kaardistamiseks. Klassifitseerimisel piisab teadmisest, kas 128×128 paanis on lamapuitu. Segmenteerimisel peab mudel õppima lamapuidu kuju ja asukohta iga piksli tasemel, mistõttu on vajalik eraldi maskiandmestik.

Lõplikus metoodikas käsitleti lamapuitu semantilise segmenteerimisena: iga kehtiva analüüsiala piksel on kas lamapuit või taust. Piksleid väljaspool kontrollitud ala ei kasutatud treeningkao arvutamisel, sest neid ei saa usaldusväärselt tõlgendada taustana.

## 2. Sisendfailid ja nende roll

Peamised sisendfailid olid:

- `data/labels/cdw_labels_MP.gpkg` — käsitsi märgendatud lamapuidu polügoonid;
- `data/labels/valid_area.gpkg` — kontrollitud ala piir, mille sees märgendamata piksleid tohib käsitleda taustana;
- `seg_pipeline/input/baseline_chm.tif` — referentsraster, mille ruudustikku, ulatust ja koordinaatsüsteemi kasutati maski rasteriseerimisel;
- `seg_pipeline/input/composite_4band.tif` — lõplik CHM sisendvariant mudeli treenimiseks.

Lamapuidu GeoPackage sisaldas {len(label_gdf)} objekti. Polügoonide kogupindala oli {label_total_area:.1f} m², keskmine pindala {label_mean_area:.2f} m² ja mediaan {label_median_area:.2f} m². Kehtiva ala polügooni pindala oli {area_polygon_area:.1f} m².

Joonis: `joonised/segmenteerimine_alusinfo/seg_gpkg_mask_workflow.png`

## 3. GPKG polügoonide rasteriseerimine

Mõlemad GeoPackage'i kihid teisendati EPSG:3301 koordinaatsüsteemi ja rasteriseeriti samale 5000×5000 pikslisele ruudustikule kui CHM. Rasteriseerimisel kasutati `rasterio.features.rasterize` loogikat ning `all_touched=True` seadet, et kitsad lamapuidu polügoonid ei kaoks rastervõrku teisendamisel ära. See on oluline, sest lamapuit on sageli kitsas objekt ja võib 0,2 m piksli juures paikneda pikslipiiridel.

Rasteriseerimise järel koostati 3-kanaliline juhendmask:

1. `target` — 1 tähendab lamapuitu, 0 tähendab tausta;
2. `valid_mask` — 1 tähendab, et pikslit kasutatakse loss'i arvutamisel, 0 tähendab ignoreeritud pikslit;
3. `ensemble_prob` — lõplikus V10 töövoos täideti nullidega, et säilitada ühilduvus varasema pipeline'iga.

Pikslite loogika oli järgmine:

- lamapuit: piksel on `valid_area.gpkg` sees ja `cdw_labels_MP.gpkg` polügooni sees;
- taust: piksel on `valid_area.gpkg` sees, kuid lamapuidu polügoonist väljas;
- ignoreeritud: piksel on kontrollitud alast väljas või CHM-is mittekehtiv.

Kogu maskis oli kehtivaid piksleid {mask_meta["n_valid"]:,} ehk {valid_pct:.2f}% rasterpinnast. Lamapuidu piksleid oli {mask_meta["n_positive"]:,}, mis moodustas {positive_valid_pct:.2f}% kehtivast analüüsialast. See näitab tugevat klasside tasakaalustamatust.

## 4. Püsiv testala ja foldide jaotus

Raster jagati viieks vertikaalseks 1000 veeru laiuseks triibuks. Kuna piksli suurus oli 0,2 m, vastas üks triip 200 m laiusele alale. Läänepoolne triip 0 jäeti püsivaks testalaks. Seda ei kasutatud mudeli valikul ega hüperparameetrite häälestamisel.

Lõplikus kahe foldiga jaotuses kasutati järgmisi rolle:

- püsiv testala: triip 0;
- fold 0 valideerimine: triip 1;
- fold 0 treening: triibud 2, 3 ja 4;
- fold 1 treening: triip 1;
- fold 1 valideerimine: triibud 2, 3 ja 4.

Joonis: `joonised/segmenteerimine_alusinfo/seg_final_spatial_split.png`

### Triipude pikslijaotus

{md_table(stripes_md, ["stripe", "cols", "role", "positive_px", "negative_px", "ignored_px", "positive_valid_pct", "valid_total_pct"], ["Triip", "Veerud", "Roll", "Lamapuit px", "Taust px", "Ignoreeritud px", "LP valid (%)", "Valid kogu (%)"])}

### Testala ja foldide pikslijaotus

{md_table(roles_md, ["role", "stripes", "positive_px", "negative_px", "ignored_px", "positive_valid_pct", "valid_total_pct"], ["Roll", "Triibud", "Lamapuit px", "Taust px", "Ignoreeritud px", "LP valid (%)", "Valid kogu (%)"])}

Joonis: `joonised/segmenteerimine_alusinfo/seg_final_pixel_distribution.png`

## 5. Paanide moodustamine mudelile

Mudeli sisendiks ei antud tervet 5000×5000 rasterpilti korraga, vaid sellest lõigati 256×256 piksliga paanid. Paanide samm oli 192 pikslit, mis tähendab 64 pikslit ülekatet. Maapinnal vastas üks paan 51,2×51,2 meetrile ning samm 38,4 meetrile.

Joonis: `joonised/segmenteerimine_alusinfo/seg_patch_extraction_scheme.png`

Paan jäeti andmestikust välja, kui selles oli vähem kui 328 kehtivat pikslit. See välistas peaaegu tühjad või valdavalt ignoreeritud piirkonnad. Iga treeningnäide koosnes kahest omavahel samas rasteraknas olevast osast:

- sisendpilt: CHM kanalid, lõplikus põhivariandis `composite_4band`;
- juhend: sama akna `target` ja `valid_mask`.

Paanide jaotus rollide kaupa oli:

{md_table(patches_md, ["role", "patches", "positive_patches", "positive_patch_pct", "positive_px_in_patches", "valid_px_in_patches"], ["Roll", "Paanid", "Pos. paanid", "Pos. paanid (%)", "LP px paanides", "Valid px paanides"])}

## 6. Mudelile antav lõplik sisend

Lõplikus metoodikas kasutati komposiitset CHM sisendit, sest see koondab mitu lamapuidu jaoks olulist infokihti. Komposiitsisendis on nii silutud kõrgusmuster, toorem kõrgussignaal, baasmudeli lokaalsed kõrgusväärtused kui ka kehtivate pikslite mask. See on lamapuidu puhul põhjendatud, sest osa objekte tuleb esile ainult tooretes lokaalse kõrguse muutustes, osa aga pigem silutud pikliku struktuurina.

Treeningul normaliseeriti CHM kanalid treeningandmestiku statistika alusel. Maskikanalit käsitleti binaarse infona. Mudelile anti korraga sisendpaan, sihtmask ja kehtivusmask; loss arvutati ainult nendel pikslitel, kus `valid_mask=1`.

## 7. Automaatne ablation ja lõpliku konfiguratsiooni valik

Lõpliku segmenteerimismetoodika valikul kasutati skripti `run_full_ablation_automated_top2.sh`. Selle eesmärk oli vältida olukorda, kus iga faasi parim üksiktulemus lukustab kogu järgneva otsinguruumi liiga vara. Selle asemel kanti igast faasist edasi kaks parimat konfiguratsiooni.

Joonis: `joonised/segmenteerimine_alusinfo/seg_top2_ablation_workflow.png`

Töövoog koosnes järgmistest sammudest:

1. `Preflight` — vajadusel seoti CHM sisendfailid `seg_pipeline/input` kausta ning ehitati iga CHM variandi jaoks uuesti `patch_index_*.csv` ja `band_stats_*.json`.
2. `Faas 2` — võrreldi CHM andmestiku variante: `baseline`, `raw`, `gauss`, `masked` ja `composite`.
3. `Faas 3` — kahe parima CHM variandi peal võrreldi mudeliarhitektuure. Lukustatud vaikeseadistuses kasutati kandidaate `3B`, `3C` ja `3E`, vastavalt U-Net++ EfficientNet-B0, U-Net++ EfficientNet-B2 ja DeepLabV3+ EfficientNet-B2.
4. `Faas 4` — kahe parima andmestiku/mudeli kombinatsiooni peal võrreldi kaofunktsioone ja nende parameetreid. Lukustatud kandidaadid olid `4A`, `4D`, `4F` ja `4H`, hõlmates DiceFocalit, tasakaalustatud Tverskyt, kõrgema täpsuse Tverskyt ning Tversky+clDice kombinatsiooni.
5. `Faas 5` — kahe parima andmestiku/mudeli/loss'i kombinatsiooni peal võrreldi augmentatsiooni ja regulariseerimise seadistusi. Lukustatud kandidaadid olid `5A`, `5D` ja `5E`, mis eristasid augmentatsioonita treeningut, täisaugmentatsiooni koos soft target'ite ja SWA-ga ning sama seadistust ilma SWA-ta.
6. `Faas 6` — pärast mudelivaliku lukustamist tehti lõplik testhinnang. Selles faasis kasutati `--evaluate-test` ja `--final-train-all` seadeid, st mudel treeniti kõigil mitte-test triipudel ning hinnati püsival testalal ehk triibul 0.

Valikumõõdik oli lukustatud `val_cldice`. See sobib lamapuidu ülesandesse paremini kui ainult pikslipõhine Dice, sest lamapuit on piklik ja katkendlik objektiklass ning oluline on säilitada objekti teljeline pidevus. Faaside 2-5 jooksul testala ei kasutatud. Konfiguratsioonid järjestati järgmise reegliga:

1. suurem keskmine `val_cldice`;
2. võrdse tulemuse korral suurem keskmine `val_f1`;
3. seejärel väiksem `val_cldice` standardhälve;
4. lõpuks suurem foldide/ridade katvus.

Selline järjestus eraldab mudelivaliku ja lõpliku hindamise. Metoodika seisukohast on oluline rõhutada, et püsivat testala ei kasutatud ei CHM variandi, arhitektuuri, loss'i ega augmentatsiooni valimiseks. Testala avati alles siis, kui lõplik konfiguratsioon oli valideerimistulemuste põhjal lukustatud.

## 8. Metoodika kirjutamise tuum lõputöös

Peatükis `Segmenteerimise metoodika` peaks rõhk olema järgmisel loogikal:

1. miks paanisildid ei ole pikslipõhise ülesande jaoks piisavad;
2. kuidas `cdw_labels_MP.gpkg` ja `valid_area.gpkg` muudeti rastermaskiks;
3. miks kontrollitud ala mask on vajalik, et märgendamata ala ei muutuks ekslikult taustaks;
4. kuidas püsiv testala ja fold0/fold1 ruumiliselt eraldati;
5. kuidas 256×256 paanid, target ja valid mask moodustasid mudeli treeningnäite;
6. miks komposiit-CHM valiti lõplikuks sisendiks;
7. kuidas automaatne top-2 ablation valis andmestiku, mudeli, loss'i ja augmentatsiooni;
8. kuidas lõplik väljund on tõenäosuskaart, millest saab läve abil binaarse lamapuidu maski.

Kõige olulisem akadeemiline mõte on see, et segmenteerimise metoodika usaldusväärsus sõltub vähem ühest mudeliarhitektuurist ja rohkem sellest, kas juhendmask eristab korrektselt kolme seisundit: lamapuit, usaldusväärne taust ja mittehinnatav ala. Ilma `valid_area.gpkg` kihita oleks suur osa märgendamata alast mudeli jaoks vale-negatiivne treeninginfo.
"""

    DOC_PATH.write_text(md, encoding="utf-8")

    summary = {
        "document": str(DOC_PATH),
        "figure_dir": str(FIG_DIR),
        "n_label_objects": int(len(label_gdf)),
        "label_area_m2": label_total_area,
        "valid_area_m2": area_polygon_area,
        "mask_meta": mask_meta,
        "stripe_pixel_stats": stripes.to_dict(orient="records"),
        "role_pixel_stats": roles.to_dict(orient="records"),
        "role_patch_stats": patches.to_dict(orient="records"),
    }
    (FIG_DIR / "seg_final_methodology_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for path in [DOC_PATH, *FIG_DIR.iterdir()]:
        if path.is_file():
            path.chmod(0o666)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

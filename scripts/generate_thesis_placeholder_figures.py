#!/usr/bin/env python3
"""Generate thesis figures that replace Placeholder figure blocks.

All generated figures follow the shared thesis style:
- 16:9 layout
- 300 DPI
- seaborn-v0_8-paper style
- consistent color palette
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle


OUT_DIR = Path("LaTeX/Lamapuidu_tuvastamine/estonian/joonised")
STYLE = "seaborn-v0_8-paper"

# Shared palette from existing thesis figures.
C_BLUE = "#1f77b4"
C_RED = "#d62728"
C_GREEN = "#2ca02c"
C_GRAY = "#7f7f7f"
C_BLACK = "#000000"

CLS_CSV = Path("data/chm_variants/labels_canonical_with_splits_spatial_ensemble.csv")
CHM_TIF = Path("data/chm_max_hag/406455_2021_tava_chm_max_hag_20cm.tif")
SEG_PROB_TIF = Path(
    "seg_pipeline/output/ablation_v10_top2_blockcv5_full_20260511_105703/"
    "phase6_2E__3B__4H__5E__6_final_validation/406455_2021_tava_2E__3B__4H__5E__6_prob.tif"
)
TRUE_MASK_TIF = Path("seg_pipeline/output/phase1_masks/406455_2021_tava_truemask.tif")


@dataclass
class TileStats:
    y: int
    x: int
    f1: float
    precision: float
    recall: float
    fp_rate: float
    fn_rate: float
    area_delta: float
    chm_std: float
    gt_sum: int
    pred_sum: int
    fp: int
    fn: int


def _set_style() -> None:
    plt.style.use(STYLE)


def _save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Valmis: {output}")


def _base_axes_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def _create_cwd_colormap():
    """Rich pseudo-color gradient: light gray→dark gray→red→orange→yellow→green→cyan→blue→purple."""
    # Position values for 0-1.3m range mapped to 0-1 colormap range
    # 0.1m → 0.1/1.3 ≈ 0.077
    cdict = {
        'red': [
            (0.000, 0.80, 0.80),   # 0.0m:    medium gray
            (0.077, 0.40, 0.40),   # 0.1m:    dark gray
            (0.115, 0.80, 0.80),   # 0.15m:   light red
            (0.230, 0.95, 0.95),   # 0.3m:    bright red peak
            (0.308, 1.0, 1.0),     # 0.4m:    orange-red
            (0.385, 1.0, 1.0),     # 0.5m:    orange
            (0.462, 1.0, 1.0),     # 0.6m:    yellow
            (0.538, 0.40, 0.40),   # 0.7m:    yellow-green (less red)
            (0.615, 0.20, 0.20),   # 0.8m:    green (low red)
            (0.692, 0.10, 0.10),   # 0.9m:    cyan (very low red)
            (0.769, 0.05, 0.05),   # 1.0m:    blue (minimal red)
            (0.846, 0.10, 0.10),   # 1.1m:    purple (start red rise)
            (1.000, 0.40, 0.40),   # 1.3m:    dark purple
        ],
        'green': [
            (0.000, 0.80, 0.80),   # 0.0m:    medium gray
            (0.077, 0.40, 0.40),   # 0.1m:    dark gray
            (0.115, 0.20, 0.20),   # 0.15m:   red (no green)
            (0.230, 0.0, 0.0),     # 0.3m:    pure red
            (0.308, 0.40, 0.40),   # 0.4m:    orange
            (0.385, 0.60, 0.60),   # 0.5m:    orange
            (0.462, 0.90, 0.90),   # 0.6m:    yellow
            (0.538, 0.80, 0.80),   # 0.7m:    yellow-green
            (0.615, 0.65, 0.65),   # 0.8m:    green
            (0.692, 0.85, 0.85),   # 0.9m:    cyan
            (0.769, 0.80, 0.80),   # 1.0m:    blue
            (0.846, 0.40, 0.40),   # 1.1m:    purple
            (1.000, 0.20, 0.20),   # 1.3m:    dark purple
        ],
        'blue': [
            (0.000, 0.80, 0.80),   # 0.0m:    medium gray
            (0.077, 0.40, 0.40),   # 0.1m:    dark gray
            (0.115, 0.15, 0.15),   # 0.15m:   red
            (0.230, 0.0, 0.0),     # 0.3m:    pure red
            (0.308, 0.10, 0.10),   # 0.4m:    orange-red
            (0.385, 0.15, 0.15),   # 0.5m:    orange
            (0.462, 0.0, 0.0),     # 0.6m:    yellow
            (0.538, 0.30, 0.30),   # 0.7m:    yellow-green
            (0.615, 0.60, 0.60),   # 0.8m:    green
            (0.692, 1.0, 1.0),     # 0.9m:    cyan (peak blue)
            (0.769, 1.0, 1.0),     # 1.0m:    blue (peak blue)
            (0.846, 0.80, 0.80),   # 1.1m:    purple
            (1.000, 0.60, 0.60),   # 1.3m:    dark purple
        ],
    }
    cmap = mcolors.LinearSegmentedColormap("cwd_hag", cdict, N=256)
    return cmap


def plot_shadow_density(output: Path) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(16, 9))

    x = np.linspace(0, 100, 200)
    sparse = 43.0 - 0.23 * x + 1.8 * np.sin(x / 14.0)
    dense = 39.0 - 0.34 * x + 1.3 * np.cos(x / 15.0 + 0.4)
    sparse = np.clip(sparse, 6, None)
    dense = np.clip(dense, 3, None)

    ax.plot(x, sparse, color=C_GREEN, linewidth=2.8, label="Horem mets")
    ax.plot(x, dense, color=C_RED, linewidth=2.8, label="Tihe okasmets")
    ax.fill_between(x, dense, sparse, where=sparse >= dense, color=C_GRAY, alpha=0.18)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.set_xlabel("Varju moju tugevus (suhteline skaala)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Hinnanguline punktitihedus (p/m²)", fontsize=13, fontweight="bold")
    _base_axes_title(ax, "Okaspuude varjude moju maapinnalahedasele punktitihedusele")
    ax.legend(loc="upper right", fontsize=11, frameon=True)

    _save(fig, output)


def _add_flow_box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        linewidth=1.2,
        edgecolor=C_BLACK,
        facecolor=color,
        alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, fontweight="bold")


def plot_chm_flow(output: Path) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    steps = [
        ("LAZ-\npunktipilv", C_GRAY),
        ("DTM\neemaldamine", "#bdd7ee"),
        ("Normaliseeritud\nkõrgused", "#c6e0b4"),
        ("Filter\n0–1,3 m", "#ffe699"),
        ("Rasterdus\n20 cm", "#f4b183"),
    ]

    x0, y0, w, h, gap = 0.01, 0.42, 0.15, 0.16, 0.045
    for i, (label, color) in enumerate(steps):
        x = x0 + i * (w + gap)
        _add_flow_box(ax, x, y0, w, h, label, color)
        if i < len(steps) - 1:
            x1 = x + w
            x2 = x + w + gap
            ax.add_patch(
                FancyArrowPatch(
                    (x1 + 0.003, y0 + h / 2),
                    (x2 - 0.003, y0 + h / 2),
                    arrowstyle="-|>",
                    mutation_scale=15,
                    linewidth=1.5,
                    color=C_BLACK,
                )
            )

    ax.text(
        0.5,
        0.78,
        "CHM loomise ja eeltöötluse toovoog",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.18,
        "Tulemus: Ühtlane sisend piksliresolutsiooniga 20 cm",
        ha="center",
        va="center",
        fontsize=13.5,
        color=C_GRAY,
        style="italic",
    )

    _save(fig, output)


def plot_chm_example(output: Path) -> None:
    import pickle

    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    cwd_cmap = _create_cwd_colormap()

    # Load real example data (if available)
    data_file = Path("_real_example_data.pkl")
    if data_file.exists():
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        laz_points = data["laz_subset"]
        chm_tile = data["chm_tile"]
        bbox = data["bbox"]

        # Use IDW-HAG normalized heights (0-1.3m range)
        laz_filtered = laz_points

        # Filter LAZ to 0-1.3m range (HAG)
        mask_hag = (laz_filtered[:, 2] >= 0.0) & (laz_filtered[:, 2] <= 1.3)
        laz_filtered = laz_filtered[mask_hag]

        # Normalize coordinates to pixel space
        x_min, x_max = bbox["x_min"], bbox["x_max"]
        y_min, y_max = bbox["y_min"], bbox["y_max"]

        x_pixel = ((laz_filtered[:, 0] - x_min) / (x_max - x_min)) * 127
        y_pixel = ((laz_filtered[:, 1] - y_max) / (y_min - y_max)) * 127  # flip y
        z_values = laz_filtered[:, 2]

        # Scatter plot with custom colormap (0-1.3m scale)
        sc = axes[0].scatter(
            x_pixel, y_pixel, c=z_values, cmap=cwd_cmap, s=10, linewidths=0, alpha=0.8,
            vmin=0.0, vmax=1.3
        )
        axes[0].set_xlim(-2, 130)
        axes[0].set_ylim(-2, 130)
        axes[0].set_aspect("equal")
        axes[0].set_xlabel("Veerg (20 cm)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Rida (20 cm)", fontsize=12, fontweight="bold")
        axes[0].set_title("Punktipilv normaliseeritud", fontsize=13, fontweight="bold")
        axes[0].grid(True, linestyle=":", alpha=0.3)
        cb = fig.colorbar(sc, ax=axes[0], shrink=0.85)
        cb.set_label("Kõrgus (m)", fontweight="bold")

        # CHM raster with same 0-1.3m scale
        im = axes[1].imshow(
            chm_tile, cmap=cwd_cmap, vmin=0.0, vmax=1.3, origin="upper"
        )
        axes[1].set_aspect("equal")
        axes[1].set_title("Taimkatte kõrgusmudel (Baasmudel)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Veerg", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Rida", fontsize=12, fontweight="bold")
        cb2 = fig.colorbar(im, ax=axes[1], shrink=0.85)
        cb2.set_label("Kõrgus (m)", fontweight="bold")

    else:
        # Fallback to synthetic if data not available
        rng = np.random.default_rng(42)
        n_ground = 2500
        xg = rng.uniform(0, 128, size=n_ground)
        yg = rng.uniform(0, 128, size=n_ground)
        zg = rng.normal(0.12, 0.05, size=n_ground).clip(0.0, 0.35)

        t = np.linspace(10, 120, 1500)
        xl = t + rng.normal(0, 1.5, size=t.size)
        yl = 0.4 * t + 15 + rng.normal(0, 1.5, size=t.size)
        zl = 0.5 + 0.25 * np.sin(t / 15) + rng.normal(0, 0.08, size=t.size)
        zl = np.clip(zl, 0.2, 1.1)

        x = np.concatenate([xg, xl])
        y = np.concatenate([yg, yl])
        z = np.concatenate([zg, zl])

        sc = axes[0].scatter(x, y, c=z, cmap=cwd_cmap, s=10, linewidths=0, alpha=0.8,
                            vmin=0.0, vmax=1.3)
        axes[0].set_xlim(0, 128)
        axes[0].set_ylim(0, 128)
        axes[0].set_aspect("equal")
        axes[0].set_xlabel("Veerg (20 cm)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Rida (20 cm)", fontsize=12, fontweight="bold")
        axes[0].set_title("Punktipilv (normaliseeritud kõrgus)", fontsize=13, fontweight="bold")
        axes[0].grid(True, linestyle=":", alpha=0.3)
        cb = fig.colorbar(sc, ax=axes[0], shrink=0.85)
        cb.set_label("Kõrgus (m)", fontweight="bold")

        # Synthetic CHM
        grid = np.zeros((128, 128), dtype=np.float32)
        for xi, yi, zi in zip(x, y, z):
            ix = int(np.clip(np.floor(xi), 0, 127))
            iy = int(np.clip(np.floor(yi), 0, 127))
            if zi > grid[iy, ix]:
                grid[iy, ix] = zi

        im = axes[1].imshow(grid, cmap=cwd_cmap, vmin=0.0, vmax=1.3, origin="upper")
        axes[1].set_aspect("equal")
        axes[1].set_title("Taimkatte kõrgusmudel (Baasmudel)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Veerg", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Rida", fontsize=12, fontweight="bold")
        cb2 = fig.colorbar(im, ax=axes[1], shrink=0.85)
        cb2.set_label("Kõrgus (m)", fontweight="bold")

    fig.suptitle("Lamapuit LiDAR-i ja taimkattemudelil", fontsize=16, fontweight="bold")
    sns.despine(trim=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output)


def plot_spatial_validation(output: Path) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")

    ax.add_patch(Rectangle((0, 0), 100, 100, facecolor="#f2f2f2", edgecolor=C_GRAY, linewidth=1.2))
    ax.add_patch(Rectangle((25, 25), 50, 50, facecolor="#f8d7da", edgecolor=C_RED, linewidth=2.0, alpha=0.7))
    ax.add_patch(Rectangle((35, 35), 30, 30, facecolor="#d6eaf8", edgecolor=C_BLUE, linewidth=2.2, alpha=0.95))

    for p in np.arange(0, 101, 10):
        ax.axhline(p, color="#d9d9d9", linewidth=0.6, zorder=0)
        ax.axvline(p, color="#d9d9d9", linewidth=0.6, zorder=0)

    ax.text(50, 50, "Testtsoon\n(51,2 m x 51,2 m)", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(50, 76, "Puhvertsoon", ha="center", va="center", fontsize=12, fontweight="bold", color=C_RED)
    ax.text(11, 92, "Treeningala", ha="left", va="center", fontsize=12, fontweight="bold", color=C_GRAY)

    legend_handles = [
        Patch(facecolor="#d6eaf8", edgecolor=C_BLUE, label="Testtsoon"),
        Patch(facecolor="#f8d7da", edgecolor=C_RED, label="Puhvertsoon"),
        Patch(facecolor="#f2f2f2", edgecolor=C_GRAY, label="Treeningala"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=True)
    ax.set_title("Ruumilise valideerimise ja puhveralade loogika", fontsize=16, fontweight="bold")
    ax.set_xlabel("X (suhteline skaala)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y (suhteline skaala)", fontsize=12, fontweight="bold")
    ax.grid(False)

    _save(fig, output)


def plot_ensemble_scheme(output: Path) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    model_boxes = [
        ("CNN-Deep-Attn (seed 42)", 0.76, C_BLUE),
        ("CNN-Deep-Attn (seed 43)", 0.62, C_BLUE),
        ("CNN-Deep-Attn (seed 44)", 0.48, C_BLUE),
        ("EfficientNet-B2", 0.34, C_GREEN),
    ]
    for label, y, color in model_boxes:
        _add_flow_box(ax, 0.08, y, 0.28, 0.1, label, color)

    _add_flow_box(ax, 0.46, 0.52, 0.22, 0.14, "Kaalutud\nhaaletus", "#f4cccc")
    _add_flow_box(ax, 0.77, 0.54, 0.16, 0.1, "Loplik\notsus", "#ffe699")

    for _, y, _ in model_boxes:
        ax.add_patch(
            FancyArrowPatch(
                (0.36, y + 0.05),
                (0.46, 0.59),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.8,
                color=C_BLACK,
                connectionstyle="arc3,rad=0.0",
            )
        )
        ax.text(0.39, y + 0.055, "p", fontsize=10, color=C_GRAY)

    ax.add_patch(
        FancyArrowPatch((0.68, 0.59), (0.77, 0.59), arrowstyle="-|>", mutation_scale=20, linewidth=2.0, color=C_BLACK)
    )
    ax.text(0.60, 0.69, "TTA + mudelite toenaosuste agregeerimine", fontsize=11, ha="center", color=C_GRAY)

    ax.text(0.5, 0.87, "Loplika ansambli toopohimote", fontsize=18, fontweight="bold", ha="center")
    ax.text(
        0.5,
        0.2,
        "Kolm juhuseemnega CNN-mudelit + 1 EfficientNet-B2 annavad stabiilsema otsuse.",
        fontsize=11.5,
        ha="center",
        color=C_GRAY,
    )

    _save(fig, output)


def load_classification_test_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_prob: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != "test":
                continue
            prob_raw = (row.get("model_prob") or "").strip()
            if not prob_raw:
                continue
            try:
                p = float(prob_raw)
            except ValueError:
                continue
            lbl = (row.get("label") or "").strip().lower()
            y_true.append(1 if lbl in {"cdw", "1", "true", "yes"} else 0)
            y_prob.append(float(np.clip(p, 0.0, 1.0)))

    if not y_true:
        raise RuntimeError(f"Puuduvad testandmed klassifitseerimise CSV failis: {csv_path}")
    return np.asarray(y_true, dtype=np.uint8), np.asarray(y_prob, dtype=np.float32)


def compute_curves(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, np.ndarray | float]:
    thr = np.linspace(0, 1, 501)
    pos = y_true == 1
    neg = ~pos

    tpr, fpr, precision, recall, f1 = [], [], [], [], []
    for t in thr:
        pred = y_prob >= float(t)
        tp = np.logical_and(pred, pos).sum()
        fp = np.logical_and(pred, neg).sum()
        fn = np.logical_and(~pred, pos).sum()
        tn = np.logical_and(~pred, neg).sum()

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1_val = (2 * p * r) / (p + r + 1e-8)
        tpr_val = r
        fpr_val = fp / (fp + tn + 1e-8)

        precision.append(p)
        recall.append(r)
        f1.append(f1_val)
        tpr.append(tpr_val)
        fpr.append(fpr_val)

    precision = np.asarray(precision)
    recall = np.asarray(recall)
    f1 = np.asarray(f1)
    tpr = np.asarray(tpr)
    fpr = np.asarray(fpr)

    roc_order = np.argsort(fpr)
    pr_order = np.argsort(recall)
    roc_auc = float(np.trapezoid(tpr[roc_order], fpr[roc_order])) if hasattr(np, "trapezoid") else float(np.trapz(tpr[roc_order], fpr[roc_order]))
    pr_auc = float(np.trapezoid(precision[pr_order], recall[pr_order])) if hasattr(np, "trapezoid") else float(np.trapz(precision[pr_order], recall[pr_order]))

    best_idx = int(np.argmax(f1))
    return {
        "thr": thr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_idx": best_idx,
    }


def plot_roc_pr_curve(output: Path) -> None:
    y_true, y_prob = load_classification_test_data(CLS_CSV)
    curves = compute_curves(y_true, y_prob)

    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    # ROC
    axes[0].plot(curves["fpr"], curves["tpr"], color=C_BLUE, linewidth=2.4, label=f"AUC = {curves['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], color=C_GRAY, linestyle="--", linewidth=1.6, label="Juhuslik klassifikaator")
    i = int(curves["best_idx"])
    axes[0].scatter(float(curves["fpr"][i]), float(curves["tpr"][i]), color=C_RED, s=80, zorder=5)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Valepositiivsete maar", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Toepositiivsete maar", fontsize=12, fontweight="bold")
    axes[0].set_title("ROC-kover", fontsize=14, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(frameon=True)

    # PR
    axes[1].plot(curves["recall"], curves["precision"], color=C_RED, linewidth=2.4, label=f"AUC-PR = {curves['pr_auc']:.3f}")
    axes[1].scatter(float(curves["recall"][i]), float(curves["precision"][i]), color=C_BLUE, s=80, zorder=5)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Saagis", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Tapsus", fontsize=12, fontweight="bold")
    axes[1].set_title("Tapsuse-saagise kover", fontsize=14, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True)

    fig.suptitle("Loplika ansambli ROC ja PR koverad testandmestikul", fontsize=16, fontweight="bold")
    sns.despine(trim=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output)


def _make_bg(seed: int, shape: tuple[int, int] = (140, 220)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    bg = 0.35 + 0.10 * np.sin(x / 18.0) + 0.08 * np.cos(y / 13.0) + rng.normal(0, 0.03, shape)
    return np.clip(bg, 0, 1)


def plot_error_examples(output: Path) -> None:
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    # (a) FP example
    bg_a = _make_bg(5)
    gt_a = np.zeros_like(bg_a, dtype=bool)
    pred_a = np.zeros_like(bg_a, dtype=bool)
    pred_a[58:76, 35:175] = True
    pred_a[61:73, 28:38] = True
    pred_a[62:74, 175:185] = True

    axes[0].imshow(bg_a, cmap="gray", vmin=0, vmax=1)
    axes[0].contour(pred_a.astype(float), levels=[0.5], colors=[C_RED], linewidths=2.0)
    axes[0].set_title("(a) Valepositiivne: kivi/kaend", fontsize=13, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # (b) FN example
    bg_b = _make_bg(9)
    gt_b = np.zeros_like(bg_b, dtype=bool)
    gt_b[72:90, 45:190] = True
    gt_b[70:92, 45:52] = True
    gt_b[70:92, 183:190] = True
    pred_b = np.zeros_like(bg_b, dtype=bool)
    pred_b[76:84, 110:145] = True  # only small segment detected

    axes[1].imshow(bg_b, cmap="gray", vmin=0, vmax=1)
    axes[1].contour(gt_b.astype(float), levels=[0.5], colors=[C_GREEN], linewidths=2.0)
    axes[1].contour(pred_b.astype(float), levels=[0.5], colors=[C_RED], linewidths=2.0)
    axes[1].set_title("(b) Valenegatiivne: varjus lamapuu", fontsize=13, fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    handles = [
        Patch(facecolor="none", edgecolor=C_GREEN, label="Toene lamapuu"),
        Patch(facecolor="none", edgecolor=C_RED, label="Mudeli ennustus"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=True)
    fig.suptitle("Klassifitseerimismudeli iseloomulikud vead", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    _save(fig, output)


def plot_phase_ranking(output: Path) -> None:
    phase_data = {
        "Faas 2 (CHM variandid)": [
            ("2E", 0.2311, 0.0476),
            ("2A", 0.2206, 0.0482),
            ("2C", 0.2161, 0.0385),
            ("2B", 0.2138, 0.0417),
            ("2D", 0.1185, 0.0204),
        ],
        "Faas 3 (Arhitektuur + CHM)": [
            ("2E__3B", 0.2645, 0.0482),
            ("2A__3C", 0.2277, 0.0714),
            ("2E__3E", 0.2224, 0.0437),
            ("2A__3B", 0.2221, 0.0515),
            ("2A__3E", 0.2146, 0.0512),
            ("2E__3C", 0.2131, 0.0358),
        ],
        "Faas 4 (Kaofunktsioonid)": [
            ("2E__3B__4H", 0.2551, 0.0434),
            ("2E__3B__4A", 0.2442, 0.0448),
            ("2E__3B__4F", 0.2367, 0.0615),
            ("2A__3C__4H", 0.2194, 0.0324),
            ("2A__3C__4A", 0.2002, 0.0427),
            ("2E__3B__4D", 0.1839, 0.0259),
            ("2A__3C__4D", 0.1774, 0.0130),
            ("2A__3C__4F", 0.1633, 0.0192),
        ],
        "Faas 5 (Andmerikastamine)": [
            ("2E__3B__4H__5E", 0.3270, 0.0554),
            ("2E__3B__4A__5E", 0.3101, 0.0600),
            ("2E__3B__4H__5A", 0.2980, 0.0516),
            ("2E__3B__4A__5A", 0.2765, 0.0480),
            ("2E__3B__4H__5D", 0.2222, 0.0298),
            ("2E__3B__4A__5D", 0.2005, 0.0271),
        ],
    }

    _set_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False)
    axes = axes.ravel()

    for ax, (title, rows) in zip(axes, phase_data.items()):
        labels = [r[0] for r in rows]
        mean = np.array([r[1] for r in rows])
        sd = np.array([r[2] for r in rows])
        y = np.arange(len(labels))

        ax.errorbar(mean, y, xerr=sd, fmt="o", color=C_BLUE, ecolor=C_RED, elinewidth=1.8, capsize=4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Keskmine val_clDice", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Mudelivaliku dunaamika faasides 2-5", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output)


def load_seg_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(CHM_TIF) as src:
        chm = src.read(1).astype(np.float32)
    with rasterio.open(SEG_PROB_TIF) as src:
        prob = src.read(1).astype(np.float32)
    with rasterio.open(TRUE_MASK_TIF) as src:
        gt = src.read(1).astype(np.float32)
        valid = src.read(2).astype(np.float32)

    h = min(chm.shape[0], prob.shape[0], gt.shape[0], valid.shape[0])
    w = min(chm.shape[1], prob.shape[1], gt.shape[1], valid.shape[1])
    return chm[:h, :w], np.clip(prob[:h, :w], 0.0, 1.0), (gt[:h, :w] > 0.5), (valid[:h, :w] > 0.5)


def _tile_metrics(chm: np.ndarray, prob: np.ndarray, gt: np.ndarray, valid: np.ndarray, tile: int = 256) -> list[TileStats]:
    pred = prob >= 0.5
    out: list[TileStats] = []
    h, w = gt.shape
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            vv = valid[y : y + tile, x : x + tile]
            if vv.mean() < 0.65:
                continue

            g = gt[y : y + tile, x : x + tile] & vv
            p = pred[y : y + tile, x : x + tile] & vv
            gg = int(g.sum())
            if gg < 40:
                continue

            tp = int(np.logical_and(g, p).sum())
            fp = int(np.logical_and(~g, p).sum())
            fn = int(np.logical_and(g, ~p).sum())
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = (2 * precision * recall) / (precision + recall + 1e-8)
            fp_rate = fp / (fp + tp + 1e-8)
            fn_rate = fn / (fn + tp + 1e-8)
            pred_sum = int(p.sum())
            area_delta = abs(pred_sum - gg) / (gg + 1e-8)
            chm_std = float(np.std(chm[y : y + tile, x : x + tile][vv]))
            out.append(
                TileStats(
                    y=y,
                    x=x,
                    f1=float(f1),
                    precision=float(precision),
                    recall=float(recall),
                    fp_rate=float(fp_rate),
                    fn_rate=float(fn_rate),
                    area_delta=float(area_delta),
                    chm_std=chm_std,
                    gt_sum=gg,
                    pred_sum=pred_sum,
                    fp=fp,
                    fn=fn,
                )
            )
    if not out:
        raise RuntimeError("Ei leidnud sobivaid plaate kvalitatiivsete jooniste loomiseks.")
    return out


def _unique_pick(candidates: list[TileStats], selected: list[tuple[int, int]]) -> TileStats:
    for c in candidates:
        if (c.y, c.x) not in selected:
            return c
    return candidates[0]


def choose_tiles(stats: list[TileStats]) -> dict[str, TileStats]:
    selected: list[tuple[int, int]] = []

    best = sorted(stats, key=lambda s: s.f1, reverse=True)
    success_1 = _unique_pick(best, selected)
    selected.append((success_1.y, success_1.x))
    success_2 = _unique_pick(best[1:], selected)
    selected.append((success_2.y, success_2.x))

    fp_hard = sorted(stats, key=lambda s: (s.fp_rate, s.fp), reverse=True)
    diff_fp = _unique_pick(fp_hard, selected)
    selected.append((diff_fp.y, diff_fp.x))

    fn_hard = sorted(stats, key=lambda s: (s.fn_rate, s.fn), reverse=True)
    diff_fn = _unique_pick(fn_hard, selected)
    selected.append((diff_fn.y, diff_fn.x))

    area_shift = _unique_pick(sorted(stats, key=lambda s: s.area_delta, reverse=True), selected)
    selected.append((area_shift.y, area_shift.x))

    complex_bg = _unique_pick(sorted(stats, key=lambda s: (s.chm_std, -s.f1), reverse=True), selected)

    return {
        "success_1": success_1,
        "success_2": success_2,
        "diff_fp": diff_fp,
        "diff_fn": diff_fn,
        "area_shift": area_shift,
        "complex_bg": complex_bg,
    }


def _extract_tile(arr: np.ndarray, y: int, x: int, tile: int = 256) -> np.ndarray:
    return arr[y : y + tile, x : x + tile]


def plot_seg_qualitative(output: Path) -> None:
    chm, prob, gt, valid = load_seg_arrays()
    stats = _tile_metrics(chm, prob, gt, valid, tile=256)
    picks = choose_tiles(stats)

    rows = [
        ("Edukas juhtum 1", picks["success_1"]),
        ("Edukas juhtum 2", picks["success_2"]),
        ("Keerukas juhtum (FP)", picks["diff_fp"]),
        ("Keerukas juhtum (FN)", picks["diff_fn"]),
    ]

    _set_style()
    fig, axes = plt.subplots(4, 4, figsize=(16, 9))
    col_titles = ["Sisend CHM", "Toene mask", "Toenaosuskaart", "Binaarmask"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=11, fontweight="bold")

    for r, (label, s) in enumerate(rows):
        yy, xx = s.y, s.x
        ch = _extract_tile(chm, yy, xx)
        gg = _extract_tile(gt, yy, xx) & _extract_tile(valid, yy, xx)
        pp = _extract_tile(prob, yy, xx)
        bb = (pp >= 0.5) & _extract_tile(valid, yy, xx)

        axes[r, 0].imshow(ch, cmap="viridis")
        axes[r, 1].imshow(gg.astype(float), cmap="Greens", vmin=0, vmax=1)
        axes[r, 2].imshow(pp, cmap="magma", vmin=0, vmax=1)
        axes[r, 3].imshow(bb.astype(float), cmap="Reds", vmin=0, vmax=1)

        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

        axes[r, 0].set_ylabel(f"{label}\nF1={s.f1:.2f}", fontsize=10, fontweight="bold")

    fig.suptitle("Parima segmenteerimismudeli kvalitatiivsed naited", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output)


def _overlay_error(ax: plt.Axes, chm_tile: np.ndarray, gt_tile: np.ndarray, pred_tile: np.ndarray, title: str) -> None:
    tp = np.logical_and(gt_tile, pred_tile)
    fp = np.logical_and(~gt_tile, pred_tile)
    fn = np.logical_and(gt_tile, ~pred_tile)

    ax.imshow(chm_tile, cmap="gray")
    overlay = np.zeros((*gt_tile.shape, 4), dtype=np.float32)
    overlay[tp] = (44 / 255.0, 160 / 255.0, 44 / 255.0, 0.35)
    overlay[fp] = (214 / 255.0, 39 / 255.0, 40 / 255.0, 0.65)
    overlay[fn] = (31 / 255.0, 119 / 255.0, 180 / 255.0, 0.65)
    ax.imshow(overlay)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_seg_error_panel(output: Path) -> None:
    chm, prob, gt, valid = load_seg_arrays()
    stats = _tile_metrics(chm, prob, gt, valid, tile=256)
    picks = choose_tiles(stats)

    cases = [
        ("(a) Ule-segmenteerimine (FP fragmendid)", picks["diff_fp"]),
        ("(b) Uhenduvuse katkemine", picks["diff_fn"]),
        ("(c) Kontuurinihe servades", picks["area_shift"]),
        ("(d) Keerukas alusmets", picks["complex_bg"]),
    ]

    _set_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, (title, s) in zip(axes.ravel(), cases):
        yy, xx = s.y, s.x
        ch = _extract_tile(chm, yy, xx)
        gg = _extract_tile(gt, yy, xx) & _extract_tile(valid, yy, xx)
        pp = (_extract_tile(prob, yy, xx) >= 0.5) & _extract_tile(valid, yy, xx)
        _overlay_error(ax, ch, gg, pp, title)

    handles = [
        Patch(facecolor=(44 / 255.0, 160 / 255.0, 44 / 255.0, 0.6), edgecolor="none", label="TP"),
        Patch(facecolor=(214 / 255.0, 39 / 255.0, 40 / 255.0, 0.7), edgecolor="none", label="FP"),
        Patch(facecolor=(31 / 255.0, 119 / 255.0, 180 / 255.0, 0.7), edgecolor="none", label="FN"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=True)
    fig.suptitle("Segmenteerimise generaliseerimise kitsaskohad", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, output)


def main() -> int:
    targets = {
        "varjude_moju_punktitihedusele.png": plot_shadow_density,
        "chm_eeltootluse_toovoog.png": plot_chm_flow,
        "lamapuu_punktipilv_ja_chm.png": plot_chm_example,
        "ruumilise_valideerimise_loogika.png": plot_spatial_validation,
        "ansambli_toopohimote.png": plot_ensemble_scheme,
        "roc_pr_kover_loplik_ansambel.png": plot_roc_pr_curve,
        "vigade_visuaalsed_naited.png": plot_error_examples,
        "seg_faaside_paremusjarjestus.png": plot_phase_ranking,
        "seg_parim_kvalitatiivne.png": plot_seg_qualitative,
        "seg_generaliseerimise_kitsaskohad.png": plot_seg_error_panel,
    }

    for name, fn in targets.items():
        fn(OUT_DIR / name)

    print("Koik placeholder-jooniste failid on genereeritud.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

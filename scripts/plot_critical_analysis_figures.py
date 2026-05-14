#!/usr/bin/env python3
"""Create high-value analytical figures for thesis improvement.

Figures:
1) Segmentation threshold sensitivity on held-out CVv5 test entries.
2) Spatial error map on held-out CVv5 test entries (threshold=0.50).
3) Classification performance by label source (test split).
4) Segmentation ablation waterfall (Phase 2->5, val_clDice only).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
SEG_SCRIPTS = ROOT / "seg_pipeline" / "scripts"
if str(SEG_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SEG_SCRIPTS))

from common.extended_metrics import boundary_iou, cldice_metric  # type: ignore
from phase2_dataset_v3 import CWDSegDataset, load_patch_index  # type: ignore
from phase3_train_v10 import build_model  # type: ignore


PALETTE = {
    "blue": "#1f77b4",
    "red": "#d62728",
    "green": "#2ca02c",
    "gray": "#7f7f7f",
    "black": "#000000",
}


@dataclass
class PatchPrediction:
    row_off: int
    col_off: int
    prob: np.ndarray
    target: np.ndarray
    valid: np.ndarray


def _style() -> None:
    plt.style.use("seaborn-v0_8-paper")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Valmis: {path}")


def _dataset_dir_candidates(base_dir: Path) -> list[Path]:
    candidates = [base_dir]
    repo_out = ROOT / "seg_pipeline" / "output"
    for extra in [
        "phase2_dataset_v10_blockcv5_full_20260511_105703",
        "phase2_dataset_v10_blockcv5_probe",
        "phase2_dataset_v10_blockcv5_smoke",
        "phase2_dataset_v10_reconstructed",
        "phase2_dataset_v3",
    ]:
        p = repo_out / extra
        if p not in candidates:
            candidates.append(p)
    return candidates


def _resolve_dataset_files(base_dir: Path, variant: str) -> tuple[Path, Path]:
    for d in _dataset_dir_candidates(base_dir):
        pidx = d / f"patch_index_{variant}.csv"
        bstats = d / f"band_stats_{variant}.json"
        if pidx.exists() and bstats.exists():
            return pidx, bstats
    tried = ", ".join(str(d) for d in _dataset_dir_candidates(base_dir))
    raise FileNotFoundError(
        f"Ei leidnud patch_index/band_stats faile variandile '{variant}'. Otsiti: {tried}"
    )


def _resolve_existing_chm_tif(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(
        f"CHM fail puudub: {path}. Anna korrektne --seg-chm-tif."
    )


def load_segmentation_test_predictions(
    phase6_dir: Path,
    dataset_dir: Path,
    chm_tif: Path,
    mask_tif: Path,
    device: str = "cpu",
    batch_size: int = 16,
) -> tuple[list[PatchPrediction], dict[str, Any]]:
    meta_path = phase6_dir / "all_train_metrics_test.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Puudub fail: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    variant = str(meta["chm_variant"])
    arch = str(meta["arch"])
    in_channels = int(meta["in_channels"])

    patch_index_path, band_stats_path = _resolve_dataset_files(dataset_dir, variant)
    patch_index = load_patch_index(patch_index_path)
    test_entries = [e for e in patch_index if int(getattr(e, "fold_id", -99)) == -1]
    if not test_entries:
        raise RuntimeError("CVv5 test-entry'd (fold_id=-1) puuduvad patch-index failis.")

    with open(band_stats_path, "r", encoding="utf-8") as f:
        band_stats = json.load(f)

    ckpt_path = phase6_dir / variant / "fold0" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Puudub checkpoint: {ckpt_path}")

    torch_device = torch.device(device)
    model = build_model(arch, in_channels=in_channels, pretrained=False).to(torch_device)
    ckpt = torch.load(ckpt_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = CWDSegDataset(
        entries=test_entries,
        chm_tif=_resolve_existing_chm_tif(chm_tif),
        mask_tif=mask_tif,
        band_stats=band_stats,
        patch_size=256,
        in_channels=in_channels,
        augment=False,
        variant=variant,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    out: list[PatchPrediction] = []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(torch_device, non_blocking=True)
            logits = model(image)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
            tgts = batch["target"][:, 0].numpy()
            vals = (batch["valid"][:, 0].numpy() > 0.5)

            n = probs.shape[0]
            chunk_entries = test_entries[cursor : cursor + n]
            cursor += n
            for e, p, t, v in zip(chunk_entries, probs, tgts, vals):
                out.append(
                    PatchPrediction(
                        row_off=int(e.row_off),
                        col_off=int(e.col_off),
                        prob=np.clip(p, 0.0, 1.0).astype(np.float32),
                        target=(t > 0.5).astype(np.uint8),
                        valid=v.astype(bool),
                    )
                )

    return out, meta


def _pixel_confusion(pred: np.ndarray, tgt: np.ndarray, valid: np.ndarray) -> tuple[int, int, int, int]:
    tv = np.logical_and(tgt == 1, valid)
    fv = np.logical_and(tgt == 0, valid)
    tp = int(np.logical_and(pred == 1, tv).sum())
    fp = int(np.logical_and(pred == 1, fv).sum())
    fn = int(np.logical_and(pred == 0, tv).sum())
    tn = int(np.logical_and(pred == 0, fv).sum())
    return tp, fp, fn, tn


def segmentation_threshold_sweep(
    patches: list[PatchPrediction],
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    precision_vals: list[float] = []
    recall_vals: list[float] = []
    f1_vals: list[float] = []
    cldice_vals: list[float] = []
    boundary_vals: list[float] = []

    for thr in thresholds:
        tp = fp = fn = tn = 0
        cl_local: list[float] = []
        bnd_local: list[float] = []

        for p in patches:
            pred_bin = (p.prob >= float(thr))
            tp_i, fp_i, fn_i, tn_i = _pixel_confusion(pred_bin, p.target, p.valid)
            tp += tp_i
            fp += fp_i
            fn += fn_i
            tn += tn_i

            # Match official per-patch extended metric behavior.
            pred_v = np.logical_and(pred_bin, p.valid).astype(np.uint8)
            gt_v = np.logical_and(p.target == 1, p.valid).astype(np.uint8)
            cl_local.append(float(cldice_metric(pred_v, gt_v)))
            bnd_local.append(float(boundary_iou(pred_v, gt_v)))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
        precision_vals.append(float(precision))
        recall_vals.append(float(recall))
        f1_vals.append(float(f1))
        cldice_vals.append(float(np.mean(cl_local)))
        boundary_vals.append(float(np.mean(bnd_local)))

    return {
        "thresholds": thresholds,
        "precision": np.asarray(precision_vals),
        "recall": np.asarray(recall_vals),
        "f1": np.asarray(f1_vals),
        "cldice": np.asarray(cldice_vals),
        "boundary_iou": np.asarray(boundary_vals),
    }


def plot_threshold_sensitivity(
    sweep: dict[str, np.ndarray],
    output_path: Path,
    official_threshold: float,
    official_f1: float,
) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    thr = sweep["thresholds"]

    best_idx = int(np.argmax(sweep["f1"]))
    best_thr = float(thr[best_idx])
    best_f1 = float(sweep["f1"][best_idx])

    ax = axes[0]
    ax.plot(thr, sweep["precision"], color=PALETTE["blue"], linewidth=2.2, label="Täpsus")
    ax.plot(thr, sweep["recall"], color=PALETTE["red"], linewidth=2.2, label="Saagis")
    ax.plot(thr, sweep["f1"], color=PALETTE["green"], linewidth=2.4, label="F1")
    ax.axvline(official_threshold, color=PALETTE["gray"], linestyle="--", linewidth=1.8, label=f"Ametlik lävi t={official_threshold:.2f}")
    ax.axvline(best_thr, color=PALETTE["black"], linestyle=":", linewidth=2.0, label=f"F1 maksimum t={best_thr:.2f}")
    ax.set_xlim(float(thr.min()), float(thr.max()))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Lävi t", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mõõdiku väärtus", fontsize=12, fontweight="bold")
    ax.set_title("Pikslitaseme mõõdikud", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower center", fontsize=10, frameon=True)

    ax2 = axes[1]
    ax2.plot(thr, sweep["cldice"], color=PALETTE["red"], linewidth=2.4, label="clDice")
    ax2.plot(thr, sweep["boundary_iou"], color=PALETTE["blue"], linewidth=2.2, label="Boundary IoU")
    ax2.axvline(official_threshold, color=PALETTE["gray"], linestyle="--", linewidth=1.8)
    ax2.set_xlim(float(thr.min()), float(thr.max()))
    ax2.set_ylim(0, max(0.65, float(np.max(sweep["cldice"])) + 0.05))
    ax2.set_xlabel("Lävi t", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mõõdiku väärtus", fontsize=12, fontweight="bold")
    ax2.set_title("Struktuuri ja piiri mõõdikud", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=10, frameon=True)

    fig.suptitle(
        "Segmenteerimismudeli läve tundlikkus (held-out test-entry valim)",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        f"Ametlik test-F1 (t={official_threshold:.2f}) = {official_f1:.4f}; "
        f"diagnostiline F1 maksimum = {best_f1:.4f} (t={best_thr:.2f})",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["gray"],
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    _save(fig, output_path)


def patch_metrics_at_threshold(patches: list[PatchPrediction], threshold: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for p in patches:
        pred = p.prob >= threshold
        tp, fp, fn, tn = _pixel_confusion(pred, p.target, p.valid)
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
        valid_n = int(p.valid.sum())
        rows.append(
            {
                "row_off": float(p.row_off),
                "col_off": float(p.col_off),
                "f1": float(f1),
                "fp_density": float(fp / (valid_n + 1e-12)),
                "fn_density": float(fn / (valid_n + 1e-12)),
                "valid_n": float(valid_n),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
                "tn": float(tn),
            }
        )
    return rows


def _plot_patch_rectangles(
    ax: plt.Axes,
    rows: list[dict[str, float]],
    key: str,
    title: str,
    cbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    x_max: float,
    y_max: float,
) -> None:
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    for r in rows:
        y = float(r["row_off"])
        x = float(r["col_off"])
        val = float(r[key])
        rect = Rectangle(
            (x, y),
            256,
            256,
            facecolor=mapper.to_rgba(val),
            edgecolor="white",
            linewidth=0.6,
        )
        ax.add_patch(rect)

    ax.set_xlim(0, x_max)
    ax.set_ylim(y_max, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Veerg (px)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Rida (px)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(False)
    cbar = plt.colorbar(mapper, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10, fontweight="bold")


def plot_spatial_error_map(rows: list[dict[str, float]], output_path: Path, threshold: float) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    x_max = max(float(r["col_off"]) for r in rows) + 256.0
    y_max = max(float(r["row_off"]) for r in rows) + 256.0

    _plot_patch_rectangles(
        axes[0],
        rows,
        key="f1",
        title="Ploki F1",
        cbar_label="F1",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        x_max=x_max,
        y_max=y_max,
    )
    _plot_patch_rectangles(
        axes[1],
        rows,
        key="fp_density",
        title="FP tihedus (FP / valid px)",
        cbar_label="Valepositiivsete tihedus",
        cmap="Reds",
        vmin=0.0,
        vmax=max(1e-4, float(max(r["fp_density"] for r in rows))),
        x_max=x_max,
        y_max=y_max,
    )
    _plot_patch_rectangles(
        axes[2],
        rows,
        key="fn_density",
        title="FN tihedus (FN / valid px)",
        cbar_label="Valenegatiivsete tihedus",
        cmap="Blues",
        vmin=0.0,
        vmax=max(1e-4, float(max(r["fn_density"] for r in rows))),
        x_max=x_max,
        y_max=y_max,
    )

    fig.suptitle(
        f"Segmenteerimise ruumiline veakaart (held-out test-entry'd, lävi t={threshold:.2f})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output_path)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    has_pos = bool((y_true == 1).any())
    has_neg = bool((y_true == 0).any())
    y_pred = y_prob >= threshold
    tp = int(np.logical_and(y_pred, y_true == 1).sum())
    fp = int(np.logical_and(y_pred, y_true == 0).sum())
    fn = int(np.logical_and(~y_pred, y_true == 1).sum())
    tn = int(np.logical_and(~y_pred, y_true == 0).sum())
    if has_pos and has_neg:
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
    else:
        precision = math.nan
        recall = math.nan
        f1 = math.nan
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    if np.unique(y_true).size < 2:
        return {"f1_lo": math.nan, "f1_hi": math.nan, "ap_lo": math.nan, "ap_hi": math.nan}

    rng = np.random.default_rng(seed)
    n = y_true.size
    f1s: list[float] = []
    aps: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if np.unique(yt).size > 1:
            met = _binary_metrics(yt, yp, threshold)
            if not math.isnan(met["f1"]):
                f1s.append(float(met["f1"]))
            aps.append(float(average_precision_score(yt, yp)))
    if f1s:
        f1_lo, f1_hi = np.percentile(f1s, [2.5, 97.5])
    else:
        f1_lo, f1_hi = math.nan, math.nan
    if aps:
        ap_lo, ap_hi = np.percentile(aps, [2.5, 97.5])
    else:
        ap_lo, ap_hi = math.nan, math.nan
    return {"f1_lo": float(f1_lo), "f1_hi": float(f1_hi), "ap_lo": float(ap_lo), "ap_hi": float(ap_hi)}


def load_classification_source_groups(csv_path: Path, split: str = "test") -> dict[str, dict[str, np.ndarray]]:
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != split:
                continue
            raw_prob = (row.get("model_prob") or "").strip()
            if not raw_prob:
                continue
            try:
                prob = float(raw_prob)
            except ValueError:
                continue
            label = (row.get("label") or "").strip().lower()
            y = 1 if label in {"cdw", "1", "true", "yes"} else 0
            source = (row.get("source") or "unknown").strip().lower()
            groups[source].append((y, float(np.clip(prob, 0.0, 1.0))))

    out: dict[str, dict[str, np.ndarray]] = {}
    for source, pairs in groups.items():
        y = np.asarray([p[0] for p in pairs], dtype=np.uint8)
        pr = np.asarray([p[1] for p in pairs], dtype=np.float32)
        out[source] = {"y_true": y, "y_prob": pr}
    return out


def load_classification_layout_rows(
    csv_path: Path,
    split_column: str = "split",
    map_sheet: str | None = None,
) -> tuple[list[dict[str, int | str]], str]:
    all_rows: list[dict[str, int | str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get(split_column) or "").strip().lower()
            if split not in {"train", "test"}:
                continue
            ms = (row.get("map_sheet") or "").strip()
            if not ms:
                continue
            chunk_raw = (row.get("chunk_size") or "").strip()
            if not chunk_raw:
                continue
            all_rows.append(
                {
                    "map_sheet": ms,
                    "split": split,
                    "row_off": int(row["row_off"]),
                    "col_off": int(row["col_off"]),
                    "chunk_size": int(chunk_raw),
                }
            )

    if not all_rows:
        raise RuntimeError(f"Klassifitseerimise CSV-s puuduvad train/test ruumilised read veerus '{split_column}'.")

    if map_sheet is None:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "test": 0})
        for r in all_rows:
            ms = str(r["map_sheet"])
            sp = str(r["split"])
            counts[ms][sp] += 1

        best_sheet = ""
        best_score = -1
        best_total = -1
        for ms, c in counts.items():
            train_n = c["train"]
            test_n = c["test"]
            score = min(train_n, test_n)
            total = train_n + test_n
            if score > best_score or (score == best_score and total > best_total):
                best_sheet = ms
                best_score = score
                best_total = total
        map_sheet = best_sheet

    sheet_rows = [r for r in all_rows if str(r["map_sheet"]) == map_sheet]
    if not sheet_rows:
        raise RuntimeError(f"Kaardilehte '{map_sheet}' ei leitud train/test ruumilistes ridades.")
    return sheet_rows, map_sheet


def plot_classification_train_test_layout(
    rows: list[dict[str, int | str]],
    map_sheet: str,
    output_path: Path,
    buffer_m: float = 12.5,
    meters_per_pixel: float = 0.2,
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor("#f7f7f7")

    unique_train: set[tuple[int, int, int]] = set()
    unique_test: set[tuple[int, int, int]] = set()
    for r in rows:
        tpl = (int(r["row_off"]), int(r["col_off"]), int(r["chunk_size"]))
        if str(r["split"]) == "test":
            unique_test.add(tpl)
        else:
            unique_train.add(tpl)

    all_tiles = list(unique_train | unique_test)
    min_col = min(c for _, c, _ in all_tiles)
    min_row = min(r for r, _, _ in all_tiles)
    max_col = max(c + s for _, c, s in all_tiles)
    max_row = max(r + s for r, _, s in all_tiles)

    buffer_px = float(buffer_m / meters_per_pixel)

    # Draw test buffers first.
    for row_off, col_off, chunk in unique_test:
        ax.add_patch(
            Rectangle(
                (col_off - buffer_px, row_off - buffer_px),
                chunk + (2.0 * buffer_px),
                chunk + (2.0 * buffer_px),
                facecolor="none",
                edgecolor="#ff7f0e",
                linewidth=0.45,
                alpha=0.35,
                zorder=1,
            )
        )

    # Draw train/test tiles as solid fills (no thin white gridlines).
    for row_off, col_off, chunk in unique_train:
        ax.add_patch(
            Rectangle(
                (col_off, row_off),
                chunk,
                chunk,
                facecolor=PALETTE["blue"],
                edgecolor="white",
                linewidth=0.55,
                alpha=0.58,
                zorder=2,
            )
        )
    for row_off, col_off, chunk in unique_test:
        ax.add_patch(
            Rectangle(
                (col_off, row_off),
                chunk,
                chunk,
                facecolor=PALETTE["red"],
                edgecolor="white",
                linewidth=0.55,
                alpha=0.58,
                zorder=3,
            )
        )

    # Count train tiles intersecting any expanded test buffer rectangle.
    test_expanded: list[tuple[float, float, float, float]] = []
    for row_off, col_off, chunk in unique_test:
        test_expanded.append(
            (
                col_off - buffer_px,
                col_off + chunk + buffer_px,
                row_off - buffer_px,
                row_off + chunk + buffer_px,
            )
        )

    train_buffer_overlap = 0
    for row_off, col_off, chunk in unique_train:
        x1, x2 = float(col_off), float(col_off + chunk)
        y1, y2 = float(row_off), float(row_off + chunk)
        intersects = False
        for bx1, bx2, by1, by2 in test_expanded:
            if x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1:
                intersects = True
                break
        if intersects:
            train_buffer_overlap += 1

    ax.set_xlim(min_col, max_col)
    ax.set_ylim(max_row, min_row)
    ax.set_aspect("equal")
    ax.set_xlabel("Veerg (px)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rida (px)", fontsize=12, fontweight="bold")
    ax.set_title(f"Klassifitseerimise paanide paiknemine (kaardileht {map_sheet})", fontsize=16, fontweight="bold")
    ax.grid(False)

    legend_items = [
        Patch(facecolor=PALETTE["blue"], edgecolor="none", label=f"Treening (unikaalsed plokid, n={len(unique_train)})"),
        Patch(facecolor=PALETTE["red"], edgecolor="none", label=f"Test (unikaalsed plokid, n={len(unique_test)})"),
        Patch(facecolor="none", edgecolor="#ff7f0e", label=f"Testi puhver {buffer_m:.1f} m"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=True, fontsize=11)

    fig.text(
        0.5,
        0.02,
        f"Puhvri kontroll: {train_buffer_overlap} / {len(unique_train)} treeninguplokki lõikub testi {buffer_m:.1f} m puhvriga.",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["gray"],
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    _save(fig, output_path)


def plot_classification_corner_zoom(
    rows: list[dict[str, int | str]],
    map_sheet: str,
    output_path: Path,
    buffer_m: float = 12.5,
    meters_per_pixel: float = 0.2,
    corner_window_px: int = 500,
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor("#f7f7f7")

    unique_train: set[tuple[int, int, int]] = set()
    unique_test: set[tuple[int, int, int]] = set()
    for r in rows:
        tpl = (int(r["row_off"]), int(r["col_off"]), int(r["chunk_size"]))
        if str(r["split"]) == "test":
            unique_test.add(tpl)
        else:
            unique_train.add(tpl)

    all_tiles = list(unique_train | unique_test)
    min_col = min(c for _, c, _ in all_tiles)
    min_row = min(r for r, _, _ in all_tiles)
    x0 = float(min_col)
    y0 = float(min_row)
    x1 = x0 + float(corner_window_px)
    y1 = y0 + float(corner_window_px)

    buffer_px = float(buffer_m / meters_per_pixel)

    def intersects_window(row_off: int, col_off: int, chunk: int) -> bool:
        tx1, tx2 = float(col_off), float(col_off + chunk)
        ty1, ty2 = float(row_off), float(row_off + chunk)
        return tx1 < x1 and tx2 > x0 and ty1 < y1 and ty2 > y0

    vis_train = [t for t in unique_train if intersects_window(*t)]
    vis_test = [t for t in unique_test if intersects_window(*t)]

    for row_off, col_off, chunk in vis_test:
        ax.add_patch(
            Rectangle(
                (col_off - buffer_px, row_off - buffer_px),
                chunk + (2.0 * buffer_px),
                chunk + (2.0 * buffer_px),
                facecolor="none",
                edgecolor="#ff7f0e",
                linewidth=0.9,
                alpha=0.45,
                zorder=1,
            )
        )

    for row_off, col_off, chunk in vis_train:
        ax.add_patch(
            Rectangle(
                (col_off, row_off),
                chunk,
                chunk,
                facecolor=PALETTE["blue"],
                edgecolor="white",
                linewidth=0.85,
                alpha=0.58,
                zorder=2,
            )
        )
    for row_off, col_off, chunk in vis_test:
        ax.add_patch(
            Rectangle(
                (col_off, row_off),
                chunk,
                chunk,
                facecolor=PALETTE["red"],
                edgecolor="white",
                linewidth=0.85,
                alpha=0.58,
                zorder=3,
            )
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_xlabel("Veerg (px)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rida (px)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Klassifitseerimine: 500x500 px nurga detail (kaart {map_sheet})",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(False)
    ax.legend(
        handles=[
            Patch(facecolor=PALETTE["blue"], edgecolor="none", label=f"Treening (n={len(vis_train)})"),
            Patch(facecolor=PALETTE["red"], edgecolor="none", label=f"Test (n={len(vis_test)})"),
            Patch(facecolor="none", edgecolor="#ff7f0e", label=f"Testi puhver {buffer_m:.1f} m"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=11,
    )
    plt.tight_layout()
    _save(fig, output_path)


def _select_dense_test_tile(
    rows: list[dict[str, int | str]],
    stride: int = 64,
    radius: int = 2,
) -> tuple[str, int, int, int]:
    """Pick a test tile with many nearby train tiles for a visible overlap example."""
    by_map_sheet: dict[str, dict[str, set[tuple[int, int]]]] = defaultdict(lambda: {"train": set(), "test": set()})
    for r in rows:
        ms = str(r["map_sheet"])
        pos = (int(r["row_off"]), int(r["col_off"]))
        split = str(r["split"])
        if split == "test":
            by_map_sheet[ms]["test"].add(pos)
        elif split == "train":
            by_map_sheet[ms]["train"].add(pos)

    best_sheet = ""
    best_pos = (0, 0)
    best_score = -1
    for ms, d in by_map_sheet.items():
        for r0, c0 in d["test"]:
            score = 0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r0 + dr * stride, c0 + dc * stride) in d["train"]:
                        score += 1
            if score > best_score:
                best_sheet = ms
                best_pos = (r0, c0)
                best_score = score
    return best_sheet, best_pos[0], best_pos[1], best_score


def plot_single_test_tile_overlap_example(
    rows: list[dict[str, int | str]],
    output_path: Path,
    window_px: int = 500,
    buffer_m: float = 12.5,
    meters_per_pixel: float = 0.2,
) -> None:
    """Plot one test tile with translucent tiles so overlaps are visible."""
    _style()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor("#f7f7f7")

    map_sheet, test_row, test_col, score = _select_dense_test_tile(rows)
    if not map_sheet:
        raise RuntimeError("Ei leidnud sobivat testpaani näitejoonise jaoks.")

    unique_train: set[tuple[int, int, int]] = set()
    unique_test: set[tuple[int, int, int]] = set()
    for r in rows:
        if str(r["map_sheet"]) != map_sheet:
            continue
        tpl = (int(r["row_off"]), int(r["col_off"]), int(r["chunk_size"]))
        if str(r["split"]) == "test":
            unique_test.add(tpl)
        elif str(r["split"]) == "train":
            unique_train.add(tpl)

    x0 = max(0.0, float(test_col - window_px // 2))
    y0 = max(0.0, float(test_row - window_px // 2))
    x1 = x0 + float(window_px)
    y1 = y0 + float(window_px)

    buffer_px = float(buffer_m / meters_per_pixel)

    def visible(tile: tuple[int, int, int]) -> bool:
        r0, c0, chunk = tile
        return (c0 < x1) and (c0 + chunk > x0) and (r0 < y1) and (r0 + chunk > y0)

    vis_train = [t for t in unique_train if visible(t)]
    vis_test = [t for t in unique_test if visible(t)]

    for row_off, col_off, chunk in vis_test:
        ax.add_patch(
            Rectangle(
                (col_off - buffer_px, row_off - buffer_px),
                chunk + (2.0 * buffer_px),
                chunk + (2.0 * buffer_px),
                facecolor="none",
                edgecolor="#ff7f0e",
                linewidth=0.85,
                alpha=0.35,
                zorder=1,
            )
        )

    for row_off, col_off, chunk in vis_train:
        ax.add_patch(
            Rectangle(
                (col_off, row_off),
                chunk,
                chunk,
                facecolor=PALETTE["blue"],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.32,
                zorder=2,
            )
        )
    for row_off, col_off, chunk in vis_test:
        ax.add_patch(
            Rectangle(
                (col_off, row_off),
                chunk,
                chunk,
                facecolor=PALETTE["red"],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.32,
                zorder=3,
            )
        )

    ax.add_patch(
        Rectangle(
            (test_col, test_row),
            128,
            128,
            facecolor="none",
            edgecolor="#111111",
            linewidth=2.0,
            zorder=4,
        )
    )
    ax.text(
        test_col + 64,
        test_row - 12,
        "valitud testpaan",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#111111",
        zorder=5,
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_xlabel("Veerg (px)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rida (px)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Testpaani läbipaistev kattuvusnäide (kaart {map_sheet}, naabreid={score})",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(False)
    ax.legend(
        handles=[
            Patch(facecolor=PALETTE["blue"], edgecolor="none", alpha=0.32, label=f"Treening (n={len(vis_train)})"),
            Patch(facecolor=PALETTE["red"], edgecolor="none", alpha=0.32, label=f"Test (n={len(vis_test)})"),
            Patch(facecolor="none", edgecolor="#ff7f0e", label=f"Testi puhver {buffer_m:.1f} m"),
            Patch(facecolor="none", edgecolor="#111111", label="Valitud testpaan"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.02,
        "Läbipaistvad ruudud näitavad, kuidas sama kaardilehe treening- ja testpaanid üksteise peale jooksevad.",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["gray"],
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    _save(fig, output_path)


def plot_source_comparison(
    groups: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    threshold: float = 0.40,
) -> None:
    ordered = [s for s in ["manual", "auto", "auto_skip"] if s in groups] + [
        s for s in sorted(groups.keys()) if s not in {"manual", "auto", "auto_skip"}
    ]

    rows: list[dict[str, float | str]] = []
    for src in ordered:
        y = groups[src]["y_true"]
        p = groups[src]["y_prob"]
        met = _binary_metrics(y, p, threshold=threshold)
        ap = float(average_precision_score(y, p)) if np.unique(y).size > 1 else math.nan
        ci = _bootstrap_ci(y, p, threshold=threshold, n_boot=1000, seed=42)
        rows.append(
            {
                "source": src,
                "n": float(y.size),
                "pos": float(int((y == 1).sum())),
                "neg": float(int((y == 0).sum())),
                "has_both_classes": float(np.unique(y).size > 1),
                "f1": met["f1"],
                "f1_lo": ci["f1_lo"],
                "f1_hi": ci["f1_hi"],
                "ap": ap,
                "ap_lo": ci["ap_lo"],
                "ap_hi": ci["ap_hi"],
            }
        )

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    colors = {
        "manual": PALETTE["green"],
        "auto": PALETTE["blue"],
        "auto_skip": PALETTE["gray"],
    }

    x = np.arange(len(rows))
    labels = [r["source"] for r in rows]
    bar_colors = [colors.get(r["source"], PALETTE["black"]) for r in rows]

    # Left: F1 at fixed threshold
    f1_raw = np.array([float(r["f1"]) for r in rows], dtype=np.float64)
    f1_plot = np.nan_to_num(f1_raw, nan=0.0)
    f1_lo = np.array(
        [
            0.0 if math.isnan(float(r["f1"])) or math.isnan(float(r["f1_lo"])) else float(r["f1"]) - float(r["f1_lo"])
            for r in rows
        ]
    )
    f1_hi = np.array(
        [
            0.0 if math.isnan(float(r["f1"])) or math.isnan(float(r["f1_hi"])) else float(r["f1_hi"]) - float(r["f1"])
            for r in rows
        ]
    )
    bars_f1 = axes[0].bar(x, f1_plot, color=bar_colors, alpha=0.9)
    axes[0].errorbar(x, f1_plot, yerr=[f1_lo, f1_hi], fmt="none", ecolor=PALETTE["black"], capsize=4, linewidth=1.2)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel(f"F1 (lävi t={threshold:.2f})", fontsize=12, fontweight="bold")
    axes[0].set_title("Lävepõhine jõudlus allikate lõikes", fontsize=13, fontweight="bold")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)

    # Right: AP (threshold-free)
    ap_raw = np.array([float(r["ap"]) for r in rows], dtype=np.float64)
    ap_plot = np.nan_to_num(ap_raw, nan=0.0)
    ap_lo = np.array([0.0 if math.isnan(float(r["ap_lo"])) or math.isnan(float(r["ap"])) else float(r["ap"]) - float(r["ap_lo"]) for r in rows])
    ap_hi = np.array([0.0 if math.isnan(float(r["ap_hi"])) or math.isnan(float(r["ap"])) else float(r["ap_hi"]) - float(r["ap"]) for r in rows])
    bars_ap = axes[1].bar(x, ap_plot, color=bar_colors, alpha=0.9)
    axes[1].errorbar(x, ap_plot, yerr=[ap_lo, ap_hi], fmt="none", ecolor=PALETTE["black"], capsize=4, linewidth=1.2)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("AP (lävevaba)", fontsize=12, fontweight="bold")
    axes[1].set_title("Tõenäosusjärjestuse kvaliteet allikate lõikes", fontsize=13, fontweight="bold")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)

    for i, r in enumerate(rows):
        if not bool(r["has_both_classes"]):
            bars_f1[i].set_hatch("//")
            bars_ap[i].set_hatch("//")
            axes[0].text(i, 0.07, "N/A", ha="center", va="bottom", fontsize=10, color=PALETTE["black"])
            axes[1].text(i, 0.07, "N/A", ha="center", va="bottom", fontsize=10, color=PALETTE["black"])

    for i, r in enumerate(rows):
        txt = f"n={int(r['n'])}\npos={int(r['pos'])}\nneg={int(r['neg'])}"
        axes[0].text(i, 0.02, txt, ha="center", va="bottom", fontsize=9, color=PALETTE["black"])

    fig.suptitle("Klassifitseerimise tulemus märgistusallikate lõikes (test split)", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Märkus: auto ja auto_skip põhinevad osaliselt automaatsetel märgenditel; "
        "tõlgenda neid kui kooskõla allikate, mitte absoluutse tõe järgi.",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["gray"],
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save(fig, output_path)


def _read_phase_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_patch_layout_rows(patch_index_path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with patch_index_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "row_off": int(r["row_off"]),
                    "col_off": int(r["col_off"]),
                    "fold_id": int(r["fold_id"]),
                }
            )
    if not rows:
        raise RuntimeError(f"Patch-index failis puuduvad read: {patch_index_path}")
    return rows


def plot_train_test_layout(
    rows: list[dict[str, int]],
    output_path: Path,
    patch_size: int = 256,
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(16, 9))

    min_col = min(r["col_off"] for r in rows)
    min_row = min(r["row_off"] for r in rows)
    max_col = max(r["col_off"] for r in rows) + patch_size
    max_row = max(r["row_off"] for r in rows) + patch_size

    n_train = 0
    n_test = 0
    for r in rows:
        is_test = (r["fold_id"] == -1)
        if is_test:
            n_test += 1
            face = PALETTE["red"]
            alpha = 0.85
            z = 3
        else:
            n_train += 1
            face = PALETTE["blue"]
            alpha = 0.55
            z = 2
        ax.add_patch(
            Rectangle(
                (r["col_off"], r["row_off"]),
                patch_size,
                patch_size,
                facecolor=face,
                edgecolor="white",
                linewidth=0.7,
                alpha=alpha,
                zorder=z,
            )
        )

    ax.set_xlim(min_col, max_col)
    ax.set_ylim(max_row, min_row)
    ax.set_aspect("equal")
    ax.set_xlabel("Veerg (px)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rida (px)", fontsize=12, fontweight="bold")
    ax.set_title("Segmenteerimise paanide ruumiline paiknemine", fontsize=16, fontweight="bold")
    ax.grid(False)

    legend_items = [
        Patch(facecolor=PALETTE["blue"], edgecolor="white", alpha=0.7, label=f"Treening (n={n_train})"),
        Patch(facecolor=PALETTE["red"], edgecolor="white", alpha=0.9, label=f"Test (n={n_test})"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=True, fontsize=11)

    fig.text(
        0.5,
        0.02,
        "Märkus: testpaane kasutatakse ainult lõpphindamiseks (fold_id = -1), treeningpaane mudeli õppimiseks (fold_id >= 0).",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["gray"],
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    _save(fig, output_path)


def plot_center_ring_schematic(
    rows: list[dict[str, int | str]],
    output_path: Path,
    window_radius: int = 4,
    stride: int = 64,
    chunk_size: int = 128,
    buffer_m: float = 12.5,
    meters_per_pixel: float = 0.2,
) -> None:
    """Schematic of the intended center-based split logic around one test tile."""
    _style()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor("#f7f7f7")

    map_sheet, test_row, test_col, score = _select_dense_test_tile(rows, stride=stride, radius=2)
    if not map_sheet:
        raise RuntimeError("Ei leidnud sobivat testpaani keskkoha skeemi jaoks.")

    center_x = float(test_col)
    center_y = float(test_row)
    buffer_px = float(buffer_m / meters_per_pixel)

    def class_for_offset(dr: int, dc: int) -> str:
        d = max(abs(dr), abs(dc))
        if d <= 1:
            return "test"
        if d <= 2:
            return "buffer"
        return "train"

    colors = {
        "test": PALETTE["red"],
        "buffer": "#fafafa",
        "train": PALETTE["blue"],
    }

    # Draw a 9x9 grid of tile centers around the selected test tile.
    for dr in range(-window_radius, window_radius + 1):
        for dc in range(-window_radius, window_radius + 1):
            cls = class_for_offset(dr, dc)
            x = center_x + dc * stride
            y = center_y + dr * stride
            alpha = 0.34 if cls == "test" else 0.98 if cls == "buffer" else 0.18
            lw = 1.8 if (dr == 0 and dc == 0) else 0.95

            # Underlay buffer halo for the selected test tile.
            if dr == 0 and dc == 0:
                ax.add_patch(
                    Rectangle(
                        (x - buffer_px, y - buffer_px),
                        chunk_size + (2.0 * buffer_px),
                        chunk_size + (2.0 * buffer_px),
                        facecolor="none",
                        edgecolor="#8f8f8f",
                        linewidth=1.0,
                        alpha=0.7,
                        zorder=1,
                    )
                )

            ax.add_patch(
                Rectangle(
                    (x - chunk_size / 2.0, y - chunk_size / 2.0),
                    chunk_size,
                    chunk_size,
                    facecolor=colors[cls],
                    edgecolor="#a9a9a9" if cls == "buffer" else "#d8d8d8",
                    linewidth=lw,
                    alpha=alpha,
                    zorder=2 if cls == "train" else 3 if cls == "buffer" else 4,
                )
            )

    # Emphasize the selected center tile.
    ax.add_patch(
        Rectangle(
            (center_x - chunk_size / 2.0, center_y - chunk_size / 2.0),
            chunk_size,
            chunk_size,
            facecolor="none",
            edgecolor="#111111",
            linewidth=2.0,
            zorder=5,
        )
    )

    # Add tiny labels for the 1D sequence along the horizontal axis.
    seq = [(-4, "train"), (-3, "buffer"), (-2, "buffer"), (-1, "test"), (0, "test"), (1, "test"), (2, "buffer"), (3, "buffer"), (4, "train")]
    for dc, cls in seq:
        x = center_x + dc * stride
        ax.text(
            x,
            center_y + window_radius * stride + 92,
            cls,
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors[cls],
            fontweight="bold",
        )

    extent = window_radius * stride + chunk_size / 2.0
    x_min = center_x - extent
    x_max = center_x + extent
    y_min = center_y - extent
    y_max = center_y + extent

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal")
    grid_vals_x = np.arange(x_min, x_max + 0.1, stride)
    grid_vals_y = np.arange(y_min, y_max + 0.1, stride)
    ax.set_xticks(grid_vals_x, minor=True)
    ax.set_yticks(grid_vals_y, minor=True)
    ax.grid(False)
    # Draw a symmetric lattice that includes both the tile centers and the
    # outer tile edges so the layout reads the same in every direction.
    for gx in grid_vals_x:
        ax.axvline(gx, color="#c0c0c0", linewidth=0.9, zorder=6, alpha=0.95)
    for gy in grid_vals_y:
        ax.axhline(gy, color="#c0c0c0", linewidth=0.9, zorder=6, alpha=0.95)
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="none",
            edgecolor="#9a9a9a",
            linewidth=1.2,
            zorder=7,
        )
    )
    ax.set_xlabel("Veerg (px)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rida (px)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Keskkoha-põhine test/buffer/train skeem (kaart {map_sheet}, näide={test_row},{test_col})",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(False)
    ax.legend(
        handles=[
            Patch(facecolor=PALETTE["red"], edgecolor="none", alpha=0.34, label="Test"),
            Patch(facecolor="#fafafa", edgecolor="#8f8f8f", alpha=0.98, label="Puhver"),
            Patch(facecolor=PALETTE["blue"], edgecolor="none", alpha=0.18, label="Treening"),
            Patch(facecolor="none", edgecolor="#111111", label=f"Valitud testpaan (naabreid={score})"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.02,
        "Skeem on ehitatud keskkohtade järgi: samm 64 px, test kuni 1 samm, buffer kuni 2 sammu, väljaspoole jääv osa on train.",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["gray"],
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    _save(fig, output_path)


def plot_center_ring_tpr_grid(
    rows: list[dict[str, int | str]],
    output_path: Path,
    window_radius: int = 4,
    stride: int = 64,
) -> None:
    """Compact grid view where each tile is labeled T/P/R by split class."""
    _style()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor("#f7f7f7")

    map_sheet, test_row, test_col, score = _select_dense_test_tile(rows, stride=stride, radius=2)
    if not map_sheet:
        raise RuntimeError("Ei leidnud sobivat testpaani T/P/R ruudustiku jaoks.")

    center_x = float(test_col)
    center_y = float(test_row)

    def class_for_offset(dr: int, dc: int) -> str:
        d = max(abs(dr), abs(dc))
        if d <= 1:
            return "T"
        if d <= 3:
            return "P"
        return "R"

    colors = {
        "T": PALETTE["red"],
        "P": "#fbfbfb",
        "R": PALETTE["blue"],
    }

    text_colors = {
        "T": "#7a0f0f",
        "P": "#666666",
        "R": "#0f3f6a",
    }

    extent = window_radius * stride + stride / 2.0
    x_min = center_x - extent
    x_max = center_x + extent
    y_min = center_y - extent
    y_max = center_y + extent

    for dr in range(-window_radius, window_radius + 1):
        for dc in range(-window_radius, window_radius + 1):
            cls = class_for_offset(dr, dc)
            x = center_x + dc * stride
            y = center_y + dr * stride
            face = colors[cls]
            alpha = 0.38 if cls == "T" else 0.96 if cls == "P" else 0.22
            ax.add_patch(
                Rectangle(
                    (x - stride / 2.0, y - stride / 2.0),
                    stride,
                    stride,
                    facecolor=face,
                    edgecolor="#bdbdbd",
                    linewidth=0.8,
                    alpha=alpha,
                    zorder=2 if cls == "R" else 3 if cls == "P" else 4,
                )
            )
            ax.text(
                x,
                y,
                cls,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_colors[cls],
                zorder=5,
            )

    ax.add_patch(
        Rectangle(
            (center_x - stride / 2.0, center_y - stride / 2.0),
            stride,
            stride,
            facecolor="none",
            edgecolor="#111111",
            linewidth=2.0,
            zorder=6,
        )
    )

    grid_vals = np.arange(x_min, x_max + 0.1, stride)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal")
    ax.set_xticks(grid_vals, minor=True)
    ax.set_yticks(grid_vals, minor=True)
    for gx in grid_vals:
        ax.axvline(gx, color="#c0c0c0", linewidth=0.9, zorder=7, alpha=0.95)
    for gy in grid_vals:
        ax.axhline(gy, color="#c0c0c0", linewidth=0.9, zorder=7, alpha=0.95)
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="none",
            edgecolor="#9a9a9a",
            linewidth=1.2,
            zorder=8,
        )
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_title(
        f"T/P/R ruudustik keskpunktide järgi (kaart {map_sheet}, näide={test_row},{test_col})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(False)
    fig.text(0.5, 0.915, "T = test, P = puhver, R = treening", ha="center", va="center", fontsize=11, color=PALETTE["gray"])
    fig.text(
        0.5,
        0.02,
        "Sama reegel kõigile: 1 samm = T, 2-3 sammu = P, ülejäänu = R.",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["gray"],
    )
    plt.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.10)
    _save(fig, output_path)


def _aggregate_mean_sd(rows: list[dict[str, str]], run_id: str, metric: str) -> tuple[float, float]:
    vals = []
    for r in rows:
        if r.get("run_id") == run_id:
            raw = r.get(metric, "")
            if raw != "":
                vals.append(float(raw))
    if not vals:
        raise RuntimeError(f"Run '{run_id}' metric '{metric}' puudub.")
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def plot_ablation_waterfall(
    ablation_dir: Path,
    output_path: Path,
) -> None:
    phase6_rows = _read_phase_rows(ablation_dir / "phase6_results_test.csv")
    best_phase6 = max(phase6_rows, key=lambda r: float(r.get("test_f1", "0") or 0.0))
    best_run = best_phase6["run_id"]  # e.g. 2E__3B__4H__5E__6
    parts = best_run.split("__")
    if len(parts) < 5:
        raise RuntimeError(f"Run-id kuju ootamatu: {best_run}")
    run_phase2 = parts[0]
    run_phase3 = "__".join(parts[:2])
    run_phase4 = "__".join(parts[:3])
    run_phase5 = "__".join(parts[:4])

    p2 = _read_phase_rows(ablation_dir / "phase2_results_val.csv")
    p3 = _read_phase_rows(ablation_dir / "phase3_results_val.csv")
    p4 = _read_phase_rows(ablation_dir / "phase4_results_val.csv")
    p5 = _read_phase_rows(ablation_dir / "phase5_results_val.csv")

    labels = [run_phase2, run_phase3, run_phase4, run_phase5]
    means = []
    sds = []
    for rows, run in zip([p2, p3, p4, p5], labels):
        m, sd = _aggregate_mean_sd(rows, run, "val_cldice")
        means.append(m)
        sds.append(sd)

    means_arr = np.asarray(means, dtype=np.float64)
    deltas = np.diff(means_arr)

    _style()
    fig, ax = plt.subplots(figsize=(16, 9))

    # Waterfall bars
    x = np.arange(len(labels))
    ax.bar([0], [means_arr[0]], color=PALETTE["blue"], alpha=0.85, label="Algväärtus (faas 2)")
    prev = means_arr[0]
    for i, d in enumerate(deltas, start=1):
        color = PALETTE["green"] if d >= 0 else PALETTE["red"]
        bottom = prev if d >= 0 else prev + d
        ax.bar([i], [abs(d)], bottom=[bottom], color=color, alpha=0.85)
        prev = prev + d

    # Point estimates + SD
    ax.errorbar(x, means_arr, yerr=np.asarray(sds), fmt="o-", color=PALETTE["black"], linewidth=1.5, capsize=4, label="Mean val_clDice ± SD")

    for i, m in enumerate(means_arr):
        ax.text(i, m + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    for i, d in enumerate(deltas, start=1):
        sign = "+" if d >= 0 else ""
        ax.text(i, means_arr[i] + 0.02, f"{sign}{d:.3f}", ha="center", va="bottom", fontsize=10, color=PALETTE["black"])

    ax.set_xticks(x, ["Faas 2", "Faas 3", "Faas 4", "Faas 5"])
    ax.set_ylabel("val_clDice (CV keskmine)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(0.4, float(np.max(means_arr) + 0.08)))
    ax.set_title("Ablatsiooni valikuraja mõju (faasid 2–5)", fontsize=16, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    test_cldice = float(best_phase6.get("cldice", "nan"))
    test_f1 = float(best_phase6.get("test_f1", "nan"))
    ax.text(
        0.99,
        0.02,
        f"Parim phase6 (held-out test): run={best_run}, test_clDice={test_cldice:.3f}, test_F1={test_f1:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=PALETTE["gray"],
    )

    plt.tight_layout()
    _save(fig, output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate critical analytical thesis figures.")
    p.add_argument(
        "--phase6-dir",
        type=Path,
        default=ROOT / "seg_pipeline/output/ablation_v10_top2_blockcv5_full_20260511_105703/phase6_2E__3B__4H__5E__6_final_validation",
    )
    p.add_argument(
        "--seg-dataset-dir",
        type=Path,
        default=ROOT / "seg_pipeline/output/phase2_dataset_v10_blockcv5_full_20260511_105703",
    )
    p.add_argument(
        "--seg-chm-tif",
        type=Path,
        default=ROOT / "source/406455_2021_tava/chm_variants_reconstructed_original_20260510/composite_4band_raw_base_mask/406455_2021_4band.tif",
    )
    p.add_argument(
        "--seg-mask-tif",
        type=Path,
        default=ROOT / "seg_pipeline/output/phase1_masks/406455_2021_tava_truemask.tif",
    )
    p.add_argument(
        "--classification-csv",
        type=Path,
        default=ROOT / "data/chm_variants/labels_canonical_with_splits_spatial_ensemble.csv",
    )
    p.add_argument(
        "--classification-map-sheet",
        default=None,
        help="Valikuline kaardilehe ID klassifitseerimise train/test paiknemise jooniseks.",
    )
    p.add_argument(
        "--classification-split-column",
        default="split",
        help="CSV veerg, mida kasutada klassifitseerimise paanimise jaoks (nt split või split_center_gap).",
    )
    p.add_argument("--classification-buffer-m", type=float, default=12.5)
    p.add_argument("--classification-meters-per-pixel", type=float, default=0.2)
    p.add_argument("--classification-corner-window-px", type=int, default=500)
    p.add_argument("--classification-example-window-px", type=int, default=500)
    p.add_argument(
        "--ablation-dir",
        type=Path,
        default=ROOT / "seg_pipeline/output/ablation_v10_top2_blockcv5_full_20260511_105703",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "LaTeX/Lamapuidu_tuvastamine/estonian/joonised",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threshold-main", type=float, default=0.50)
    p.add_argument("--threshold-source", type=float, default=0.40)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    patches, meta = load_segmentation_test_predictions(
        phase6_dir=args.phase6_dir,
        dataset_dir=args.seg_dataset_dir,
        chm_tif=args.seg_chm_tif,
        mask_tif=args.seg_mask_tif,
        device=args.device,
        batch_size=args.batch_size,
    )

    thresholds = np.linspace(0.0, 1.0, 101)
    sweep = segmentation_threshold_sweep(patches, thresholds)
    plot_threshold_sensitivity(
        sweep=sweep,
        output_path=args.output_dir / "seg_lave_tundlikkus_testvalim.png",
        official_threshold=float(meta.get("optimal_threshold", args.threshold_main)),
        official_f1=float(meta.get("test_f1", np.nan)),
    )

    patch_rows = patch_metrics_at_threshold(patches, threshold=args.threshold_main)
    plot_spatial_error_map(
        rows=patch_rows,
        output_path=args.output_dir / "seg_ruumiline_veakaart_testplokid.png",
        threshold=args.threshold_main,
    )

    source_groups = load_classification_source_groups(args.classification_csv, split="test")
    plot_source_comparison(
        groups=source_groups,
        output_path=args.output_dir / "klassifitseerimine_margistusallikad_vordlus.png",
        threshold=args.threshold_source,
    )
    cls_layout_rows, cls_sheet = load_classification_layout_rows(
        args.classification_csv,
        split_column=args.classification_split_column,
        map_sheet=args.classification_map_sheet,
    )
    layout_suffix = "" if args.classification_split_column == "split" else f"_{args.classification_split_column}"
    plot_classification_train_test_layout(
        rows=cls_layout_rows,
        map_sheet=cls_sheet,
        output_path=args.output_dir / f"klassifitseerimine_treening_test_paanid_kaart_{cls_sheet}{layout_suffix}.png",
        buffer_m=args.classification_buffer_m,
        meters_per_pixel=args.classification_meters_per_pixel,
    )
    plot_classification_corner_zoom(
        rows=cls_layout_rows,
        map_sheet=cls_sheet,
        output_path=args.output_dir / f"klassifitseerimine_treening_test_paanid_kaart_{cls_sheet}{layout_suffix}_nurk_{args.classification_corner_window_px}px.png",
        buffer_m=args.classification_buffer_m,
        meters_per_pixel=args.classification_meters_per_pixel,
        corner_window_px=args.classification_corner_window_px,
    )
    plot_single_test_tile_overlap_example(
        rows=cls_layout_rows,
        output_path=args.output_dir / f"klassifitseerimine_testpaani_kattuvus_naidis{layout_suffix}.png",
        window_px=args.classification_example_window_px,
        buffer_m=args.classification_buffer_m,
        meters_per_pixel=args.classification_meters_per_pixel,
    )
    plot_center_ring_schematic(
        rows=cls_layout_rows,
        output_path=args.output_dir / f"klassifitseerimise_keskkoha_ringid_skeem{layout_suffix}.png",
        window_radius=4,
        stride=64,
        chunk_size=128,
        buffer_m=args.classification_buffer_m,
        meters_per_pixel=args.classification_meters_per_pixel,
    )
    plot_center_ring_tpr_grid(
        rows=cls_layout_rows,
        output_path=args.output_dir / f"klassifitseerimise_TPR_ruudustik{layout_suffix}.png",
        window_radius=4,
        stride=64,
    )

    plot_ablation_waterfall(
        ablation_dir=args.ablation_dir,
        output_path=args.output_dir / "seg_ablation_waterfall_faasid2_5.png",
    )

    patch_index_path, _ = _resolve_dataset_files(args.seg_dataset_dir, str(meta.get("chm_variant", "composite")))
    layout_rows = load_patch_layout_rows(patch_index_path)
    plot_train_test_layout(
        rows=layout_rows,
        output_path=args.output_dir / "seg_treening_test_paanide_paiknemine.png",
        patch_size=256,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

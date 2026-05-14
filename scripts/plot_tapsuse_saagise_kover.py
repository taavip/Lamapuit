#!/usr/bin/env python3
"""Generate separate thesis-ready Precision-Recall figures for:
1) best classification model
2) best segmentation model

Fixes:
- no artificial (0, 0) PR point
- thresholds use full score resolution (not coarse manual steps)
- improved classifier visualization with an honest zoomed panel
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sns
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve


DEFAULT_CLASSIFICATION_CSV = Path(
    "data/chm_variants/labels_canonical_with_splits_spatial_ensemble.csv"
)
DEFAULT_SEGMENTATION_PHASE6_DIR = Path(
    "seg_pipeline/output/ablation_v10_top2_blockcv5_full_20260511_105703/"
    "phase6_2E__3B__4H__5E__6_final_validation"
)
DEFAULT_SEGMENTATION_DATASET_DIR = Path(
    "seg_pipeline/output/phase2_dataset_v10_blockcv5_full_20260511_105703"
)
DEFAULT_SEGMENTATION_CHM_TIF = Path(
    "source/406455_2021_tava/chm_variants_reconstructed_original_20260510/"
    "composite_4band_raw_base_mask/406455_2021_4band.tif"
)
DEFAULT_SEGMENTATION_PROB_TIF = Path(
    "seg_pipeline/output/ablation_v10_top2_blockcv5_full_20260511_105703/"
    "phase6_2E__3B__4H__5E__6_final_validation/406455_2021_tava_2E__3B__4H__5E__6_prob.tif"
)
DEFAULT_SEGMENTATION_MASK_TIF = Path(
    "seg_pipeline/output/phase1_masks/406455_2021_tava_truemask.tif"
)

DEFAULT_OUT_CLASSIFICATION = Path(
    "LaTeX/Lamapuidu_tuvastamine/estonian/joonised/"
    "tapsuse_saagise_kover_parim_klassifitseerimine.png"
)
DEFAULT_OUT_SEGMENTATION = Path(
    "LaTeX/Lamapuidu_tuvastamine/estonian/joonised/"
    "tapsuse_saagise_kover_parim_segmenteerimine.png"
)

C_BLUE = "#1f77b4"
C_RED = "#d62728"
C_GRAY = "#7f7f7f"


def _parse_label_to_int(label: str) -> int:
    v = (label or "").strip().lower()
    if v in {"cdw", "1", "true", "yes"}:
        return 1
    return 0


def load_classification_test_data(
    csv_path: Path,
    prob_col: str = "model_prob",
    label_col: str = "label",
    split_col: str = "split",
    split_value: str = "test",
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_prob: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split_col and split_value and row.get(split_col, "") != split_value:
                continue
            raw_prob = (row.get(prob_col) or "").strip()
            if not raw_prob:
                continue
            try:
                p = float(raw_prob)
            except ValueError:
                continue
            y_true.append(_parse_label_to_int(row.get(label_col, "")))
            y_prob.append(float(np.clip(p, 0.0, 1.0)))

    if not y_true:
        raise RuntimeError(f"Klassifitseerimise andmeid ei leitud: {csv_path}")
    return np.asarray(y_true, dtype=np.uint8), np.asarray(y_prob, dtype=np.float32)


def load_segmentation_data_from_prob_tif(
    prob_tif: Path,
    mask_tif: Path,
    use_test_stripe: bool = True,
    test_stripe: int = 0,
    n_stripes: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(prob_tif) as src_prob:
        prob = src_prob.read(1).astype(np.float32)

    with rasterio.open(mask_tif) as src_mask:
        gt = src_mask.read(1).astype(np.float32)
        valid = src_mask.read(2).astype(np.float32)

    if use_test_stripe:
        width = gt.shape[1]
        stripe_width = max(1, width // n_stripes)
        start = test_stripe * stripe_width
        end = width if test_stripe >= (n_stripes - 1) else min(start + stripe_width, width)
        prob = prob[:, start:end]
        gt = gt[:, start:end]
        valid = valid[:, start:end]

    valid_mask = valid > 0.5
    if not np.any(valid_mask):
        raise RuntimeError("Segmenteerimise maskis puuduvad kehtivad pikslid.")

    y_true = (gt[valid_mask] > 0.5).astype(np.uint8)
    y_prob = np.clip(prob[valid_mask], 0.0, 1.0).astype(np.float32)
    return y_true, y_prob


def _dataset_dir_candidates(base_dir: Path) -> list[Path]:
    candidates = [base_dir]
    repo_out = Path("seg_pipeline/output")
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
    fallback = Path(
        "source/406455_2021_tava/chm_variants_reconstructed_original_20260510/"
        "composite_4band_raw_base_mask/406455_2021_4band.tif"
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Segmenteerimise CHM faili ei leitud: {path}. "
        f"Kontrolli teed või anna --seg-chm-tif."
    )


def load_segmentation_data_cv5_entries(
    phase6_dir: Path,
    dataset_dir: Path,
    chm_tif: Path,
    mask_tif: Path,
    device: str = "cpu",
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    root = Path(__file__).resolve().parents[1]
    seg_scripts = root / "seg_pipeline" / "scripts"
    if str(seg_scripts) not in sys.path:
        sys.path.insert(0, str(seg_scripts))

    from phase2_dataset_v3 import CWDSegDataset, load_patch_index  # type: ignore
    from phase3_train_v10 import build_model  # type: ignore

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

    model = build_model(arch, in_channels=in_channels, pretrained=False).to(torch.device(device))
    ckpt = torch.load(ckpt_path, map_location=torch.device(device), weights_only=False)
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

    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(torch.device(device), non_blocking=True)
            logits = model(image)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
            tgts = batch["target"][:, 0].numpy()
            vals = batch["valid"][:, 0].numpy() > 0.5

            for p, t, v in zip(probs, tgts, vals):
                y_true_all.append((t[v] > 0.5).astype(np.uint8))
                y_prob_all.append(np.clip(p[v], 0.0, 1.0).astype(np.float32))

    if not y_true_all:
        raise RuntimeError("CVv5 test-entry hindamisel ei kogunenud ühtegi valid pikslit.")

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    info = {
        "reported_test_f1": float(meta.get("test_f1", np.nan)),
        "reported_optimal_threshold": float(meta.get("optimal_threshold", np.nan)),
        "reported_precision": float(meta.get("precision", np.nan)),
        "reported_recall": float(meta.get("recall", np.nan)),
        "n_test_entries": float(len(test_entries)),
    }
    return y_true, y_prob, info


def compute_pr_stats(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, np.ndarray | float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns recall from 1 -> 0. Reverse for plotting on x=[0..1].
    precision_plot = precision[::-1]
    recall_plot = recall[::-1]

    f1_vals = (2.0 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1_vals))

    ap = float(average_precision_score(y_true, y_prob))
    auc_pr = float(auc(recall_plot, precision_plot))

    return {
        "precision_plot": precision_plot.astype(np.float64),
        "recall_plot": recall_plot.astype(np.float64),
        "thresholds": thresholds.astype(np.float64),
        "best_threshold": float(thresholds[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "best_f1": float(f1_vals[best_idx]),
        "ap": ap,
        "auc_pr": auc_pr,
        "n_curve_points": float(precision_plot.size),
        "n_unique_scores": float(np.unique(y_prob).size),
        "class_prevalence": float(np.mean(y_true == 1)),
    }


def _base_style(ax: plt.Axes) -> None:
    ax.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def plot_segmentation_pr(
    stats: dict[str, np.ndarray | float],
    output_path: Path,
    reported: dict[str, float] | None = None,
) -> None:
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(16, 9))

    rec = stats["recall_plot"]
    prec = stats["precision_plot"]

    ax.step(
        rec,
        prec,
        where="post",
        color=C_RED,
        linewidth=2.4,
        label=f"Segmenteerimine (AP={stats['ap']:.3f}, AUC-PR={stats['auc_pr']:.3f})",
    )
    ax.scatter(
        [stats["best_recall"]],
        [stats["best_precision"]],
        color=C_BLUE,
        s=80,
        zorder=5,
        label=f"Parim F1={stats['best_f1']:.3f} (t={stats['best_threshold']:.3f})",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Saagis", fontsize=14, fontweight="bold")
    ax.set_ylabel("Tapsus", fontsize=14, fontweight="bold")
    ax.set_title("Tapsuse-saagise kover: parim segmenteerimismudel", fontsize=16, fontweight="bold")
    _base_style(ax)
    ax.legend(loc="lower left", fontsize=11, frameon=True)

    ax.text(
        0.99,
        0.02,
        f"Punktid: {int(stats['n_curve_points'])}, unikaalseid skooritasemeid: {int(stats['n_unique_scores'])}",
        ha="right",
        va="bottom",
        fontsize=10,
        color=C_GRAY,
    )
    if reported is not None:
        rf1 = reported.get("reported_test_f1")
        if isinstance(rf1, float) and np.isfinite(rf1):
            ax.text(
                0.01,
                0.02,
                f"Raportis test-F1: {rf1:.3f} (CVv5 test-entry valim)",
                ha="left",
                va="bottom",
                fontsize=10,
                color=C_GRAY,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _zoom_bounds(recall_plot: np.ndarray, precision_plot: np.ndarray, best_recall: float, best_precision: float) -> tuple[float, float]:
    mask = recall_plot >= 0.75
    if np.any(mask):
        y_min = float(np.min(precision_plot[mask]))
        x_min = float(np.min(recall_plot[mask]))
    else:
        y_min = float(np.min(precision_plot))
        x_min = float(np.min(recall_plot))

    x0 = max(0.0, min(x_min, best_recall) - 0.03)
    y0 = max(0.0, min(y_min, best_precision) - 0.03)
    return x0, y0


def plot_classification_pr(stats: dict[str, np.ndarray | float], output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    rec = np.asarray(stats["recall_plot"], dtype=np.float64)
    prec = np.asarray(stats["precision_plot"], dtype=np.float64)
    br = float(stats["best_recall"])
    bp = float(stats["best_precision"])

    # Full panel
    ax = axes[0]
    ax.step(
        rec,
        prec,
        where="post",
        color=C_BLUE,
        linewidth=2.4,
        label=f"Klassifitseerimine (AP={stats['ap']:.3f}, AUC-PR={stats['auc_pr']:.3f})",
    )
    ax.scatter([br], [bp], color=C_RED, s=80, zorder=5, label=f"Parim F1={stats['best_f1']:.3f}")
    ax.hlines(float(stats["class_prevalence"]), 0.0, 1.0, colors=C_GRAY, linestyles="--", linewidth=1.5, label="Baasmaar")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Saagis", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tapsus", fontsize=12, fontweight="bold")
    ax.set_title("Taisskaala", fontsize=13, fontweight="bold")
    _base_style(ax)
    ax.legend(loc="lower left", fontsize=10, frameon=True)

    # Zoomed panel
    axz = axes[1]
    axz.step(rec, prec, where="post", color=C_BLUE, linewidth=2.4)
    axz.scatter([br], [bp], color=C_RED, s=80, zorder=5)
    x0, y0 = _zoom_bounds(rec, prec, br, bp)
    axz.set_xlim(x0, 1.0)
    axz.set_ylim(y0, 1.005)
    axz.set_xlabel("Saagis (suurendus)", fontsize=12, fontweight="bold")
    axz.set_ylabel("Tapsus (suurendus)", fontsize=12, fontweight="bold")
    axz.set_title("Informatiivne vaade korge tapsuse piirkonnale", fontsize=13, fontweight="bold")
    _base_style(axz)

    fig.suptitle("Tapsuse-saagise kover: parim klassifitseerimismudel", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        f"Punktid: {int(stats['n_curve_points'])}, unikaalseid skooritasemeid: {int(stats['n_unique_scores'])}",
        ha="center",
        va="center",
        fontsize=10,
        color=C_GRAY,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Loo kaks eraldi PR joonist (klassifitseerimine + segmenteerimine) "
            "parimatele mudelitele."
        )
    )
    parser.add_argument("--classification-csv", type=Path, default=DEFAULT_CLASSIFICATION_CSV)
    parser.add_argument("--classification-split", default="test")
    parser.add_argument("--classification-prob-col", default="model_prob")
    parser.add_argument("--classification-label-col", default="label")
    parser.add_argument(
        "--seg-source",
        choices=["cv5_entries", "prob_tif"],
        default="cv5_entries",
        help="Segmenteerimise PR andmeallikas: ametlik CVv5 test-entry inference või prob_tif.",
    )
    parser.add_argument("--seg-phase6-dir", type=Path, default=DEFAULT_SEGMENTATION_PHASE6_DIR)
    parser.add_argument("--seg-dataset-dir", type=Path, default=DEFAULT_SEGMENTATION_DATASET_DIR)
    parser.add_argument("--seg-chm-tif", type=Path, default=DEFAULT_SEGMENTATION_CHM_TIF)
    parser.add_argument("--seg-prob-tif", type=Path, default=DEFAULT_SEGMENTATION_PROB_TIF)
    parser.add_argument("--seg-mask-tif", type=Path, default=DEFAULT_SEGMENTATION_MASK_TIF)
    parser.add_argument("--use-test-stripe", action="store_true", default=True)
    parser.add_argument("--all-valid-pixels", action="store_true")
    parser.add_argument("--test-stripe", type=int, default=0)
    parser.add_argument("--n-stripes", type=int, default=5)
    parser.add_argument("--seg-device", default="cpu")
    parser.add_argument("--seg-batch-size", type=int, default=16)
    parser.add_argument("--out-classification", type=Path, default=DEFAULT_OUT_CLASSIFICATION)
    parser.add_argument("--out-segmentation", type=Path, default=DEFAULT_OUT_SEGMENTATION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    y_cls, p_cls = load_classification_test_data(
        csv_path=args.classification_csv,
        prob_col=args.classification_prob_col,
        label_col=args.classification_label_col,
        split_value=args.classification_split,
    )
    cls_stats = compute_pr_stats(y_cls, p_cls)

    seg_info: dict[str, float] | None = None
    if args.seg_source == "cv5_entries":
        y_seg, p_seg, seg_info = load_segmentation_data_cv5_entries(
            phase6_dir=args.seg_phase6_dir,
            dataset_dir=args.seg_dataset_dir,
            chm_tif=args.seg_chm_tif,
            mask_tif=args.seg_mask_tif,
            device=args.seg_device,
            batch_size=args.seg_batch_size,
        )
    else:
        y_seg, p_seg = load_segmentation_data_from_prob_tif(
            prob_tif=args.seg_prob_tif,
            mask_tif=args.seg_mask_tif,
            use_test_stripe=(args.use_test_stripe and not args.all_valid_pixels),
            test_stripe=args.test_stripe,
            n_stripes=args.n_stripes,
        )
    seg_stats = compute_pr_stats(y_seg, p_seg)

    plot_classification_pr(cls_stats, args.out_classification)
    plot_segmentation_pr(seg_stats, args.out_segmentation, reported=seg_info)

    print(f"Valmis: {args.out_classification}")
    print(f"Valmis: {args.out_segmentation}")
    print(
        "Klassifitseerimine: "
        f"AP={cls_stats['ap']:.4f}, AUC-PR={cls_stats['auc_pr']:.4f}, "
        f"F1max={cls_stats['best_f1']:.4f}, t={cls_stats['best_threshold']:.4f}"
    )
    print(
        "Segmenteerimine: "
        f"AP={seg_stats['ap']:.4f}, AUC-PR={seg_stats['auc_pr']:.4f}, "
        f"F1max={seg_stats['best_f1']:.4f}, t={seg_stats['best_threshold']:.4f}"
    )
    if seg_info is not None:
        print(
            "Segmenteerimise raporti kontroll: "
            f"test_F1={seg_info['reported_test_f1']:.4f}, "
            f"t={seg_info['reported_optimal_threshold']:.2f}, "
            f"test_entries={int(seg_info['n_test_entries'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

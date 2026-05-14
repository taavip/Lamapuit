#!/usr/bin/env python3
"""Create train/validation loss-style curves for best segmentation and classification models.

Note:
- Existing logs contain explicit train loss.
- Explicit val_loss is not stored in these runs, so validation curve uses
  proxy `1 - val_F1` and is labeled accordingly.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


SEG_HISTORY_JSONL = Path(
    "seg_pipeline/output/ablation_v10_top2_blockcv5_full_20260511_105703/"
    "phase6_2E__3B__4H__5E__6_final_validation/composite/fold0/fold_history.jsonl"
)
CLS_ENSEMBLE_LOG = Path("output/classification_140526_train_buffer_test/train_ensemble.log")
OUT_PNG = Path(
    "LaTeX/Lamapuidu_tuvastamine/estonian/joonised/"
    "treening_valideerimine_kadu_parimad_mudelid.png"
)


_SECTION_RE = re.compile(r"^\[train_ensemble\]\s+([a-zA-Z0-9_]+)\s+epochs=(\d+)")
_EPOCH_RE = re.compile(
    r"^\s*epoch\s+(\d+)/(\d+)\s+loss=([0-9.]+)\s+val_AUC=([0-9.]+)\s+F1=([0-9.]+)@([0-9.]+)"
)
_SAVED_RE = re.compile(
    r"^\[train_ensemble\]\s+([a-zA-Z0-9_]+)\s+saved.+val_AUC=([0-9.]+)\s+F1=([0-9.]+)"
)


def load_seg_history(history_jsonl: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs: list[int] = []
    train_loss: list[float] = []
    val_proxy_loss: list[float] = []

    with history_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            epoch = int(row["epoch"])
            tr_loss = float(row["train_loss"])
            val_f1 = float(row["val_f1"])
            epochs.append(epoch)
            train_loss.append(tr_loss)
            val_proxy_loss.append(1.0 - val_f1)

    if not epochs:
        raise RuntimeError(f"Segmenteerimise ajalugu tühi: {history_jsonl}")

    return (
        np.asarray(epochs, dtype=np.int32),
        np.asarray(train_loss, dtype=np.float64),
        np.asarray(val_proxy_loss, dtype=np.float64),
    )


def _parse_classification_log(log_path: Path) -> dict[str, dict]:
    sections: dict[str, dict] = {}
    current_model = None

    with log_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            m_sec = _SECTION_RE.match(line)
            if m_sec:
                model = m_sec.group(1)
                epochs_total = int(m_sec.group(2))
                sections.setdefault(
                    model,
                    {
                        "epochs_total": epochs_total,
                        "rows": [],
                        "final_val_f1": None,
                        "final_val_auc": None,
                    },
                )
                current_model = model
                continue

            m_epoch = _EPOCH_RE.match(line)
            if m_epoch and current_model is not None:
                ep = int(m_epoch.group(1))
                ep_total = int(m_epoch.group(2))
                tr_loss = float(m_epoch.group(3))
                val_auc = float(m_epoch.group(4))
                val_f1 = float(m_epoch.group(5))
                sections[current_model]["rows"].append(
                    {
                        "epoch": ep,
                        "epochs_total": ep_total,
                        "progress": round(ep / max(ep_total, 1), 3),
                        "train_loss": tr_loss,
                        "val_auc": val_auc,
                        "val_f1": val_f1,
                    }
                )
                continue

            m_saved = _SAVED_RE.match(line)
            if m_saved:
                model = m_saved.group(1)
                if model in sections:
                    sections[model]["final_val_auc"] = float(m_saved.group(2))
                    sections[model]["final_val_f1"] = float(m_saved.group(3))

    return sections


def aggregate_ensemble_training_curves(
    log_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sections = _parse_classification_log(log_path)
    if not sections:
        raise RuntimeError(f"Klassifitseerimise logi ei sisaldanud epohhi ridu: {log_path}")

    # Kasutame lõpliku ansambli 4 baasmudelit.
    selected = ["cnn_seed42", "cnn_seed43", "cnn_seed44", "effnet_b2"]
    for tag in selected:
        if tag not in sections or not sections[tag]["rows"]:
            raise RuntimeError(f"Puudub klassifitseerimise sektsioon: {tag}")

    progress_levels = sorted(
        {round(row["progress"], 1) for tag in selected for row in sections[tag]["rows"]}
    )

    prog_vals: list[float] = []
    train_vals: list[float] = []
    val_proxy_vals: list[float] = []

    for p in progress_levels:
        tr_samples: list[float] = []
        vp_samples: list[float] = []
        for tag in selected:
            # Leia selle mudeli lähim logitud progress.
            rows = sections[tag]["rows"]
            nearest = min(rows, key=lambda r: abs(round(r["progress"], 1) - p))
            if abs(round(nearest["progress"], 1) - p) > 1e-9:
                continue
            tr_samples.append(float(nearest["train_loss"]))
            vp_samples.append(1.0 - float(nearest["val_f1"]))

        if tr_samples:
            prog_vals.append(float(p))
            train_vals.append(float(np.mean(tr_samples)))
            val_proxy_vals.append(float(np.mean(vp_samples)))

    if not prog_vals:
        raise RuntimeError("Ansambli agregatsioon ebaõnnestus: progressi punkte ei tekkinud.")

    progress_pct = np.asarray(prog_vals, dtype=np.float64) * 100.0
    return (
        progress_pct,
        np.asarray(train_vals, dtype=np.float64),
        np.asarray(val_proxy_vals, dtype=np.float64),
    )


def plot_curves(
    seg_epochs: np.ndarray,
    seg_train: np.ndarray,
    seg_val_proxy: np.ndarray,
    cls_progress_pct: np.ndarray,
    cls_train: np.ndarray,
    cls_val_proxy: np.ndarray,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    c_train = "#1f77b4"
    c_val = "#d62728"

    # Segmenteerimine
    ax = axes[0]
    ax.plot(seg_epochs, seg_train, color=c_train, linewidth=2.4, label="Treeningkadu")
    ax.plot(
        seg_epochs,
        seg_val_proxy,
        color=c_val,
        linewidth=2.2,
        linestyle="--",
        label="Valideerimise kaoproksy (1 - F1)",
    )
    ax.set_title("Parim segmenteerimismudel (2E__3B__4H__5E__6)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epohh", fontsize=12, fontweight="bold")
    ax.set_ylabel("Kao väärtus", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Klassifitseerimine (ansambli baasmudelite keskmine)
    ax = axes[1]
    ax.plot(
        cls_progress_pct,
        cls_train,
        color=c_train,
        linewidth=2.4,
        label="Treeningkadu (ansambli liikmete keskmine)",
    )
    ax.plot(
        cls_progress_pct,
        cls_val_proxy,
        color=c_val,
        linewidth=2.2,
        linestyle="--",
        label="Valideerimise kaoproksy (1 - F1, keskmine)",
    )
    ax.set_title("Parim klassifitseerimismudel (lõplik ansambel)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Treeningu edenemine (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Kao väärtus", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=11,
        frameon=True,
    )

    fig.suptitle(
        "Treening- ja valideerimisgrupi kaofunktsiooni kõverad üle epohhide",
        fontsize=16,
        fontweight="bold",
    )
    sns.despine(trim=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joonista parima segmenteerimis- ja klassifitseerimismudeli treening/valideerimine kõverad."
    )
    p.add_argument("--seg-history", type=Path, default=SEG_HISTORY_JSONL)
    p.add_argument("--cls-log", type=Path, default=CLS_ENSEMBLE_LOG)
    p.add_argument("--output", type=Path, default=OUT_PNG)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    seg_epochs, seg_train, seg_val_proxy = load_seg_history(args.seg_history)
    cls_progress_pct, cls_train, cls_val_proxy = aggregate_ensemble_training_curves(args.cls_log)

    plot_curves(
        seg_epochs=seg_epochs,
        seg_train=seg_train,
        seg_val_proxy=seg_val_proxy,
        cls_progress_pct=cls_progress_pct,
        cls_train=cls_train,
        cls_val_proxy=cls_val_proxy,
        output_path=args.output,
    )

    print(f"Valmis: {args.output}")
    print(
        "Märkus: valideerimiskao otsest veergu logides ei olnud; "
        "kasutatud proksy `1 - val_F1`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

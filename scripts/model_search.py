#!/usr/bin/env python3
"""
Comprehensive CDW model search with leakage-aware validation and paper-ready reporting.

Pipeline summary
----------------
1) Analyze training data quality, class balance, source-bias, raster-bias, and pixel stats.
2) Build a staged model search:
   - Stage 1: broad baseline sweep across candidate architectures.
   - Stage 2: expand top models with data strategy / loss / regularization variants.
3) Select top-5 by CV F1 (mean, then variance).
4) Retrain top-5 on full non-test training data and evaluate once on held-out test set.
5) Evaluate top-5 soft-vote ensemble and (if feasible) stacking ensemble.
6) Save all artifacts for reproducibility and research writing.

Notes
-----
- Test split is never used for model or threshold selection.
- Spatial leakage is mitigated via grouped CV; optionally with coordinate-based spatial fences across rasters/years.
- Designed to run in the existing Lamapuit project with `fine_tune_cnn.py` helpers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc as _auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

# matplotlib backend must be set before pyplot import
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _import_fine_tune_helpers(script_path: Path):
    spec = importlib.util.spec_from_file_location("fine_tune_cnn", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "load_labels": mod._load_labels,
        "build_deep_cnn": mod._build_deep_cnn_attn_net,
        "norm_tile": mod._norm_tile,
    }


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[x >= 0]
    if len(x) == 0 or np.allclose(x.sum(), 0.0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))


@dataclass
class Record:
    raster: str
    row_off: int
    col_off: int
    chunk_size: int
    label: int
    source: str
    weight: float
    model_prob: float | None

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.raster, self.row_off, self.col_off)


@dataclass
class ExperimentConfig:
    experiment_id: str
    model_name: str
    data_strategy: str
    loss_name: str
    regularization: str
    epochs: int
    batch_size: int
    lr_head: float
    lr_backbone: float
    label_smoothing: float
    patience: int
    seed: int
    use_tta: bool = False


def _setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("model_search")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(output_dir / "model_search.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _write_progress_state(output_dir: Path, payload: dict) -> None:
    progress_path = output_dir / "progress.json"
    safe = {
        **payload,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    progress_path.write_text(json.dumps(safe, indent=2))


def _canonical_model_name(name: str) -> str:
    canon = name.strip().lower()
    aliases = {
        "resnet18": "resnet18",
        "resnet50": "resnet50",
        "resnet50v2": "resnet50v2",
        "resnet50_v2": "resnet50v2",
        "deep_cnn_attn_base": "deep_cnn_attn",
        "deep_cnn_attn_dropout": "deep_cnn_attn_dropout_tuned",
        "maxvit_small_tf_224": "maxvit_small",
        "maxvit_small": "maxvit_small",
        "convnext_v2_small": "convnextv2_small",
        "convnextv2_small": "convnextv2_small",
        "convnextv2-s": "convnextv2_small",
        "eva02_small": "eva02_small",
        "eva-02-small": "eva02_small",
        "eva02_small_patch14_224": "eva02_small",
    }
    return aliases.get(canon, canon)


def _clean_state_dict(state_dict: dict) -> dict:
    """Return a cleaned copy of a state_dict:
    - strip common DataParallel/module prefixes
    - drop obvious SWA/averaging metadata keys
    """
    if not isinstance(state_dict, dict):
        return state_dict

    out: dict = {}
    for k, v in state_dict.items():
        # drop SWA bookkeeping keys that sometimes appear inside state_dict
        if k in ("n_averaged", "param_averages"):
            continue

        nk = k
        # common wrappers
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]

        out[nk] = v
    return out


def _is_deprioritized_model(name: str) -> bool:
    m = _canonical_model_name(name)
    deprioritized_prefixes = ("mobilenet", "mnasnet", "shufflenet", "squeezenet")
    return any(m.startswith(prefix) for prefix in deprioritized_prefixes)


def _stable_experiment_id(prefix: str, *parts: str) -> str:
    clean = ["".join(ch if ch.isalnum() else "_" for ch in p.lower()) for p in parts]
    base = "_".join(x for x in clean if x)
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{base}_{digest}" if base else f"{prefix}_{digest}"


def _load_existing_experiment_summaries(output_dir: Path) -> dict[str, dict]:
    exp_dir = output_dir / "experiments"
    if not exp_dir.exists():
        return {}

    out: dict[str, dict] = {}
    for p in sorted(exp_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            continue
        exp_id = str(
            summary.get("experiment_id") or payload.get("config", {}).get("experiment_id") or p.stem
        )
        summary["experiment_id"] = exp_id
        out[exp_id] = summary
    return out


def _completed_stage1_models(existing_summaries: dict[str, dict]) -> set[str]:
    completed: set[str] = set()
    for exp_id, summary in existing_summaries.items():
        if not str(exp_id).startswith("s1_"):
            continue
        model_name = summary.get("model_name")
        if model_name:
            completed.add(_canonical_model_name(str(model_name)))
    return completed


def _maybe_reuse_fold_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_idx: np.ndarray,
    build_deep_cnn: Callable,
    device,
    use_tta_eval: bool = False,
):
    if not checkpoint_path.exists():
        return None

    import torch

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = state.get("state_dict")
        if not isinstance(state_dict, dict):
            return None

        net, _ = _build_model(model_name, build_deep_cnn, input_size=X_val.shape[-1])
        net = net.to(device)
        cleaned = _clean_state_dict(state_dict)
        res = net.load_state_dict(cleaned, strict=False)
        # log any mismatches for visibility
        lg = logging.getLogger("model_search")
        if hasattr(res, "missing_keys"):
            if res.missing_keys:
                lg.warning("Checkpoint load missing keys: %s", res.missing_keys)
            if res.unexpected_keys:
                lg.warning("Checkpoint load unexpected keys: %s", res.unexpected_keys)

        val_eval = _evaluate_classifier(net, X_val, y_val, device=device, tta=use_tta_eval)
        best = state.get("best_metrics", {}) if isinstance(state, dict) else {}
        return {
            "auc": float(best.get("auc", val_eval["auc"])),
            "f1": float(best.get("f1", val_eval["f1"])),
            "precision": float(best.get("precision", val_eval["precision"])),
            "recall": float(best.get("recall", val_eval["recall"])),
            "threshold": float(best.get("threshold", val_eval["threshold"])),
            "epoch": int(best.get("epoch", 0)),
            "val_probs": val_eval["probs"],
            "val_idx": val_idx,
        }
    except Exception:
        return None


def _load_records_with_probs(
    labels_dir: Path,
    source_weights: dict[str, float],
) -> list[Record]:
    records: list[Record] = []
    for csv_path in sorted(labels_dir.glob("*_labels.csv")):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                lbl = row.get("label", "")
                if lbl not in ("cdw", "no_cdw"):
                    continue
                src = row.get("source", "")
                w = float(source_weights.get(src, source_weights.get("", 0.75)))
                mp_raw = row.get("model_prob", "")
                mp = None
                if mp_raw not in (None, ""):
                    try:
                        mp = float(mp_raw)
                    except ValueError:
                        mp = None

                records.append(
                    Record(
                        raster=row["raster"],
                        row_off=int(row["row_off"]),
                        col_off=int(row["col_off"]),
                        chunk_size=int(row.get("chunk_size", 128)),
                        label=1 if lbl == "cdw" else 0,
                        source=src,
                        weight=w,
                        model_prob=mp,
                    )
                )
    return records


def _load_test_keys(test_split: Path | None) -> set[tuple[str, int, int]]:
    if test_split is None or not test_split.exists():
        return set()
    data = json.loads(test_split.read_text())
    return {(r, int(ro), int(co)) for r, ro, co in data.get("keys", [])}


def _build_arrays_with_meta(
    records: list[Record],
    chm_dir: Path,
    norm_tile: Callable[[np.ndarray], np.ndarray],
    canonical_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    by_raster: dict[str, list[Record]] = {}
    for rec in records:
        by_raster.setdefault(rec.raster, []).append(rec)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    w_list: list[float] = []
    meta: list[dict] = []

    for raster_name, recs in by_raster.items():
        raster_path = chm_dir / raster_name
        if not raster_path.exists():
            matches = list(chm_dir.glob(f"{Path(raster_name).stem}*"))
            if not matches:
                continue
            raster_path = matches[0]
        try:
            with rasterio.open(raster_path) as src:
                for rec in recs:
                    x_center = None
                    y_center = None
                    try:
                        center_row = int(rec.row_off + rec.chunk_size // 2)
                        center_col = int(rec.col_off + rec.chunk_size // 2)
                        x_center, y_center = src.xy(center_row, center_col, offset="center")
                    except Exception:
                        x_center = None
                        y_center = None

                    raw = src.read(
                        1,
                        window=Window(rec.col_off, rec.row_off, rec.chunk_size, rec.chunk_size),
                        boundless=True,
                        fill_value=0,
                    ).astype(np.float32)
                    if raw.shape != (canonical_size, canonical_size):
                        import cv2

                        raw = cv2.resize(raw, (canonical_size, canonical_size))

                    x = norm_tile(raw)
                    X_list.append(x)
                    y_list.append(rec.label)
                    w_list.append(rec.weight)
                    meta.append(
                        {
                            "raster": rec.raster,
                            "row_off": rec.row_off,
                            "col_off": rec.col_off,
                            "source": rec.source,
                            "model_prob": rec.model_prob,
                            "key": f"{rec.raster}|{rec.row_off}|{rec.col_off}",
                            "x_center": (float(x_center) if x_center is not None else None),
                            "y_center": (float(y_center) if y_center is not None else None),
                        }
                    )
        except Exception:
            continue

    if not X_list:
        raise RuntimeError("No tiles were loaded from CHM rasters.")

    X = np.array(X_list, dtype=np.float32)[:, np.newaxis, :, :]
    y = np.array(y_list, dtype=np.int64)
    w = np.array(w_list, dtype=np.float32)
    return X, y, w, meta


def _save_data_analysis(
    output_dir: Path,
    records_train: list[Record],
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: list[dict],
    logger: logging.Logger,
) -> dict:
    analysis_dir = output_dir / "data_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # class balance
    n_total = len(records_train)
    n_cdw = int(sum(r.label == 1 for r in records_train))
    n_no = n_total - n_cdw
    class_balance = {
        "n_total": n_total,
        "n_cdw": n_cdw,
        "n_no_cdw": n_no,
        "cdw_ratio": float(n_cdw / max(n_total, 1)),
        "imbalance_ratio_no_to_cdw": float(n_no / max(n_cdw, 1)),
    }
    (analysis_dir / "class_balance.json").write_text(json.dumps(class_balance, indent=2))

    # per-raster distribution
    rows = []
    by_raster: dict[str, list[Record]] = {}
    for r in records_train:
        by_raster.setdefault(r.raster, []).append(r)
    for raster, recs in sorted(by_raster.items()):
        tot = len(recs)
        cdw = sum(x.label == 1 for x in recs)
        no = tot - cdw
        rows.append(
            {
                "raster": raster,
                "total": tot,
                "cdw": cdw,
                "no_cdw": no,
                "cdw_ratio": cdw / max(tot, 1),
            }
        )
    per_raster_df = pd.DataFrame(rows)
    per_raster_df.to_csv(analysis_dir / "per_raster_stats.csv", index=False)

    # source distribution
    source_counts: dict[str, int] = {}
    for r in records_train:
        key = r.source if r.source else "<empty>"
        source_counts[key] = source_counts.get(key, 0) + 1
    source_payload = {
        "counts": source_counts,
        "fractions": {k: v / max(n_total, 1) for k, v in source_counts.items()},
    }
    (analysis_dir / "source_distribution.json").write_text(json.dumps(source_payload, indent=2))

    # pixel statistics
    px = X_train[:, 0, :, :]
    px_cdw = px[y_train == 1]
    px_no = px[y_train == 0]

    def _stats(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    nodata_fraction = (px <= 1e-6).mean(axis=(1, 2))
    pixel_stats = {
        "overall": _stats(px),
        "cdw": _stats(px_cdw),
        "no_cdw": _stats(px_no),
        "nodata_fraction": {
            "mean": float(nodata_fraction.mean()),
            "p90": float(np.quantile(nodata_fraction, 0.90)),
            "p95": float(np.quantile(nodata_fraction, 0.95)),
            "n_tiles_gt_50pct": int(np.sum(nodata_fraction > 0.50)),
        },
    }
    (analysis_dir / "pixel_stats.json").write_text(json.dumps(pixel_stats, indent=2))

    # suspect label candidates from model_prob disagreement
    suspect_rows = []
    for rec in records_train:
        if rec.model_prob is None:
            continue
        if rec.label == 1 and rec.model_prob <= 0.05:
            suspect_rows.append({**asdict(rec), "reason": "label_cdw_prob_very_low"})
        elif rec.label == 0 and rec.model_prob >= 0.95:
            suspect_rows.append({**asdict(rec), "reason": "label_no_cdw_prob_very_high"})
    pd.DataFrame(suspect_rows).to_csv(analysis_dir / "suspect_labels.csv", index=False)

    # charts
    plt.figure(figsize=(6, 4))
    plt.bar(["CDW", "No CDW"], [n_cdw, n_no])
    plt.title("Class Balance (Training only)")
    plt.tight_layout()
    plt.savefig(analysis_dir / "class_balance.png", dpi=140)
    plt.close()

    if not per_raster_df.empty:
        plt.figure(figsize=(10, 4))
        tmp = per_raster_df.sort_values("cdw_ratio", ascending=False)
        plt.bar(tmp["raster"], tmp["cdw_ratio"])
        plt.xticks(rotation=75, ha="right", fontsize=8)
        plt.ylabel("CDW Ratio")
        plt.title("Per-raster CDW ratio")
        plt.tight_layout()
        plt.savefig(analysis_dir / "per_raster_cdw.png", dpi=140)
        plt.close()

    if source_counts:
        plt.figure(figsize=(6, 4))
        labels = list(source_counts.keys())
        values = [source_counts[x] for x in labels]
        plt.bar(labels, values)
        plt.xticks(rotation=30, ha="right")
        plt.title("Label Source Distribution")
        plt.tight_layout()
        plt.savefig(analysis_dir / "source_weights.png", dpi=140)
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(px.flatten(), bins=50, alpha=0.6, label="all")
    if px_cdw.size > 0:
        plt.hist(px_cdw.flatten(), bins=50, alpha=0.4, label="cdw")
    if px_no.size > 0:
        plt.hist(px_no.flatten(), bins=50, alpha=0.4, label="no_cdw")
    plt.title("Pixel intensity distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "pixel_histograms.png", dpi=140)
    plt.close()

    gini_cdw = _gini(per_raster_df["cdw"].to_numpy() if not per_raster_df.empty else np.array([]))
    analysis_summary = {
        "class_balance": class_balance,
        "n_rasters": int(per_raster_df.shape[0]),
        "gini_cdw_per_raster": gini_cdw,
        "n_suspect_labels": int(len(suspect_rows)),
        "pixel_stats": pixel_stats,
    }
    (analysis_dir / "analysis_summary.json").write_text(json.dumps(analysis_summary, indent=2))
    logger.info(
        "Data analysis complete | n=%d, cdw=%d, no_cdw=%d, rasters=%d, gini=%.3f",
        n_total,
        n_cdw,
        n_no,
        analysis_summary["n_rasters"],
        gini_cdw,
    )
    return analysis_summary


def _set_module_by_name(root, module_name: str, new_module):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _adapt_first_conv_to_1ch(model) -> None:
    import torch.nn as nn

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
            )
            with np.errstate(all="ignore"):
                new_conv.weight.data = module.weight.data.mean(dim=1, keepdim=True)
            if module.bias is not None:
                new_conv.bias.data = module.bias.data.clone()
            _set_module_by_name(model, name, new_conv)
            return


def _replace_classifier_head(model, num_classes: int = 2):
    import torch.nn as nn

    if hasattr(model, "reset_classifier"):
        try:
            model.reset_classifier(num_classes=num_classes)
            return
        except TypeError:
            model.reset_classifier(num_classes)
            return
        except Exception:
            pass

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return
    if hasattr(model, "classifier"):
        cls = model.classifier
        if isinstance(cls, nn.Linear):
            model.classifier = nn.Linear(cls.in_features, num_classes)
            return
        if isinstance(cls, nn.Sequential):
            for i in range(len(cls) - 1, -1, -1):
                if isinstance(cls[i], nn.Linear):
                    cls[i] = nn.Linear(cls[i].in_features, num_classes)
                    return
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
        return
    raise RuntimeError("Unsupported model head replacement path")


def _build_deep_cnn_variant(name: str, build_deep_cnn: Callable):
    import torch.nn as nn

    m = _canonical_model_name(name)
    net = build_deep_cnn()
    if m == "deep_cnn_attn":
        return net

    if not hasattr(net, "head") or not isinstance(net.head, nn.Sequential):
        return net

    head = net.head
    first_linear = None
    for layer in head:
        if isinstance(layer, nn.Linear):
            first_linear = layer
            break
    if first_linear is None:
        return net

    in_features = int(first_linear.in_features)

    if m == "deep_cnn_attn_headlight":
        hidden = 256
        net.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(hidden, 2),
        )
        return net

    if m == "deep_cnn_attn_headwide":
        hidden = 768
        net.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(hidden, 2),
        )
        return net

    if m == "deep_cnn_attn_dropout_tuned":
        for layer in net.head:
            if isinstance(layer, nn.Dropout):
                layer.p = 0.35
        return net

    return net


def _build_model(name: str, build_deep_cnn: Callable, input_size: int | None = None):
    import torch
    import torch.nn as nn
    from torchvision import models as tvm

    def _try_timm(model_name: str):
        try:
            import timm
        except Exception as exc:
            raise RuntimeError(f"timm not available for model '{model_name}': {exc}") from exc

        # Pass explicit image size to timm when available so models that depend
        # on input resolution (window/partitioning) are configured correctly.
        kwargs: dict = {"in_chans": 1, "num_classes": 2}
        if input_size is not None:
            kwargs["img_size"] = (int(input_size), int(input_size))
        try:
            return timm.create_model(model_name, pretrained=True, **kwargs)
        except Exception:
            return timm.create_model(model_name, pretrained=False, **kwargs)

    def _try_timm_candidates(candidates: list[str]):
        last_exc = None
        for cand in candidates:
            try:
                return _try_timm(cand)
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(
            f"None of the timm candidates could be created: {candidates}; last_error={last_exc}"
        )

    name = _canonical_model_name(name)

    def _try_weights(factory, weights_attr: str | None = None):
        try:
            if weights_attr and hasattr(tvm, weights_attr):
                return factory(weights=getattr(tvm, weights_attr).DEFAULT)
            return factory(weights="DEFAULT")
        except Exception:
            return factory(weights=None)

    if name.startswith("deep_cnn_attn"):
        return _build_deep_cnn_variant(name, build_deep_cnn), []
    if name == "resnet18":
        m = _try_weights(tvm.resnet18, "ResNet18_Weights")
    elif name == "resnet34":
        m = _try_weights(tvm.resnet34, "ResNet34_Weights")
    elif name == "resnet50":
        m = _try_weights(tvm.resnet50, "ResNet50_Weights")
    elif name == "efficientnet_b0":
        m = _try_weights(tvm.efficientnet_b0, "EfficientNet_B0_Weights")
    elif name == "efficientnet_b2":
        m = _try_weights(tvm.efficientnet_b2, "EfficientNet_B2_Weights")
    elif name == "efficientnet_b4":
        m = _try_weights(tvm.efficientnet_b4, "EfficientNet_B4_Weights")
    elif name == "densenet121":
        m = _try_weights(tvm.densenet121, "DenseNet121_Weights")
    elif name == "mobilenet_v3_large":
        m = _try_weights(tvm.mobilenet_v3_large, "MobileNet_V3_Large_Weights")
    elif name == "convnext_tiny":
        m = _try_weights(tvm.convnext_tiny, "ConvNeXt_Tiny_Weights")
    elif name == "convnext_small":
        try:
            m = _try_weights(tvm.convnext_small, "ConvNeXt_Small_Weights")
        except Exception:
            m = _try_timm("convnext_small")
    elif name == "swin_t":
        m = _try_weights(tvm.swin_t, "Swin_T_Weights")
    elif name == "regnety_004":
        try:
            m = _try_weights(tvm.regnet_y_400mf, "RegNet_Y_400MF_Weights")
        except Exception:
            m = _try_timm("regnety_004")
    elif name == "maxvit_small":
        try:
            m = _try_timm_candidates(
                [
                    "maxvit_small_tf_224",
                    "maxvit_small_tf_224.in1k",
                    "maxvit_small_tf_224.in21k_ft_in1k",
                ]
            )
        except Exception:
            m = _try_weights(tvm.maxvit_t, "MaxVit_T_Weights")
    elif name == "convnextv2_small":
        m = _try_timm_candidates(
            [
                "convnextv2_small",
                "convnextv2_small.fcmae_ft_in22k_in1k",
            ]
        )
    elif name == "eva02_small":
        m = _try_timm_candidates(
            [
                "eva02_small_patch14_224.mim_in22k_ft_in1k",
                "eva02_small_patch14_224",
            ]
        )
    elif name == "resnet50v2":
        m = _try_timm("resnetv2_50")
    else:
        raise ValueError(f"Unknown model: {name}")

    _adapt_first_conv_to_1ch(m)
    _replace_classifier_head(m, 2)

    # param groups: lower LR for backbone, higher for head
    head_params = []
    if hasattr(m, "fc"):
        head_params += list(m.fc.parameters())
    if hasattr(m, "head"):
        head_params += list(m.head.parameters())
    if hasattr(m, "classifier"):
        head_params += list(m.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone = [p for p in m.parameters() if id(p) not in head_ids]
    param_group = [
        {"params": backbone, "lr": 1.0},
        {"params": head_params, "lr": 1.0},
    ]
    return m, param_group


class TileDataset:
    def __init__(
        self,
        X,
        y,
        w,
        augment: bool = False,
        augment_random_nodata_frac: float = 0.0,
        augment_pattern_nodata_frac: float = 0.0,
    ):
        import torch

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.w = torch.from_numpy(w.astype(np.float32))
        self.augment = augment
        self.augment_random_nodata_frac = float(max(0.0, min(1.0, augment_random_nodata_frac)))
        self.augment_pattern_nodata_frac = float(max(0.0, min(1.0, augment_pattern_nodata_frac)))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        import torch

        x = self.X[idx].clone()
        y = self.y[idx]
        w = self.w[idx]
        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [-1])
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [-2])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                x = torch.rot90(x, k, [-2, -1])
            if torch.rand(1).item() > 0.7:
                x = (x + torch.randn_like(x) * 0.015).clamp(0.0, 1.0)
            if torch.rand(1).item() > 0.80:
                alpha = 0.85 + torch.rand(1).item() * 0.30
                beta = (torch.rand(1).item() - 0.5) * 0.06
                x = (x * alpha + beta).clamp(0.0, 1.0)

            # Random nodata mask (e.g. 0.50 => 50% pixels set to nodata/0).
            if self.augment_random_nodata_frac > 0 and torch.rand(1).item() > 0.5:
                rnd_mask = torch.rand_like(x) < self.augment_random_nodata_frac
                x = x.masked_fill(rnd_mask, 0.0)

            # Repeating nodata pattern mask (0.75 uses 2x2 keep-one/drop-three pattern).
            if self.augment_pattern_nodata_frac > 0 and torch.rand(1).item() > 0.5:
                h, w_px = x.shape[-2], x.shape[-1]
                yy = torch.arange(h, device=x.device).view(-1, 1)
                xx = torch.arange(w_px, device=x.device).view(1, -1)

                oy = int(torch.randint(0, 2, (1,), device=x.device).item())
                ox = int(torch.randint(0, 2, (1,), device=x.device).item())
                keep2d = ((yy % 2) == oy) & ((xx % 2) == ox)  # keeps ~25%
                nodata2d = ~keep2d

                if self.augment_pattern_nodata_frac < 0.75:
                    scale = self.augment_pattern_nodata_frac / 0.75
                    nodata2d = nodata2d & (torch.rand_like(nodata2d.float()) < scale)

                pattern_mask = nodata2d.unsqueeze(0)
                x = x.masked_fill(pattern_mask, 0.0)
        return x, y, w


def _mixup_batch(xb, yb, wb, alpha: float = 0.3):
    import torch

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(xb.size(0), device=xb.device)
    return (
        lam * xb + (1.0 - lam) * xb[perm],
        yb,
        yb[perm],
        lam,
        lam * wb + (1.0 - lam) * wb[perm],
    )


def _cutmix_batch(xb, yb, wb, alpha: float = 1.0):
    import torch

    b, _, h, w = xb.shape
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(b, device=xb.device)

    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    xb2 = xb.clone()
    xb2[:, :, y1:y2, x1:x2] = xb[perm, :, y1:y2, x1:x2]
    lam_adj = 1.0 - ((x2 - x1) * (y2 - y1) / float(h * w))
    return xb2, yb, yb[perm], lam_adj, lam_adj * wb + (1.0 - lam_adj) * wb[perm]


def _forward_probs(net, xb, tta: bool = False):
    import torch

    if not tta:
        return torch.softmax(net(xb), dim=1)[:, 1]
    views = []
    for k in range(4):
        v = torch.rot90(xb, k, [-2, -1])
        views.append(torch.softmax(net(v), dim=1)[:, 1])
        views.append(torch.softmax(net(torch.flip(v, [-1])), dim=1)[:, 1])
    return torch.stack(views, dim=0).mean(dim=0)


def _evaluate_classifier(net, X, y, device, batch_size: int = 64, tta: bool = False) -> dict:
    import torch

    net.eval()
    X_t = torch.from_numpy(X).to(device)
    probs: list[float] = []

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            pb = _forward_probs(net, X_t[i : i + batch_size], tta=tta)
            probs.extend(pb.cpu().numpy().tolist())

    probs_arr = np.array(probs, dtype=np.float64)
    labels = y.astype(int)
    if len(np.unique(labels)) < 2:
        return {
            "auc": 0.5,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "threshold": 0.5,
            "probs": probs_arr,
        }

    auc_v = float(roc_auc_score(labels, probs_arr))
    best = {"f1": -1.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (probs_arr >= thr).astype(int)
        f1 = float(f1_score(labels, pred, zero_division=0))
        if f1 > best["f1"]:
            best = {
                "f1": f1,
                "threshold": float(thr),
                "precision": float(precision_score(labels, pred, zero_division=0)),
                "recall": float(recall_score(labels, pred, zero_division=0)),
            }

    out = {
        "auc": auc_v,
        **best,
        "probs": probs_arr,
    }
    return out


def _focal_loss_per_sample(logits, targets, class_weights, gamma: float = 2.0):
    import torch
    import torch.nn.functional as F

    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma) * ce


def _build_cv_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(y))
    try:
        return list(splitter.split(idx, y, groups=groups))
    except Exception:
        fallback = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(fallback.split(idx, y))


def _select_models_after_analysis(summary: dict, smoke_test: bool) -> list[str]:
    if smoke_test:
        return ["deep_cnn_attn"]

    base = [
        "deep_cnn_attn",
        "deep_cnn_attn_headlight",
        "deep_cnn_attn_headwide",
        "deep_cnn_attn_dropout_tuned",
        "resnet18",
        "efficientnet_b2",
        "densenet121",
        "convnext_small",
    ]
    if summary["class_balance"]["n_total"] >= 10000:
        base += ["resnet34", "convnext_tiny", "swin_t", "regnety_004", "resnet50v2"]
    if summary["gini_cdw_per_raster"] >= 0.45:
        base += ["resnet50", "efficientnet_b4"]
    # preserve order + uniqueness
    uniq = []
    seen = set()
    for x in base:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _strategy_indices(meta: list[dict], strategy: str) -> np.ndarray:
    if strategy == "full":
        return np.arange(len(meta))
    if strategy == "manual_only":
        return np.array(
            [i for i, m in enumerate(meta) if (m["source"] in ("manual", "auto_reviewed"))],
            dtype=np.int64,
        )
    if strategy == "balanced":
        return np.arange(len(meta))
    raise ValueError(f"Unknown data strategy: {strategy}")


def _spatial_fence_id(m: dict, block_size_px: int = 128, fence_m: float = 0.0) -> str:
    if fence_m > 0:
        x = m.get("x_center")
        y = m.get("y_center")
        if x is not None and y is not None:
            gx = int(math.floor(float(x) / float(fence_m)))
            gy = int(math.floor(float(y) / float(fence_m)))
            return f"xy|{gx}|{gy}"
    return f"px|{m['raster']}|{m['row_off']//block_size_px}|{m['col_off']//block_size_px}"


def _cv_group_id(m: dict, block_size_px: int = 128, block_m: float = 0.0) -> str:
    if block_m > 0:
        x = m.get("x_center")
        y = m.get("y_center")
        if x is not None and y is not None:
            gx = int(math.floor(float(x) / float(block_m)))
            gy = int(math.floor(float(y) / float(block_m)))
            return f"xy|{gx}|{gy}"
    return f"px|{m['raster']}|{m['row_off']//block_size_px}|{m['col_off']//block_size_px}"


def _make_groups(
    meta: list[dict],
    idx: np.ndarray,
    block_size: int = 128,
    cv_block_m: float = 0.0,
) -> np.ndarray:
    groups = []
    for i in idx:
        m = meta[int(i)]
        groups.append(_cv_group_id(m, block_size_px=block_size, block_m=cv_block_m))
    return np.array(groups)


def _parse_float_candidates(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = float(token)
        except Exception:
            continue
        if val > 0:
            out.append(val)
    uniq = []
    seen = set()
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        uniq.append(v)
    return uniq


def _probe_cv_block_size_m(
    y: np.ndarray,
    meta: list[dict],
    n_folds: int,
    seed: int,
    group_block_size_px: int,
    candidates_m: list[float],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []

    for cand_m in candidates_m:
        groups = _make_groups(
            meta,
            np.arange(len(meta), dtype=np.int64),
            block_size=group_block_size_px,
            cv_block_m=float(cand_m),
        )
        unique_groups = int(len(np.unique(groups)))

        row: dict[str, Any] = {
            "candidate_block_m": float(cand_m),
            "unique_groups": unique_groups,
            "ok": False,
        }

        if unique_groups < max(n_folds, 2):
            row["reason"] = "too_few_groups"
            row["score"] = 1e9
            rows.append(row)
            continue

        try:
            splits = _build_cv_splits(y, groups, n_splits=n_folds, seed=seed)
        except Exception as exc:
            row["reason"] = f"split_error:{exc}"
            row["score"] = 1e9
            rows.append(row)
            continue

        fold_sizes = []
        fold_pos_rates = []
        fold_pos_counts = []
        for _tr_idx, va_idx in splits:
            ys = y[va_idx]
            fold_sizes.append(int(len(ys)))
            pos_count = int(np.sum(ys == 1))
            fold_pos_counts.append(pos_count)
            fold_pos_rates.append(float(np.mean(ys == 1)) if len(ys) else 0.0)

        size_cv = float(np.std(fold_sizes) / max(np.mean(fold_sizes), 1.0))
        pos_rate_std = float(np.std(fold_pos_rates))
        min_pos = int(min(fold_pos_counts) if fold_pos_counts else 0)

        # Lower is better: balanced fold size + stable class ratio, while avoiding zero-positive folds.
        score = size_cv + pos_rate_std
        if min_pos <= 0:
            score += 1.0

        row.update(
            {
                "ok": True,
                "score": float(score),
                "size_cv": size_cv,
                "pos_rate_std": pos_rate_std,
                "min_pos_in_fold": min_pos,
                "fold_sizes": fold_sizes,
                "fold_pos_rates": fold_pos_rates,
            }
        )
        rows.append(row)

    valid = [r for r in rows if bool(r.get("ok"))]
    if not valid:
        return float(candidates_m[0]), rows

    valid.sort(key=lambda r: (float(r.get("score", 1e9)), -int(r.get("unique_groups", 0))))
    return float(valid[0]["candidate_block_m"]), rows


def _train_fold(
    cfg: ExperimentConfig,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    build_deep_cnn: Callable,
    checkpoint_path: Path,
    device,
    logger: logging.Logger | None = None,
    fold_tag: str | None = None,
    use_tta_eval: bool = False,
    augment_random_nodata_frac: float = 0.0,
    augment_pattern_nodata_frac: float = 0.0,
) -> dict:
    import torch
    from torch.optim.swa_utils import AveragedModel
    from torch.utils.data import DataLoader, WeightedRandomSampler

    _set_seed(cfg.seed)
    net, param_groups = _build_model(model_name, build_deep_cnn, input_size=X.shape[-1])
    net = net.to(device)

    x_tr = X[train_idx]
    y_tr = y[train_idx]
    w_tr = w[train_idx]
    x_va = X[val_idx]
    y_va = y[val_idx]

    n_neg = int(np.sum(y_tr == 0))
    n_pos = int(np.sum(y_tr == 1))
    w_pos = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, w_pos], dtype=torch.float32, device=device)

    if not param_groups:
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr_head, weight_decay=1e-4)
    else:
        param_groups[0]["lr"] = cfg.lr_backbone
        param_groups[1]["lr"] = cfg.lr_head
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    train_ds = TileDataset(
        x_tr,
        y_tr,
        w_tr,
        augment=True,
        augment_random_nodata_frac=augment_random_nodata_frac,
        augment_pattern_nodata_frac=augment_pattern_nodata_frac,
    )

    sampler = None
    if cfg.data_strategy == "balanced":
        class_inv = {0: 1.0 / max(np.sum(y_tr == 0), 1), 1: 1.0 / max(np.sum(y_tr == 1), 1)}
        sw = np.array([class_inv[int(lbl)] for lbl in y_tr], dtype=np.float64)
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
    )

    best_state = None
    best_metrics = {"f1": -1.0, "auc": 0.0, "threshold": 0.5, "epoch": 0}
    bad_epochs = 0

    use_swa = cfg.regularization == "mixup_swa"
    swa_model = AveragedModel(net) if use_swa else None
    log_every = max(1, cfg.epochs // 5)

    if logger is not None:
        logger.info(
            "[%s] training start | n_train=%d n_val=%d epochs=%d batch=%d strategy=%s loss=%s reg=%s",
            fold_tag or "fold",
            len(train_idx),
            len(val_idx),
            cfg.epochs,
            cfg.batch_size,
            cfg.data_strategy,
            cfg.loss_name,
            cfg.regularization,
        )

    for epoch in range(1, cfg.epochs + 1):
        net.train()
        epoch_loss = 0.0
        epoch_steps = 0
        for xb, yb, wb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device).float()

            optimizer.zero_grad()

            if cfg.regularization in ("mixup", "mixup_swa") and np.random.rand() < 0.5:
                xb2, ya, yb2, lam, wb2 = _mixup_batch(xb, yb, wb, alpha=0.3)
                logits = net(xb2)
                if cfg.loss_name == "focal":
                    l1 = _focal_loss_per_sample(logits, ya, class_weights)
                    l2 = _focal_loss_per_sample(logits, yb2, class_weights)
                else:
                    l1 = torch.nn.functional.cross_entropy(
                        logits,
                        ya,
                        weight=class_weights,
                        reduction="none",
                        label_smoothing=cfg.label_smoothing,
                    )
                    l2 = torch.nn.functional.cross_entropy(
                        logits,
                        yb2,
                        weight=class_weights,
                        reduction="none",
                        label_smoothing=cfg.label_smoothing,
                    )
                loss = (lam * l1 + (1.0 - lam) * l2) * wb2
            elif cfg.regularization == "cutmix" and np.random.rand() < 0.5:
                xb2, ya, yb2, lam, wb2 = _cutmix_batch(xb, yb, wb, alpha=1.0)
                logits = net(xb2)
                if cfg.loss_name == "focal":
                    l1 = _focal_loss_per_sample(logits, ya, class_weights)
                    l2 = _focal_loss_per_sample(logits, yb2, class_weights)
                else:
                    l1 = torch.nn.functional.cross_entropy(
                        logits,
                        ya,
                        weight=class_weights,
                        reduction="none",
                        label_smoothing=cfg.label_smoothing,
                    )
                    l2 = torch.nn.functional.cross_entropy(
                        logits,
                        yb2,
                        weight=class_weights,
                        reduction="none",
                        label_smoothing=cfg.label_smoothing,
                    )
                loss = (lam * l1 + (1.0 - lam) * l2) * wb2
            else:
                logits = net(xb)
                if cfg.loss_name == "focal":
                    loss = _focal_loss_per_sample(logits, yb, class_weights) * wb
                else:
                    loss = (
                        torch.nn.functional.cross_entropy(
                            logits,
                            yb,
                            weight=class_weights,
                            reduction="none",
                            label_smoothing=cfg.label_smoothing,
                        )
                        * wb
                    )

            loss.mean().backward()
            optimizer.step()
            epoch_loss += float(loss.mean().item())
            epoch_steps += 1

        scheduler.step()
        if use_swa and epoch >= (cfg.epochs - 10):
            swa_model.update_parameters(net)

        eval_model = swa_model if (use_swa and epoch >= (cfg.epochs - 10)) else net
        m = _evaluate_classifier(eval_model, x_va, y_va, device=device, tta=use_tta_eval)

        improved = (m["f1"] > best_metrics["f1"]) or (
            np.isclose(m["f1"], best_metrics["f1"]) and m["auc"] > best_metrics["auc"]
        )
        if improved:
            best_metrics = {
                "f1": float(m["f1"]),
                "auc": float(m["auc"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "threshold": float(m["threshold"]),
                "epoch": epoch,
            }
            state_src = swa_model if (use_swa and epoch >= (cfg.epochs - 10)) else net
            best_state = {k: v.detach().cpu().clone() for k, v in state_src.state_dict().items()}
            bad_epochs = 0
            if logger is not None and (epoch == 1 or epoch % log_every == 0 or epoch == cfg.epochs):
                logger.info(
                    "[%s] epoch %d/%d | loss=%.4f val_auc=%.4f val_f1=%.4f best_f1=%.4f",
                    fold_tag or "fold",
                    epoch,
                    cfg.epochs,
                    epoch_loss / max(epoch_steps, 1),
                    m["auc"],
                    m["f1"],
                    best_metrics["f1"],
                )
        else:
            bad_epochs += 1
            if logger is not None and (epoch == 1 or epoch % log_every == 0 or epoch == cfg.epochs):
                logger.info(
                    "[%s] epoch %d/%d | loss=%.4f val_auc=%.4f val_f1=%.4f no-improve=%d/%d",
                    fold_tag or "fold",
                    epoch,
                    cfg.epochs,
                    epoch_loss / max(epoch_steps, 1),
                    m["auc"],
                    m["f1"],
                    bad_epochs,
                    cfg.patience,
                )

        if bad_epochs >= cfg.patience:
            if logger is not None:
                logger.info(
                    "[%s] early stop at epoch %d (patience=%d), best_f1=%.4f @ epoch=%d",
                    fold_tag or "fold",
                    epoch,
                    cfg.patience,
                    best_metrics["f1"],
                    best_metrics["epoch"],
                )
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "model_name": model_name,
            "experiment": asdict(cfg),
            "best_metrics": best_metrics,
        },
        checkpoint_path,
    )

    cleaned_best = _clean_state_dict(best_state)
    res = net.load_state_dict(cleaned_best, strict=False)
    lg = logging.getLogger("model_search")
    if hasattr(res, "missing_keys"):
        if res.missing_keys:
            lg.warning("Best-state load missing keys: %s", res.missing_keys)
        if res.unexpected_keys:
            lg.warning("Best-state load unexpected keys: %s", res.unexpected_keys)
    val_eval = _evaluate_classifier(net, x_va, y_va, device=device, tta=use_tta_eval)
    if logger is not None:
        logger.info(
            "[%s] training done | best_auc=%.4f best_f1=%.4f thr=%.2f best_epoch=%d",
            fold_tag or "fold",
            best_metrics["auc"],
            best_metrics["f1"],
            best_metrics["threshold"],
            best_metrics["epoch"],
        )
    return {
        **best_metrics,
        "val_probs": val_eval["probs"],
        "val_idx": val_idx,
    }


def _run_cv_experiment(
    cfg: ExperimentConfig,
    X_all: np.ndarray,
    y_all: np.ndarray,
    w_all: np.ndarray,
    meta_all: list[dict],
    build_deep_cnn: Callable,
    output_dir: Path,
    n_folds: int,
    group_block_size_px: int,
    cv_spatial_block_m: float,
    device,
    logger: logging.Logger,
    resume: bool = True,
    augment_random_nodata_frac: float = 0.0,
    augment_pattern_nodata_frac: float = 0.0,
) -> dict:
    t0 = time.monotonic()

    idx = _strategy_indices(meta_all, cfg.data_strategy)
    if len(idx) < max(200, n_folds * 20):
        raise RuntimeError(f"Too few samples for strategy={cfg.data_strategy}: {len(idx)}")

    X = X_all[idx]
    y = y_all[idx]
    w = w_all[idx]
    meta = [meta_all[i] for i in idx]
    groups = _make_groups(
        meta,
        np.arange(len(meta)),
        block_size=group_block_size_px,
        cv_block_m=cv_spatial_block_m,
    )

    splits = _build_cv_splits(y, groups, n_splits=n_folds, seed=cfg.seed)
    fold_rows = []
    oof_rows = []

    logger.info(
        "Experiment %s start | model=%s strategy=%s loss=%s reg=%s tta=%s | n=%d folds=%d",
        cfg.experiment_id,
        cfg.model_name,
        cfg.data_strategy,
        cfg.loss_name,
        cfg.regularization,
        cfg.use_tta,
        len(idx),
        len(splits),
    )

    for fold_i, (tr_idx, va_idx) in enumerate(splits, start=1):
        logger.info(
            "Experiment %s | fold %d/%d start (train=%d val=%d)",
            cfg.experiment_id,
            fold_i,
            len(splits),
            len(tr_idx),
            len(va_idx),
        )
        ckpt = output_dir / "checkpoints" / cfg.experiment_id / f"fold{fold_i}.pt"
        fold_metrics = None
        if resume:
            fold_metrics = _maybe_reuse_fold_checkpoint(
                checkpoint_path=ckpt,
                model_name=cfg.model_name,
                X_val=X[va_idx],
                y_val=y[va_idx],
                val_idx=va_idx,
                build_deep_cnn=build_deep_cnn,
                device=device,
                use_tta_eval=cfg.use_tta,
            )
            if fold_metrics is not None:
                logger.info(
                    "Experiment %s | fold %d/%d reused from checkpoint",
                    cfg.experiment_id,
                    fold_i,
                    len(splits),
                )

        if fold_metrics is None:
            fold_metrics = _train_fold(
                cfg=cfg,
                model_name=cfg.model_name,
                X=X,
                y=y,
                w=w,
                train_idx=tr_idx,
                val_idx=va_idx,
                build_deep_cnn=build_deep_cnn,
                checkpoint_path=ckpt,
                device=device,
                logger=logger,
                fold_tag=f"{cfg.experiment_id}:fold{fold_i}/{len(splits)}",
                use_tta_eval=cfg.use_tta,
                augment_random_nodata_frac=augment_random_nodata_frac,
                augment_pattern_nodata_frac=augment_pattern_nodata_frac,
            )
        logger.info(
            "Experiment %s | fold %d/%d done | auc=%.4f f1=%.4f p=%.4f r=%.4f thr=%.2f",
            cfg.experiment_id,
            fold_i,
            len(splits),
            fold_metrics["auc"],
            fold_metrics["f1"],
            fold_metrics["precision"],
            fold_metrics["recall"],
            fold_metrics["threshold"],
        )
        fold_rows.append(
            {
                "experiment_id": cfg.experiment_id,
                "fold": fold_i,
                "auc": fold_metrics["auc"],
                "f1": fold_metrics["f1"],
                "precision": fold_metrics["precision"],
                "recall": fold_metrics["recall"],
                "threshold": fold_metrics["threshold"],
                "best_epoch": fold_metrics["epoch"],
            }
        )
        for local_idx, prob in zip(fold_metrics["val_idx"], fold_metrics["val_probs"]):
            m = meta[int(local_idx)]
            oof_rows.append(
                {
                    "key": m["key"],
                    "label": int(y[int(local_idx)]),
                    "prob": float(prob),
                    "experiment_id": cfg.experiment_id,
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    elapsed = time.monotonic() - t0

    summary = {
        **asdict(cfg),
        "n_samples": int(len(idx)),
        "mean_cv_auc": float(fold_df["auc"].mean()),
        "std_cv_auc": float(fold_df["auc"].std(ddof=0)),
        "mean_cv_f1": float(fold_df["f1"].mean()),
        "std_cv_f1": float(fold_df["f1"].std(ddof=0)),
        "mean_cv_precision": float(fold_df["precision"].mean()),
        "mean_cv_recall": float(fold_df["recall"].mean()),
        "best_fold_f1": float(fold_df["f1"].max()),
        "worst_fold_f1": float(fold_df["f1"].min()),
        "mean_threshold": float(fold_df["threshold"].mean()),
        "training_time_s": float(elapsed),
    }

    exp_dir = output_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / f"{cfg.experiment_id}.json").write_text(
        json.dumps({"config": asdict(cfg), "summary": summary, "folds": fold_rows}, indent=2)
    )
    pd.DataFrame(oof_rows).to_csv(exp_dir / f"{cfg.experiment_id}_oof.csv", index=False)
    logger.info(
        "Experiment %s done | model=%s strategy=%s loss=%s reg=%s | CV F1=%.4f±%.4f",
        cfg.experiment_id,
        cfg.model_name,
        cfg.data_strategy,
        cfg.loss_name,
        cfg.regularization,
        summary["mean_cv_f1"],
        summary["std_cv_f1"],
    )
    return summary


def _final_train_and_test_eval(
    top_rows: pd.DataFrame,
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    w_train_all: np.ndarray,
    meta_train_all: list[dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: list[dict],
    build_deep_cnn: Callable,
    output_dir: Path,
    device,
    logger: logging.Logger,
    augment_random_nodata_frac: float = 0.0,
    augment_pattern_nodata_frac: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    import torch

    final_dir = output_dir / "final_models"
    final_dir.mkdir(parents=True, exist_ok=True)

    results = []
    test_probs_by_model: dict[str, np.ndarray] = {}
    test_preds_by_model: dict[str, np.ndarray] = {}

    total_final = len(top_rows)
    for rank_i, (_, row) in enumerate(top_rows.iterrows(), start=1):
        cfg = ExperimentConfig(
            experiment_id=f"final_{row['experiment_id']}",
            model_name=row["model_name"],
            data_strategy=row["data_strategy"],
            loss_name=row["loss_name"],
            regularization=row["regularization"],
            epochs=int(row["epochs"]),
            batch_size=int(row["batch_size"]),
            lr_head=float(row["lr_head"]),
            lr_backbone=float(row["lr_backbone"]),
            label_smoothing=float(row["label_smoothing"]),
            patience=int(row["patience"]),
            seed=int(row["seed"]),
            use_tta=bool(row.get("use_tta", False)),
        )

        idx = _strategy_indices(meta_train_all, cfg.data_strategy)
        Xs, ys, ws = X_train_all[idx], y_train_all[idx], w_train_all[idx]
        tr_idx, va_idx = train_test_split(
            np.arange(len(Xs)),
            test_size=0.10,
            random_state=cfg.seed,
            stratify=ys,
        )

        ckpt_path = (
            final_dir
            / f"{cfg.model_name}_{cfg.data_strategy}_{cfg.loss_name}_{cfg.regularization}.pt"
        )
        logger.info(
            "Final retrain %d/%d | source=%s model=%s strategy=%s loss=%s reg=%s",
            rank_i,
            total_final,
            row["experiment_id"],
            cfg.model_name,
            cfg.data_strategy,
            cfg.loss_name,
            cfg.regularization,
        )

        fit = _train_fold(
            cfg=cfg,
            model_name=cfg.model_name,
            X=Xs,
            y=ys,
            w=ws,
            train_idx=tr_idx,
            val_idx=va_idx,
            build_deep_cnn=build_deep_cnn,
            checkpoint_path=ckpt_path,
            device=device,
            logger=logger,
            fold_tag=f"final:{rank_i}/{total_final}",
            use_tta_eval=cfg.use_tta,
            augment_random_nodata_frac=augment_random_nodata_frac,
            augment_pattern_nodata_frac=augment_pattern_nodata_frac,
        )

        # Load best state and evaluate on test with TTA
        net, _ = _build_model(cfg.model_name, build_deep_cnn, input_size=Xs.shape[-1])
        net = net.to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        cleaned = _clean_state_dict(state.get("state_dict", {}))
        res = net.load_state_dict(cleaned, strict=False)
        lg = logging.getLogger("model_search")
        if hasattr(res, "missing_keys"):
            if res.missing_keys:
                lg.warning("Final load missing keys: %s", res.missing_keys)
            if res.unexpected_keys:
                lg.warning("Final load unexpected keys: %s", res.unexpected_keys)
        m_test = _evaluate_classifier(net, X_test, y_test, device=device, tta=True)

        thr = float(fit["threshold"])
        pred = (m_test["probs"] >= thr).astype(int)
        cm = confusion_matrix(y_test, pred).tolist()
        res = {
            "model_id": ckpt_path.stem,
            "experiment_source": row["experiment_id"],
            "model_name": cfg.model_name,
            "data_strategy": cfg.data_strategy,
            "loss_name": cfg.loss_name,
            "regularization": cfg.regularization,
            "test_auc": float(m_test["auc"]),
            "test_f1": float(f1_score(y_test, pred, zero_division=0)),
            "test_precision": float(precision_score(y_test, pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, pred, zero_division=0)),
            "threshold_from_val": thr,
            "confusion_matrix": cm,
        }
        results.append(res)

        key = res["model_id"]
        test_probs_by_model[key] = m_test["probs"]
        test_preds_by_model[key] = pred
        logger.info(
            "Final test %s | AUC=%.4f F1=%.4f P=%.4f R=%.4f @thr=%.2f",
            key,
            res["test_auc"],
            res["test_f1"],
            res["test_precision"],
            res["test_recall"],
            thr,
        )

    return pd.DataFrame(results), test_probs_by_model, test_preds_by_model


def _soft_vote_ensemble(
    test_probs_by_model: dict[str, np.ndarray], thresholds: list[float], y_test: np.ndarray
):
    probs = np.mean(np.stack(list(test_probs_by_model.values()), axis=0), axis=0)
    thr = float(np.mean(thresholds)) if thresholds else 0.5
    pred = (probs >= thr).astype(int)
    return {
        "test_auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.5,
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "threshold": thr,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "probs": probs,
    }


def _stacking_ensemble(
    exp_dir: Path,
    top_experiment_ids: list[str],
    top_model_ids: list[str],
    y_test: np.ndarray,
    test_probs_by_model: dict[str, np.ndarray],
) -> dict | None:
    oof_tables = []
    for exp_id in top_experiment_ids:
        p = exp_dir / f"{exp_id}_oof.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        df = df[["key", "label", "prob"]].rename(columns={"prob": f"prob_{exp_id}"})
        oof_tables.append(df)

    merged = oof_tables[0]
    for df in oof_tables[1:]:
        merged = merged.merge(df, on=["key", "label"], how="inner")
    if len(merged) < 200:
        return None

    feature_cols = [c for c in merged.columns if c.startswith("prob_")]
    X_meta = merged[feature_cols].to_numpy()
    y_meta = merged["label"].to_numpy().astype(int)

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_meta, y_meta)

    test_feat = np.column_stack([test_probs_by_model[mid] for mid in top_model_ids])
    test_prob = clf.predict_proba(test_feat)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_meta, clf.predict_proba(X_meta)[:, 1])
    f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    if len(thresholds) > 0:
        thr = float(thresholds[int(np.nanargmax(f1s[:-1]))])
    else:
        thr = 0.5

    test_pred = (test_prob >= thr).astype(int)
    return {
        "test_auc": float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else 0.5,
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "threshold": thr,
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }


def _write_research_report(
    output_dir: Path,
    analysis_summary: dict,
    experiment_summary_df: pd.DataFrame,
    final_df: pd.DataFrame,
    ensemble_row: dict | None,
    stacking_row: dict | None,
) -> None:
    def _table_text(df: pd.DataFrame) -> str:
        if df.empty:
            return "No rows."
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_csv(index=False)

    top10 = experiment_summary_df.sort_values(
        ["mean_cv_f1", "std_cv_f1"], ascending=[False, True]
    ).head(10)

    lines = []
    lines.append("# CDW Model Search Research Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Data Summary")
    cb = analysis_summary["class_balance"]
    lines.append(f"- Training tiles: {cb['n_total']}")
    lines.append(f"- CDW: {cb['n_cdw']} ({cb['cdw_ratio']:.2%})")
    lines.append(f"- No-CDW: {cb['n_no_cdw']} ({1.0 - cb['cdw_ratio']:.2%})")
    lines.append(f"- No/CDW imbalance ratio: {cb['imbalance_ratio_no_to_cdw']:.2f}")
    lines.append(f"- Raster CDW concentration Gini: {analysis_summary['gini_cdw_per_raster']:.3f}")
    lines.append(f"- Suspect labels: {analysis_summary['n_suspect_labels']}")
    lines.append("")
    lines.append("## Top-10 CV Experiments")
    lines.append("")
    lines.append(_table_text(top10))
    lines.append("")
    lines.append("## Final Test Results (Top-5 retrained)")
    lines.append("")
    lines.append(_table_text(final_df) if not final_df.empty else "No final results.")
    lines.append("")
    if ensemble_row:
        lines.append("## Soft-vote Ensemble")
        lines.append("")
        lines.append(json.dumps(ensemble_row, indent=2))
        lines.append("")
    if stacking_row:
        lines.append("## Stacking Ensemble")
        lines.append("")
        lines.append(json.dumps(stacking_row, indent=2))
        lines.append("")

    (output_dir / "RESEARCH_REPORT.md").write_text("\n".join(lines))


def _run_raster_loocv(
    top_rows: pd.DataFrame,
    X_all: np.ndarray,
    y_all: np.ndarray,
    w_all: np.ndarray,
    meta_all: list[dict],
    build_deep_cnn: Callable,
    output_dir: Path,
    device,
    logger: logging.Logger,
) -> None:
    rasters = sorted({m["raster"] for m in meta_all})
    rows = []
    logger.info("Raster LOOCV start | top_models=%d rasters=%d", len(top_rows), len(rasters))
    for _, row in top_rows.iterrows():
        model = row["model_name"]
        cfg = ExperimentConfig(
            experiment_id=f"rastercv_{row['experiment_id']}",
            model_name=model,
            data_strategy="full",
            loss_name=row["loss_name"],
            regularization=row["regularization"],
            epochs=max(10, int(row["epochs"] // 2)),
            batch_size=int(row["batch_size"]),
            lr_head=float(row["lr_head"]),
            lr_backbone=float(row["lr_backbone"]),
            label_smoothing=float(row["label_smoothing"]),
            patience=max(4, int(row["patience"] // 2)),
            seed=int(row["seed"]),
            use_tta=False,
        )
        for raster in rasters:
            train_idx = np.array([i for i, m in enumerate(meta_all) if m["raster"] != raster])
            val_idx = np.array([i for i, m in enumerate(meta_all) if m["raster"] == raster])
            if len(val_idx) < 30 or len(np.unique(y_all[val_idx])) < 2:
                continue
            ckpt = output_dir / "checkpoints" / "raster_loocv" / f"{cfg.model_name}_{raster}.pt"
            logger.info(
                "Raster LOOCV | model=%s raster=%s start (train=%d val=%d)",
                cfg.model_name,
                raster,
                len(train_idx),
                len(val_idx),
            )
            metrics = _train_fold(
                cfg=cfg,
                model_name=cfg.model_name,
                X=X_all,
                y=y_all,
                w=w_all,
                train_idx=train_idx,
                val_idx=val_idx,
                build_deep_cnn=build_deep_cnn,
                checkpoint_path=ckpt,
                device=device,
                logger=logger,
                fold_tag=f"raster_loocv:{cfg.model_name}:{raster}",
            )
            rows.append(
                {
                    "model_name": cfg.model_name,
                    "source_experiment": row["experiment_id"],
                    "held_out_raster": raster,
                    "auc": metrics["auc"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                }
            )
            logger.info("Raster-LOOCV %s | %s | F1=%.4f", cfg.model_name, raster, metrics["f1"])

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "raster_cv_results.csv", index=False)
        logger.info("Raster LOOCV done | rows=%d", len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive CDW model search with leakage-safe validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labels", default="output/tile_labels", help="Directory with *_labels.csv"
    )
    parser.add_argument("--chm-dir", default="chm_max_hag", help="Directory with CHM rasters")
    parser.add_argument("--test-split", default="output/tile_labels/cnn_test_split.json")
    parser.add_argument("--output", default="output/model_search", help="Output root directory")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-scratch", type=int, default=60)
    parser.add_argument("--epochs-pretrained", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--canonical-size",
        type=int,
        default=128,
        help="Canonical tile size in pixels used when resizing tiles before training",
    )
    parser.add_argument(
        "--only-model",
        default="",
        help="If set, run Stage 1 only for this model name (canonicalized)",
    )
    parser.add_argument(
        "--cv-group-block-size",
        type=int,
        default=128,
        help="Spatial group block size in pixels when coordinate fence is unavailable",
    )
    parser.add_argument(
        "--spatial-fence-m",
        type=float,
        default=0.0,
        help="If >0, enforce train/test spatial fence in meters and use coordinate-based CV grouping",
    )
    parser.add_argument(
        "--cv-spatial-block-m",
        type=float,
        default=0.0,
        help="CV block size in meters. If 0, falls back to --spatial-fence-m (or pixel blocks).",
    )
    parser.add_argument(
        "--auto-cv-block-size",
        action="store_true",
        help="Probe candidate CV block sizes in meters and choose the best-balanced folds.",
    )
    parser.add_argument(
        "--cv-block-candidates-m",
        default="26,39,52,78,104",
        help="Comma-separated candidate CV block sizes (meters) used with --auto-cv-block-size",
    )
    parser.add_argument(
        "--augment-random-nodata-frac",
        type=float,
        default=0.0,
        help="Random nodata augmentation fraction (e.g. 0.50 means 50% random pixels set to nodata)",
    )
    parser.add_argument(
        "--augment-pattern-nodata-frac",
        type=float,
        default=0.0,
        help="Repeating-pattern nodata augmentation fraction (0.75 ~= keep-one/drop-three pattern)",
    )
    parser.add_argument("--top-k-expand", type=int, default=8)
    parser.add_argument("--top-k-final", type=int, default=5)
    parser.add_argument("--max-extended", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--run-raster-loocv", action="store_true", help="Run raster-level LOOCV for top models"
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny smoke pipeline")
    parser.add_argument("--add-models", default="", help="Comma-separated extra model names")
    parser.add_argument(
        "--stage2-pilot", action="store_true", help="Run focused Stage 2 pilot only"
    )
    parser.add_argument(
        "--stage2-pilot-top-models",
        type=int,
        default=4,
        help="Top eligible models for Stage 2 pilot",
    )
    parser.add_argument(
        "--stage2-keep-models",
        default="deep_cnn_attn,convnext_small,convnext_tiny,efficientnet_b2,densenet121,swin_t",
        help="Comma-separated focused Stage 2 model set",
    )
    parser.add_argument(
        "--stage2-strategies",
        default="full,focal,mixup_swa,tta",
        help="Comma-separated Stage 2 strategies from: full,focal,mixup_swa,tta",
    )
    parser.add_argument(
        "--stage2-epochs", type=int, default=60, help="Epochs for Stage 2 experiments"
    )
    parser.add_argument(
        "--stage2-patience", type=int, default=10, help="Patience for Stage 2 experiments"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    logger = _setup_logging(output_dir)
    t0 = time.monotonic()

    try:
        _write_progress_state(
            output_dir,
            {
                "status": "starting",
                "stage": "bootstrap",
                "message": "Initializing model search",
            },
        )

        _set_seed(args.seed)
        logger.info("Starting comprehensive model search")

        fine_tune_path = Path(__file__).parent / "fine_tune_cnn.py"
        helpers = _import_fine_tune_helpers(fine_tune_path)
        load_labels = helpers["load_labels"]
        build_deep_cnn = helpers["build_deep_cnn"]
        norm_tile = helpers["norm_tile"]

        labels_dir = Path(args.labels)
        chm_dir = Path(args.chm_dir)
        test_split = Path(args.test_split) if args.test_split else None

        # load labels from both helper and full parser to preserve compatibility + model_prob support
        source_weights = {
            "manual": 1.00,
            "auto_reviewed": 0.85,
            "": 0.75,
            "auto": 0.60,
            "auto_skip": 0.30,
        }
        _legacy_records, _ = load_labels(labels_dir)
        records = _load_records_with_probs(labels_dir, source_weights)
        if not records:
            raise RuntimeError("No valid labeled records found.")

        test_keys = _load_test_keys(test_split)
        rec_train = [r for r in records if r.key not in test_keys]
        rec_test = [r for r in records if r.key in test_keys]

        if len(rec_test) == 0:
            logger.warning(
                "No test records matched test-split keys. Final test evaluation will be skipped."
            )

        # leakage assertion
        overlap = set(r.key for r in rec_train).intersection(set(r.key for r in rec_test))
        if overlap:
            raise RuntimeError(f"Leakage detected: {len(overlap)} overlapping train/test keys")

        logger.info("Records loaded | train=%d test=%d", len(rec_train), len(rec_test))
        _write_progress_state(
            output_dir,
            {
                "status": "running",
                "stage": "data_loading",
                "n_train_records": len(rec_train),
                "n_test_records": len(rec_test),
                "message": "Records loaded and leakage check passed",
            },
        )

        X_train, y_train, w_train, meta_train = _build_arrays_with_meta(
            rec_train, chm_dir, norm_tile, canonical_size=args.canonical_size
        )
        X_test = y_test = w_test = None
        meta_test: list[dict] = []
        if rec_test:
            X_test, y_test, w_test, meta_test = _build_arrays_with_meta(
                rec_test, chm_dir, norm_tile, canonical_size=args.canonical_size
            )

        if args.spatial_fence_m > 0 and meta_test:
            test_fences = {
                _spatial_fence_id(
                    m,
                    block_size_px=args.cv_group_block_size,
                    fence_m=args.spatial_fence_m,
                )
                for m in meta_test
            }

            keep_mask = np.array(
                [
                    _spatial_fence_id(
                        m,
                        block_size_px=args.cv_group_block_size,
                        fence_m=args.spatial_fence_m,
                    )
                    not in test_fences
                    for m in meta_train
                ],
                dtype=bool,
            )

            dropped = int((~keep_mask).sum())
            if dropped > 0:
                dropped_keys = {meta_train[i]["key"] for i in np.where(~keep_mask)[0]}
                rec_train = [r for r in rec_train if r.key not in dropped_keys]
                X_train = X_train[keep_mask]
                y_train = y_train[keep_mask]
                w_train = w_train[keep_mask]
                meta_train = [m for i, m in enumerate(meta_train) if keep_mask[i]]
                logger.info(
                    "Spatial fence applied | fence_m=%.2f dropped_train_tiles=%d remaining_train=%d",
                    args.spatial_fence_m,
                    dropped,
                    len(meta_train),
                )

            train_fences = {
                _spatial_fence_id(
                    m,
                    block_size_px=args.cv_group_block_size,
                    fence_m=args.spatial_fence_m,
                )
                for m in meta_train
            }
            shared_fences = train_fences.intersection(test_fences)
            if shared_fences:
                raise RuntimeError(
                    f"Spatial leakage detected after fence filtering: {len(shared_fences)} shared fence cells"
                )

        # CV grouping block size is independent from train/test leakage fence.
        cv_spatial_block_m = (
            float(args.cv_spatial_block_m)
            if float(args.cv_spatial_block_m) > 0
            else (float(args.spatial_fence_m) if float(args.spatial_fence_m) > 0 else 0.0)
        )
        cv_probe_rows: list[dict[str, Any]] = []
        if args.auto_cv_block_size:
            candidates_m = _parse_float_candidates(args.cv_block_candidates_m)
            if candidates_m:
                chosen_m, cv_probe_rows = _probe_cv_block_size_m(
                    y=y_train,
                    meta=meta_train,
                    n_folds=(2 if args.smoke_test else args.n_folds),
                    seed=args.seed,
                    group_block_size_px=args.cv_group_block_size,
                    candidates_m=candidates_m,
                )
                cv_spatial_block_m = float(chosen_m)
                logger.info(
                    "CV block probe | selected_block_m=%.2f candidates=%s",
                    cv_spatial_block_m,
                    ",".join(str(v) for v in candidates_m),
                )

                (output_dir / "cv_block_probe.json").write_text(
                    json.dumps(
                        {
                            "selected_block_m": cv_spatial_block_m,
                            "candidates_m": candidates_m,
                            "rows": cv_probe_rows,
                        },
                        indent=2,
                    )
                )

        analysis_summary = _save_data_analysis(
            output_dir, rec_train, X_train, y_train, meta_train, logger
        )
        _write_progress_state(
            output_dir,
            {
                "status": "running",
                "stage": "analysis",
                "message": "Data analysis complete",
                "analysis_summary": analysis_summary,
            },
        )

        selected_models = _select_models_after_analysis(analysis_summary, args.smoke_test)
        if args.add_models.strip():
            extras = [_canonical_model_name(m) for m in args.add_models.split(",") if m.strip()]
            selected_models = list(dict.fromkeys(selected_models + extras))

        selected_models = [_canonical_model_name(m) for m in selected_models]

        if args.only_model and args.only_model.strip():
            only_m = _canonical_model_name(args.only_model.strip())
            logger.info("Only-model mode enabled; restricting selected models to: %s", only_m)
            selected_models = [only_m]

        existing_summaries = _load_existing_experiment_summaries(output_dir)
        completed_stage1_models = _completed_stage1_models(existing_summaries)

        filtered_models: list[str] = []
        for m in selected_models:
            if _is_deprioritized_model(m) and m not in completed_stage1_models:
                logger.info(
                    "Skipping deprioritized model not yet completed in Stage 1: %s",
                    m,
                )
                continue
            filtered_models.append(m)
        selected_models = filtered_models

        # filter models that can be built in this environment
        available = []
        for m in selected_models:
            try:
                _build_model(m, build_deep_cnn, input_size=args.canonical_size)
                available.append(m)
            except Exception as exc:
                logger.warning("Skipping model %s (unavailable): %s", m, exc)

        if not available:
            raise RuntimeError("No model architectures are available in this environment")

        logger.info("Model candidates after data analysis: %s", available)
        _write_progress_state(
            output_dir,
            {
                "status": "running",
                "stage": "model_selection",
                "message": "Model candidates selected",
                "models": available,
            },
        )

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", device)

        summary_by_id: dict[str, dict] = dict(existing_summaries)
        if summary_by_id:
            logger.info("Resume discovery | existing completed experiments=%d", len(summary_by_id))

        # Stage 1 baseline sweep
        logger.info("Stage 1: baseline sweep")
        n_stage1 = len(available)
        for i, model_name in enumerate(available, start=1):
            logger.info("Stage 1 progress %d/%d | model=%s", i, n_stage1, model_name)
            _write_progress_state(
                output_dir,
                {
                    "status": "running",
                    "stage": "stage1",
                    "stage_progress": {"current": i, "total": n_stage1},
                    "current_model": model_name,
                    "message": "Running Stage 1 baseline experiment",
                },
            )
            is_pretrained = not _canonical_model_name(model_name).startswith("deep_cnn_attn")
            stage1_id = _stable_experiment_id(
                "s1",
                model_name,
                "full",
                "ce",
                "mixup",
                str(
                    3
                    if args.smoke_test
                    else (args.epochs_pretrained if is_pretrained else args.epochs_scratch)
                ),
                str(2 if args.smoke_test else args.n_folds),
            )
            legacy_stage1_id = f"s1_{i:02d}_{model_name}"
            if stage1_id in summary_by_id or legacy_stage1_id in summary_by_id:
                logger.info(
                    "Stage 1 skip | model=%s completed (id=%s or legacy=%s)",
                    model_name,
                    stage1_id,
                    legacy_stage1_id,
                )
                continue

            cfg = ExperimentConfig(
                experiment_id=stage1_id,
                model_name=model_name,
                data_strategy="full",
                loss_name="ce",
                regularization="mixup",
                epochs=(
                    3
                    if args.smoke_test
                    else (args.epochs_pretrained if is_pretrained else args.epochs_scratch)
                ),
                batch_size=8 if args.smoke_test else args.batch_size,
                lr_head=5e-4,
                lr_backbone=5e-5 if is_pretrained else 5e-4,
                label_smoothing=0.05,
                patience=2 if args.smoke_test else args.patience,
                seed=args.seed + i,
                use_tta=False,
            )
            try:
                summary = _run_cv_experiment(
                    cfg=cfg,
                    X_all=X_train,
                    y_all=y_train,
                    w_all=w_train,
                    meta_all=meta_train,
                    build_deep_cnn=build_deep_cnn,
                    output_dir=output_dir,
                    n_folds=2 if args.smoke_test else args.n_folds,
                    group_block_size_px=args.cv_group_block_size,
                    cv_spatial_block_m=cv_spatial_block_m,
                    device=device,
                    logger=logger,
                    resume=True,
                    augment_random_nodata_frac=args.augment_random_nodata_frac,
                    augment_pattern_nodata_frac=args.augment_pattern_nodata_frac,
                )
                summary_by_id[summary["experiment_id"]] = summary
            except Exception as exc:
                logger.warning("Stage 1 experiment failed (%s): %s", stage1_id, exc)

        logger.info("Stage 1 complete")

        stage1_summaries = [
            s for s in summary_by_id.values() if str(s.get("experiment_id", "")).startswith("s1_")
        ]
        stage1_df = pd.DataFrame(stage1_summaries)
        stage1_sorted = stage1_df.sort_values(["mean_cv_f1", "std_cv_f1"], ascending=[False, True])
        top_models = (
            stage1_sorted["model_name"].head(min(args.top_k_expand, len(stage1_sorted))).tolist()
        )

        # Stage 2 focused matrix
        if not args.smoke_test:
            logger.info("Stage 2: focused matrix on top models")
            ext_configs = []
            stage2_keep = {
                _canonical_model_name(m) for m in args.stage2_keep_models.split(",") if m.strip()
            }

            def _stage2_allowed(model_name: str) -> bool:
                m = _canonical_model_name(model_name)
                if m.startswith("deep_cnn_attn"):
                    return True
                if _is_deprioritized_model(m):
                    return False
                if m in ("regnety_004", "efficientnet_b4", "resnet50v2"):
                    return False
                return m in stage2_keep

            strategy_library = {
                "full": {
                    "data_strategy": "full",
                    "loss_name": "ce",
                    "regularization": "mixup",
                    "use_tta": False,
                },
                "focal": {
                    "data_strategy": "full",
                    "loss_name": "focal",
                    "regularization": "mixup",
                    "use_tta": False,
                },
                "mixup_swa": {
                    "data_strategy": "full",
                    "loss_name": "ce",
                    "regularization": "mixup_swa",
                    "use_tta": False,
                },
                "tta": {
                    "data_strategy": "full",
                    "loss_name": "ce",
                    "regularization": "mixup",
                    "use_tta": True,
                },
            }

            requested_strategies = [
                s.strip().lower() for s in args.stage2_strategies.split(",") if s.strip()
            ]
            stage2_strategies = [s for s in requested_strategies if s in strategy_library]
            if not stage2_strategies:
                stage2_strategies = ["full", "focal", "mixup_swa", "tta"]

            eligible_models = [m for m in top_models if _stage2_allowed(m)]
            if args.stage2_pilot:
                eligible_models = eligible_models[: max(1, args.stage2_pilot_top_models)]
                logger.info(
                    "Stage 2 pilot enabled | top_models=%d strategies=%d planned=%d",
                    len(eligible_models),
                    len(stage2_strategies),
                    len(eligible_models) * len(stage2_strategies),
                )

            exp_idx = 0
            for model in eligible_models:
                is_pretrained = not _canonical_model_name(model).startswith("deep_cnn_attn")
                for strategy in stage2_strategies:
                    strategy_cfg = strategy_library[strategy]
                    exp_idx += 1
                    exp_id = _stable_experiment_id(
                        "s2",
                        model,
                        strategy_cfg["data_strategy"],
                        strategy_cfg["loss_name"],
                        strategy_cfg["regularization"],
                        f"tta_{int(strategy_cfg['use_tta'])}",
                        f"ep_{args.stage2_epochs}",
                        "pilot" if args.stage2_pilot else "focused",
                    )
                    ext_configs.append(
                        ExperimentConfig(
                            experiment_id=exp_id,
                            model_name=model,
                            data_strategy=strategy_cfg["data_strategy"],
                            loss_name=strategy_cfg["loss_name"],
                            regularization=strategy_cfg["regularization"],
                            epochs=args.stage2_epochs,
                            batch_size=args.batch_size,
                            lr_head=5e-4,
                            lr_backbone=5e-5 if is_pretrained else 5e-4,
                            label_smoothing=0.05,
                            patience=args.stage2_patience,
                            seed=args.seed + 1000 + exp_idx,
                            use_tta=bool(strategy_cfg["use_tta"]),
                        )
                    )

            if args.max_extended > 0:
                ext_configs = ext_configs[: args.max_extended]

            logger.info("Stage 2 planned experiments: %d", len(ext_configs))

            for i, cfg in enumerate(ext_configs, start=1):
                logger.info(
                    "Stage 2 progress %d/%d | model=%s strategy=%s loss=%s reg=%s tta=%s",
                    i,
                    len(ext_configs),
                    cfg.model_name,
                    cfg.data_strategy,
                    cfg.loss_name,
                    cfg.regularization,
                    cfg.use_tta,
                )
                _write_progress_state(
                    output_dir,
                    {
                        "status": "running",
                        "stage": "stage2",
                        "stage_progress": {"current": i, "total": len(ext_configs)},
                        "experiment_id": cfg.experiment_id,
                        "use_tta": cfg.use_tta,
                        "message": "Running Stage 2 extended experiment",
                    },
                )
                if cfg.experiment_id in summary_by_id:
                    logger.info("Stage 2 skip | completed experiment=%s", cfg.experiment_id)
                    continue
                try:
                    summary = _run_cv_experiment(
                        cfg=cfg,
                        X_all=X_train,
                        y_all=y_train,
                        w_all=w_train,
                        meta_all=meta_train,
                        build_deep_cnn=build_deep_cnn,
                        output_dir=output_dir,
                        n_folds=args.n_folds,
                        group_block_size_px=args.cv_group_block_size,
                        cv_spatial_block_m=cv_spatial_block_m,
                        device=device,
                        logger=logger,
                        resume=True,
                        augment_random_nodata_frac=args.augment_random_nodata_frac,
                        augment_pattern_nodata_frac=args.augment_pattern_nodata_frac,
                    )
                    summary_by_id[summary["experiment_id"]] = summary
                except Exception as exc:
                    logger.warning("Extended experiment failed (%s): %s", cfg.experiment_id, exc)

            logger.info("Stage 2 complete")

        experiment_summaries = list(summary_by_id.values())
        summary_df = pd.DataFrame(experiment_summaries)
        summary_df = summary_df.sort_values(["mean_cv_f1", "std_cv_f1"], ascending=[False, True])
        summary_df.to_csv(output_dir / "experiment_summary.csv", index=False)

        # Choose top-k final by CV
        top_final = summary_df.head(min(args.top_k_final, len(summary_df))).copy()
        if top_final.empty:
            raise RuntimeError("No experiments completed successfully.")

        final_results_df = pd.DataFrame()
        ensemble_row = None
        stacking_row = None

        if (
            X_test is not None
            and y_test is not None
            and len(y_test) > 0
            and len(np.unique(y_test)) > 1
        ):
            _write_progress_state(
                output_dir,
                {
                    "status": "running",
                    "stage": "final_test",
                    "message": "Retraining top models and evaluating test set",
                    "top_k": int(len(top_final)),
                },
            )
            final_results_df, test_probs_by_model, _ = _final_train_and_test_eval(
                top_rows=top_final,
                X_train_all=X_train,
                y_train_all=y_train,
                w_train_all=w_train,
                meta_train_all=meta_train,
                X_test=X_test,
                y_test=y_test,
                meta_test=meta_test,
                build_deep_cnn=build_deep_cnn,
                output_dir=output_dir,
                device=device,
                logger=logger,
                augment_random_nodata_frac=args.augment_random_nodata_frac,
                augment_pattern_nodata_frac=args.augment_pattern_nodata_frac,
            )
            final_results_df.to_csv(output_dir / "final_test_results.csv", index=False)

            if not final_results_df.empty:
                ensemble_eval = _soft_vote_ensemble(
                    test_probs_by_model=test_probs_by_model,
                    thresholds=final_results_df["threshold_from_val"].tolist(),
                    y_test=y_test,
                )
                ensemble_row = {
                    "model_id": "soft_vote_top5",
                    "experiment_source": "ensemble",
                    "model_name": "soft_vote",
                    "data_strategy": "mixed",
                    "loss_name": "mixed",
                    "regularization": "mixed",
                    "test_auc": ensemble_eval["test_auc"],
                    "test_f1": ensemble_eval["test_f1"],
                    "test_precision": ensemble_eval["test_precision"],
                    "test_recall": ensemble_eval["test_recall"],
                    "threshold_from_val": ensemble_eval["threshold"],
                    "confusion_matrix": ensemble_eval["confusion_matrix"],
                }
                final_results_df = pd.concat(
                    [final_results_df, pd.DataFrame([ensemble_row])], ignore_index=True
                )
                final_results_df.to_csv(output_dir / "final_test_results.csv", index=False)

                exp_dir = output_dir / "experiments"
                top_exp_ids = top_final["experiment_id"].tolist()
                top_model_ids = final_results_df[final_results_df["model_name"] != "soft_vote"][
                    "model_id"
                ].tolist()
                stacking_eval = _stacking_ensemble(
                    exp_dir=exp_dir,
                    top_experiment_ids=top_exp_ids,
                    top_model_ids=top_model_ids,
                    y_test=y_test,
                    test_probs_by_model=test_probs_by_model,
                )
                if stacking_eval is not None:
                    stacking_row = {
                        "model_id": "stacking_top5",
                        "experiment_source": "stacking",
                        "model_name": "stacking_logreg",
                        "data_strategy": "mixed",
                        "loss_name": "mixed",
                        "regularization": "mixed",
                        "test_auc": stacking_eval["test_auc"],
                        "test_f1": stacking_eval["test_f1"],
                        "test_precision": stacking_eval["test_precision"],
                        "test_recall": stacking_eval["test_recall"],
                        "threshold_from_val": stacking_eval["threshold"],
                        "confusion_matrix": stacking_eval["confusion_matrix"],
                    }
                    final_results_df = pd.concat(
                        [final_results_df, pd.DataFrame([stacking_row])], ignore_index=True
                    )
                    final_results_df.to_csv(output_dir / "final_test_results.csv", index=False)

        if args.run_raster_loocv and not args.smoke_test:
            _write_progress_state(
                output_dir,
                {
                    "status": "running",
                    "stage": "raster_loocv",
                    "message": "Running raster-level leave-one-out CV",
                },
            )
            _run_raster_loocv(
                top_rows=top_final,
                X_all=X_train,
                y_all=y_train,
                w_all=w_train,
                meta_all=meta_train,
                build_deep_cnn=build_deep_cnn,
                output_dir=output_dir,
                device=device,
                logger=logger,
            )

        _write_research_report(
            output_dir=output_dir,
            analysis_summary=analysis_summary,
            experiment_summary_df=summary_df,
            final_df=final_results_df,
            ensemble_row=ensemble_row,
            stacking_row=stacking_row,
        )

        run_manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": args.seed,
            "n_records_train": int(len(rec_train)),
            "n_records_test": int(len(rec_test)),
            "n_experiments": int(len(summary_df)),
            "top_k_final": int(min(args.top_k_final, len(summary_df))),
            "device": str(device),
            "elapsed_s": float(time.monotonic() - t0),
            "selected_models": available,
            "smoke_test": bool(args.smoke_test),
        }
        (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))

        _write_progress_state(
            output_dir,
            {
                "status": "done",
                "stage": "completed",
                "message": "Model search run completed",
                "run_manifest": run_manifest,
            },
        )

        logger.info("Done in %.1fs | artifacts: %s", run_manifest["elapsed_s"], output_dir)

    except KeyboardInterrupt:
        _write_progress_state(
            output_dir,
            {
                "status": "interrupted",
                "stage": "interrupted",
                "message": "Model search interrupted by user; safe to rerun for resume",
            },
        )
        logger.warning("Interrupted by user")
        raise SystemExit(130)
    except Exception as exc:
        _write_progress_state(
            output_dir,
            {
                "status": "failed",
                "stage": "failed",
                "message": f"Model search failed: {exc}",
            },
        )
        logger.exception("Model search failed")
        raise


if __name__ == "__main__":
    main()

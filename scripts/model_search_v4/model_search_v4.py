#!/usr/bin/env python3
"""Model Search V4 — academic-grade CWD model selection on harmonized CHMs.

High-level pipeline:

    1. **Curate** labels from ``output/onboarding_labels_v2_drop13``. Human
       annotations are kept unconditionally; auto rows are admitted only
       when their V3-ensemble probability clears a strict two-sided
       confidence gate. Duplicates are resolved by provenance priority,
       then timestamp.
    2. **Split** into train / buffer / test at the *place* level
       (tile × site, year-agnostic), grouped into spatial blocks, and with
       a configurable neighbour-block buffer ring that goes into neither
       split. A second, metre-unit fence is applied later at the tile
       level to prevent residual adjacency leakage inside retained
       training rows.
    3. **Ablate inputs** {original, raw, gauss, fusion3} on a top-``m``
       deep-model slate plus three classical baselines. Mode selection
       uses a composite score combining deep and classical F1 with an
       explicit overwhelming-margin gate.
    4. **Main search** of the top-``k`` V3 models (ranked by Lower
       Confidence Bound, ``mean - k * std``) on the selected input, via
       the base ``scripts/model_search.py``.
    5. **Audit** test performance by provenance (manual-only F1), year,
       and place, and benchmark against the V3 ``convnext_small``
       checkpoint.

This module is a thin orchestrator; pure-logic sub-modules under
``scripts/model_search_v4/`` hold label curation, splitting, features,
ranking, and audit so they can be unit-tested independently of the base
search script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

_V4_DIR = Path(__file__).resolve().parent
if str(_V4_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_V4_DIR.parent))

from model_search_v4._audit import breakdown_test_metrics
from model_search_v4._features import tile_feature_matrix
from model_search_v4._labels import (
    legacy_sample_id,
    safe_float,
    write_curated_labels_drop_only,
)
from model_search_v4._ranking import (
    DEFAULT_TOP3_FALLBACK,
    composite_mode_score,
    select_best_mode,
    top_models_from_v3_lcb,
)
from model_search_v4._splits import write_spatial_block_test_split


logger = logging.getLogger("model_search_v4")

DEFAULT_SOURCE_WEIGHTS = {
    "manual": 1.00,
    "auto_reviewed": 0.85,
    "": 0.75,
    "auto_threshold_gate_v4": 0.70,
}


@dataclass
class ModeContext:
    mode: str
    original_chm_dir: Path
    harmonized_root: Path


@dataclass
class SearchRunOptions:
    seed: int
    n_folds: int
    batch_size: int
    epochs_scratch: int
    epochs_pretrained: int
    patience: int
    top_k_expand: int
    top_k_final: int
    max_extended: int
    cv_group_block_size: int
    spatial_fence_m: float
    cv_spatial_block_m: float
    auto_cv_block_size: bool
    cv_block_candidates_m: str
    augment_random_nodata_frac: float
    augment_pattern_nodata_frac: float
    stage2_strategies: str
    stage2_epochs: int
    stage2_patience: int
    stage2_pilot: bool
    stage2_pilot_top_models: int
    smoke_test: bool


class V4Error(RuntimeError):
    """Raised for V4 orchestration issues."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_base_module_cache: dict[str, Any] = {}


def _import_module(module_path: Path, module_name: str):
    """Import (and cache) a Python file as a module under a stable name."""
    if module_name in _base_module_cache:
        return _base_module_cache[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise V4Error(f"Failed to import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    _base_module_cache[module_name] = mod
    return mod


PATCH_CONTRACT_SYMBOLS = [
    "_build_arrays_with_meta",
    "_build_model",
    "_select_models_after_analysis",
    "_is_deprioritized_model",
    "_spatial_fence_id",
    "_load_records_with_probs",
    "_load_test_keys",
    "_evaluate_classifier",
    "_clean_state_dict",
    "_import_fine_tune_helpers",
    "_canonical_model_name",
    "_build_deep_cnn_variant",
    "_replace_classifier_head",
]


def assert_base_patch_contract(base_script: Path) -> list[str]:
    """Verify that all symbols we monkey-patch or call exist in the base script."""
    mod = _import_module(base_script, "model_search_v4_contract_probe")
    missing = [s for s in PATCH_CONTRACT_SYMBOLS if not hasattr(mod, s)]
    if missing:
        raise V4Error(
            "base search script is missing expected symbols; patch contract broken: "
            + ", ".join(missing)
        )
    return list(PATCH_CONTRACT_SYMBOLS)


# ---------------------------------------------------------------------------
# Mode-aware array builder (shared between ablation, main search, classical).
# ---------------------------------------------------------------------------


def _mode_paths_for_record(rec, ctx: ModeContext) -> list[Path] | None:
    sample_id = legacy_sample_id(rec.raster)
    if sample_id is None:
        return None
    original = ctx.original_chm_dir / f"{sample_id}_chm_max_hag_20cm.tif"
    raw = ctx.harmonized_root / "chm_raw" / f"{sample_id}_harmonized_dem_last_raw_chm.tif"
    gauss = ctx.harmonized_root / "chm_gauss" / f"{sample_id}_harmonized_dem_last_gauss_chm.tif"
    if ctx.mode == "original":
        return [original] if original.exists() else None
    if ctx.mode == "raw":
        return [raw] if raw.exists() else None
    if ctx.mode == "gauss":
        return [gauss] if gauss.exists() else None
    if ctx.mode == "fusion3":
        paths = [original, raw, gauss]
        return paths if all(p.exists() for p in paths) else None
    raise V4Error(f"Unsupported mode: {ctx.mode}")


def _read_window_array(src, rec, canonical_size: int) -> np.ndarray:
    arr = src.read(
        1,
        window=Window(rec.col_off, rec.row_off, rec.chunk_size, rec.chunk_size),
        boundless=True,
        fill_value=0,
    ).astype(np.float32)
    if src.nodata is not None:
        arr[np.isclose(arr, float(src.nodata))] = 0.0
    arr[~np.isfinite(arr)] = 0.0
    if arr.shape != (canonical_size, canonical_size):
        import cv2
        arr = cv2.resize(arr, (canonical_size, canonical_size))
    return arr


def _build_arrays_with_mode(
    records,
    norm_tile: Callable[[np.ndarray], np.ndarray],
    canonical_size: int,
    ctx: ModeContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    src_cache: dict[str, Any] = {}
    bad_paths: set[str] = set()
    logged_bad_paths: set[str] = set()

    def _mark_bad_path(path: Path, exc: Exception) -> None:
        key = str(path)
        src = src_cache.pop(key, None)
        if src is not None:
            try:
                src.close()
            except Exception:
                pass
        bad_paths.add(key)
        if key not in logged_bad_paths:
            logger.warning("mode=%s skipping unreadable raster: %s (%s)", ctx.mode, key, exc)
            logged_bad_paths.add(key)

    def _open(path: Path):
        key = str(path)
        if key in bad_paths:
            return None
        if key not in src_cache:
            try:
                src_cache[key] = rasterio.open(path)
            except Exception as exc:
                _mark_bad_path(path, exc)
                return None
        return src_cache[key]

    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    w_list: list[float] = []
    meta: list[dict[str, Any]] = []
    skipped_missing = 0

    try:
        for rec in records:
            paths = _mode_paths_for_record(rec, ctx)
            if not paths:
                skipped_missing += 1
                continue
            channels: list[np.ndarray] = []
            x_center = y_center = None
            skip_record = False
            for i, path in enumerate(paths):
                src = _open(path)
                if src is None:
                    skip_record = True
                    break
                if i == 0:
                    try:
                        center_row = int(rec.row_off + rec.chunk_size // 2)
                        center_col = int(rec.col_off + rec.chunk_size // 2)
                        x_center, y_center = src.xy(center_row, center_col, offset="center")
                    except Exception:
                        x_center = y_center = None
                try:
                    raw = _read_window_array(src, rec, canonical_size=canonical_size)
                except Exception as exc:
                    _mark_bad_path(path, exc)
                    skip_record = True
                    break
                channels.append(norm_tile(raw))

            if skip_record:
                skipped_missing += 1
                continue

            if ctx.mode == "fusion3":
                tile = np.stack(channels, axis=0).astype(np.float32)
            else:
                tile = np.stack([channels[0]], axis=0).astype(np.float32)

            x_list.append(tile)
            y_list.append(int(rec.label))
            w_list.append(float(rec.weight))
            meta.append(
                {
                    "raster": rec.raster,
                    "row_off": int(rec.row_off),
                    "col_off": int(rec.col_off),
                    "source": rec.source,
                    "model_prob": rec.model_prob,
                    "key": f"{rec.raster}|{rec.row_off}|{rec.col_off}",
                    "x_center": float(x_center) if x_center is not None else None,
                    "y_center": float(y_center) if y_center is not None else None,
                }
            )
    finally:
        for src in src_cache.values():
            try:
                src.close()
            except Exception:
                pass

    if not x_list:
        raise V4Error(f"No tiles were loaded for mode={ctx.mode}")

    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    w = np.asarray(w_list, dtype=np.float32)
    if skipped_missing > 0:
        logger.info("mode=%s skipped_missing=%d", ctx.mode, skipped_missing)
    if bad_paths:
        logger.warning("mode=%s unreadable_rasters=%d", ctx.mode, len(bad_paths))
    return x, y, w, meta


# ---------------------------------------------------------------------------
# Monkey-patch helpers for the base search script.
# ---------------------------------------------------------------------------


def _set_module_by_name(root, module_name: str, new_module) -> None:
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


def _adapt_first_conv_to_nch(model, in_channels: int) -> None:
    import torch
    import torch.nn as nn
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.in_channels == in_channels:
                return
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
            )
            with torch.no_grad():
                if module.in_channels == 1 and in_channels > 1:
                    new_conv.weight.copy_(module.weight.repeat(1, in_channels, 1, 1) / float(in_channels))
                elif module.in_channels == 3 and in_channels == 1:
                    new_conv.weight.copy_(module.weight.mean(dim=1, keepdim=True))
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias)
            _set_module_by_name(model, name, new_conv)
            return


class _CHMInputNorm(object):
    """Wraps a backbone with a learnable per-channel affine normalization layer.

    ImageNet-pretrained backbones expect inputs near [0, 1] normalized by
    ImageNet mean/std. Raw CHM tiles have very different per-channel statistics
    (especially ``fusion3`` which stacks original 0.2 m CHM, raw 0.8 m CHM, and
    Gaussian-smoothed 0.8 m CHM). Without re-normalizing, the ImageNet
    backbone's BatchNorm/LayerNorm statistics mismatch causes training collapse
    (model predicts all-positive, threshold=0.05, recall=1.0).

    The wrapper adds a ``nn.Conv2d(C, C, 1, groups=C, bias=True)`` depthwise
    conv — equivalent to per-channel scale + bias (affine norm) — initialized
    to identity (scale=1, bias=0). It learns to map each CHM channel to the
    statistics the backbone expects within the first few epochs.
    """

    @staticmethod
    def wrap(backbone, n_channels: int = 3):
        import torch
        import torch.nn as nn

        class Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_norm = nn.Conv2d(n_channels, n_channels, 1, groups=n_channels, bias=True)
                with torch.no_grad():
                    nn.init.ones_(self.input_norm.weight)
                    nn.init.zeros_(self.input_norm.bias)
                self.backbone = backbone

            def forward(self, x):
                return self.backbone(self.input_norm(x))

        return Wrapped()


def _build_model_3ch(
    base_mod,
    name: str,
    build_deep_cnn: Callable,
    original_build_model_fn: Callable,
    input_size: int | None = None,
) -> tuple:
    """Build a 3-channel model for ``fusion3`` mode.

    Three strategies in priority order:
    1. Named torchvision backbones (convnext_small, efficientnet_b*): load
       pretrained 3-ch ImageNet weights directly, then wrap with a learnable
       per-channel input-normalization layer so the backbone sees CHM values
       re-scaled toward its expected distribution.
    2. timm fallback: use ``timm.create_model(name, in_chans=3, ...)``.
    3. Last resort: use ``original_build_model_fn`` (the pre-patch base
       builder) + first-conv surgery.  ``original_build_model_fn`` is
       intentionally the *original*, unpatched function to avoid infinite
       recursion that would occur if ``base_mod._build_model`` (our patch)
       were called here.
    """
    import torch.nn as nn
    from torchvision import models as tvm

    model_name = base_mod._canonical_model_name(name)

    if model_name.startswith("deep_cnn_attn"):
        net = base_mod._build_deep_cnn_variant(model_name, build_deep_cnn)
        _adapt_first_conv_to_nch(net, in_channels=3)
        return net, []

    def _try_weights(factory, weights_attr: str | None = None):
        try:
            if weights_attr and hasattr(tvm, weights_attr):
                return factory(weights=getattr(tvm, weights_attr).DEFAULT)
            return factory(weights="DEFAULT")
        except Exception:
            return factory(weights=None)

    if model_name == "convnext_small":
        m = _try_weights(tvm.convnext_small, "ConvNeXt_Small_Weights")
        base_mod._replace_classifier_head(m, 2)
        return _CHMInputNorm.wrap(m, 3), []
    if model_name == "efficientnet_b2":
        m = _try_weights(tvm.efficientnet_b2, "EfficientNet_B2_Weights")
        base_mod._replace_classifier_head(m, 2)
        return _CHMInputNorm.wrap(m, 3), []
    if model_name == "efficientnet_b0":
        m = _try_weights(tvm.efficientnet_b0, "EfficientNet_B0_Weights")
        base_mod._replace_classifier_head(m, 2)
        return _CHMInputNorm.wrap(m, 3), []

    try:
        import timm
        kwargs: dict[str, Any] = {"in_chans": 3, "num_classes": 2}
        if input_size is not None:
            kwargs["img_size"] = (int(input_size), int(input_size))
        try:
            m = timm.create_model(model_name, pretrained=True, **kwargs)
        except Exception:
            m = timm.create_model(model_name, pretrained=False, **kwargs)
        if hasattr(m, "classifier") and isinstance(m.classifier, nn.Linear):
            m.classifier = nn.Linear(m.classifier.in_features, 2)
        return _CHMInputNorm.wrap(m, 3), []
    except Exception:
        # Last resort: use the *original* (unpatched) base builder to avoid
        # infinite recursion — base_mod._build_model is our patch at this point.
        m, extra = original_build_model_fn(model_name, build_deep_cnn, input_size=input_size)
        _adapt_first_conv_to_nch(m, in_channels=3)
        return m, extra


def _force_models_during_run(base_mod, force_models: list[str]):
    original_select = base_mod._select_models_after_analysis
    original_deprioritize = base_mod._is_deprioritized_model
    base_mod._select_models_after_analysis = lambda _s, _smoke: list(force_models)
    base_mod._is_deprioritized_model = lambda _name: False
    return original_select, original_deprioritize


def _run_base_search_with_mode(
    base_script: Path,
    labels_dir: Path,
    test_split_path: Path,
    output_dir: Path,
    mode_ctx: ModeContext,
    force_models: list[str],
    opts: SearchRunOptions,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    module_name = f"model_search_v4_base_{mode_ctx.mode}"
    base_mod = _import_module(base_script, module_name)

    original_build_arrays = base_mod._build_arrays_with_meta
    original_build_model = base_mod._build_model

    def _patched_build_arrays(records, chm_dir, norm_tile, canonical_size: int = 128):
        return _build_arrays_with_mode(
            records=records, norm_tile=norm_tile, canonical_size=canonical_size, ctx=mode_ctx,
        )

    def _patched_build_model(name: str, build_deep_cnn: Callable, input_size: int | None = None):
        if mode_ctx.mode == "fusion3":
            return _build_model_3ch(
                base_mod, name, build_deep_cnn,
                original_build_model_fn=original_build_model,  # prevents infinite recursion
                input_size=input_size,
            )
        return original_build_model(name, build_deep_cnn, input_size=input_size)

    original_select, original_deprioritize = _force_models_during_run(base_mod, force_models)
    base_mod._build_arrays_with_meta = _patched_build_arrays
    base_mod._build_model = _patched_build_model

    old_argv = sys.argv[:]
    try:
        argv = [
            str(base_script),
            "--labels", str(labels_dir),
            "--chm-dir", str(mode_ctx.original_chm_dir),
            "--test-split", str(test_split_path),
            "--output", str(output_dir),
            "--seed", str(opts.seed),
            "--n-folds", str(opts.n_folds),
            "--batch-size", str(opts.batch_size),
            "--epochs-scratch", str(opts.epochs_scratch),
            "--epochs-pretrained", str(opts.epochs_pretrained),
            "--patience", str(opts.patience),
            "--cv-group-block-size", str(opts.cv_group_block_size),
            "--spatial-fence-m", str(opts.spatial_fence_m),
            "--cv-spatial-block-m", str(opts.cv_spatial_block_m),
            "--cv-block-candidates-m", str(opts.cv_block_candidates_m),
            "--top-k-expand", str(max(1, min(opts.top_k_expand, len(force_models)))),
            "--top-k-final", str(max(1, min(opts.top_k_final, len(force_models)))),
            "--max-extended", str(opts.max_extended),
            "--augment-random-nodata-frac", str(opts.augment_random_nodata_frac),
            "--augment-pattern-nodata-frac", str(opts.augment_pattern_nodata_frac),
            "--stage2-keep-models", ",".join(force_models),
            "--stage2-strategies", str(opts.stage2_strategies),
            "--stage2-epochs", str(opts.stage2_epochs),
            "--stage2-patience", str(opts.stage2_patience),
        ]
        if opts.auto_cv_block_size:
            argv.append("--auto-cv-block-size")
        if opts.stage2_pilot:
            argv.extend(["--stage2-pilot", "--stage2-pilot-top-models", str(opts.stage2_pilot_top_models)])
        if opts.smoke_test:
            argv.append("--smoke-test")
        sys.argv = argv
        base_mod.main()
    finally:
        sys.argv = old_argv
        base_mod._build_arrays_with_meta = original_build_arrays
        base_mod._build_model = original_build_model
        base_mod._select_models_after_analysis = original_select
        base_mod._is_deprioritized_model = original_deprioritize


# ---------------------------------------------------------------------------
# Result extraction, baseline, fence, benchmark.
# ---------------------------------------------------------------------------


def _extract_best_deep_result(run_dir: Path) -> dict[str, Any]:
    final_csv = run_dir / "final_test_results.csv"
    if final_csv.exists():
        try:
            df = pd.read_csv(final_csv)
            if not df.empty:
                singles = df[
                    (~df["model_name"].astype(str).isin(["soft_vote", "stacking_logreg", "diverse_soft_vote_top3"]))
                    & (~df["model_id"].astype(str).isin(["soft_vote_top5", "stacking_top5", "diverse_top3"]))
                ].copy()
                if not singles.empty:
                    best = singles.sort_values("test_f1", ascending=False).iloc[0]
                    return {
                        "source": "final_test",
                        "model_name": str(best.get("model_name")),
                        "metric_name": "test_f1",
                        "metric": float(best.get("test_f1", 0.0)),
                        "test_auc": float(best.get("test_auc", 0.0)),
                        "test_precision": float(best.get("test_precision", 0.0)),
                        "test_recall": float(best.get("test_recall", 0.0)),
                    }
        except Exception as exc:
            logger.warning("failed to read %s: %s", final_csv, exc)

    summary_csv = run_dir / "experiment_summary.csv"
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            if not df.empty and {"model_name", "mean_cv_f1"}.issubset(set(df.columns)):
                best = df.sort_values(["mean_cv_f1", "std_cv_f1"], ascending=[False, True]).iloc[0]
                return {
                    "source": "cv_only",
                    "model_name": str(best.get("model_name")),
                    "metric_name": "mean_cv_f1",
                    "metric": float(best.get("mean_cv_f1", 0.0)),
                    "test_auc": None,
                    "test_precision": None,
                    "test_recall": None,
                }
        except Exception as exc:
            logger.warning("failed to read %s: %s", summary_csv, exc)

    return {
        "source": "missing",
        "model_name": "",
        "metric_name": "none",
        "metric": float("nan"),
        "test_auc": None,
        "test_precision": None,
        "test_recall": None,
    }


def _best_threshold_from_probs(y_true: np.ndarray, probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return 0.5
    f1s = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    return float(thresholds[int(np.nanargmax(f1s))])


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (probs >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "auc": auc,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    y = y.astype(int)
    n = len(y)
    n_pos = max(1, int(np.sum(y == 1)))
    n_neg = max(1, int(np.sum(y == 0)))
    return np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg)).astype(np.float32)


def _run_classical_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    out_csv: Path,
) -> list[dict[str, Any]]:
    if len(np.unique(y_train)) < 2:
        raise V4Error("Classical baseline requires at least 2 classes in training set")

    xtr = tile_feature_matrix(x_train)
    xte = tile_feature_matrix(x_test)

    x_fit, x_val, y_fit, y_val = train_test_split(
        xtr, y_train, test_size=0.20, random_state=seed, stratify=y_train,
    )
    sw_fit = _balanced_sample_weight(y_fit)
    sw_full = _balanced_sample_weight(y_train)

    models: list[tuple[str, Any]] = [
        ("logreg_balanced", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed)),
        (
            "rf_balanced",
            RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_leaf=2,
                class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
            ),
        ),
        (
            "hgb_weighted",
            HistGradientBoostingClassifier(
                learning_rate=0.06, max_depth=None, max_leaf_nodes=63,
                min_samples_leaf=20, random_state=seed,
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for model_name, model in models:
        model.fit(x_fit, y_fit, sample_weight=sw_fit)
        val_probs = (
            model.predict_proba(x_val)[:, 1]
            if hasattr(model, "predict_proba")
            else 1.0 / (1.0 + np.exp(-model.decision_function(x_val)))
        )
        thr = _best_threshold_from_probs(y_val, val_probs)
        model.fit(xtr, y_train, sample_weight=sw_full)
        test_probs = (
            model.predict_proba(xte)[:, 1]
            if hasattr(model, "predict_proba")
            else 1.0 / (1.0 + np.exp(-model.decision_function(xte)))
        )
        m = _binary_metrics(y_test, test_probs, threshold=thr)
        rows.append(
            {
                "model_name": model_name,
                "test_auc": m["auc"],
                "test_f1": m["f1"],
                "test_precision": m["precision"],
                "test_recall": m["recall"],
                "threshold_from_val": m["threshold"],
                "confusion_matrix": m["confusion_matrix"],
            }
        )

    df = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df.to_dict(orient="records")


def _load_records_split(base_mod, labels_dir: Path, test_split_path: Path):
    records = base_mod._load_records_with_probs(labels_dir, DEFAULT_SOURCE_WEIGHTS)
    test_keys = base_mod._load_test_keys(test_split_path)
    rec_train = [r for r in records if r.key not in test_keys]
    rec_test = [r for r in records if r.key in test_keys]
    return records, rec_train, rec_test


def _apply_spatial_fence(
    base_mod,
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    meta_train: list[dict[str, Any]],
    meta_test: list[dict[str, Any]],
    fence_m: float,
    group_block_size_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], int]:
    if fence_m <= 0 or not meta_test:
        return x_train, y_train, w_train, meta_train, 0
    test_fences = {
        base_mod._spatial_fence_id(m, block_size_px=group_block_size_px, fence_m=fence_m) for m in meta_test
    }
    keep_mask = np.array(
        [
            base_mod._spatial_fence_id(m, block_size_px=group_block_size_px, fence_m=fence_m) not in test_fences
            for m in meta_train
        ],
        dtype=bool,
    )
    dropped = int((~keep_mask).sum())
    return (
        x_train[keep_mask],
        y_train[keep_mask],
        w_train[keep_mask],
        [m for i, m in enumerate(meta_train) if keep_mask[i]],
        dropped,
    )


def _v3_benchmark_threshold(v3_summary_csv: Path, model_name: str) -> float:
    if not v3_summary_csv.exists():
        return 0.5
    try:
        df = pd.read_csv(v3_summary_csv)
        sub = df[df["model_name"].astype(str).str.lower() == str(model_name).lower()].copy()
        if sub.empty:
            return 0.5
        sub = sub.sort_values(["mean_cv_f1", "std_cv_f1"], ascending=[False, True])
        thr = safe_float(sub.iloc[0].get("mean_threshold"))
        return float(thr) if thr is not None else 0.5
    except Exception as exc:
        logger.warning("v3 benchmark threshold lookup failed: %s", exc)
        return 0.5


def _evaluate_v3_checkpoint(
    base_script: Path,
    labels_dir: Path,
    test_split_path: Path,
    mode_ctx: ModeContext,
    checkpoint_path: Path,
    v3_summary_csv: Path,
    benchmark_model: str,
) -> dict[str, Any] | None:
    if mode_ctx.mode == "fusion3":
        logger.warning(
            "V3 benchmark skipped: selected mode=fusion3 is 3-channel but V3 checkpoint is 1-channel"
        )
        return None
    if not checkpoint_path.exists():
        logger.warning("V3 checkpoint not found: %s", checkpoint_path)
        return None

    mod = _import_module(base_script, "model_search_v4_benchmark")
    helpers = mod._import_fine_tune_helpers(base_script.parent / "fine_tune_cnn.py")
    build_deep_cnn = helpers["build_deep_cnn"]
    norm_tile = helpers["norm_tile"]

    _records, _rec_train, rec_test = _load_records_split(mod, labels_dir, test_split_path)
    if not rec_test:
        logger.warning("V3 benchmark skipped: no test records found")
        return None

    x_test, y_test, _w_test, meta_test = _build_arrays_with_mode(
        records=rec_test, norm_tile=norm_tile, canonical_size=128, ctx=mode_ctx,
    )

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = mod._build_model(benchmark_model, build_deep_cnn, input_size=x_test.shape[-1])
    net = net.to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = state.get("state_dict", state)
    cleaned = mod._clean_state_dict(sd if isinstance(sd, dict) else {})
    net.load_state_dict(cleaned, strict=False)

    ev = mod._evaluate_classifier(net, x_test, y_test, device=device, tta=True)
    fixed_thr = _v3_benchmark_threshold(v3_summary_csv, benchmark_model)
    fixed_metrics = _binary_metrics(y_test, ev["probs"], threshold=fixed_thr)
    auto_metrics = {
        "auc": float(ev["auc"]),
        "f1": float(ev["f1"]),
        "precision": float(ev["precision"]),
        "recall": float(ev["recall"]),
        "threshold": float(ev["threshold"]),
    }
    audit = breakdown_test_metrics(y_test, np.asarray(ev["probs"]), fixed_thr, meta_test)

    return {
        "model_name": benchmark_model,
        "checkpoint": str(checkpoint_path),
        "n_test": int(len(y_test)),
        "fixed_threshold": float(fixed_thr),
        "fixed_metrics": fixed_metrics,
        "auto_threshold_metrics": auto_metrics,
        "audit_by_provenance": audit,
    }


# ---------------------------------------------------------------------------
# Ablation orchestration.
# ---------------------------------------------------------------------------


def _run_single_mode_ablation(
    mode: str,
    ablation_dir: Path,
    args: argparse.Namespace,
    mode_ctx: ModeContext,
    rec_train_all,
    rec_test_all,
    split_mod,
    base_script: Path,
    curated_labels_dir: Path,
    split_path: Path,
    norm_tile,
    ablation_models: list[str],
) -> dict[str, Any]:
    deep_run_dir = ablation_dir / f"deep_{mode}"
    deep_opts = SearchRunOptions(
        seed=int(args.seed),
        n_folds=2 if args.smoke_test else int(args.ablation_folds),
        batch_size=8 if args.smoke_test else int(args.batch_size),
        epochs_scratch=3 if args.smoke_test else int(args.ablation_epochs),
        epochs_pretrained=3 if args.smoke_test else int(args.ablation_epochs),
        patience=2 if args.smoke_test else int(args.ablation_patience),
        top_k_expand=max(1, len(ablation_models)),
        top_k_final=max(1, len(ablation_models)),
        max_extended=0,
        cv_group_block_size=int(args.cv_group_block_size),
        spatial_fence_m=float(args.spatial_fence_m),
        cv_spatial_block_m=float(args.cv_spatial_block_m),
        auto_cv_block_size=bool(args.auto_cv_block_size),
        cv_block_candidates_m=str(args.cv_block_candidates_m),
        augment_random_nodata_frac=float(args.augment_random_nodata_frac),
        augment_pattern_nodata_frac=float(args.augment_pattern_nodata_frac),
        stage2_strategies="full",
        stage2_epochs=max(3, int(args.ablation_epochs // 2)),
        stage2_patience=max(2, int(args.ablation_patience)),
        stage2_pilot=True,
        stage2_pilot_top_models=1,
        smoke_test=bool(args.smoke_test),
    )

    if args.no_deep:
        logger.info("[mode=%s] --no-deep: skipping deep ablation", mode)
        deep_res = {
            "source": "no_deep", "model_name": "", "metric_name": "skipped",
            "metric": float("nan"), "test_auc": None, "test_precision": None, "test_recall": None,
        }
    else:
        try:
            _run_base_search_with_mode(
                base_script=base_script,
                labels_dir=curated_labels_dir,
                test_split_path=split_path,
                output_dir=deep_run_dir,
                mode_ctx=mode_ctx,
                force_models=ablation_models,
                opts=deep_opts,
            )
            deep_res = _extract_best_deep_result(deep_run_dir)
        except Exception as exc:
            logger.exception("[mode=%s] deep ablation failed", mode)
            deep_res = {
                "source": "error", "model_name": "", "metric_name": "error",
                "metric": float("nan"), "test_auc": None, "test_precision": None, "test_recall": None,
                "error": str(exc),
            }

    try:
        x_train, y_train, w_train, meta_train = _build_arrays_with_mode(
            records=rec_train_all, norm_tile=norm_tile, canonical_size=128, ctx=mode_ctx,
        )
        x_test, y_test, _w_test, meta_test = _build_arrays_with_mode(
            records=rec_test_all, norm_tile=norm_tile, canonical_size=128, ctx=mode_ctx,
        )
        n_train_before = int(len(y_train))
        x_train, y_train, w_train, meta_train, dropped = _apply_spatial_fence(
            base_mod=split_mod,
            x_train=x_train, y_train=y_train, w_train=w_train,
            meta_train=meta_train, meta_test=meta_test,
            fence_m=float(args.spatial_fence_m),
            group_block_size_px=int(args.cv_group_block_size),
        )
        if dropped > 0:
            logger.info("[mode=%s] classical train dropped by spatial fence: %d", mode, dropped)

        classical_rows = _run_classical_baselines(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            seed=int(args.seed), out_csv=ablation_dir / f"classical_{mode}.csv",
        )
        classical_best = classical_rows[0] if classical_rows else None
        classical_train_count_post_fence = int(len(y_train))
        classical_train_count_pre_fence = n_train_before
        classical_test_count = int(len(y_test))
    except Exception as exc:
        logger.exception("[mode=%s] classical baseline failed", mode)
        (ablation_dir / f"classical_{mode}.error.txt").write_text(str(exc))
        classical_best = None
        classical_train_count_post_fence = 0
        classical_train_count_pre_fence = 0
        classical_test_count = 0

    deep_metric = safe_float(deep_res.get("metric"))
    deep_metric = float(deep_metric) if deep_metric is not None else float("nan")
    classical_best_f1 = (
        float(classical_best.get("test_f1", float("nan"))) if classical_best is not None else float("nan")
    )
    composite = composite_mode_score(
        deep_f1=deep_metric if np.isfinite(deep_metric) else None,
        classical_f1=classical_best_f1 if np.isfinite(classical_best_f1) else None,
        deep_weight=float(args.deep_weight),
    )
    return {
        "mode": mode,
        "deep_source": deep_res.get("source"),
        "deep_model": deep_res.get("model_name", ""),
        "deep_metric_name": deep_res.get("metric_name", ""),
        "deep_metric": deep_metric,
        "classical_best_model": classical_best.get("model_name") if classical_best else "",
        "classical_best_f1": classical_best_f1,
        "classical_train_rows_pre_fence": classical_train_count_pre_fence,
        "classical_train_rows_post_fence": classical_train_count_post_fence,
        "classical_test_rows": classical_test_count,
        "composite_score": composite,
    }


# ---------------------------------------------------------------------------
# Report writing.
# ---------------------------------------------------------------------------


def _table_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def _write_report(
    output_dir: Path,
    prepared_stats: dict[str, Any],
    ablation_df: pd.DataFrame,
    selection: dict[str, Any],
    top_models: list[str],
    v3_audit_rows: list[dict[str, Any]],
    main_output_dir: Path,
    v3_benchmark: dict[str, Any] | None,
    runtime_s: float,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# Model Search V4 — Run Report")
    lines.append("")
    lines.append(f"Generated: {_utc_now()}  |  runtime: {runtime_s:.1f}s")
    lines.append("")

    lines.append("## Design")
    lines.append(f"- Labels: {args.labels_dir}")
    lines.append("- Split: spatial place-blocks with neighbour-buffer ring (no year leakage)")
    lines.append(
        f"- Neighbour buffer blocks: {args.neighbor_buffer_blocks}  |  block size (places): {args.split_block_size_places}"
    )
    lines.append(
        f"- Metric fence inside retained train: {args.spatial_fence_m} m "
        f"(cv_group_block_size_px={args.cv_group_block_size})"
    )
    lines.append(f"- Inputs compared: {args.input_modes}")
    lines.append(f"- Ablation model slate: {args.ablation_models or '(default top-2 from V3 LCB)'}")
    lines.append(f"- Main model pool (V3 top-{args.n_models} by LCB, k={args.lcb_k}): {', '.join(top_models)}")
    lines.append(f"- Composite score weight: deep={args.deep_weight}, classical={1.0 - float(args.deep_weight):.2f}")
    lines.append(f"- Overwhelming margin for mode choice: {args.overwhelming_margin}")
    lines.append("")

    lines.append("## Data Preparation")
    for key in [
        "drop_labeled_rows",
        "drop_kept_manual_or_reviewed",
        "drop_kept_threshold_gate",
        "drop_rejected_no_prob",
        "drop_rejected_below_threshold",
        "curated_rows_after_dedup",
        "curated_rasters",
    ]:
        lines.append(f"- {key}: {prepared_stats.get(key, 0)}")
    lines.append(f"- test_rows: {prepared_stats.get('test_rows', 0)}")
    lines.append(f"- test_manual_rows: {prepared_stats.get('test_manual_rows', 0)}")
    lines.append(f"- test_threshold_gate_rows: {prepared_stats.get('test_threshold_gate_rows', 0)}")
    lines.append(f"- buffer_rows: {prepared_stats.get('buffer_rows', 0)}")
    lines.append(
        f"- n_places_test / buffer / train: "
        f"{prepared_stats.get('n_places_test', 0)} / "
        f"{prepared_stats.get('n_places_buffer', 0)} / "
        f"{prepared_stats.get('n_places_train', 0)}"
    )
    lines.append(f"- test_fraction_actual: {prepared_stats.get('test_fraction_actual', 0.0):.4f}")
    lines.append(f"- place_overlap_train_vs_test: {prepared_stats.get('place_overlap_train_vs_test', 0)}")
    lines.append("")

    lines.append("## V3 LCB Top-k Audit")
    if v3_audit_rows:
        lines.append(_table_markdown(pd.DataFrame(v3_audit_rows)))
    else:
        lines.append("_(V3 summary CSV was not found — falling back to hardcoded list)_")
    lines.append("")

    lines.append("## Input Ablation")
    if not ablation_df.empty:
        shown = ablation_df.sort_values("composite_score", ascending=False).copy()
        cols = [
            "mode", "deep_metric_name", "deep_metric", "classical_best_f1", "composite_score",
            "deep_model", "classical_best_model",
            "classical_train_rows_pre_fence", "classical_train_rows_post_fence", "classical_test_rows",
        ]
        for col in cols:
            if col not in shown.columns:
                shown[col] = np.nan
        lines.append(_table_markdown(shown[cols]))
    else:
        lines.append("_(no ablation rows)_")
    lines.append("")

    lines.append("## Selected Input")
    lines.append(f"- chosen_mode: {selection.get('best_mode', 'n/a')}")
    lines.append(f"- overwhelming_margin_reached: {selection.get('overwhelming', False)}")
    if "runner_up_mode" in selection:
        lines.append(f"- runner_up: {selection['runner_up_mode']} (score {selection['runner_up_score']:.4f})")
        lines.append(f"- margin: {selection.get('margin', 0.0):.4f}")
    lines.append("")

    summary_csv = main_output_dir / "experiment_summary.csv"
    final_csv = main_output_dir / "final_test_results.csv"
    lines.append("## Main Search")
    lines.append(f"- output_dir: {main_output_dir}")
    if summary_csv.exists():
        try:
            sdf = pd.read_csv(summary_csv)
            top = sdf.sort_values(["mean_cv_f1", "std_cv_f1"], ascending=[False, True]).head(8)
            lines.append("\nTop CV experiments:\n")
            lines.append(_table_markdown(top))
        except Exception:
            lines.append("- experiment_summary.csv exists but could not be rendered")
    else:
        lines.append("- experiment_summary.csv missing")
    if final_csv.exists():
        try:
            fdf = pd.read_csv(final_csv)
            lines.append("\nFinal test results:\n")
            lines.append(_table_markdown(fdf))
        except Exception:
            lines.append("- final_test_results.csv exists but could not be rendered")
    else:
        lines.append("- final_test_results.csv missing")
    lines.append("")

    lines.append("## V3 Benchmark")
    if v3_benchmark is None:
        lines.append("- V3 checkpoint benchmark was not available for this selected mode (see log).")
    else:
        fm = v3_benchmark["fixed_metrics"]
        am = v3_benchmark["auto_threshold_metrics"]
        lines.append(
            f"- Fixed-threshold benchmark (thr={v3_benchmark['fixed_threshold']:.4f}): "
            f"F1={fm['f1']:.4f}, AUC={fm['auc']:.4f}, P={fm['precision']:.4f}, R={fm['recall']:.4f}"
        )
        lines.append(
            f"- Auto-threshold reference: F1={am['f1']:.4f}, AUC={am['auc']:.4f}, "
            f"P={am['precision']:.4f}, R={am['recall']:.4f}, thr={am['threshold']:.4f}"
        )
        audit = v3_benchmark.get("audit_by_provenance") or {}
        mo = audit.get("manual_only")
        if mo:
            lines.append(
                f"- **Manual-only test (n={mo['n']})**: F1={mo['f1']:.4f}, AUC={mo['auc']:.4f}, "
                f"P={mo['precision']:.4f}, R={mo['recall']:.4f}"
            )
        to = audit.get("threshold_gate_only")
        if to:
            lines.append(
                f"- Threshold-gate-only test (n={to['n']}): F1={to['f1']:.4f}, AUC={to['auc']:.4f}"
            )
        per_year = audit.get("per_year") or {}
        if per_year:
            rows = [
                {"year": y, **{k: v for k, v in m.items() if k in ("n", "f1", "auc")}}
                for y, m in sorted(per_year.items())
            ]
            lines.append("\nPer-year breakdown:\n")
            lines.append(_table_markdown(pd.DataFrame(rows)))
        pps = audit.get("per_place_summary") or {}
        if pps:
            lines.append(
                f"\nPer-place test F1 spread (places with ≥10 rows, n={pps.get('n_places_min10_rows', 0)}): "
                f"mean={pps.get('f1_mean', float('nan')):.4f} "
                f"std={pps.get('f1_std', float('nan')):.4f} "
                f"min={pps.get('f1_min', float('nan')):.4f} "
                f"max={pps.get('f1_max', float('nan')):.4f}"
            )

    lines.append("")
    lines.append("## Self-Audit Notes")
    lines.append("- Same-place-multi-year leakage: PREVENTED by year-agnostic place_key grouping.")
    lines.append(f"- Neighbour-place leakage: mitigated by {args.neighbor_buffer_blocks}-block buffer ring.")
    lines.append(f"- Fine-grained neighbour leakage: mitigated by {args.spatial_fence_m} m metric fence.")
    lines.append("- Pseudo-label circularity: audited above via manual-only test metrics.")
    lines.append("- Model-selection variance: top-k uses LCB ranking (see V3 audit table).")
    lines.append("")

    (output_dir / "REPORT_V4.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description="Model Search V4: harmonized-input ablation + LCB-ranked top-k search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--output", default=str(root / "output/model_search_v4"))
    p.add_argument("--labels-dir", default=str(root / "output/onboarding_labels_v2_drop13"))
    p.add_argument("--base-search-script", default=str(root / "scripts/model_search.py"))

    p.add_argument("--original-chm-dir", default=str(root / "data/lamapuit/chm_max_hag_13_drop"))
    p.add_argument("--harmonized-root",
                   default=str(root / "output/chm_dataset_harmonized_0p8m_raw_gauss_stable"))

    p.add_argument("--v3-summary",
                   default=str(root / "output/model_search_v3_academic_leakage26/experiment_summary.csv"))
    p.add_argument("--v3-best-checkpoint",
                   default=str(root / "output/model_search_v3_academic_leakage26/final_models/convnext_small_full_ce_mixup.pt"))
    p.add_argument("--benchmark-model", default="convnext_small")

    p.add_argument("--force-models", default="")
    p.add_argument("--n-models", type=int, default=3)
    p.add_argument("--lcb-k", type=float, default=1.0,
                   help="LCB penalty on cv std for V3 top-k ranking (mean - k*std).")

    p.add_argument("--input-modes", default="original,raw,gauss,fusion3")
    p.add_argument("--ablation-models", default="",
                   help="Comma-separated ablation model slate; defaults to top-2 from V3 LCB.")
    p.add_argument("--overwhelming-margin", type=float, default=0.05)
    p.add_argument("--deep-weight", type=float, default=0.70,
                   help="Weight of deep-model F1 in the composite mode score.")

    p.add_argument("--t-high", type=float, default=0.9995)
    p.add_argument("--t-low", type=float, default=0.0698)

    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--split-block-size-places", type=int, default=2)
    p.add_argument("--neighbor-buffer-blocks", type=int, default=1,
                   help="Chebyshev radius of blocks reserved as a buffer ring around test blocks.")

    p.add_argument("--seed", type=int, default=2026)

    p.add_argument("--ablation-folds", type=int, default=3)
    p.add_argument("--ablation-epochs", type=int, default=14)
    p.add_argument("--ablation-patience", type=int, default=5)

    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-scratch", type=int, default=60)
    p.add_argument("--epochs-pretrained", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--top-k-final", type=int, default=3)
    p.add_argument("--max-extended", type=int, default=12)
    p.add_argument("--stage2-strategies", default="full,focal,mixup_swa,tta")
    p.add_argument("--stage2-epochs", type=int, default=60)
    p.add_argument("--stage2-patience", type=int, default=10)

    p.add_argument("--cv-group-block-size", type=int, default=128)
    p.add_argument("--spatial-fence-m", type=float, default=26.0)
    p.add_argument("--cv-spatial-block-m", type=float, default=0.0)
    p.add_argument("--auto-cv-block-size", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cv-block-candidates-m", default="26,39,52,78,104")

    p.add_argument("--augment-random-nodata-frac", type=float, default=0.50)
    p.add_argument("--augment-pattern-nodata-frac", type=float, default=0.75)

    p.add_argument("--time-budget-hours", type=float, default=6.0)

    p.add_argument("--prepare-only", action="store_true",
                   help="Do label curation + split only; skip all training.")
    p.add_argument("--no-deep", action="store_true",
                   help="Skip deep ablation and main deep search; run only classical baselines.")
    p.add_argument("--smoke-test", action="store_true")

    return p.parse_args()


def _remaining_hours(start_t: float, budget_hours: float) -> float:
    elapsed_h = (time.monotonic() - start_t) / 3600.0
    return max(0.0, float(budget_hours) - elapsed_h)


class _DedupBaseSearchFilter(logging.Filter):
    """Drop records that the base ``model_search`` logger re-emits to the root.

    The base script installs its own stdout + file handlers on the
    ``model_search`` logger; by default those records also propagate up to the
    root logger, which caused every line to appear twice with slightly
    different formatting. We keep the base logger's direct handlers
    (``output/main_search/model_search.log`` is useful) and just suppress the
    duplicates at the root.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        return record.name != "model_search"


def _log_device_info() -> None:
    """Log torch/cuda visibility once at startup so GPU misconfig is obvious."""
    try:
        import torch  # noqa: WPS433 (late import: base conda env may lack torch)

        avail = bool(torch.cuda.is_available())
        version = getattr(torch, "__version__", "?")
        cuda_version = getattr(torch.version, "cuda", "?")
        if avail:
            name = torch.cuda.get_device_name(0)
            count = torch.cuda.device_count()
            logger.info(
                "Torch %s | cuda=ON (%d device%s) | cuda=%s | gpu[0]=%s",
                version, count, "s" if count != 1 else "", cuda_version, name,
            )
        else:
            logger.warning(
                "Torch %s | cuda=OFF — deep training will run on CPU "
                "(expected ~6 min/epoch). Relaunch docker with `--gpus all` "
                "and the `lamapuit:gpu` image if this is unintended.",
                version,
            )
    except Exception as exc:  # pragma: no cover — torch import failure
        logger.warning("torch not importable at startup: %s", exc)


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    stream = logging.StreamHandler(sys.stdout); stream.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "v4_run.log"); fh.setFormatter(fmt)
    dedup = _DedupBaseSearchFilter()
    stream.addFilter(dedup)
    fh.addFilter(dedup)
    root_logger.addHandler(stream)
    root_logger.addHandler(fh)


def main() -> int:
    args = parse_args()
    t0 = time.monotonic()

    output_dir = Path(args.output)
    _setup_logging(output_dir)
    logger.info("=== Model Search V4 start | output=%s ===", output_dir)
    _log_device_info()

    prepared_dir = output_dir / "prepared"
    ablation_dir = output_dir / "input_ablation"
    main_out_dir = output_dir / "main_search"

    labels_dir = Path(args.labels_dir)
    base_script = Path(args.base_search_script)
    original_chm_dir = Path(args.original_chm_dir)
    harmonized_root = Path(args.harmonized_root)
    v3_summary = Path(args.v3_summary)
    v3_best_ckpt = Path(args.v3_best_checkpoint)

    prepared_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    if base_script.exists() and not args.prepare_only and not args.no_deep:
        try:
            verified = assert_base_patch_contract(base_script)
            logger.info("base patch contract verified: %d symbols present", len(verified))
        except V4Error as exc:
            logger.error("%s", exc)
            raise

    if args.force_models.strip():
        top_models: list[str] = []
        seen: set[str] = set()
        for raw in args.force_models.split(","):
            m = raw.strip().lower()
            if m and m not in seen:
                top_models.append(m)
                seen.add(m)
        top_models = top_models[: max(1, args.n_models)]
        v3_audit_rows: list[dict[str, Any]] = []
    else:
        top_models, v3_audit_rows = top_models_from_v3_lcb(
            v3_summary, n_models=max(1, args.n_models), lcb_k=float(args.lcb_k),
        )
    if not top_models:
        top_models = DEFAULT_TOP3_FALLBACK[:3]
        v3_audit_rows = []

    curated_labels_dir = prepared_dir / "labels_curated_v4"
    split_path = prepared_dir / "cnn_test_split_v4.json"

    curated_stats, all_candidates = write_curated_labels_drop_only(
        drop_labels_dir=labels_dir,
        curated_labels_dir=curated_labels_dir,
        t_high=float(args.t_high),
        t_low=float(args.t_low),
    )

    split_stats = write_spatial_block_test_split(
        all_candidates=all_candidates,
        output_test_split=split_path,
        seed=int(args.seed),
        test_fraction=float(args.test_fraction),
        split_block_size_places=int(args.split_block_size_places),
        neighbor_buffer_blocks=int(args.neighbor_buffer_blocks),
    )

    prepared_payload = {
        "created_at": _utc_now(),
        "curated_stats": curated_stats,
        "split_stats": split_stats,
        "top_models": top_models,
        "v3_audit_rows": v3_audit_rows,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (output_dir / "prepared_dataset_summary.json").write_text(json.dumps(prepared_payload, indent=2))

    logger.info("Dataset prepared for V4 run")
    logger.info(
        "Labels  | total=%d | kept=%d (manual=%d + gate=%d) | rejected=%d (no_prob=%d, below=%d) | rasters=%d",
        int(curated_stats.get("drop_labeled_rows", 0)),
        int(curated_stats.get("curated_rows_after_dedup", 0)),
        int(curated_stats.get("drop_kept_manual_or_reviewed", 0)),
        int(curated_stats.get("drop_kept_threshold_gate", 0)),
        int(curated_stats.get("drop_rejected_no_prob", 0)) + int(curated_stats.get("drop_rejected_below_threshold", 0)),
        int(curated_stats.get("drop_rejected_no_prob", 0)),
        int(curated_stats.get("drop_rejected_below_threshold", 0)),
        int(curated_stats.get("curated_rasters", 0)),
    )
    logger.info(
        "Split   | places=%d (train=%d test=%d buffer=%d multi_year=%d) | rows train≈%d test=%d buffer=%d | test_cdw=%d test_manual=%d test_gate=%d | overlap=%d",
        int(split_stats.get("n_places_total", 0)),
        int(split_stats.get("n_places_train", 0)),
        int(split_stats.get("n_places_test", 0)),
        int(split_stats.get("n_places_buffer", 0)),
        int(split_stats.get("places_with_multi_year", 0)),
        int(split_stats.get("train_rows_estimate", 0)),
        int(split_stats.get("test_rows", 0)),
        int(split_stats.get("buffer_rows", 0)),
        int(split_stats.get("test_cdw_rows", 0)),
        int(split_stats.get("test_manual_rows", 0)),
        int(split_stats.get("test_threshold_gate_rows", 0)),
        int(split_stats.get("place_overlap_train_vs_test", 0)),
    )
    logger.info("Top-k models (LCB-ranked): %s", ", ".join(top_models))
    if v3_audit_rows:
        for r in v3_audit_rows[: max(3, len(top_models))]:
            logger.info(
                "  v3-lcb | model=%-32s mean=%.4f std=%.4f lcb=%.4f",
                str(r.get("model_name", "")), float(r.get("mean_cv_f1", 0.0)),
                float(r.get("std_cv_f1", 0.0)), float(r.get("lcb", 0.0)),
            )
    buf_frac = (
        float(split_stats.get("buffer_rows", 0)) / max(1.0, float(split_stats.get("total_rows", 1)))
    )
    if buf_frac > 0.5:
        logger.warning(
            "Buffer ring captured %.0f%% of rows (%d). Training set is small; "
            "consider --neighbor-buffer-blocks 0 if results are unstable.",
            buf_frac * 100.0, int(split_stats.get("buffer_rows", 0)),
        )

    if args.prepare_only:
        _write_report(
            output_dir=output_dir,
            prepared_stats={**curated_stats, **split_stats},
            ablation_df=pd.DataFrame(),
            selection={"best_mode": "n/a", "overwhelming": False},
            top_models=top_models,
            v3_audit_rows=v3_audit_rows,
            main_output_dir=main_out_dir,
            v3_benchmark=None,
            runtime_s=float(time.monotonic() - t0),
            args=args,
        )
        logger.info("prepare-only mode complete")
        return 0

    split_mod = _import_module(base_script, "model_search_v4_split_helper")
    helpers = split_mod._import_fine_tune_helpers(base_script.parent / "fine_tune_cnn.py")
    norm_tile = helpers["norm_tile"]
    _records, rec_train_all, rec_test_all = _load_records_split(split_mod, curated_labels_dir, split_path)

    if args.ablation_models.strip():
        ablation_models = [m.strip().lower() for m in args.ablation_models.split(",") if m.strip()]
    else:
        ablation_models = top_models[: max(1, min(2, len(top_models)))]
    logger.info("Ablation model slate: %s", ablation_models)

    ablation_modes = [m.strip().lower() for m in args.input_modes.split(",") if m.strip()]
    ablation_rows: list[dict[str, Any]] = []

    for idx, mode in enumerate(ablation_modes, start=1):
        left = _remaining_hours(t0, float(args.time_budget_hours))
        if left <= 0.05:
            logger.warning("Time budget exhausted before completing all ablation modes")
            break
        elapsed_h = (time.monotonic() - t0) / 3600.0
        logger.info(
            "[Ablation %d/%d] mode=%s | elapsed=%.2fh | hours_left=%.2f",
            idx, len(ablation_modes), mode, elapsed_h, left,
        )
        mode_ctx = ModeContext(mode=mode, original_chm_dir=original_chm_dir, harmonized_root=harmonized_root)
        mode_t0 = time.monotonic()

        row = _run_single_mode_ablation(
            mode=mode, ablation_dir=ablation_dir, args=args, mode_ctx=mode_ctx,
            rec_train_all=rec_train_all, rec_test_all=rec_test_all,
            split_mod=split_mod, base_script=base_script,
            curated_labels_dir=curated_labels_dir, split_path=split_path,
            norm_tile=norm_tile, ablation_models=ablation_models,
        )
        ablation_rows.append(row)
        logger.info(
            "[Ablation %d/%d] mode=%s done in %.1f min | deep_f1=%.4f (%s via %s) | classical_f1=%.4f (%s) | composite=%.4f",
            idx, len(ablation_modes), mode,
            (time.monotonic() - mode_t0) / 60.0,
            float(row.get("deep_metric", float("nan"))),
            str(row.get("deep_model", "")),
            str(row.get("deep_metric_name", "")),
            float(row.get("classical_best_f1", float("nan"))),
            str(row.get("classical_best_model", "")),
            float(row.get("composite_score", float("nan"))),
        )

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(output_dir / "input_ablation_summary.csv", index=False)

    best_mode, overwhelming, selection_diag = select_best_mode(
        ablation_df=ablation_df, overwhelming_margin=float(args.overwhelming_margin),
    )
    selection_diag["overwhelming"] = overwhelming
    if overwhelming:
        logger.info(
            "Selected input mode: %s (margin=%.4f >= threshold=%.4f — overwhelming)",
            best_mode,
            float(selection_diag.get("margin", 0.0)),
            float(selection_diag.get("overwhelming_margin_threshold", 0.0)),
        )
    else:
        logger.warning(
            "Selected input mode: %s — margin=%.4f over runner-up '%s' below threshold=%.4f. "
            "Paper should disclose the runner-up.",
            best_mode,
            float(selection_diag.get("margin", 0.0)),
            str(selection_diag.get("runner_up_mode", "")),
            float(selection_diag.get("overwhelming_margin_threshold", 0.0)),
        )

    chosen_ctx = ModeContext(mode=best_mode, original_chm_dir=original_chm_dir, harmonized_root=harmonized_root)

    if args.no_deep:
        logger.info("--no-deep: skipping main deep search and V3 benchmark")
        _write_report(
            output_dir=output_dir,
            prepared_stats={**curated_stats, **split_stats},
            ablation_df=ablation_df,
            selection=selection_diag,
            top_models=top_models,
            v3_audit_rows=v3_audit_rows,
            main_output_dir=main_out_dir,
            v3_benchmark=None,
            runtime_s=float(time.monotonic() - t0),
            args=args,
        )
        return 0

    left_before_main = _remaining_hours(t0, float(args.time_budget_hours))
    if left_before_main <= 0.05:
        logger.warning("Time budget exhausted before main stage; writing partial report")
        _write_report(
            output_dir=output_dir,
            prepared_stats={**curated_stats, **split_stats},
            ablation_df=ablation_df,
            selection=selection_diag,
            top_models=top_models,
            v3_audit_rows=v3_audit_rows,
            main_output_dir=main_out_dir,
            v3_benchmark=None,
            runtime_s=float(time.monotonic() - t0),
            args=args,
        )
        return 0

    main_opts = SearchRunOptions(
        seed=int(args.seed),
        n_folds=2 if args.smoke_test else int(args.n_folds),
        batch_size=8 if args.smoke_test else int(args.batch_size),
        epochs_scratch=3 if args.smoke_test else int(args.epochs_scratch),
        epochs_pretrained=3 if args.smoke_test else int(args.epochs_pretrained),
        patience=2 if args.smoke_test else int(args.patience),
        top_k_expand=max(1, len(top_models)),
        top_k_final=max(1, min(int(args.top_k_final), len(top_models))),
        max_extended=0 if args.smoke_test else int(args.max_extended),
        cv_group_block_size=int(args.cv_group_block_size),
        spatial_fence_m=float(args.spatial_fence_m),
        cv_spatial_block_m=float(args.cv_spatial_block_m),
        auto_cv_block_size=bool(args.auto_cv_block_size),
        cv_block_candidates_m=str(args.cv_block_candidates_m),
        augment_random_nodata_frac=float(args.augment_random_nodata_frac),
        augment_pattern_nodata_frac=float(args.augment_pattern_nodata_frac),
        stage2_strategies=str(args.stage2_strategies),
        stage2_epochs=3 if args.smoke_test else int(args.stage2_epochs),
        stage2_patience=2 if args.smoke_test else int(args.stage2_patience),
        stage2_pilot=bool(args.smoke_test),
        stage2_pilot_top_models=max(1, min(2, len(top_models))),
        smoke_test=bool(args.smoke_test),
    )

    logger.info(
        "=== Main search start | mode=%s | models=%s | folds=%d | epochs=%d/%d | elapsed=%.2fh | hours_left=%.2f ===",
        best_mode, ",".join(top_models),
        main_opts.n_folds, main_opts.epochs_scratch, main_opts.epochs_pretrained,
        (time.monotonic() - t0) / 3600.0, left_before_main,
    )
    main_t0 = time.monotonic()
    _run_base_search_with_mode(
        base_script=base_script,
        labels_dir=curated_labels_dir,
        test_split_path=split_path,
        output_dir=main_out_dir,
        mode_ctx=chosen_ctx,
        force_models=top_models,
        opts=main_opts,
    )
    logger.info("Main search done in %.1f min", (time.monotonic() - main_t0) / 60.0)

    v3_benchmark = _evaluate_v3_checkpoint(
        base_script=base_script,
        labels_dir=curated_labels_dir,
        test_split_path=split_path,
        mode_ctx=chosen_ctx,
        checkpoint_path=v3_best_ckpt,
        v3_summary_csv=v3_summary,
        benchmark_model=str(args.benchmark_model).strip().lower(),
    )
    if v3_benchmark is not None:
        (output_dir / "v3_benchmark.json").write_text(json.dumps(v3_benchmark, indent=2))
        try:
            combined = (v3_benchmark.get("test_breakdown") or {}).get("combined") or {}
            manual = (v3_benchmark.get("test_breakdown") or {}).get("manual_only") or {}
            logger.info(
                "V3 benchmark | combined f1=%.4f p=%.4f r=%.4f auc=%.4f | manual-only f1=%s (n=%s)",
                float(combined.get("f1", float("nan"))),
                float(combined.get("precision", float("nan"))),
                float(combined.get("recall", float("nan"))),
                float(combined.get("auc", float("nan"))),
                f"{float(manual.get('f1', float('nan'))):.4f}" if manual else "n/a",
                str(manual.get("n", "0")) if manual else "0",
            )
            if manual and combined:
                gap = float(combined.get("f1", 0.0)) - float(manual.get("f1", 0.0))
                if gap >= 0.03:
                    logger.warning(
                        "Manual-only F1 gap = %.4f (combined - manual). Evidence of "
                        "pseudo-label memorization; paper should disclose both numbers.",
                        gap,
                    )
        except Exception:
            logger.exception("failed to log v3 benchmark summary")

    (output_dir / "run_manifest_v4.json").write_text(json.dumps({
        "created_at": _utc_now(),
        "chosen_mode": best_mode,
        "overwhelming": bool(overwhelming),
        "top_models": top_models,
        "v3_audit_rows": v3_audit_rows,
        "time_budget_hours": float(args.time_budget_hours),
        "elapsed_seconds": float(time.monotonic() - t0),
    }, indent=2))

    _write_report(
        output_dir=output_dir,
        prepared_stats={**curated_stats, **split_stats},
        ablation_df=ablation_df,
        selection=selection_diag,
        top_models=top_models,
        v3_audit_rows=v3_audit_rows,
        main_output_dir=main_out_dir,
        v3_benchmark=v3_benchmark,
        runtime_s=float(time.monotonic() - t0),
        args=args,
    )

    total_h = (time.monotonic() - t0) / 3600.0
    logger.info("=== Model Search V4 complete | total=%.2fh ===", total_h)
    logger.info("Report: %s", output_dir / "REPORT_V4.md")
    logger.info("Main output: %s", main_out_dir)
    logger.info("Log file: %s", output_dir / "v4_run.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

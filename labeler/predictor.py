"""Classifier-driven CWD confidence and saliency heatmap overlays for one CHM chip."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from types import ModuleType

import numpy as np
from matplotlib import cm
from PIL import Image

from .renderer import read_chip


def _resolve_label_tiles_path() -> Path | None:
    candidates = [
        # Typical workspace path during local development.
        Path(__file__).resolve().parents[1] / "scripts" / "label_tiles.py",
        # Typical docker-compose path used by the labeler service.
        Path("/workspace/scripts/label_tiles.py"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=1)
def _load_label_tiles_module() -> ModuleType | None:
    path = _resolve_label_tiles_path()
    if path is None:
        return None
    try:
        spec = importlib.util.spec_from_file_location("labeler_label_tiles_runtime", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001 — gracefully degrade when module import fails
        return None


@lru_cache(maxsize=2)
def _load_classifier(model_path: str):
    """Load and cache the best tile-classifier predictor from label_tiles.py."""

    mod = _load_label_tiles_module()
    if mod is None:
        return None

    model_file = Path(model_path)
    if not model_file.exists():
        return None

    try:
        predictor = mod.CNNPredictor()
    except Exception:  # noqa: BLE001
        return None

    try:
        if model_file.suffix.lower() == ".json":
            ok = bool(predictor.load_ensemble_meta(model_file))
            return predictor if ok else None

        ok = bool(predictor.load_from_disk(model_file))
        if not ok:
            sidecar_meta = model_file.parent / "ensemble_meta.json"
            if sidecar_meta.exists():
                ok = bool(predictor.load_ensemble_meta(sidecar_meta))
        return predictor if ok else None
    except Exception:  # noqa: BLE001
        return None


_HEATMAP_CACHE_MAX = 256
_HEATMAP_CACHE: dict[tuple[str, str, str, int, int], np.ndarray] = {}


def _compute_classifier_heatmap(
    mode: str,
    predictor,
    model_tag: str,
    raster_path: str,
    row_off: int,
    col_off: int,
    chip: np.ndarray,
) -> np.ndarray:
    mod = _load_label_tiles_module()
    if mod is None:
        return np.zeros(chip.shape, dtype=np.uint8)

    key = (model_tag, mode, raster_path, row_off, col_off)
    cached = _HEATMAP_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        sal = mod._compute_heatmap(mode, predictor, chip, row_off, col_off, {})
        sal_u8 = np.asarray(sal, dtype=np.uint8)
    except Exception:  # noqa: BLE001
        sal_u8 = np.zeros(chip.shape, dtype=np.uint8)

    if sal_u8.shape != chip.shape:
        sal_u8 = np.array(
            Image.fromarray(sal_u8).resize((chip.shape[1], chip.shape[0]), Image.BILINEAR),
            dtype=np.uint8,
        )

    _HEATMAP_CACHE[key] = sal_u8
    if len(_HEATMAP_CACHE) > _HEATMAP_CACHE_MAX:
        # Drop an arbitrary oldest inserted key (dict preserves insertion order).
        _HEATMAP_CACHE.pop(next(iter(_HEATMAP_CACHE)))
    return sal_u8


def predict_chip(
    raster_path: str,
    row_off: int,
    col_off: int,
    chip_size: int,
    model_path: str | Path,
    confidence: float = 0.15,
    mode: str = "IntGrad",
) -> bytes:
    """Return an RGBA PNG saliency overlay from the best classifier heatmap."""

    _ = confidence  # Kept for API compatibility.
    model_path = str(model_path)
    predictor = _load_classifier(model_path)
    if predictor is None:
        return _empty_overlay(chip_size)

    chip = read_chip(raster_path, row_off, col_off, chip_size)
    sal_u8 = _compute_classifier_heatmap(mode, predictor, model_path, raster_path, row_off, col_off, chip)

    cmap = cm.get_cmap("inferno")
    rgba = (cmap(sal_u8.astype(np.float32) / 255.0) * 255).astype(np.uint8)
    # Keep low activations subtle, make strong regions prominent.
    alpha = np.clip(sal_u8.astype(np.float32) * 0.75, 0.0, 220.0).astype(np.uint8)
    rgba[..., 3] = alpha

    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def predict_chip_score(
    raster_path: str,
    row_off: int,
    col_off: int,
    chip_size: int,
    model_path: str | Path,
    confidence: float = 0.15,
) -> float:
    """Return tile-level classifier confidence P(CDW) in [0, 1]."""

    _ = confidence  # Kept for API compatibility.
    model_path = str(model_path)
    predictor = _load_classifier(model_path)
    if predictor is None:
        return 0.0

    try:
        chip = read_chip(raster_path, row_off, col_off, chip_size)
        prob = predictor.predict_proba_cdw(chip)
        if prob is None:
            return 0.0
        return float(np.clip(prob, 0.0, 1.0))
    except Exception:  # noqa: BLE001 — queue scoring should degrade gracefully
        return 0.0


def _empty_overlay(chip_size: int) -> bytes:
    arr = np.zeros((chip_size, chip_size, 4), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()

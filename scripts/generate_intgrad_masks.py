#!/usr/bin/env python3
"""
Generate CWD segmentation masks from baseline CHM tiles using ensemble
Integrated Gradients (Captum) with TTA aggregation.

Inputs
------
- Labels CSV: data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv
- Baseline CHM tiles: data/chm_variants/baseline_chm_20cm
- Ensemble checkpoints: output/tile_labels/ensemble_meta.json (4 models)

Outputs
-------
- *_cam.npy : normalized IntGrad heatmap in [0,1]
- *_mask.npy: binary mask in {0,1}
- manifest.csv with tile metadata and mask paths
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

try:
    import torch
except Exception as exc:
    print(f"ERROR: torch is required ({exc})")
    sys.exit(1)

try:
    from captum.attr import IntegratedGradients
except Exception as exc:
    print(f"ERROR: captum is required ({exc})")
    sys.exit(1)


_DEFAULT_LABELS = "data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv"
_DEFAULT_CHM_DIR = "data/chm_variants/baseline_chm_20cm"
_DEFAULT_META = "output/tile_labels/ensemble_meta.json"
_DEFAULT_OUTPUT = "output/intgrad_masks"

_CHM_MAX = 1.3
_MODEL_SIZE = 128

_SLD_BREAKPOINTS = [
    (0.000, "#580a0c"),
    (0.065, "#f2854e"),
    (0.130, "#f9ab66"),
    (0.195, "#fcbf75"),
    (0.260, "#fec57b"),
    (0.325, "#fed68f"),
    (0.390, "#fee29e"),
    (0.455, "#fdedaa"),
    (0.520, "#f7f4b3"),
    (0.585, "#e4f2b4"),
    (0.650, "#d6eeb1"),
    (0.715, "#c9e9ae"),
    (0.780, "#bce4a9"),
    (0.845, "#addca8"),
    (0.910, "#9dd3a7"),
    (0.975, "#8bc6aa"),
    (1.040, "#78b9ad"),
    (1.105, "#65acb0"),
    (1.170, "#529eb4"),
    (1.235, "#3e91b7"),
    (1.300, "#2b83ba"),
]
_DARK_THRESHOLD = 0.15 / _CHM_MAX

_FILENAME_RE = re.compile(r"^(?P<grid>\d+)_(?P<year>\d{4})_(?P<source>[^_]+)_(?P<tail>.+)$")


def _parse_name(stem: str) -> tuple[str | None, str | None, str | None]:
    m = _FILENAME_RE.match(stem)
    if not m:
        return None, None, None
    return m.group("grid"), m.group("year"), m.group("source")


def _safe_float(val: object, default: float = 0.0) -> float:
    try:
        return float(str(val).strip())
    except Exception:
        return default


def _safe_int(val: object, default: int | None = None) -> int | None:
    try:
        return int(str(val).strip())
    except Exception:
        return default


def _normalize_chm(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(tile, dtype=np.float32)
    valid = np.isfinite(arr)
    arr = np.where(valid, arr, 0.0)
    arr = np.clip(arr, 0.0, _CHM_MAX) / _CHM_MAX
    return arr.astype(np.float32), valid


def _normalize01(arr: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if valid is not None and np.any(valid):
        vals = x[valid]
    else:
        vals = x.reshape(-1)
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    mn = float(np.min(vals))
    mx = float(np.max(vals))
    if mx <= mn + 1e-8:
        out = np.zeros_like(x, dtype=np.float32)
    else:
        out = (x - mn) / (mx - mn)
    out = np.clip(out, 0.0, 1.0)
    if valid is not None:
        out[~valid] = 0.0
    return out.astype(np.float32)


def _make_sld_cmap():
    import matplotlib.colors as mcolors

    vals = [v / _CHM_MAX for v, _ in _SLD_BREAKPOINTS]
    colors = [c for _, c in _SLD_BREAKPOINTS]
    return mcolors.LinearSegmentedColormap.from_list("sld_terrain", list(zip(vals, colors)))


def _apply_sld(tile: np.ndarray) -> np.ndarray:
    """Return RGB uint8 image using SLD terrain colormap (0–1.3 m)."""
    nodata = ~np.isfinite(tile)
    is_zero = tile <= 0
    black_mask = nodata | is_zero
    t = tile.copy().astype(np.float32)
    t[black_mask] = 0.0
    t = np.clip(t, 0.0, _CHM_MAX) / _CHM_MAX
    cmap = _make_sld_cmap()
    rgb = (cmap(t)[:, :, :3] * 255).astype(np.uint8)
    dark_factor = np.where(
        t < _DARK_THRESHOLD,
        (t / _DARK_THRESHOLD) ** 0.7,
        1.0,
    ).astype(np.float32)
    rgb = (rgb.astype(np.float32) * dark_factor[:, :, np.newaxis]).astype(np.uint8)
    rgb[black_mask] = 0
    return rgb


def _fit_annotation_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    arr_f = np.asarray(arr, dtype=np.float32)
    if arr_f.shape == target_shape:
        return arr_f

    th, tw = target_shape
    h, w = arr_f.shape

    src_r0 = max(0, (h - th) // 2)
    src_c0 = max(0, (w - tw) // 2)
    src_r1 = min(h, src_r0 + th)
    src_c1 = min(w, src_c0 + tw)
    cropped = arr_f[src_r0:src_r1, src_c0:src_c1]

    out = np.zeros((th, tw), dtype=np.float32)
    ch, cw = cropped.shape
    dst_r0 = max(0, (th - ch) // 2)
    dst_c0 = max(0, (tw - cw) // 2)
    out[dst_r0 : dst_r0 + ch, dst_c0 : dst_c0 + cw] = cropped
    return out


def _apply_tta(arr: np.ndarray, rot_k: int, flip: str | None) -> np.ndarray:
    out = np.rot90(arr, rot_k) if rot_k else arr
    if flip == "h":
        out = np.fliplr(out)
    elif flip == "v":
        out = np.flipud(out)
    return out


def _invert_tta(arr: np.ndarray, rot_k: int, flip: str | None) -> np.ndarray:
    out = arr
    if flip == "h":
        out = np.fliplr(out)
    elif flip == "v":
        out = np.flipud(out)
    if rot_k:
        out = np.rot90(out, -rot_k)
    return out


class RasterCache:
    def __init__(self, max_open: int = 6) -> None:
        self._max_open = max(1, int(max_open))
        self._cache: OrderedDict[str, rasterio.DatasetReader] = OrderedDict()

    def get(self, path: Path) -> rasterio.DatasetReader:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        ds = rasterio.open(path)
        self._cache[key] = ds
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_open:
            _, old = self._cache.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return ds

    def close(self) -> None:
        for ds in list(self._cache.values()):
            try:
                ds.close()
            except Exception:
                pass
        self._cache.clear()


def _load_tile(cache: RasterCache, raster_path: Path, row_off: int, col_off: int, size: int) -> np.ndarray:
    ds = cache.get(raster_path)
    window = Window(col_off, row_off, size, size)
    data = ds.read(1, window=window, boundless=True, fill_value=0)
    data = data.astype(np.float32)
    if ds.nodata is not None:
        data[data == ds.nodata] = np.nan
    return data


_LABEL_TILES = None


def _load_label_tiles_module():
    global _LABEL_TILES
    if _LABEL_TILES is not None:
        return _LABEL_TILES
    lt_path = Path(__file__).with_name("label_tiles.py")
    if not lt_path.exists():
        _LABEL_TILES = None
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("label_tiles", lt_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LABEL_TILES = mod
        return mod
    except Exception:
        _LABEL_TILES = None
        return None


def _get_build_fn(name: str):
    lt = _load_label_tiles_module()
    if lt is not None:
        fn = getattr(lt, "_get_build_fn", None)
        if callable(fn):
            try:
                return fn(name)
            except Exception:
                return None
    return None


def _instantiate_model(build_fn):
    try:
        sig = inspect.signature(build_fn)
        if "pretrained" in sig.parameters:
            return build_fn(pretrained=False)
    except Exception:
        pass
    return build_fn()


def _load_models(
    *,
    ensemble_meta: Path | None,
    model_paths: list[Path] | None,
    device: torch.device,
    max_models: int,
) -> list[tuple[str, torch.nn.Module]]:
    models: list[tuple[str, torch.nn.Module]] = []

    paths: list[Path] = []
    if model_paths:
        paths = [Path(p) for p in model_paths]
    elif ensemble_meta is not None and ensemble_meta.exists():
        meta = json.loads(ensemble_meta.read_text())
        ckpts = meta.get("checkpoints", {})
        for tag in sorted(ckpts):
            entry = ckpts[tag]
            if isinstance(entry, dict):
                path = Path(entry.get("path", ""))
            else:
                path = Path(str(entry))
            if path.exists():
                paths.append(path)

    if not paths:
        raise RuntimeError("No model checkpoints found")

    paths = paths[: max_models]
    for path in paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        build_name = ""
        if isinstance(ckpt, dict):
            build_name = str(ckpt.get("build_fn_name", "")).strip()
            meta = ckpt.get("meta", {})
            if not build_name:
                build_name = str(meta.get("build_fn_name", "")).strip()
        if not build_name:
            build_name = "_build_deep_cnn_attn"

        build_fn = _get_build_fn(build_name)
        if build_fn is None:
            raise RuntimeError(f"Unknown build_fn '{build_name}' for {path}")
        net = _instantiate_model(build_fn).to(device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        net.load_state_dict(state, strict=False)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        models.append((path.name, net))

    if len(models) < max_models:
        raise RuntimeError(f"Expected {max_models} models, found {len(models)}")
    return models


def _ig_map(
    ig: IntegratedGradients,
    model: torch.nn.Module,
    tile: np.ndarray,
    device: torch.device,
    steps: int,
    target: int = 1,
) -> np.ndarray:
    x = torch.from_numpy(tile[np.newaxis, np.newaxis]).float().to(device)
    x.requires_grad_(True)
    baseline = torch.zeros_like(x)
    attr = ig.attribute(x, baselines=baseline, target=target, n_steps=steps)
    out = attr.detach().cpu().numpy()[0, 0]
    return np.abs(out.astype(np.float32))


def _compute_intgrad_tta(
    models: list[tuple[str, torch.nn.Module]],
    tile_norm: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
    steps: int,
    tta_mode: str,
) -> np.ndarray:
    flips = [None, "h"] if tta_mode == "8" else [None, "h", "v"]
    heat = np.zeros(tile_norm.shape, dtype=np.float32)
    count = 0

    for _, model in models:
        ig = IntegratedGradients(model)
        for rot_k in range(4):
            for flip in flips:
                tta_tile = _apply_tta(tile_norm, rot_k, flip)
                hmap = _ig_map(ig, model, tta_tile, device, steps=steps)
                hmap = _invert_tta(hmap, rot_k, flip)
                heat += hmap
                count += 1

    heat = heat / max(count, 1)
    heat[~valid] = 0.0
    return heat


def _threshold_heatmap(
    heat: np.ndarray,
    valid: np.ndarray,
    method: str,
    fixed_thr: float,
) -> tuple[np.ndarray, float]:
    if not np.any(valid):
        return np.zeros_like(heat, dtype=np.float32), 0.0

    if method == "fixed":
        thr = float(fixed_thr)
        mask = (heat >= thr).astype(np.float32)
        mask[~valid] = 0.0
        return mask, thr

    hm_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    if int(hm_u8.max()) == 0:
        return np.zeros_like(heat, dtype=np.float32), 0.0
    thr_u8, _ = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(thr_u8) / 255.0
    mask = (heat >= thr).astype(np.float32)
    mask[~valid] = 0.0
    return mask, thr


def _resolve_manual_mask(
    *,
    manual_mask_dir: Path | None,
    manual_map: dict[tuple[str, int, int], Path],
    raster_stem: str,
    row_off: int,
    col_off: int,
    chunk_size: int,
) -> Path | None:
    key = (raster_stem, row_off, col_off)
    if key in manual_map:
        return manual_map[key]
    if manual_mask_dir is None:
        return None

    candidates = [
        manual_mask_dir / f"{raster_stem}__r{row_off}_c{col_off}_s{chunk_size}_mask.npy",
        manual_mask_dir / f"{raster_stem}__r{row_off}_c{col_off}_mask.npy",
        manual_mask_dir / f"{raster_stem}_{row_off}_{col_off}_mask.npy",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _load_manual_mask(path: Path, target_shape: tuple[int, int]) -> np.ndarray | None:
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None
    arr = _fit_annotation_to_shape(arr, target_shape)
    return (arr >= 0.5).astype(np.float32)


def _load_manual_map(manifest: Path | None) -> dict[tuple[str, int, int], Path]:
    if manifest is None or not manifest.exists():
        return {}
    out: dict[tuple[str, int, int], Path] = {}
    with manifest.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raster = str(row.get("raster", "")).strip()
            row_off = _safe_int(row.get("row_off"))
            col_off = _safe_int(row.get("col_off"))
            if not raster or row_off is None or col_off is None:
                continue
            mask_raw = str(row.get("mask_path", "")).strip() or str(row.get("mask_file", "")).strip()
            if not mask_raw:
                continue
            path = Path(mask_raw)
            out[(Path(raster).stem, row_off, col_off)] = path
    return out


def _save_preview(
    chm: np.ndarray,
    manual_mask: np.ndarray | None,
    pred_mask: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chm_rgb = _apply_sld(chm)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")

    axes[0].imshow(chm_rgb)
    axes[0].set_title("Baseline CHM (SLD)", color="white")
    if manual_mask is None:
        axes[1].imshow(np.zeros_like(pred_mask), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Manual Mask (missing)", color="white")
    else:
        axes[1].imshow(manual_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Manual Mask", color="white")
    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted Mask", color="white")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10, color="white")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IntGrad masks for CWD detection")
    parser.add_argument("--labels", default=_DEFAULT_LABELS)
    parser.add_argument("--baseline-chm-dir", default=_DEFAULT_CHM_DIR)
    parser.add_argument("--ensemble-meta", default=_DEFAULT_META)
    parser.add_argument("--model-path", action="append", default=[], help="Explicit checkpoint path(s)")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT)
    parser.add_argument("--sources", default="manual,auto_skip")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--tta", choices=["8", "12"], default="12")
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--threshold", choices=["otsu", "fixed"], default="otsu")
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--model-size", type=int, default=_MODEL_SIZE)
    parser.add_argument("--manual-mask-dir", default="")
    parser.add_argument("--manual-mask-manifest", default="")
    parser.add_argument("--preview-count", type=int, default=0)
    parser.add_argument("--preview-dir", default="")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    baseline_dir = Path(args.baseline_chm_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manual_dir = Path(args.manual_mask_dir) if str(args.manual_mask_dir).strip() else None
    manual_manifest = Path(args.manual_mask_manifest) if str(args.manual_mask_manifest).strip() else None
    manual_map = _load_manual_map(manual_manifest)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    models = _load_models(
        ensemble_meta=Path(args.ensemble_meta) if args.ensemble_meta else None,
        model_paths=[Path(p) for p in args.model_path] if args.model_path else None,
        device=device,
        max_models=4,
    )
    print(f"Loaded {len(models)} model(s): {', '.join(name for name, _ in models)}")

    sources = {s.strip().lower() for s in str(args.sources).split(",") if s.strip()}
    max_items = int(args.limit) if args.limit > 0 else None

    cache = RasterCache()
    processed = 0
    skipped = 0
    missing_raster = 0
    missing_manual = 0
    previews: list[tuple[np.ndarray, np.ndarray | None, np.ndarray, str]] = []

    manifest_path = out_dir / "manifest.csv"
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    with labels_path.open("r", newline="", encoding="utf-8") as fh, manifest_path.open(
        "w", newline="", encoding="utf-8"
    ) as mh:
        reader = csv.DictReader(fh)
        writer = csv.DictWriter(
            mh,
            fieldnames=[
                "tile_id",
                "sample_id",
                "mapsheet",
                "year",
                "raster_path",
                "row_off",
                "col_off",
                "chunk_size",
                "label",
                "source",
                "mask_file",
                "cam_file",
                "threshold_method",
                "threshold_value",
                "mask_positive_ratio",
                "cam_mean",
                "cam_max",
                "model_count",
                "tta_count",
                "timestamp",
            ],
        )
        writer.writeheader()

        for row in reader:
            src = str(row.get("source", "")).strip().lower()
            if src not in sources:
                continue
            label = str(row.get("label", "")).strip().lower()
            if label not in {"cdw", "no_cdw"}:
                continue

            raster = str(row.get("raster", "")).strip()
            row_off = _safe_int(row.get("row_off"))
            col_off = _safe_int(row.get("col_off"))
            chunk_size = _safe_int(row.get("chunk_size"), _MODEL_SIZE)
            if not raster or row_off is None or col_off is None or chunk_size is None:
                skipped += 1
                continue

            raster_path = baseline_dir / raster
            if not raster_path.exists():
                missing_raster += 1
                continue

            raw_tile = _load_tile(cache, raster_path, row_off, col_off, chunk_size)
            tile_norm, valid = _normalize_chm(raw_tile)

            model_size = int(args.model_size)
            if tile_norm.shape != (model_size, model_size):
                tile_norm = cv2.resize(tile_norm, (model_size, model_size), interpolation=cv2.INTER_LINEAR)
                valid = cv2.resize(valid.astype(np.uint8), (model_size, model_size), interpolation=cv2.INTER_NEAREST) > 0

            if label == "no_cdw" or src == "auto_skip":
                cam = np.zeros((model_size, model_size), dtype=np.float32)
                mask = np.zeros((model_size, model_size), dtype=np.float32)
                thr = 0.0
            else:
                heat = _compute_intgrad_tta(
                    models=models,
                    tile_norm=tile_norm,
                    valid=valid,
                    device=device,
                    steps=int(args.ig_steps),
                    tta_mode=str(args.tta),
                )
                cam = _normalize01(heat, valid)
                mask, thr = _threshold_heatmap(
                    cam,
                    valid,
                    method=str(args.threshold),
                    fixed_thr=float(args.fixed_threshold),
                )

            if cam.shape != (chunk_size, chunk_size):
                cam = cv2.resize(cam, (chunk_size, chunk_size), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (chunk_size, chunk_size), interpolation=cv2.INTER_NEAREST)

            manual_mask = None
            manual_path = _resolve_manual_mask(
                manual_mask_dir=manual_dir,
                manual_map=manual_map,
                raster_stem=Path(raster).stem,
                row_off=row_off,
                col_off=col_off,
                chunk_size=chunk_size,
            )
            if manual_path is not None and manual_path.exists():
                manual_mask = _load_manual_mask(manual_path, (chunk_size, chunk_size))
                if manual_mask is not None:
                    mask = (mask > 0.5) & (manual_mask > 0.5)
                    mask = mask.astype(np.float32)
            else:
                if manual_dir is not None or manual_map:
                    missing_manual += 1

            stem = f"{Path(raster).stem}__r{row_off}_c{col_off}_s{chunk_size}"
            mask_file = f"{stem}_mask.npy"
            cam_file = f"{stem}_cam.npy"
            np.save(out_dir / cam_file, cam.astype(np.float32))
            np.save(out_dir / mask_file, mask.astype(np.float32))

            grid, year, source_token = _parse_name(Path(raster).stem)
            sample_id = f"{grid}_{year}_{source_token}" if grid and year and source_token else Path(raster).stem
            tile_id = f"{Path(raster).stem}__r{row_off}_c{col_off}"

            writer.writerow(
                {
                    "tile_id": tile_id,
                    "sample_id": sample_id,
                    "mapsheet": grid or "",
                    "year": year or "",
                    "raster_path": str(raster_path),
                    "row_off": int(row_off),
                    "col_off": int(col_off),
                    "chunk_size": int(chunk_size),
                    "label": label,
                    "source": src,
                    "mask_file": mask_file,
                    "cam_file": cam_file,
                    "threshold_method": str(args.threshold),
                    "threshold_value": f"{thr:.6f}",
                    "mask_positive_ratio": f"{float(mask.mean()):.6f}",
                    "cam_mean": f"{float(cam.mean()):.6f}",
                    "cam_max": f"{float(cam.max()):.6f}",
                    "model_count": len(models),
                    "tta_count": 4 * (2 if args.tta == "8" else 3),
                    "timestamp": timestamp,
                }
            )

            if (
                args.preview_count > 0
                and label == "cdw"
                and len(previews) < int(args.preview_count)
            ):
                previews.append((raw_tile, manual_mask, mask.copy(), tile_id))

            processed += 1
            if max_items is not None and processed >= max_items:
                break

    cache.close()

    if args.preview_count > 0:
        preview_dir = Path(args.preview_dir) if str(args.preview_dir).strip() else out_dir
        preview_dir.mkdir(parents=True, exist_ok=True)
        for idx, (raw_tile, manual_mask, pred_mask, title) in enumerate(previews, start=1):
            out_png = preview_dir / f"preview_{idx:02d}.png"
            _save_preview(raw_tile, manual_mask, pred_mask, title, out_png)

    print("Done.")
    print(f"  processed: {processed}")
    print(f"  skipped rows: {skipped}")
    print(f"  missing rasters: {missing_raster}")
    if manual_dir is not None or manual_map:
        print(f"  missing manual masks: {missing_manual}")
    print(f"  output: {out_dir}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()

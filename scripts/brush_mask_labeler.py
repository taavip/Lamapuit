#!/usr/bin/env python3
"""Interactive brush labeler for manual CWD mask refinement.

Outputs training-compatible artifacts:
- <stem>_mask.npy: binary mask in {0,1}
- <stem>_cam.npy: confidence map in [0,1]
- <stem>_neg.npy: explicit negative stroke mask in {0,1}

Modes:
- Single tile mode: --image ...
- Browser mode: --tile-csv ... (N/P navigation)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _configure_qt_env() -> None:
    # OpenCV Qt HighGUI in some environments cannot find bundled fonts.
    # Use common system font directories to avoid repetitive runtime warnings.
    if "QT_QPA_FONTDIR" not in os.environ:
        for font_dir in (
            "/usr/share/fonts/truetype/dejavu",
            "/usr/share/fonts/dejavu",
            "/usr/share/fonts/truetype/liberation2",
            "/usr/share/fonts/truetype/freefont",
        ):
            if Path(font_dir).exists():
                os.environ["QT_QPA_FONTDIR"] = font_dir
                break

    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false;qt.text.*=false")


_configure_qt_env()

import cv2
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.windows import bounds as window_bounds
except Exception:  # pragma: no cover - optional fallback when TIFF support is unavailable
    rasterio = None
    Window = None
    window_bounds = None

try:
    from cdw_detect.wms_utils import build_wms_layer_name, fetch_wms_for_bbox
except Exception:
    _repo_root = Path(__file__).resolve().parents[1]
    _src_dir = _repo_root / "src"
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    try:
        from cdw_detect.wms_utils import build_wms_layer_name, fetch_wms_for_bbox
    except Exception:  # pragma: no cover - optional fallback if WMS helpers are unavailable
        build_wms_layer_name = None
        fetch_wms_for_bbox = None


_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class TileEntry:
    tile_id: str
    output_stem: str
    image_path: Optional[Path] = None
    raster_path: Optional[Path] = None
    row_off: Optional[int] = None
    col_off: Optional[int] = None
    chunk_size: Optional[int] = None
    seed_mask_path: Optional[Path] = None
    seed_cam_path: Optional[Path] = None
    hotspot_path: Optional[Path] = None
    rank_score: Optional[float] = None


@dataclass
class LoadedTileState:
    entry: TileEntry
    mask: np.ndarray
    edited_neg: np.ndarray
    init_cam: Optional[np.ndarray]
    hotspot_map: Optional[np.ndarray]
    base_bgr: np.ndarray
    wms_layer: Optional[str]
    wms_bbox: Optional[tuple[float, float, float, float]]
    ortho_bgr: Optional[np.ndarray] = None
    ortho_attempted: bool = False
    cursor: tuple[int, int] = (-1, -1)
    active_label: Optional[int] = None
    dirty: bool = False


def _safe_stem(text: str) -> str:
    cleaned = _SAFE_STEM_RE.sub("_", str(text)).strip("_")
    return cleaned or "tile"


def _parse_int(text: str, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(str(text).strip())
    except Exception:
        return default


def _parse_float(text: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(str(text).strip())
    except Exception:
        return default


def _load_image_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D image after squeeze, got shape={arr.shape}")
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if suffix in {".tif", ".tiff"}:
        if rasterio is None:
            raise RuntimeError("rasterio is required to open TIFF inputs")
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def _load_array_if_exists(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        return None
    return _load_image_array(path)


def _fit_to_shape(arr: np.ndarray, target_shape: tuple[int, int], *, is_mask: bool) -> np.ndarray:
    if arr.shape == target_shape:
        return arr.astype(np.float32)

    th, tw = target_shape
    h, w = arr.shape

    if th >= h and tw >= w:
        out = np.zeros(target_shape, dtype=np.float32)
        r0 = (th - h) // 2
        c0 = (tw - w) // 2
        out[r0 : r0 + h, c0 : c0 + w] = arr.astype(np.float32)
        return out

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(arr.astype(np.float32), (tw, th), interpolation=interp).astype(np.float32)


def _fit_annotation_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Center-crop/pad an annotation map to match the tile shape.

    We avoid interpolation here to preserve existing manual strokes and avoid
    introducing resized artifacts when recovering from older mismatched outputs.
    """
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


def _mouse_wheel_delta(flags: int) -> int:
    try:
        get_delta = getattr(cv2, "getMouseWheelDelta", None)
        if callable(get_delta):
            return int(get_delta(flags))
    except Exception:
        pass

    # Fallback decode for builds without cv2.getMouseWheelDelta.
    delta = (int(flags) >> 16) & 0xFFFF
    if delta >= 0x8000:
        delta -= 0x10000
    return int(delta)


def _clip_text_for_width(text: str, max_px: int, font: int, scale: float, thickness: int) -> str:
    if max_px <= 8:
        return ""

    line = str(text)
    width = cv2.getTextSize(line, font, scale, thickness)[0][0]
    if width <= max_px:
        return line

    ellipsis = "..."
    lo = 0
    hi = len(line)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cand = line[:mid].rstrip() + ellipsis
        cand_w = cv2.getTextSize(cand, font, scale, thickness)[0][0]
        if cand_w <= max_px:
            lo = mid
        else:
            hi = mid - 1

    if lo <= 0:
        return ellipsis
    return line[:lo].rstrip() + ellipsis


def _panel_width_for_tile(tile_width: int) -> int:
    return int(max(220, min(360, round(float(tile_width) * 0.28))))


def _wrap_text_for_width(text: str, max_px: int, font: int, scale: float, thickness: int) -> list[str]:
    raw = str(text).strip()
    if not raw:
        return []

    words = raw.split()
    if not words:
        clipped = _clip_text_for_width(raw, max_px, font, scale, thickness)
        return [clipped] if clipped else []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        cand_w = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if cand_w <= max_px:
            current = candidate
            continue

        clipped = _clip_text_for_width(current, max_px, font, scale, thickness)
        if clipped:
            lines.append(clipped)
        current = word

    clipped = _clip_text_for_width(current, max_px, font, scale, thickness)
    if clipped:
        lines.append(clipped)
    return lines


def _canvas_to_tile_xy(x: int, y: int, tile_shape: tuple[int, int]) -> Optional[tuple[int, int]]:
    tile_h, tile_w = tile_shape
    panel_w = _panel_width_for_tile(tile_w)
    tx = int(x) - panel_w
    ty = int(y)
    if 0 <= tx < tile_w and 0 <= ty < tile_h:
        return tx, ty
    return None


def _normalize_for_display(chm: np.ndarray) -> np.ndarray:
    values = chm[np.isfinite(chm)]
    if values.size == 0:
        return np.zeros(chm.shape, dtype=np.uint8)

    lo = float(np.percentile(values, 2.0))
    hi = float(np.percentile(values, 98.0))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    norm = np.clip((chm - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _resolve_path(raw_value: str, csv_dir: Path, tile_root: Optional[Path]) -> Optional[Path]:
    value = str(raw_value).strip()
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    if tile_root is not None:
        candidates.append(tile_root / path)
    candidates.append(csv_dir / path)

    for cand in candidates:
        if cand.exists():
            return cand

    if tile_root is not None:
        return (tile_root / path).resolve()
    return (csv_dir / path).resolve()


def _first_nonempty(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _read_raster_tile(
    raster_path: Path,
    row_off: int,
    col_off: int,
    chunk_size: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if rasterio is None or Window is None or window_bounds is None:
        raise RuntimeError("rasterio is required for raster-window browser mode")

    with rasterio.open(raster_path) as src:
        win = Window(int(col_off), int(row_off), int(chunk_size), int(chunk_size))
        arr = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
        bbox = window_bounds(win, src.transform)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr, (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _load_entries_from_csv(csv_path: Path, output_dir: Path, tile_root: Optional[Path]) -> list[TileEntry]:
    entries: list[TileEntry] = []
    skipped = 0

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, start=1):
            tile_id = str(row.get("tile_id", "")).strip()
            if not tile_id:
                sample_id = str(row.get("sample_id", "")).strip()
                tile_id = sample_id if sample_id else f"tile_{idx:06d}"

            image_raw = _first_nonempty(row, ["image", "image_path", "chm_path", "chm_file", "tile_path"])
            raster_raw = _first_nonempty(row, ["raster_path"])
            mask_raw = _first_nonempty(row, ["init_mask", "mask_path", "mask_file"])
            cam_raw = _first_nonempty(row, ["init_cam", "cam_path", "cam_file"])
            hotspot_raw = _first_nonempty(row, ["hotspot_path", "hotspot_file", "cam_path", "cam_file"])

            image_path = _resolve_path(image_raw, csv_path.parent, tile_root) if image_raw else None
            raster_path = _resolve_path(raster_raw, csv_path.parent, tile_root) if raster_raw else None
            seed_mask_path = _resolve_path(mask_raw, csv_path.parent, tile_root) if mask_raw else None
            seed_cam_path = _resolve_path(cam_raw, csv_path.parent, tile_root) if cam_raw else None
            hotspot_path = _resolve_path(hotspot_raw, csv_path.parent, tile_root) if hotspot_raw else None

            row_off = _parse_int(str(row.get("row_off", "")).strip())
            col_off = _parse_int(str(row.get("col_off", "")).strip())
            chunk_size = _parse_int(str(row.get("chunk_size", "")).strip())
            rank_score = _parse_float(
                _first_nonempty(row, ["rank_score", "score", "final_score"]),
                default=None,
            )

            output_stem_raw = str(row.get("output_stem", "")).strip()
            if output_stem_raw:
                output_stem = _safe_stem(output_stem_raw)
            else:
                output_stem = _safe_stem(tile_id)

            has_image = image_path is not None
            has_raster_window = (
                raster_path is not None
                and row_off is not None
                and col_off is not None
                and chunk_size is not None
                and chunk_size > 0
            )
            if not has_image and not has_raster_window:
                skipped += 1
                continue

            entries.append(
                TileEntry(
                    tile_id=tile_id,
                    output_stem=output_stem,
                    image_path=image_path,
                    raster_path=raster_path,
                    row_off=row_off,
                    col_off=col_off,
                    chunk_size=chunk_size,
                    seed_mask_path=seed_mask_path,
                    seed_cam_path=seed_cam_path,
                    hotspot_path=hotspot_path,
                    rank_score=rank_score,
                )
            )

    if skipped > 0:
        print(f"[browser] skipped {skipped} csv row(s) without usable image/raster info")
    print(f"[browser] loaded {len(entries)} tile(s) from {csv_path}")

    _ = output_dir
    return entries


def _prepare_loaded_tile(entry: TileEntry, output_dir: Path, negative_conf: float) -> LoadedTileState:
    output_mask = output_dir / f"{entry.output_stem}_mask.npy"
    output_cam = output_dir / f"{entry.output_stem}_cam.npy"
    output_neg = output_dir / f"{entry.output_stem}_neg.npy"

    init_mask_path = output_mask if output_mask.exists() else entry.seed_mask_path
    init_cam_path = output_cam if output_cam.exists() else entry.seed_cam_path

    init_mask_raw = _load_array_if_exists(init_mask_path)
    init_cam_raw = _load_array_if_exists(init_cam_path)

    hotspot_path = entry.hotspot_path if entry.hotspot_path is not None else init_cam_path
    hotspot_raw = _load_array_if_exists(hotspot_path)

    bbox: Optional[tuple[float, float, float, float]] = None
    if (
        entry.raster_path is not None
        and entry.row_off is not None
        and entry.col_off is not None
        and entry.chunk_size is not None
        and entry.chunk_size > 0
    ):
        chm_raw, bbox = _read_raster_tile(
            entry.raster_path,
            row_off=int(entry.row_off),
            col_off=int(entry.col_off),
            chunk_size=int(entry.chunk_size),
        )
    elif entry.image_path is not None:
        chm_raw = _load_image_array(entry.image_path)
    else:
        raise RuntimeError(f"Tile has no image source: {entry.tile_id}")

    # Always keep tile display edge-to-edge from the source tile geometry.
    chm = chm_raw.astype(np.float32)
    h, w = chm.shape
    target_shape = (h, w)

    if init_mask_raw is not None:
        init_mask = _fit_annotation_to_shape(init_mask_raw, target_shape)
        mask = (init_mask >= 0.5).astype(np.float32)
    else:
        mask = np.zeros((h, w), dtype=np.float32)

    if init_cam_raw is not None:
        init_cam = np.clip(_fit_annotation_to_shape(init_cam_raw, target_shape), 0.0, 1.0).astype(np.float32)
    else:
        init_cam = None

    if hotspot_raw is not None:
        hotspot_map = np.clip(_fit_annotation_to_shape(hotspot_raw, target_shape), 0.0, 1.0).astype(np.float32)
    elif init_cam is not None:
        hotspot_map = init_cam.copy()
    else:
        hotspot_map = None

    if output_neg.exists():
        neg_raw = _load_image_array(output_neg)
        neg_map = _fit_annotation_to_shape(neg_raw, target_shape)
        edited_neg = neg_map >= 0.5
    else:
        edited_neg = np.zeros((h, w), dtype=bool)
        if init_cam is not None and output_cam.exists() and output_mask.exists():
            neg_conf = float(np.clip(negative_conf, 0.0, 1.0))
            edited_neg = (mask < 0.5) & (np.abs(init_cam - neg_conf) <= 1e-4)

    base_u8 = _normalize_for_display(chm)
    base_bgr = cv2.applyColorMap(base_u8, cv2.COLORMAP_VIRIDIS)

    wms_layer: Optional[str] = None
    wms_bbox: Optional[tuple[float, float, float, float]] = None
    if entry.raster_path is not None and bbox is not None and build_wms_layer_name is not None:
        wms_layer = build_wms_layer_name(entry.raster_path.name)
        if wms_layer is not None:
            wms_bbox = bbox

    return LoadedTileState(
        entry=entry,
        mask=mask,
        edited_neg=edited_neg,
        init_cam=init_cam,
        hotspot_map=hotspot_map,
        base_bgr=base_bgr,
        wms_layer=wms_layer,
        wms_bbox=wms_bbox,
    )


def _save_outputs(
    output_dir: Path,
    stem: str,
    mask: np.ndarray,
    edited_neg: np.ndarray,
    init_cam: Optional[np.ndarray],
    positive_conf: float,
    negative_conf: float,
    unedited_conf: float,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_bin = (mask >= 0.5).astype(np.float32)
    conf = np.clip(
        init_cam.copy() if init_cam is not None else np.full(mask_bin.shape, float(unedited_conf), dtype=np.float32),
        0.0,
        1.0,
    ).astype(np.float32)

    conf[mask_bin > 0.5] = float(np.clip(positive_conf, 0.0, 1.0))
    conf[edited_neg] = float(np.clip(negative_conf, 0.0, 1.0))

    mask_path = output_dir / f"{stem}_mask.npy"
    cam_path = output_dir / f"{stem}_cam.npy"
    neg_path = output_dir / f"{stem}_neg.npy"
    preview_path = output_dir / f"{stem}_preview.png"
    meta_path = output_dir / f"{stem}_manual_label_meta.json"

    np.save(mask_path, mask_bin.astype(np.float32))
    np.save(cam_path, conf.astype(np.float32))
    np.save(neg_path, edited_neg.astype(np.float32))

    preview = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    preview[mask_bin > 0.5] = (20, 20, 235)
    preview[edited_neg] = (220, 220, 20)
    cv2.imwrite(str(preview_path), preview)

    meta = {
        "mask_file": str(mask_path),
        "cam_file": str(cam_path),
        "negative_mask_file": str(neg_path),
        "preview_file": str(preview_path),
        "shape": [int(mask.shape[0]), int(mask.shape[1])],
        "positive_fraction": float(mask_bin.mean()),
        "explicit_negative_fraction": float(np.mean(edited_neg)),
        "positive_confidence": float(np.clip(positive_conf, 0.0, 1.0)),
        "negative_confidence": float(np.clip(negative_conf, 0.0, 1.0)),
        "unedited_confidence": float(np.clip(unedited_conf, 0.0, 1.0)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "mask": str(mask_path),
        "cam": str(cam_path),
        "neg": str(neg_path),
        "preview": str(preview_path),
        "meta": str(meta_path),
    }


def _cache_put(
    cache: dict[tuple, np.ndarray],
    order: list[tuple],
    key: tuple,
    value: np.ndarray,
    max_items: int,
) -> None:
    if key in cache:
        if key in order:
            order.remove(key)
    cache[key] = value
    order.append(key)
    while len(order) > max_items:
        old = order.pop(0)
        cache.pop(old, None)


def _ensure_orthophoto(
    tile: LoadedTileState,
    *,
    show_orthophoto: bool,
    wms_timeout: float,
    cache: dict[tuple, np.ndarray],
    order: list[tuple],
    cache_max: int,
) -> str:
    if not show_orthophoto:
        return ""
    if tile.ortho_attempted:
        return "" if tile.ortho_bgr is not None else "Orthophoto unavailable"

    tile.ortho_attempted = True
    if fetch_wms_for_bbox is None or tile.wms_layer is None or tile.wms_bbox is None:
        tile.ortho_bgr = None
        return "Orthophoto unavailable for this tile"

    h, w = tile.mask.shape
    bbox = tile.wms_bbox
    cache_key = (
        tile.wms_layer,
        round(float(bbox[0]), 3),
        round(float(bbox[1]), 3),
        round(float(bbox[2]), 3),
        round(float(bbox[3]), 3),
        int(w),
        int(h),
    )

    img = cache.get(cache_key)
    if img is None:
        img = fetch_wms_for_bbox(
            layer=tile.wms_layer,
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            width=int(w),
            height=int(h),
            timeout=float(max(0.1, wms_timeout)),
        )
        if img is not None:
            _cache_put(cache, order, cache_key, img, max_items=cache_max)

    if img is None:
        tile.ortho_bgr = None
        return "Orthophoto request failed"

    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    tile.ortho_bgr = img.astype(np.uint8)
    return ""


def _build_display_frame(
    *,
    tile: LoadedTileState,
    tile_index: int,
    tile_count: int,
    browser_mode: bool,
    show_hotspot: bool,
    show_orthophoto: bool,
    autosave_nav: bool,
    brush_radius: int,
    notice_text: str,
) -> np.ndarray:
    frame = (
        tile.ortho_bgr.copy()
        if (show_orthophoto and tile.ortho_bgr is not None)
        else tile.base_bgr.copy()
    )

    if show_hotspot and tile.hotspot_map is not None:
        hotspot_u8 = np.clip(tile.hotspot_map * 255.0, 0.0, 255.0).astype(np.uint8)
        hotspot_color = cv2.applyColorMap(hotspot_u8, cv2.COLORMAP_INFERNO)
        frame = cv2.addWeighted(frame, 0.58, hotspot_color, 0.42, 0.0)

    overlay = frame.copy()
    overlay[tile.mask > 0.5] = (20, 20, 235)
    overlay[tile.edited_neg] = (220, 220, 20)
    frame = cv2.addWeighted(frame, 0.62, overlay, 0.38, 0.0)

    cx, cy = tile.cursor
    if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
        color = (20, 220, 20) if tile.active_label == 1 else (220, 220, 20) if tile.active_label == 0 else (235, 235, 235)
        cv2.circle(frame, (cx, cy), brush_radius, color, 1, lineType=cv2.LINE_AA)

    pos_ratio = float(tile.mask.mean())
    neg_ratio = float(np.mean(tile.edited_neg))
    base_mode = "ORTHO" if (show_orthophoto and tile.ortho_bgr is not None) else "CHM"
    hot_mode = "ON" if show_hotspot else "OFF"
    dirty_flag = "*" if tile.dirty else ""

    panel_w = _panel_width_for_tile(frame.shape[1])
    tile_h, tile_w = frame.shape[:2]
    canvas_w = panel_w + tile_w + panel_w
    canvas = np.zeros((tile_h, canvas_w, 3), dtype=np.uint8)

    left = canvas[:, :panel_w]
    center = canvas[:, panel_w : panel_w + tile_w]
    right = canvas[:, panel_w + tile_w :]

    left[:, :] = (24, 24, 22)
    right[:, :] = (22, 24, 26)
    center[:, :] = frame

    cv2.line(canvas, (panel_w - 1, 0), (panel_w - 1, tile_h - 1), (58, 58, 58), 1)
    cv2.line(canvas, (panel_w + tile_w, 0), (panel_w + tile_w, tile_h - 1), (58, 58, 58), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.58
    text_scale = 0.43
    thickness = 1
    text_color = (234, 234, 234)
    accent = (120, 215, 255)

    def draw_panel(panel: np.ndarray, title: str, lines: list[str], *, notice: str = "") -> None:
        ph, pw = panel.shape[:2]
        cv2.rectangle(panel, (0, 0), (pw - 1, 34), (36, 36, 36), thickness=-1)
        cv2.putText(panel, title, (10, 23), font, title_scale, (250, 250, 250), 1, cv2.LINE_AA)

        max_w = max(16, pw - 20)
        y = 52
        for line in lines:
            wrapped = _wrap_text_for_width(line, max_w, font, text_scale, thickness)
            if not wrapped:
                y += 6
                continue
            for sub in wrapped:
                if y >= ph - 16:
                    break
                cv2.putText(panel, sub, (10, y), font, text_scale, text_color, thickness, cv2.LINE_AA)
                y += 16
            y += 2
            if y >= ph - 16:
                break

        if notice:
            notice_lines = _wrap_text_for_width(f"notice: {notice}", max_w - 8, font, text_scale, thickness)
            if notice_lines:
                box_h = 10 + len(notice_lines) * 16
                y0 = max(40, ph - box_h - 10)
                cv2.rectangle(panel, (8, y0), (pw - 8, y0 + box_h), (57, 73, 88), thickness=-1)
                ty = y0 + 17
                for line in notice_lines:
                    cv2.putText(panel, line, (12, ty), font, text_scale, accent, thickness, cv2.LINE_AA)
                    ty += 16

    rank_txt = f"{tile.entry.rank_score:.4f}" if tile.entry.rank_score is not None else "n/a"
    tile_line = f"{tile_index + 1}/{tile_count}" if browser_mode else "single tile"

    left_lines = [
        f"brush: {brush_radius}px",
        f"paint mode: {'positive' if tile.active_label == 1 else 'negative' if tile.active_label == 0 else 'idle'}",
        "",
        "LMB: paint positive CWD",
        "RMB: paint explicit negative",
        "wheel: larger/smaller brush",
        "[: smaller brush",
        "]: larger brush",
        "C: clear mask",
        "S: save tile",
    ]
    if browser_mode:
        left_lines.extend(
            [
                "N / P: next / previous",
                "A: toggle autosave nav",
            ]
        )
    left_lines.extend(["H: hotspot overlay", "O: orthophoto base", "Q or Esc: quit"])

    right_lines = [
        f"tile: {tile_line}",
        f"id: {tile.entry.tile_id}{dirty_flag}",
        f"score: {rank_txt}",
        f"output stem: {tile.entry.output_stem}",
        "",
        f"pos ratio: {pos_ratio:.4f}",
        f"neg ratio: {neg_ratio:.4f}",
        f"base: {base_mode}",
        f"hotspot: {hot_mode}",
        f"autosave nav: {'ON' if autosave_nav else 'OFF'}",
    ]

    draw_panel(left, "TOOLS", left_lines)
    draw_panel(right, "INFO", right_lines, notice=str(notice_text) if notice_text else "")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple brush labeler for manual CWD masks")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Single CHM/image tile to label (.npy, .tif, .png, .jpg)")
    input_group.add_argument("--tile-csv", type=str, help="CSV tile list for browser mode (supports N/P navigation)")

    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write *_mask.npy and *_cam.npy")
    parser.add_argument("--tile-id", type=str, default="", help="Tile id used to build output artifact stem")
    parser.add_argument("--output-stem", type=str, default="", help="Override output artifact stem")
    parser.add_argument("--init-mask", type=str, default="", help="Optional initial mask to start from")
    parser.add_argument("--init-cam", type=str, default="", help="Optional initial confidence map")
    parser.add_argument("--hotspot-map", type=str, default="", help="Optional hotspot map used for overlay")
    parser.add_argument("--tile-root", type=str, default="", help="Base directory to resolve relative CSV paths")
    parser.add_argument("--start-index", type=int, default=0, help="Initial index in browser mode")
    parser.add_argument("--no-autosave-nav", action="store_true", help="Disable autosave when pressing N/P")
    parser.add_argument("--show-hotspot", action="store_true", help="Start with hotspot overlay ON")
    parser.add_argument("--show-orthophoto", action="store_true", help="Start with orthophoto WMS base ON")
    parser.add_argument("--wms-timeout", type=float, default=8.0, help="Orthophoto WMS request timeout in seconds")
    parser.add_argument("--brush-radius", type=int, default=7)
    parser.add_argument("--positive-conf", type=float, default=1.0, help="Confidence written for positive painted pixels")
    parser.add_argument("--negative-conf", type=float, default=0.95, help="Confidence written for explicit negative strokes")
    parser.add_argument("--unedited-conf", type=float, default=0.05, help="Confidence assigned to untouched background")
    parser.add_argument("--window-name", type=str, default="CWD Brush Mask Labeler")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    entries: list[TileEntry] = []
    browser_mode = bool(args.tile_csv)
    if browser_mode:
        csv_path = Path(str(args.tile_csv))
        tile_root = Path(args.tile_root) if str(args.tile_root).strip() else None
        entries = _load_entries_from_csv(csv_path, output_dir=output_dir, tile_root=tile_root)
        if not entries:
            raise RuntimeError(f"No usable entries found in {csv_path}")
    else:
        image_path = Path(str(args.image))
        tile_id = args.tile_id.strip() if args.tile_id else image_path.stem
        output_stem = _safe_stem(args.output_stem if args.output_stem else tile_id)
        entries = [
            TileEntry(
                tile_id=tile_id,
                output_stem=output_stem,
                image_path=image_path,
                seed_mask_path=Path(args.init_mask) if args.init_mask else None,
                seed_cam_path=Path(args.init_cam) if args.init_cam else None,
                hotspot_path=Path(args.hotspot_map) if args.hotspot_map else (Path(args.init_cam) if args.init_cam else None),
            )
        ]

    idx = int(max(0, min(len(entries) - 1, int(args.start_index))))
    autosave_nav = browser_mode and (not bool(args.no_autosave_nav))

    ui: dict[str, object] = {
        "index": idx,
        "tile": _prepare_loaded_tile(entries[idx], output_dir=output_dir, negative_conf=float(args.negative_conf)),
        "show_hotspot": bool(args.show_hotspot),
        "show_orthophoto": bool(args.show_orthophoto),
        "autosave_nav": autosave_nav,
        "notice": "",
        "wms_cache": {},
        "wms_cache_order": [],
    }

    if ui["show_orthophoto"]:
        notice = _ensure_orthophoto(
            ui["tile"],  # type: ignore[arg-type]
            show_orthophoto=True,
            wms_timeout=float(args.wms_timeout),
            cache=ui["wms_cache"],  # type: ignore[arg-type]
            order=ui["wms_cache_order"],  # type: ignore[arg-type]
            cache_max=16,
        )
        if notice:
            ui["notice"] = notice
            ui["show_orthophoto"] = False

    state = {
        "brush_radius": max(1, int(args.brush_radius)),
    }

    def paint(x: int, y: int, label: int) -> None:
        tile: LoadedTileState = ui["tile"]  # type: ignore[assignment]
        h, w = tile.mask.shape
        brush = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(brush, (x, y), int(state["brush_radius"]), 1, thickness=-1)
        region = brush.astype(bool)

        if label == 1:
            tile.mask[region] = 1.0
            tile.edited_neg[region] = False
        else:
            tile.mask[region] = 0.0
            tile.edited_neg[region] = True
        tile.dirty = True

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        _ = param
        tile: LoadedTileState = ui["tile"]  # type: ignore[assignment]
        tile_xy = _canvas_to_tile_xy(x, y, tile.mask.shape)
        tile.cursor = tile_xy if tile_xy is not None else (-1, -1)

        if event == cv2.EVENT_LBUTTONDOWN:
            if tile_xy is None:
                return
            tile.active_label = 1
            paint(int(tile_xy[0]), int(tile_xy[1]), 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if tile_xy is None:
                return
            tile.active_label = 0
            paint(int(tile_xy[0]), int(tile_xy[1]), 0)
        elif event == getattr(cv2, "EVENT_MOUSEWHEEL", -1):
            if tile_xy is None:
                return
            delta = _mouse_wheel_delta(flags)
            if delta != 0:
                step = max(1, abs(int(delta)) // 120)
                if delta > 0:
                    state["brush_radius"] = min(256, int(state["brush_radius"]) + step)
                else:
                    state["brush_radius"] = max(1, int(state["brush_radius"]) - step)
                ui["notice"] = f"Brush radius: {int(state['brush_radius'])}"
        elif event == cv2.EVENT_MOUSEMOVE and tile.active_label is not None:
            if tile_xy is not None:
                paint(int(tile_xy[0]), int(tile_xy[1]), int(tile.active_label))
        elif event in {cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP}:
            tile.active_label = None

    def save_current(reason: str = "manual") -> dict[str, str]:
        tile: LoadedTileState = ui["tile"]  # type: ignore[assignment]
        saved = _save_outputs(
            output_dir=output_dir,
            stem=tile.entry.output_stem,
            mask=tile.mask,
            edited_neg=tile.edited_neg,
            init_cam=tile.init_cam,
            positive_conf=float(args.positive_conf),
            negative_conf=float(args.negative_conf),
            unedited_conf=float(args.unedited_conf),
        )
        tile.dirty = False
        if reason == "autosave":
            print(f"[autosave] {tile.entry.tile_id}")
        else:
            print("Saved:")
            print(f"  mask: {saved['mask']}")
            print(f"  cam: {saved['cam']}")
            print(f"  neg: {saved['neg']}")
            print(f"  preview: {saved['preview']}")
            print(f"  meta: {saved['meta']}")
        return saved

    def load_index(new_index: int) -> None:
        tile = _prepare_loaded_tile(
            entries[new_index],
            output_dir=output_dir,
            negative_conf=float(args.negative_conf),
        )
        ui["tile"] = tile
        ui["index"] = int(new_index)
        ui["notice"] = ""

        if ui["show_orthophoto"]:
            notice = _ensure_orthophoto(
                tile,
                show_orthophoto=True,
                wms_timeout=float(args.wms_timeout),
                cache=ui["wms_cache"],  # type: ignore[arg-type]
                order=ui["wms_cache_order"],  # type: ignore[arg-type]
                cache_max=16,
            )
            if notice:
                ui["notice"] = notice
                ui["show_orthophoto"] = False

        print(f"[tile {new_index + 1}/{len(entries)}] {tile.entry.tile_id}")

    def navigate(step: int) -> None:
        if not browser_mode:
            return

        tile: LoadedTileState = ui["tile"]  # type: ignore[assignment]
        if tile.dirty:
            if bool(ui["autosave_nav"]):
                save_current(reason="autosave")
            else:
                ui["notice"] = "Unsaved edits. Press S or enable autosave with A."
                return

        current_index = int(ui["index"])
        next_index = (current_index + int(step)) % len(entries)
        load_index(next_index)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(args.window_name, on_mouse)

    print("Brush labeler controls:")
    print("  Left mouse: paint positive CWD")
    print("  Right mouse: paint explicit negative/background")
    print("  Mouse wheel: larger/smaller brush")
    print("  [: smaller brush, ]: larger brush")
    print("  C: clear mask")
    print("  S: save current tile")
    if browser_mode:
        print("  N/P: next/previous tile")
        print("  A: toggle autosave on navigation")
    print("  H: toggle hotspot overlay")
    print("  O: toggle orthophoto WMS base")
    print("  Q or Esc: quit")

    while True:
        tile: LoadedTileState = ui["tile"]  # type: ignore[assignment]

        if bool(ui["show_orthophoto"]):
            notice = _ensure_orthophoto(
                tile,
                show_orthophoto=True,
                wms_timeout=float(args.wms_timeout),
                cache=ui["wms_cache"],  # type: ignore[arg-type]
                order=ui["wms_cache_order"],  # type: ignore[arg-type]
                cache_max=16,
            )
            if notice:
                ui["notice"] = notice
                ui["show_orthophoto"] = False

        frame = _build_display_frame(
            tile=tile,
            tile_index=int(ui["index"]),
            tile_count=len(entries),
            browser_mode=browser_mode,
            show_hotspot=bool(ui["show_hotspot"]),
            show_orthophoto=bool(ui["show_orthophoto"]),
            autosave_nav=bool(ui["autosave_nav"]),
            brush_radius=int(state["brush_radius"]),
            notice_text=str(ui["notice"]),
        )
        cv2.imshow(args.window_name, frame)
        key = cv2.waitKey(16) & 0xFF

        if key in {27, ord("q"), ord("Q")}:
            if tile.dirty and bool(ui["autosave_nav"]):
                save_current(reason="autosave")
                print("Exit after autosave.")
                break
            if tile.dirty and not bool(ui["autosave_nav"]):
                ui["notice"] = "Unsaved edits. Press S to save, or A to enable autosave."
                continue
            print("Exit without additional save.")
            break

        if key == ord("["):
            state["brush_radius"] = max(1, int(state["brush_radius"]) - 1)
            continue

        if key == ord("]"):
            state["brush_radius"] = min(256, int(state["brush_radius"]) + 1)
            continue

        if key in {ord("c"), ord("C")}:
            tile.mask[:] = 0.0
            tile.edited_neg[:] = False
            tile.dirty = True
            ui["notice"] = "Mask cleared"
            continue

        if key in {ord("s"), ord("S")}:
            save_current(reason="manual")
            ui["notice"] = "Saved"
            if not browser_mode:
                break
            continue

        if key in {ord("n"), ord("N")}:
            navigate(+1)
            continue

        if key in {ord("p"), ord("P")}:
            navigate(-1)
            continue

        if key in {ord("h"), ord("H")}:
            ui["show_hotspot"] = not bool(ui["show_hotspot"])
            ui["notice"] = f"Hotspot {'ON' if ui['show_hotspot'] else 'OFF'}"
            continue

        if key in {ord("o"), ord("O")}:
            ui["show_orthophoto"] = not bool(ui["show_orthophoto"])
            if bool(ui["show_orthophoto"]):
                tile.ortho_attempted = False
                notice = _ensure_orthophoto(
                    tile,
                    show_orthophoto=True,
                    wms_timeout=float(args.wms_timeout),
                    cache=ui["wms_cache"],  # type: ignore[arg-type]
                    order=ui["wms_cache_order"],  # type: ignore[arg-type]
                    cache_max=16,
                )
                if notice:
                    ui["notice"] = notice
                    ui["show_orthophoto"] = False
                else:
                    ui["notice"] = "Orthophoto ON"
            else:
                ui["notice"] = "Orthophoto OFF"
            continue

        if key in {ord("a"), ord("A")} and browser_mode:
            ui["autosave_nav"] = not bool(ui["autosave_nav"])
            ui["notice"] = f"Autosave nav {'ON' if ui['autosave_nav'] else 'OFF'}"
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

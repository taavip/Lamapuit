"""Fetch Estonian Maa-amet orthophotos for a tile via WMS."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import rasterio
from PIL import Image
from rasterio.windows import Window, bounds as window_bounds

from .tile_index import TileChip


def _tile_bbox(chip: TileChip) -> tuple[float, float, float, float]:
    with rasterio.open(chip.raster_path) as src:
        win = Window(chip.col_off, chip.row_off, chip.chip_size, chip.chip_size)
        return window_bounds(win, src.transform)


def _context_bbox(
    chip: TileChip,
    *,
    row_radius: int,
    col_radius: int,
) -> tuple[float, float, float, float]:
    with rasterio.open(chip.raster_path) as src:
        full = Window(0, 0, src.width, src.height)
        win = Window(
            chip.col_off - col_radius * chip.chip_size,
            chip.row_off - row_radius * chip.chip_size,
            chip.chip_size * (2 * col_radius + 1),
            chip.chip_size * (2 * row_radius + 1),
        ).intersection(full)
        return window_bounds(win, src.transform)


@lru_cache(maxsize=256)
def _cached_fetch(
    layer: str,
    bbox_key: tuple[float, float, float, float],
    out_width: int,
    out_height: int,
) -> bytes | None:
    try:
        from cdw_detect.wms_utils import fetch_wms_for_bbox
    except Exception:
        return None
    rgb = fetch_wms_for_bbox(layer=layer, bbox=bbox_key, width=out_width, height=out_height)
    if rgb is None:
        return None
    img = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def fetch_orthophoto(chip: TileChip, out_size: int = 640) -> bytes | None:
    """Return PNG bytes of the orthophoto covering this tile, or None if unavailable."""
    try:
        from cdw_detect.wms_utils import build_wms_layer_name
    except Exception:
        return None

    layer = build_wms_layer_name(Path(chip.raster_path).name)
    if layer is None:
        return None
    bbox = _tile_bbox(chip)
    return _cached_fetch(layer, tuple(round(v, 3) for v in bbox), out_size, out_size)


def fetch_orthophoto_context(
    chip: TileChip,
    *,
    row_radius: int,
    col_radius: int,
    out_chip_size: int | None = None,
) -> bytes | None:
    """Return one orthophoto PNG that covers the entire context neighborhood."""

    try:
        from cdw_detect.wms_utils import build_wms_layer_name
    except Exception:
        return None

    row_radius = max(0, int(row_radius))
    col_radius = max(0, int(col_radius))
    chip_px = int(out_chip_size or chip.chip_size)
    out_width = max(chip_px, chip_px * (2 * col_radius + 1))
    out_height = max(chip_px, chip_px * (2 * row_radius + 1))

    layer = build_wms_layer_name(Path(chip.raster_path).name)
    if layer is None:
        return None

    bbox = _context_bbox(chip, row_radius=row_radius, col_radius=col_radius)
    return _cached_fetch(
        layer,
        tuple(round(v, 3) for v in bbox),
        out_width,
        out_height,
    )

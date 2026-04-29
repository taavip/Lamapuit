"""Render CHM chips as PNG with colormap / hillshade overlays."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import rasterio
from matplotlib import cm
from matplotlib import colors as mcolors
from PIL import Image
from rasterio.windows import Window


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
_MAX_HAG = 1.3
_DARK_THRESHOLD = 0.15 / _MAX_HAG


def _make_sld_cmap() -> mcolors.LinearSegmentedColormap:
    vals = [v / _MAX_HAG for v, _ in _SLD_BREAKPOINTS]
    colors = [c for _, c in _SLD_BREAKPOINTS]
    return mcolors.LinearSegmentedColormap.from_list("sld_terrain", list(zip(vals, colors)))


_SLD_CMAP = _make_sld_cmap()


def _apply_sld(chip: np.ndarray) -> np.ndarray:
    """Apply the same SLD terrain style used in scripts/label_tiles.py."""

    nodata = ~np.isfinite(chip)
    is_zero = chip <= 0
    black_mask = nodata | is_zero

    t = chip.astype("float32", copy=True)
    t[black_mask] = 0.0
    t = np.clip(t, 0.0, _MAX_HAG) / _MAX_HAG

    rgb = (_SLD_CMAP(t)[:, :, :3] * 255).astype("uint8")
    dark_factor = np.where(
        t < _DARK_THRESHOLD,
        (t / _DARK_THRESHOLD) ** 0.7,
        1.0,
    ).astype("float32")
    rgb = (rgb.astype("float32") * dark_factor[:, :, np.newaxis]).astype("uint8")
    rgb[black_mask] = 0
    return rgb


def read_chip(
    raster_path: str, row_off: int, col_off: int, chip_size: int = 640
) -> np.ndarray:
    """Read a windowed chip from a CHM raster, padding edges with NaN."""

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        win = Window(col_off, row_off, chip_size, chip_size)
        # rasterio handles reads that extend past bounds by returning zeros;
        # we clip the window and pad manually so we can mark pad as NaN.
        win_clip = win.intersection(Window(0, 0, src.width, src.height))
        data = src.read(1, window=win_clip).astype("float32")

    chip = np.full((chip_size, chip_size), np.nan, dtype="float32")
    h, w = data.shape
    chip[:h, :w] = data

    # Nodata & negatives → NaN
    if nodata is not None:
        chip = np.where(chip == nodata, np.nan, chip)
    chip = np.where(chip < 0, np.nan, chip)
    return chip


def smooth_chip(chip: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Apply a small Gaussian blur to a CHM chip while preserving NaN pixels."""

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return chip

    valid = ~np.isnan(chip)
    filled = np.where(valid, chip, 0.0)
    weight = valid.astype("float32")
    num = gaussian_filter(filled, sigma=sigma)
    den = gaussian_filter(weight, sigma=sigma)
    out = np.where(den > 0, num / np.maximum(den, 1e-6), np.nan)
    out[~valid] = np.nan
    return out.astype("float32")


def render_chm_heatmap(
    chip: np.ndarray,
    lo: float = 0.1,
    hi: float = 1.5,
    mode: str = "IntGrad",
) -> bytes:
    """CHM-intrinsic inferno heatmap styled like label_tiles guidance views.

    `mode` is accepted for UI parity with label_tiles heatmap cycling.
    """

    valid = ~np.isnan(chip)
    filled = np.where(valid, chip, 0.0)
    in_band = valid & (chip >= lo) & (chip <= hi)
    dy, dx = np.gradient(filled)
    grad = np.hypot(dx, dy)
    score = np.where(in_band, np.clip(grad / 0.4, 0.0, 1.0), 0.0)

    _ = mode  # mode names are UI-facing; rendering remains CHM-intrinsic.
    cmap = cm.get_cmap("inferno")
    rgba = (cmap(score) * 255).astype("uint8")
    alpha = (score * 220).astype("uint8")
    rgba[..., 3] = alpha
    img = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_png(
    chip: np.ndarray,
    colormap: str = "sld",
    vmin: float = 0.0,
    vmax: float = 1.3,
) -> bytes:
    """Apply CHM symbology and return PNG bytes."""

    if colormap == "sld":
        rgb = _apply_sld(chip)
        rgba = np.zeros((*chip.shape, 4), dtype="uint8")
        rgba[..., :3] = rgb
        rgba[..., 3] = 255

        img = Image.fromarray(rgba, mode="RGBA")
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    valid = ~np.isnan(chip)
    norm = np.zeros_like(chip, dtype="float32")
    norm[valid] = np.clip((chip[valid] - vmin) / (vmax - vmin), 0.0, 1.0)

    cmap = cm.get_cmap(colormap)
    rgba = (cmap(norm) * 255).astype("uint8")
    # NaN pixels → transparent
    rgba[..., 3] = np.where(valid, 255, 0)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_hillshade(chip: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0) -> bytes:
    """Compute a simple numpy hillshade from the CHM and return grayscale PNG."""

    filled = np.where(np.isnan(chip), 0.0, chip)
    dy, dx = np.gradient(filled)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)

    az = np.deg2rad(360.0 - azimuth + 90.0)
    alt = np.deg2rad(altitude)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shaded = np.clip(shaded, 0.0, 1.0)

    rgba = np.zeros((*chip.shape, 4), dtype="uint8")
    gray = (shaded * 255).astype("uint8")
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = np.where(np.isnan(chip), 0, 255).astype("uint8")

    img = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_blank_mask(chip_size: int = 640) -> bytes:
    """Return a fully-transparent PNG of the chip size (for missing masks)."""

    arr = np.zeros((chip_size, chip_size, 4), dtype="uint8")
    buf = BytesIO()
    Image.fromarray(arr, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()

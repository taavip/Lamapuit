"""WMS tile fetching utilities for Estonian LiDAR data.

Requires: pip install cdw-detect[wms]
"""

from __future__ import annotations

import re
from io import BytesIO

import numpy as np
import requests
from PIL import Image

WMS_BASE_URL = "https://kaart.maaamet.ee/wms/ajalooline"
WMS_CRS = "EPSG:3301"

_CHM_FILENAME_RE = re.compile(r"^(?P<grid>\d+)_(?P<year>\d{4})_(?P<token>[a-z]+)_chm_max_hag")
_TOKEN_MAP = {
    "madal": "asulad",
    "tava": "aeropildistamine",
}


def parse_chm_filename(filename: str) -> tuple[int, str] | None:
    """Parse CHM filename and return (year, area_token), or None if unsupported."""
    match = _CHM_FILENAME_RE.match(filename)
    if not match:
        return None
    return int(match.group("year")), match.group("token")


def map_campaign_token(token: str) -> str | None:
    """Map filename token to WMS campaign token, strictly for supported values."""
    return _TOKEN_MAP.get(token.strip().lower())


def build_wms_layer_name(filename: str) -> str | None:
    """Build layer name like 'of2024asulad' from CHM filename metadata."""
    parsed = parse_chm_filename(filename)
    if parsed is None:
        return None
    year, token = parsed
    mapped = map_campaign_token(token)
    if mapped is None:
        return None
    return f"of{year}{mapped}"


def build_wms_getmap_params(
    *,
    layer: str,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    crs: str = WMS_CRS,
) -> dict[str, str]:
    """Create WMS GetMap parameters for Maa-amet historical orthophoto service."""
    minx, miny, maxx, maxy = bbox
    return {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": layer,
        "styles": "",
        "bbox": f"{minx:.3f},{miny:.3f},{maxx:.3f},{maxy:.3f}",
        "width": str(int(width)),
        "height": str(int(height)),
        "srs": crs,
        "format": "image/png",
        "transparent": "FALSE",
    }


def fetch_wms_for_bbox(
    *,
    layer: str,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    timeout: float = 8.0,
    base_url: str = WMS_BASE_URL,
) -> np.ndarray | None:
    """Fetch WMS image and return RGB uint8 array, or None on failure."""
    params = build_wms_getmap_params(layer=layer, bbox=bbox, width=width, height=height)
    try:
        resp = requests.get(base_url, params=params, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None
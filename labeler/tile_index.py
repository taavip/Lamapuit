"""Tile index: scan CHM rasters, build 640px chip inventory, track mask status."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import rasterio


FILENAME_RE = re.compile(
    r"^(?P<grid>\d+)_(?P<year>\d{4})_(?P<source>[^_]+)_(?P<tail>.+)$"
)


@dataclass
class TileChip:
    tile_id: str
    raster_path: str
    row_off: int
    col_off: int
    chip_size: int
    raster_stem: str
    grid: str | None = None
    year: str | None = None
    source: str | None = None
    product: str | None = None
    location_key: str | None = None
    label_key: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _parse_name(stem: str) -> tuple[str | None, str | None, str | None]:
    m = FILENAME_RE.match(stem)
    if not m:
        return None, None, None
    return m.group("grid"), m.group("year"), m.group("source")


def _infer_product(stem: str) -> str | None:
    name = stem.lower()
    if "_harmonized_dem_last_raw_chm" in name:
        return "harmonized_raw"
    if "_harmonized_dem_last_gauss_chm" in name:
        return "harmonized_gauss"
    if "_chm_max_hag" in name:
        return "chm_max_hag"
    return None


def _location_key(grid: str | None, year: str | None, source: str | None, stem: str) -> str:
    if grid and year and source:
        return f"{grid}_{year}_{source}"
    return stem


def _label_key(
    *,
    tile_id: str,
    grid: str | None,
    year: str | None,
    source: str | None,
    row_off: int,
    col_off: int,
    chip_size: int,
) -> str:
    if not (grid and year):
        return tile_id
    source_token = source or "unknown"
    return f"{grid}_{year}_{source_token}__r{row_off}_c{col_off}_s{chip_size}"


def preferred_label_key(tile_id: str, by_id: dict[str, TileChip] | None = None) -> str:
    if by_id is None:
        return tile_id
    chip = by_id.get(tile_id)
    if chip is None:
        return tile_id
    return chip.label_key or tile_id


def candidate_label_keys(tile_id: str, by_id: dict[str, TileChip] | None = None) -> list[str]:
    keys = [preferred_label_key(tile_id, by_id)]
    if tile_id not in keys:
        keys.append(tile_id)
    return keys


def _normalize_chip(chip: TileChip) -> TileChip:
    chip.product = chip.product or _infer_product(chip.raster_stem)
    chip.location_key = chip.location_key or _location_key(
        chip.grid, chip.year, chip.source, chip.raster_stem
    )
    chip.label_key = chip.label_key or _label_key(
        tile_id=chip.tile_id,
        grid=chip.grid,
        year=chip.year,
        source=chip.source,
        row_off=chip.row_off,
        col_off=chip.col_off,
        chip_size=chip.chip_size,
    )
    return chip


def scan(chm_dir: Path | str, chip_size: int = 640, stride: int | None = None) -> list[TileChip]:
    """Scan *chm_dir* for GeoTIFFs, yield non-overlapping chips.

    Edge chips that extend past the raster bounds are still returned; the
    renderer is responsible for padding.
    """

    chm_dir = Path(chm_dir)
    stride = stride or chip_size
    chips: list[TileChip] = []

    direct = sorted(chm_dir.glob("*.tif"))
    candidates = direct if direct else sorted(chm_dir.rglob("*.tif"))

    for tif in candidates:
        try:
            with rasterio.open(tif) as src:
                w, h = src.width, src.height
        except rasterio.RasterioIOError:
            continue

        stem = tif.stem
        grid, year, source = _parse_name(stem)
        product = _infer_product(stem)
        # Keep only known CHM products; skip unrelated TIFFs under dataset roots.
        if product is None:
            continue
        for row_off in range(0, h, stride):
            for col_off in range(0, w, stride):
                tile_id = f"{stem}__r{row_off}_c{col_off}"
                chips.append(
                    _normalize_chip(
                        TileChip(
                            tile_id=tile_id,
                            raster_path=str(tif),
                            row_off=row_off,
                            col_off=col_off,
                            chip_size=chip_size,
                            raster_stem=stem,
                            grid=grid,
                            year=year,
                            source=source,
                            product=product,
                            location_key=_location_key(grid, year, source, stem),
                            label_key=_label_key(
                                tile_id=tile_id,
                                grid=grid,
                                year=year,
                                source=source,
                                row_off=row_off,
                                col_off=col_off,
                                chip_size=chip_size,
                            ),
                        )
                    )
                )
    return chips


def status(tile_id: str, mask_dir: Path | str, by_id: dict[str, TileChip] | None = None) -> str:
    mask_dir = Path(mask_dir)
    keys = candidate_label_keys(tile_id, by_id)

    for key in keys:
        meta_path = mask_dir / f"{key}_meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("review_flag"):
                return "needs-review"
        except (json.JSONDecodeError, OSError):
            pass

    for key in keys:
        if (mask_dir / f"{key}_mask.png").exists():
            return "labeled"

    return "unlabeled"


def neighbors(
    chip: TileChip,
    all_chips: list[TileChip],
    *,
    row_radius: int = 1,
    col_radius: int = 1,
) -> dict[str, str | None]:
    """Return neighborhood tile_ids around *chip* for requested radii.

    Keys are "dr_dc" with dr in [-row_radius..row_radius] and
    dc in [-col_radius..col_radius], excluding 0_0.
    """

    cs = chip.chip_size
    by_pos: dict[tuple[str, int, int], str] = {
        (c.raster_stem, c.row_off, c.col_off): c.tile_id for c in all_chips
    }
    out: dict[str, str | None] = {}
    for dr in range(-row_radius, row_radius + 1):
        for dc in range(-col_radius, col_radius + 1):
            key = (dr, dc)
            if key == (0, 0):
                continue
            row = chip.row_off + dr * cs
            col = chip.col_off + dc * cs
            out[f"{dr}_{dc}"] = by_pos.get((chip.raster_stem, row, col))
    return out


def load_or_build(cache_path: Path | str, chm_dir: Path | str, chip_size: int = 640) -> list[TileChip]:
    """Load the tile index from JSON cache, or rebuild and write it."""

    cache_path = Path(cache_path)
    chm_dir = Path(chm_dir)
    chm_dir_key = str(chm_dir.resolve())
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            if isinstance(data, list):
                # Legacy cache format lacked source metadata (chm_dir/chip_size).
                # Rebuild once so product catalogs stay correct when CHM_DIR changes.
                data = None

            if isinstance(data, dict):
                meta = data.get("meta", {})
                cached_dir = str(meta.get("chm_dir", ""))
                cached_chip = int(meta.get("chip_size", -1))
                rows = data.get("chips", [])
                if cached_dir == chm_dir_key and cached_chip == int(chip_size):
                    return [_normalize_chip(TileChip(**c)) for c in rows]
        except (json.JSONDecodeError, TypeError, OSError):
            pass

    chips = scan(chm_dir, chip_size=chip_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "meta": {
                    "version": 2,
                    "chm_dir": chm_dir_key,
                    "chip_size": int(chip_size),
                },
                "chips": [c.as_dict() for c in chips],
            },
            indent=2,
        )
    )
    return chips

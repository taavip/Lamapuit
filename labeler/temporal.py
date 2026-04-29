"""Cross-year and cross-product tile discovery helpers."""

from __future__ import annotations

from .tile_index import TileChip


_PRODUCT_ORDER = {
    "chm_max_hag": 0,
    "harmonized_raw": 1,
    "harmonized_gauss": 2,
}


def _to_int_year(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chip_product(chip: TileChip) -> str:
    if chip.product:
        return chip.product
    name = chip.raster_stem.lower()
    if "_harmonized_dem_last_raw_chm" in name:
        return "harmonized_raw"
    if "_harmonized_dem_last_gauss_chm" in name:
        return "harmonized_gauss"
    if "_chm_max_hag" in name:
        return "chm_max_hag"
    return "unknown"


def _site_key(chip: TileChip) -> tuple[str, str] | None:
    if not chip.grid or not chip.source:
        return None
    return chip.grid, chip.source


def year_views_for_tile(tile_id: str, by_id: dict[str, TileChip]) -> list[dict]:
    """Return same-site tiles across years for the current CHM product."""

    chip = by_id.get(tile_id)
    if chip is None:
        return []

    site = _site_key(chip)
    current_year = _to_int_year(chip.year)
    current_product = _chip_product(chip)
    if site is None or current_year is None:
        return [
            {
                "year": chip.year,
                "token": chip.source or "",
                "product": current_product,
                "tile_id": tile_id,
                "label_key": chip.label_key,
                "is_current": True,
            }
        ]

    views: list[dict] = []
    for other in by_id.values():
        if _site_key(other) != site:
            continue
        if _chip_product(other) != current_product:
            continue
        if other.row_off != chip.row_off or other.col_off != chip.col_off:
            continue
        year = _to_int_year(other.year)
        if year is None:
            continue
        views.append(
            {
                "year": year,
                "token": other.source or "",
                "product": _chip_product(other),
                "tile_id": other.tile_id,
                "label_key": other.label_key,
                "is_current": other.tile_id == tile_id,
            }
        )

    if not views:
        return [
            {
                "year": chip.year,
                "token": chip.source or "",
                "product": current_product,
                "tile_id": tile_id,
                "label_key": chip.label_key,
                "is_current": True,
            }
        ]

    by_year: dict[int, dict] = {}
    for view in views:
        year = int(view["year"])
        existing = by_year.get(year)
        if existing is None or (view["is_current"] and not existing["is_current"]):
            by_year[year] = view
    return sorted(by_year.values(), key=lambda v: int(v["year"]))


def product_views_for_tile(tile_id: str, by_id: dict[str, TileChip]) -> list[dict]:
    """Return same-site, same-year tiles across CHM products."""

    chip = by_id.get(tile_id)
    if chip is None:
        return []

    site = _site_key(chip)
    current_year = _to_int_year(chip.year)
    current_product = _chip_product(chip)
    if site is None or current_year is None:
        return [
            {
                "product": current_product,
                "year": chip.year,
                "token": chip.source or "",
                "tile_id": tile_id,
                "label_key": chip.label_key,
                "is_current": True,
            }
        ]

    views: list[dict] = []
    for other in by_id.values():
        if _site_key(other) != site:
            continue
        if _to_int_year(other.year) != current_year:
            continue
        if other.row_off != chip.row_off or other.col_off != chip.col_off:
            continue
        views.append(
            {
                "product": _chip_product(other),
                "year": _to_int_year(other.year),
                "token": other.source or "",
                "tile_id": other.tile_id,
                "label_key": other.label_key,
                "is_current": other.tile_id == tile_id,
            }
        )

    if not views:
        return [
            {
                "product": current_product,
                "year": chip.year,
                "token": chip.source or "",
                "tile_id": tile_id,
                "label_key": chip.label_key,
                "is_current": True,
            }
        ]

    by_product: dict[str, dict] = {}
    for view in views:
        product = str(view["product"])
        existing = by_product.get(product)
        if existing is None or (view["is_current"] and not existing["is_current"]):
            by_product[product] = view
    return sorted(
        by_product.values(),
        key=lambda v: (_PRODUCT_ORDER.get(str(v["product"]), 99), str(v["product"])),
    )

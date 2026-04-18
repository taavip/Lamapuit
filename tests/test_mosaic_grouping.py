import importlib.util
from pathlib import Path


def _load_build_components():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "mosaic_and_tile_by_year.py"
    spec = importlib.util.spec_from_file_location("mosaic_and_tile_by_year", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.build_components


build_components = _load_build_components()


def _tile(code: str, year: int, minx: float, miny: float, maxx: float, maxy: float, res: float = 0.2):
    return {
        "path": f"{code}_{year}.tif",
        "code": code,
        "year": year,
        "minx": minx,
        "miny": miny,
        "maxx": maxx,
        "maxy": maxy,
        "width": int((maxx - minx) / res),
        "height": int((maxy - miny) / res),
        "res": res,
    }


def test_build_components_separates_non_adjacent_groups():
    # Group A: isolated single tile
    t1 = _tile("471656", 2018, 656000.0, 6470999.99, 657000.0, 6471999.99)
    # Group B: contiguous row of 5 adjacent tiles
    t2 = _tile("474657", 2018, 657000.0, 6473999.99, 658000.0, 6474999.99)
    t3 = _tile("474658", 2018, 658000.0, 6473999.99, 659000.0, 6474999.99)
    t4 = _tile("474659", 2018, 659000.0, 6473999.99, 660000.0, 6474999.99)
    t5 = _tile("474660", 2018, 660000.0, 6473999.99, 661000.0, 6474999.99)
    t6 = _tile("474661", 2018, 661000.0, 6473999.99, 662000.0, 6474999.99)

    comps_by_year = build_components([t1, t2, t3, t4, t5, t6], tolerance_px=1.0)
    comps = comps_by_year[2018]

    assert len(comps) == 2
    comp_codes = sorted([sorted([m["code"] for m in c]) for c in comps], key=len)
    assert comp_codes[0] == ["471656"]
    assert comp_codes[1] == ["474657", "474658", "474659", "474660", "474661"]

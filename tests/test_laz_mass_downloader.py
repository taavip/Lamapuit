import importlib.util
import sys
from pathlib import Path

# Load laz_mass_downloader as a module from scripts/ to avoid package import issues
spec = importlib.util.spec_from_file_location("laz_mass_downloader", Path(__file__).parent.parent / "scripts" / "laz_mass_downloader.py")
laz_mod = importlib.util.module_from_spec(spec)
sys.modules["laz_mass_downloader"] = laz_mod
spec.loader.exec_module(laz_mod)
parse_id_list = laz_mod.parse_id_list
bytes_from_text = laz_mod.bytes_from_text
extract_links_from_page = laz_mod.extract_links_from_page


def test_parse_id_list():
    raw = """
    584590
    abc
    584590
    584591
    """
    ids = parse_id_list(raw)
    assert ids == ["584590", "584591"]


def test_bytes_from_text():
    assert bytes_from_text("( 121.1 MB )") == 121100000
    assert bytes_from_text("( 917 KB )") == 917000
    assert bytes_from_text("( 1.2 GB )") == 1200000000


def test_extract_links_from_fixture():
    fixture = Path(__file__).parent / "fixtures" / "maaamet_584590.html"
    html = fixture.read_text(encoding="utf-8")
    items = extract_links_from_page(html, "584590")
    assert any("584590_2022_tava.laz" in f[0] for f in items)

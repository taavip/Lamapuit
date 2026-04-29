#!/usr/bin/env python3
"""Mass downloader for LAZ files from Maa-amet geoportaal.

Usage examples:
  python scripts/laz_mass_downloader.py --ids-file ids.txt --out data/laz_downloads --workers 8
  python scripts/laz_mass_downloader.py --ids "584590,584591" --dry-run

Features:
- Parse multiline ID lists or file input
- Fetch search page for each `kaardiruut` and extract LAZ links/sizes
- Filter files >= 100 MB and download concurrently with retries
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

BASE_SEARCH_URL = "https://geoportaal.maaamet.ee/index.php"
DEFAULT_OUTDIR = Path("data/laz_downloads")
SIZE_THRESHOLD_BYTES = 100_000_000  # 100 MB (decimal)


def setup_session(timeout: int = 30) -> requests.Session:
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "Lamapuit-LAZ-Downloader/1.0 (+https://github.com)"})
    s.request_timeout = timeout
    return s


def parse_id_list(text: str) -> List[str]:
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d{6})$", line)
        if m:
            ids.append(m.group(1))
    # preserve order, deduplicate
    seen = set()
    out = []
    for v in ids:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def bytes_from_text(s: str) -> Optional[int]:
    # examples: ( 121.1 MB ), ( 917 KB )
    s = s.strip()
    m = re.search(r"([0-9,.]+)\s*(KB|MB|GB|B)", s, flags=re.I)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    unit = m.group(2).lower()
    if unit == "b":
        return int(val)
    if unit == "kb":
        return int(val * 1_000)
    if unit == "mb":
        return int(val * 1_000_000)
    if unit == "gb":
        return int(val * 1_000_000_000)
    return None


def extract_links_from_page(html: str, kaardiruut: str) -> List[Tuple[str, str, Optional[int]]]:
    """Return list of tuples: (filename, url, size_bytes)"""
    soup = BeautifulSoup(html, "html.parser")
    results = []
    # look for list under heading mentioning LAZ
    for h3 in soup.find_all("h3"):
        if "LAZ" in h3.get_text().upper() or "laZ" in h3.get_text():
            ul = h3.find_next_sibling("ul")
            if not ul:
                continue
            for li in ul.find_all("li"):
                a = li.find("a")
                if not a or not a.get("href"):
                    continue
                href = a["href"]
                filename = a.get_text().strip()
                size_text = li.get_text()
                size = bytes_from_text(size_text)
                # resolve relative URL
                if href.startswith("index.php"):
                    url = f"https://geoportaal.maaamet.ee/{href}"
                elif href.startswith("http"):
                    url = href
                else:
                    url = f"https://geoportaal.maaamet.ee/{href.lstrip('/') }"
                results.append((filename, url, size))
    # fallback: scan anchors with .laz
    if not results:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text()
            if ".laz" in text.lower() or href.lower().endswith(".laz"):
                filename = text.strip() or Path(href).name
                size = None
                # try parent text for size
                parent = a.parent
                size = bytes_from_text(parent.get_text() if parent else "")
                if href.startswith("index.php"):
                    url = f"https://geoportaal.maaamet.ee/{href}"
                elif href.startswith("http"):
                    url = href
                else:
                    url = f"https://geoportaal.maaamet.ee/{href.lstrip('/') }"
                results.append((filename, url, size))
    # filter by kaardiruut presence in filename as a sanity check
    filtered = [r for r in results if r[0].startswith(kaardiruut)]
    return filtered if filtered else results


def fetch_manifest_for_id(
    session: requests.Session, kaardiruut: str, andmetyyp: str = "lidar_laz_tava"
) -> List[Dict]:
    params = {
        "lang_id": 1,
        "plugin_act": "otsing",
        "page_id": 614,
        "kaardiruut": kaardiruut,
        "andmetyyp": andmetyyp,
    }
    resp = session.get(
        BASE_SEARCH_URL, params=params, timeout=getattr(session, "request_timeout", 30)
    )
    resp.raise_for_status()
    items = extract_links_from_page(resp.text, kaardiruut)
    out = []
    for filename, url, size in items:
        out.append({"kaardiruut": kaardiruut, "filename": filename, "url": url, "size_bytes": size})
    return out


def write_manifest_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        # derive fieldnames from rows if possible, otherwise use a sensible default
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = ["kaardiruut", "filename", "url", "size_bytes"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def download_file(
    session: requests.Session,
    url: str,
    out_path: Path,
    expected_size: Optional[int] = None,
    timeout: int = 60,
) -> Tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with session.get(
            url, stream=True, timeout=getattr(session, "request_timeout", timeout)
        ) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        # verify size if provided
        if expected_size is not None:
            got = tmp.stat().st_size
            # allow 5% tolerance for size differences (some servers provide slightly different byte counts)
            if abs(got - expected_size) > max(1024, expected_size * 0.05):
                return False, f"size_mismatch: expected={expected_size} got={got}"
        tmp.replace(out_path)
        return True, "ok"
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False, str(e)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Mass LAZ downloader for Maa-amet geoportaal")
    parser.add_argument("--ids", help="Comma separated list of kaardiruut ids")
    parser.add_argument("--ids-file", help="File with one id per line")
    parser.add_argument("--out", default=str(DEFAULT_OUTDIR), help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel downloads")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only discover and write manifests, do not download"
    )
    parser.add_argument("--threshold-mb", type=float, default=100.0, help="Size threshold in MB")
    args = parser.parse_args(argv)

    raw = ""
    if args.ids_file:
        raw = Path(args.ids_file).read_text()
    if args.ids:
        raw = (raw + "\n" + args.ids) if raw else args.ids
    if not raw:
        print("Provide --ids or --ids-file", file=sys.stderr)
        return 2
    ids = parse_id_list(raw)
    if not ids:
        print("No valid 6-digit IDs found", file=sys.stderr)
        return 3
    outdir = Path(args.out)

    session = setup_session()

    discovery = []
    for kid in ids:
        try:
            items = fetch_manifest_for_id(session, kid)
            discovery.extend(items)
        except Exception as e:
            print(f"Error fetching {kid}: {e}", file=sys.stderr)

    write_manifest_csv(outdir / "discovery_manifest.csv", discovery)

    threshold_bytes = int(args.threshold_mb * 1_000_000)
    eligible = [r for r in discovery if (r.get("size_bytes") or 0) >= threshold_bytes]
    write_manifest_csv(outdir / "eligible_manifest.csv", eligible)

    if args.dry_run:
        print(
            f"Dry-run complete. Found {len(discovery)} items, {len(eligible)} eligible (>= {args.threshold_mb} MB)"
        )
        return 0

    # prepare download tasks
    tasks = []
    for r in eligible:
        filename = r["filename"]
        url = r["url"]
        expected = r.get("size_bytes")
        tasks.append((url, outdir / filename, expected))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(download_file, session, url, path, expected): (url, path)
            for url, path, expected in tasks
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            url, path = futures[f]
            try:
                ok, msg = f.result()
            except Exception as e:
                ok = False
                msg = str(e)
            results.append({"url": url, "path": str(path), "success": ok, "message": msg})

    write_manifest_csv(
        outdir / "download_results.csv",
        [
            {
                "kaardiruut": "",
                "filename": r["path"],
                "url": r["url"],
                "size_bytes": None,
                "success": r.get("success"),
                "message": r.get("message"),
            }
            for r in results
        ],
    )
    succ = sum(1 for r in results if r["success"])
    fail = len(results) - succ
    if fail:
        print("Some downloads failed:", file=sys.stderr)
        for r in results:
            if not r.get("success"):
                print(f"  {r.get('url')} -> {r.get('message')}", file=sys.stderr)
    print(f"Download complete: success={succ} failed={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

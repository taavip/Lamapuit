#!/usr/bin/env python3
"""Convert remediated split metadata CSV back into cnn_test_split JSON keys."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Build cnn_test_split JSON from metadata CSV")
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    keys: list[list[object]] = []
    seen: set[tuple[str, int, int]] = set()

    with open(args.metadata, newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("split", "")).strip() != "test":
                continue
            k = (str(row.get("raster", "")).strip(), int(float(row.get("row_off", 0))), int(float(row.get("col_off", 0))))
            if not k[0] or k in seen:
                continue
            seen.add(k)
            keys.append([k[0], k[1], k[2]])

    payload = {
        "keys": keys,
        "meta": {
            "source_metadata": str(args.metadata),
            "seed": args.seed,
            "test_tiles": len(keys),
            "note": "Derived from remediated metadata split=test rows",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"test_tiles={len(keys)}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

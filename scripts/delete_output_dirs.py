#!/usr/bin/env python3
"""
Safe removal tool for large output folders used in the project.

This script shows sizes and contents, and then either moves the
directories to a local `trash/` folder (safe default) or permanently
deletes them when `--permanent` is used. Use `--dry-run` first.

Usage examples:
  python scripts/delete_output_dirs.py           # interactive, moves to trash
  python scripts/delete_output_dirs.py -y        # same, no prompt
  python scripts/delete_output_dirs.py --permanent -y  # delete permanently
  python scripts/delete_output_dirs.py --dirs output/foo output/bar

Defaults target the four folders you asked to remove.
"""

from __future__ import annotations

import argparse
import shutil
import os
import sys
import time
from pathlib import Path
from typing import List

DEFAULT_DIRS = [
    "output/chm_dataset_lastreturns_hag0_1p3/labels",
    "output/chm_smoke_parallel_srsfix/labels",
    "output/chm_smoke_stability_after_cleanup/labels",
    "output/onboarding_labels_v3_drop13_diverse_top3",
]


def sizeof_fmt(num: int, suffix: str = "B") -> str:
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"


def dir_size(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except Exception:
                # ignore files that disappear / permission errors
                pass
    return total


def unique_dest(base: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = base.with_name(f"{base.name}_{ts}")
    i = 0
    while dest.exists():
        i += 1
        dest = base.with_name(f"{base.name}_{ts}_{i}")
    return dest


def confirm(prompt: str) -> bool:
    try:
        ans = input(prompt).strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Safely remove project output folders")
    p.add_argument(
        "--dirs",
        nargs="+",
        default=DEFAULT_DIRS,
        help="Directories to remove (default: the four label/output folders)",
    )
    p.add_argument("--trash", action="store_true", help="Move folders to `trash/` instead of deleting")
    p.add_argument("--trash-dir", default="trash", help="Parent folder for moved directories")
    p.add_argument("--permanent", action="store_true", help="Permanently delete (rm -rf) instead of moving")
    p.add_argument("-y", "--yes", action="store_true", help="Assume yes for prompts")
    p.add_argument("--dry-run", action="store_true", help="Do not modify anything; just print actions")

    args = p.parse_args(argv)

    targets: List[Path] = [Path(d) for d in args.dirs]

    existing = [t for t in targets if t.exists() and t.is_dir()]
    missing = [t for t in targets if not (t.exists() and t.is_dir())]

    if not existing:
        print("No target directories found. Nothing to do.")
        return 0

    print("Will process the following directories:")
    total_sum = 0
    for t in existing:
        sz = dir_size(t)
        total_sum += sz
        print(f"  - {t}  size={sizeof_fmt(sz)}")

    if missing:
        print("\nMissing (skipped):")
        for t in missing:
            print(f"  - {t}")

    action = "move to trash" if args.trash and not args.permanent else ("permanently delete" if args.permanent else "move to trash")
    print(f"\nPlanned action: {action}")
    print(f"Total size of existing targets: {sizeof_fmt(total_sum)}")

    if args.dry_run:
        print("\nDry-run mode: no changes made.")
        return 0

    if not args.yes:
        ok = confirm(f"Proceed with {action} for {len(existing)} directories? [y/N]: ")
        if not ok:
            print("Aborted by user.")
            return 1

    # perform actions
    if args.permanent:
        for t in existing:
            try:
                shutil.rmtree(t)
                print(f"REMOVED: {t}")
            except Exception as e:
                print(f"ERROR removing {t}: {e}")
    else:
        trash_root = Path(args.trash_dir)
        trash_root.mkdir(parents=True, exist_ok=True)
        for t in existing:
            dest = unique_dest(trash_root / t.name)
            try:
                shutil.move(str(t), str(dest))
                print(f"MOVED: {t} -> {dest}")
            except Exception as e:
                print(f"ERROR moving {t} -> {dest}: {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

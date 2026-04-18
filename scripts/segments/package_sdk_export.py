#!/usr/bin/env python3
"""Package Segments SDK export files into a Label-Studio-style COCO ZIP.

This script expects the Segments SDK export to have produced:
- a COCO JSON file named like `export_coco-instance_{owner}_{dataset}_{release}.json` in the CWD
- images stored under `segments/{owner}_{dataset}/{release}`

It will create a ZIP with `annotations.json` and an `images/` folder containing the SDK images.
"""
import argparse
import os
import glob
import zipfile
import shutil


def find_export_json(owner, dataset, release):
    pattern = f"export_coco-instance_{owner}_{dataset}_*{release}*.json"
    matches = glob.glob(pattern)
    if matches:
        return os.path.abspath(matches[0])
    # fallback: any export json containing owner_dataset
    pattern2 = f"export_coco-instance_*{owner}_{dataset}*.json"
    matches = glob.glob(pattern2)
    return os.path.abspath(matches[0]) if matches else None


def find_images_dir(owner, dataset, release):
    base = os.path.abspath('segments')
    candidate = os.path.join(base, f"{owner}_{dataset}", release)
    if os.path.isdir(candidate):
        return candidate
    # try without release subdir
    candidate2 = os.path.join(base, f"{owner}_{dataset}")
    if os.path.isdir(candidate2):
        return candidate2
    return None


def package(export_json, images_dir, outzip):
    with zipfile.ZipFile(outzip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(export_json, arcname='annotations.json')
        for root, dirs, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    absf = os.path.join(root, f)
                    arc = os.path.join('images', f)
                    zf.write(absf, arcname=arc)


def main():
    p = argparse.ArgumentParser(description='Package Segments SDK export into COCO zip')
    p.add_argument('--owner', '-u', required=True)
    p.add_argument('--dataset', '-d', required=True)
    p.add_argument('--release', '-r', required=True)
    p.add_argument('--out', '-o', default=None)
    args = p.parse_args()

    export_json = find_export_json(args.owner, args.dataset, args.release)
    if not export_json or not os.path.exists(export_json):
        raise SystemExit('Export JSON not found. Run the SDK export first.')

    images_dir = find_images_dir(args.owner, args.dataset, args.release)
    if not images_dir or not os.path.isdir(images_dir):
        raise SystemExit('Images directory not found under segments/. Ensure SDK downloaded images.')

    outzip = args.out or os.path.join('output','segments', f"{args.dataset}_{args.release}_sdk_coco.zip")
    os.makedirs(os.path.dirname(outzip), exist_ok=True)
    package(export_json, images_dir, outzip)
    print('Wrote', outzip)


if __name__ == '__main__':
    main()

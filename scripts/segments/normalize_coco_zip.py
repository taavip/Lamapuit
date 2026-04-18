#!/usr/bin/env python3
"""Normalize COCO ZIP exported by Segments/Label Studio.

This script:
- extracts the given ZIP (or reads a folder)
- finds the COCO JSON file
- for each image entry that references a local-files URL (e.g. '/data/local-files/?d=path/to/file'),
  it copies the referenced file from the workspace into an `images/` folder and updates
  the `file_name` to the basename so the COCO JSON points to the actual image file.
- writes a new ZIP containing `annotations.json` (normalized JSON) and `images/`.

Example:
  python scripts/segments/normalize_coco_zip.py --zip output/segments/pen_coin_coco.zip

If the referenced files are not present in the workspace, the script will try to download
HTTP URLs if they are reachable.
"""
import argparse
import os
import sys
import tempfile
import zipfile
import json
import shutil
from urllib.parse import urlparse, parse_qs


def find_coco_json(root):
    for dirpath, dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith('.json'):
                # heuristics: accept files with 'coco' or 'export' or 'result' or 'annotations' in name
                if any(x in f.lower() for x in ('coco','export','result','annotation','instances')):
                    return os.path.join(dirpath, f)
    return None


def copy_from_workspace(relpath, dest_dir):
    # relpath may be filesystem path relative to repo root
    repo_root = os.path.abspath('.')
    candidate = os.path.join(repo_root, relpath)
    if os.path.exists(candidate):
        dst = os.path.join(dest_dir, os.path.basename(relpath))
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(candidate, dst)
        return dst
    return None


def download_url(url, dest_dir):
    try:
        import requests
    except Exception:
        return None
    r = requests.get(url, stream=True, timeout=30)
    if r.status_code == 200:
        os.makedirs(dest_dir, exist_ok=True)
        fn = os.path.basename(urlparse(url).path)
        dst = os.path.join(dest_dir, fn)
        with open(dst, 'wb') as fh:
            for chunk in r.iter_content(1024*32):
                fh.write(chunk)
        return dst
    return None


def normalize(coco_path, extracted_root):
    coco = json.load(open(coco_path, 'r', encoding='utf8'))
    images = coco.get('images', [])
    images_dir = os.path.join(extracted_root, 'images')
    os.makedirs(images_dir, exist_ok=True)

    for img in images:
        # Prefer 'file_name' if it already matches an existing file in the extracted folder
        fn = img.get('file_name')
        if fn:
            candidate = os.path.join(extracted_root, fn)
            candidate_in_images = os.path.join(images_dir, os.path.basename(fn))
            if os.path.exists(candidate):
                # copy into images/ with basename
                shutil.copy2(candidate, candidate_in_images)
                img['file_name'] = os.path.basename(fn)
                continue

        # fallback: check 'path' or 'url' fields
        path = img.get('path') or img.get('url') or img.get('file_path')
        if path:
            parsed = urlparse(path)
            # check for label-studio local-files pattern ?d=relative/path
            q = parse_qs(parsed.query)
            if 'd' in q and q['d']:
                rel = q['d'][0]
                copied = copy_from_workspace(rel, images_dir)
                if copied:
                    img['file_name'] = os.path.basename(copied)
                    # remove path to avoid referencing local-files
                    img.pop('path', None)
                    img.pop('url', None)
                    continue

            # if the path is an HTTP URL (not pointing to localhost), try download
            if parsed.scheme in ('http','https') and not parsed.hostname in ('localhost','127.0.0.1'):
                downloaded = download_url(path, images_dir)
                if downloaded:
                    img['file_name'] = os.path.basename(downloaded)
                    img.pop('path', None)
                    img.pop('url', None)
                    continue

        # If nothing matched, leave file_name as-is but ensure it's basename
        if fn:
            img['file_name'] = os.path.basename(fn)

    # write normalized JSON next to coco_path
    norm_path = os.path.join(os.path.dirname(coco_path), 'annotations.json')
    json.dump(coco, open(norm_path, 'w', encoding='utf8'), indent=2)
    return norm_path, images_dir


def package_zip(norm_json_path, images_dir, outzip):
    with zipfile.ZipFile(outzip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(norm_json_path, arcname='annotations.json')
        for root, dirs, files in os.walk(images_dir):
            for f in files:
                absf = os.path.join(root, f)
                arc = os.path.join('images', f)
                zf.write(absf, arcname=arc)


def main():
    p = argparse.ArgumentParser(description='Normalize COCO ZIP and ensure images are included')
    p.add_argument('--zip', '-z', help='Path to COCO ZIP to normalize')
    p.add_argument('--out', '-o', help='Output ZIP path (defaults to <input>_normalized.zip)')
    args = p.parse_args()

    if not args.zip:
        print('Please pass --zip path/to/file.zip')
        sys.exit(2)

    zip_path = os.path.abspath(args.zip)
    if not os.path.exists(zip_path):
        print('Zip not found:', zip_path)
        sys.exit(3)

    tmp = tempfile.mkdtemp(prefix='coco_norm_')
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmp)

        coco_path = find_coco_json(tmp)
        if not coco_path:
            print('No COCO JSON found in zip')
            sys.exit(4)

        norm_json, images_dir = normalize(coco_path, tmp)
        outzip = args.out or (os.path.splitext(zip_path)[0] + '_normalized.zip')
        package_zip(norm_json, images_dir, outzip)
        print('Wrote normalized zip to', outzip)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()

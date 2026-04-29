#!/usr/bin/env python3
"""Download a Segments dataset and export to COCO format.

Usage examples:
  python scripts/segments/download_coco_from_segments.py --api-key YOUR_KEY --user Pealik --dataset pen_coin
  python scripts/segments/download_coco_from_segments.py --api-key YOUR_KEY --list

The script will try to discover datasets/releases via the SegmentsClient.
If the Segments SDK is not available or a method is missing, it will print
clear instructions for the user.
"""
import argparse
import sys
import os


def package_coco_zip(outdir: str):
    """Find a COCO JSON and related images under outdir and create a zip.

    The resulting zip will contain a single JSON file at the root (annotations.json)
    and an `images/` folder with the image files. The zip is written next to outdir
    as <outdir_basename>_coco.zip.
    """
    import zipfile
    import fnmatch

    outdir = os.path.abspath(outdir)
    # search for coco json files and images
    coco_files = []
    image_files = []
    for root, dirs, files in os.walk(outdir):
        for f in files:
            if f.lower().endswith('.json') and ('coco' in f.lower() or 'annotation' in f.lower() or 'instances' in f.lower()):
                coco_files.append(os.path.join(root, f))
            if fnmatch.fnmatch(f.lower(), '*.jpg') or fnmatch.fnmatch(f.lower(), '*.jpeg') or fnmatch.fnmatch(f.lower(), '*.png'):
                image_files.append(os.path.join(root, f))

    # Also search the repository root for exported COCO JSONs (export may write json to cwd)
    repo_root = os.path.abspath('.')
    # If export used a different default folder (./segments), include that too
    segments_default = os.path.abspath('segments')
    if segments_default != outdir and os.path.isdir(segments_default):
        for root, dirs, files in os.walk(segments_default):
            for f in files:
                if f.lower().endswith('.json') and ('coco' in f.lower() or 'annotation' in f.lower() or 'instances' in f.lower()):
                    coco_files.append(os.path.join(root, f))
                if fnmatch.fnmatch(f.lower(), '*.jpg') or fnmatch.fnmatch(f.lower(), '*.jpeg') or fnmatch.fnmatch(f.lower(), '*.png'):
                    image_files.append(os.path.join(root, f))

    # Also check repo root for exported json and images
    if repo_root != outdir:
        for root, dirs, files in os.walk(repo_root):
            for f in files:
                if f.lower().endswith('.json') and ('coco' in f.lower() or 'annotation' in f.lower() or 'instances' in f.lower() or f.startswith('export_')):
                    coco_files.append(os.path.join(root, f))
            for f in files:
                if fnmatch.fnmatch(f.lower(), '*.jpg') or fnmatch.fnmatch(f.lower(), '*.jpeg') or fnmatch.fnmatch(f.lower(), '*.png'):
                    # avoid adding lots of unrelated images at repo root; only add images under 'segments' or outdir
                    pass

    if not coco_files:
        raise RuntimeError('No COCO JSON found under ' + outdir)

    coco_path = coco_files[0]
    zip_name = os.path.join(os.path.dirname(outdir), os.path.basename(outdir.rstrip('/')) + '_coco.zip')
    print('Packaging', coco_path, 'and', len(image_files), 'images into', zip_name)

    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # add annotations.json at root
        zf.write(coco_path, arcname='annotations.json')
        # add images under images/
        for img in image_files:
            arc = os.path.join('images', os.path.basename(img))
            zf.write(img, arcname=arc)

    print('Wrote', zip_name)


def main():
    p = argparse.ArgumentParser(description='Download Segments dataset and export to COCO')
    p.add_argument('--api-key', '-k', required=True, help='Segments API key')
    p.add_argument('--user', '-u', required=True, help='Segments username (owner)')
    p.add_argument('--dataset', '-d', help='Dataset name (repo). If omitted, the script will try to discover datasets for the user')
    p.add_argument('--release', '-r', default=None, help='Release name (e.g. v1.0). If omitted latest/recommended will be used if available')
    p.add_argument('--outdir', '-o', default='output/segments', help='Output directory for images and COCO export')
    p.add_argument('--list', action='store_true', help='Only list datasets/releases for the user and exit')
    args = p.parse_args()

    try:
        from segments import SegmentsClient, SegmentsDataset
        from segments.utils import export_dataset
    except Exception as e:
        print('ERROR: failed to import segments SDK:', e)
        print('Install via: pip install segments-io')
        sys.exit(2)

    client = SegmentsClient(args.api_key)

    dataset_ident = None
    release_name = args.release

    # Try to discover datasets belonging to the user
    datasets = None
    try:
        if hasattr(client, 'list_datasets'):
            datasets = client.list_datasets()
        elif hasattr(client, 'get_datasets'):
            datasets = client.get_datasets()
        else:
            # Some SDK versions use different method names; attempt list_releases fallback
            datasets = None
    except Exception:
        datasets = None

    # Filter datasets by owner if we managed to list them
    found = []
    if datasets:
        try:
            for ds in datasets:
                # ds may be a dict-like or object
                name = getattr(ds, 'name', None) or ds.get('name') if isinstance(ds, dict) else None
                owner = getattr(ds, 'owner', None) or ds.get('owner') if isinstance(ds, dict) else None
                full = getattr(ds, 'full_name', None) or ds.get('full_name') if isinstance(ds, dict) else None
                if not full and owner and name:
                    full = f"{owner}/{name}"
                if full and full.lower().startswith(args.user.lower() + '/'):
                    found.append(full)
        except Exception:
            found = []

    # If user requested list, print what we found and exit
    if args.list:
        if found:
            print('Datasets found for user', args.user)
            for f in found:
                print(' -', f)
        else:
            print('No datasets discovered via SDK. You can still provide --dataset USER/NAME directly.')
        return

    # Determine dataset identifier
    if args.dataset:
        dataset_ident = f"{args.user}/{args.dataset}" if '/' not in args.dataset else args.dataset
    else:
        if len(found) == 1:
            dataset_ident = found[0]
            print('Auto-discovered dataset:', dataset_ident)
        elif len(found) > 1:
            print('Multiple datasets found for user', args.user)
            for f in found:
                print(' -', f)
            print('Please re-run with --dataset DATASET_NAME to select which one to download.')
            sys.exit(3)
        else:
            print('No dataset provided and none discovered. Please pass --dataset USER/REPO')
            sys.exit(4)

    # Now find release. Try to use client.get_release if available, otherwise try list_releases
    release = None
    try:
        if hasattr(client, 'get_release'):
            # If release name is not provided, try 'v1.0' then 'latest'
            tried = []
            if release_name:
                tried.append(release_name)
            else:
                tried.extend(['v1.0', 'latest'])
            for rname in tried:
                try:
                    rel = client.get_release(dataset_ident, rname)
                    release = rel
                    release_name = rname
                    break
                except Exception:
                    continue
        # fallback: list releases
        if release is None and hasattr(client, 'list_releases'):
            rels = client.list_releases(dataset_ident)
            if rels:
                # pick latest if not provided
                release = rels[0]
                release_name = getattr(release, 'name', None) or (release[0] if isinstance(release, (list,tuple)) else None)
    except Exception as e:
        print('ERROR while fetching release info:', e)

    if release is None:
        print('Failed to resolve a release for', dataset_ident)
        print('Try specifying --release RELEASE_NAME or verify the dataset identifier is correct.')
        sys.exit(5)

    # Initialize dataset (this will download images into the SDK default dir or provided outdir)
    print('Initializing SegmentsDataset for', dataset_ident, 'release', release_name)
    try:
        ds = SegmentsDataset(release)
    except Exception as e:
        print('ERROR: failed to initialize SegmentsDataset:', e)
        sys.exit(6)

    # Ensure output dir exists
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # Export to COCO
    print('Exporting dataset to COCO in', outdir)
    try:
        export_dataset(ds, export_format='coco-instance', output_dir=outdir)
    except TypeError:
        # older/newer signatures may not have output_dir kwarg
        try:
            export_dataset(ds, 'coco-instance')
            print('export_dataset used default output location; check ./segments or SDK docs')
        except Exception as e:
            print('ERROR: export_dataset failed:', e)
            sys.exit(7)
    except Exception as e:
        print('ERROR: export failed:', e)
        sys.exit(8)

    print('Done. COCO export completed in', outdir)
    # Package COCO JSON + images into a single zip file similar to Label Studio export
    try:
        package_coco_zip(outdir)
    except Exception as e:
        print('WARNING: failed to package into zip:', e)


if __name__ == '__main__':
    main()


def package_coco_zip(outdir: str):
    """Find a COCO JSON and related images under outdir and create a zip.

    The resulting zip will contain a single JSON file at the root (annotations.json)
    and an `images/` folder with the image files. The zip is written next to outdir
    as <outdir_basename>_coco.zip.
    """
    import zipfile
    import fnmatch

    outdir = os.path.abspath(outdir)
    # search for coco json files and images
    coco_files = []
    image_files = []
    for root, dirs, files in os.walk(outdir):
        for f in files:
            if f.lower().endswith('.json') and ('coco' in f.lower() or 'annotation' in f.lower() or 'instances' in f.lower()):
                coco_files.append(os.path.join(root, f))
            if fnmatch.fnmatch(f.lower(), '*.jpg') or fnmatch.fnmatch(f.lower(), '*.jpeg') or fnmatch.fnmatch(f.lower(), '*.png'):
                image_files.append(os.path.join(root, f))

    # If export used a different default folder (./segments), include that too
    segments_default = os.path.abspath('segments')
    if segments_default != outdir and os.path.isdir(segments_default):
        for root, dirs, files in os.walk(segments_default):
            for f in files:
                if f.lower().endswith('.json') and ('coco' in f.lower() or 'annotation' in f.lower() or 'instances' in f.lower()):
                    coco_files.append(os.path.join(root, f))
                if fnmatch.fnmatch(f.lower(), '*.jpg') or fnmatch.fnmatch(f.lower(), '*.jpeg') or fnmatch.fnmatch(f.lower(), '*.png'):
                    image_files.append(os.path.join(root, f))

    if not coco_files:
        raise RuntimeError('No COCO JSON found under ' + outdir)

    coco_path = coco_files[0]
    zip_name = os.path.join(os.path.dirname(outdir), os.path.basename(outdir.rstrip('/')) + '_coco.zip')
    print('Packaging', coco_path, 'and', len(image_files), 'images into', zip_name)

    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # add annotations.json at root
        zf.write(coco_path, arcname='annotations.json')
        # add images under images/
        for img in image_files:
            arc = os.path.join('images', os.path.basename(img))
            zf.write(img, arcname=arc)

    print('Wrote', zip_name)

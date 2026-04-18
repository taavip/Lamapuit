"""Semi-manual SAM demo orchestrator.

Usage:
  - Ensure the SAM service is running (see `docker/sam/app_sam.py`).
  - Run: `python prototypes/sam_demo_orchestrator.py --image IMAGE_URL_OR_PATH`
  - The script will call the SAM endpoint with a center click and save a JSON output
    into `prototypes/output/<image_basename>_sam.json`.

Notes:
  - If you have a local Segment Anything model and the `segment_anything` package,
    you can adapt the `call_sam_local` function to run inference directly.
"""
import os
import argparse
import requests
import json
from pathlib import Path

SAM_URL = os.environ.get('SAM_SERVICE_URL', 'http://localhost:8000/predict')

def call_sam_service(image, clicks):
    payload = {'image': image, 'clicks': [{'x': int(x),'y': int(y),'is_positive': True} for x,y in clicks]}
    r = requests.post(SAM_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def compute_center_click(image_size):
    # image_size: (width, height)
    w,h = image_size
    return [(w//2, h//2)]

def ensure_output_dir(path='prototypes/output'):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True, help='Image URL or local path accessible to the SAM service')
    p.add_argument('--width', type=int, help='Image width (pixels) for center click calculation')
    p.add_argument('--height', type=int, help='Image height (pixels) for center click calculation')
    p.add_argument('--clicks', help='Optional manually-specified clicks as x1,y1;x2,y2 (overrides center)')
    args = p.parse_args()

    # Determine clicks
    if args.clicks:
        clicks = [tuple(map(int, pair.split(','))) for pair in args.clicks.split(';')]
    elif args.width and args.height:
        clicks = compute_center_click((args.width, args.height))
    else:
        # best-effort: request SAM with a single center click but image size unknown
        # Many SAM servers can accept clicks in pixel coordinates relative to original image.
        # If you don't know size, try center at (512,512) as a fallback.
        clicks = [(512,512)]

    print(f'Calling SAM service at {SAM_URL} for image {args.image} with clicks={clicks}')
    try:
        out = call_sam_service(args.image, clicks)
    except Exception as e:
        print('SAM service call failed:', e)
        return

    out_dir = ensure_output_dir()
    image_basename = Path(args.image).stem if '/' in args.image or args.image.endswith(('.jpg','.png')) else 'image'
    out_path = out_dir / f"{image_basename}_sam.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print('Saved SAM output to', out_path)
    # If masks are returned as polygons or binary arrays you can extend this script
    # to visualize overlays (requires numpy + PIL/matplotlib).

if __name__ == '__main__':
    main()

import os
import io
import json
import traceback
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import requests
import cv2

app = FastAPI(title="sam-service")

# Try to import segment_anything; if not present, we'll fallback to a mock response.
SAM_AVAILABLE = False
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False

# Load model if path provided
SAM_MODEL_PATH = os.environ.get('SAM_MODEL_PATH')
predictor = None
if SAM_AVAILABLE and SAM_MODEL_PATH:
    try:
        sam = sam_model_registry.get('default') or sam_model_registry['default']
        model = sam(checkpoint=SAM_MODEL_PATH)
        predictor = SamPredictor(model)
        print('SAM model loaded from', SAM_MODEL_PATH)
    except Exception:
        print('Failed to load SAM model:')
        traceback.print_exc()
        predictor = None


class Click(BaseModel):
    x: int
    y: int
    is_positive: bool = True


class SamRequest(BaseModel):
    image: str
    clicks: Optional[List[Click]] = None


def load_image_from_source(image_src: str):
    # Accept URL or local path
    if image_src.startswith('http://') or image_src.startswith('https://'):
        r = requests.get(image_src, timeout=30)
        r.raise_for_status()
        arr = np.asarray(bytearray(r.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(image_src, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not load image from ' + image_src)
    # convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def mask_to_polygons(mask: np.ndarray):
    # mask: 2D boolean array
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        pts = c.reshape(-1, 2).tolist()
        polygons.append(pts)
    return polygons


@app.post('/predict')
async def predict(req: SamRequest):
    # If predictor loaded, run real model; otherwise return a mock response
    if predictor is None:
        # Mock response for validation/testing
        return {
            'image': req.image,
            'clicks': [c.dict() for c in (req.clicks or [])],
            'masks': [
                {
                    'id': 1,
                    'polygon': [[10,10],[110,10],[110,60],[10,60]],
                    'bbox': [10,10,110,60],
                    'score': 0.95
                }
            ]
        }

    # Real predictor path
    try:
        img = load_image_from_source(req.image)
        predictor.set_image(img)

        points = []
        labels = []
        for c in (req.clicks or []):
            points.append([c.x, c.y])
            labels.append(1 if c.is_positive else 0)

        if len(points) == 0:
            # Use a single center point if none provided
            h, w = img.shape[:2]
            points = [[w//2, h//2]]
            labels = [1]

        input_points = np.array(points)
        input_labels = np.array(labels)

        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )

        out_masks = []
        for i, m in enumerate(masks):
            mask_bool = m.astype(bool)
            polys = mask_to_polygons(mask_bool)
            ys, xs = np.where(mask_bool)
            if len(xs) and len(ys):
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                bbox = [x_min, y_min, x_max, y_max]
            else:
                bbox = [0,0,0,0]

            out_masks.append({
                'id': i,
                'polygon': polys[0] if polys else [],
                'bbox': bbox,
                'score': float(scores[i]) if scores is not None else 1.0
            })

        return {'image': req.image, 'clicks': [c.dict() for c in (req.clicks or [])], 'masks': out_masks}

    except Exception:
        traceback.print_exc()
        return {'error': 'prediction_failed', 'trace': traceback.format_exc()}

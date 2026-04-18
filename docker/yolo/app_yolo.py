from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="yolo-prototype")

class DetectRequest(BaseModel):
    image: str

@app.post('/detect')
async def detect(req: DetectRequest):
    # Prototype: replace with ultralytics.YOLO integration if installed
    return {
        'image': req.image,
        'detections': [
            {
                'label': 'car',
                'confidence': 0.88,
                'bbox': [50, 30, 180, 120]
            }
        ]
    }

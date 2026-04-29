"""Small helper to convert detection+mask outputs into Label Studio prediction format.
This is a minimal prototype; adjust to your Label Studio API and auth.
"""
import json

def build_prediction(image_url, detections, masks):
    # detections: list of {label, confidence, bbox}
    # masks: list of {id, rle or polygon, bbox}
    preds = []
    for d in detections:
        # Label Studio rectangle expects x,y,width,height in percentages or pixels depending on config.
        preds.append({
            'result': [
                {
                    'from_name': 'bboxes',
                    'to_name': 'image',
                    'type': 'rectangle',
                    'value': {
                        'x': d['bbox'][0],
                        'y': d['bbox'][1],
                        'width': d['bbox'][2]-d['bbox'][0],
                        'height': d['bbox'][3]-d['bbox'][1]
                    }
                }
            ],
            'score': d.get('confidence', 1.0)
        })
    return {
        'data': {'image': image_url},
        'predictions': [
            {
                'result': sum([p['result'] for p in preds], []),
                'score': max([p['score'] for p in preds]) if preds else 1.0
            }
        ]
    }

if __name__ == '__main__':
    # quick smoke example
    print(json.dumps(build_prediction('http://example.com/img.jpg', [{'label':'car','confidence':0.9,'bbox':[50,40,200,150]}], []), indent=2))

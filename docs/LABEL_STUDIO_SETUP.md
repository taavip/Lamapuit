# Label Studio + YOLO/SAM Semi-Automated Labeling

This guide sets up a Dockerized labeling stack for fast polygon instance segmentation on CHM car tiles in:

- [data/car/chm_2m_label](data/car/chm_2m_label)

It includes:
- Label Studio UI
- YOLO segmentation pre-annotations (YOLO26s-seg preferred, YOLO11 fallback)
- Optional SAM refinement on top of YOLO proposals
- Export to YOLO format for iterative retraining

## 1. Prerequisites

- Docker + Docker Compose
- NVIDIA Container Toolkit for GPU (recommended)
- Internet access for initial model download

Create or refresh the project image once:

```bash
docker build -t lamapuit-dev .
```

Run project Python commands via Docker + Conda instead of local venv:

```bash
bash scripts/run_python_in_docker_conda.sh <script.py> <args>
```

Optional but recommended:
- Existing model checkpoints in [output/labelstudio_pipeline/models](output/labelstudio_pipeline/models)

## 2. Download Models

Run from repo root:

```bash
bash scripts/labelstudio/download_labeling_models.sh
```

Expected files:
- [output/labelstudio_pipeline/models/yolo26s-seg.pt](output/labelstudio_pipeline/models/yolo26s-seg.pt)
- [output/labelstudio_pipeline/models/yolo11s-seg.pt](output/labelstudio_pipeline/models/yolo11s-seg.pt)
- [output/labelstudio_pipeline/models/yolo11n-seg.pt](output/labelstudio_pipeline/models/yolo11n-seg.pt)

Backend selection order is configurable via `YOLO_MODEL_CANDIDATES`.

## 3. Start Services

```bash
docker compose -f docker-compose.labelstudio.yml up -d --build
```

Check health:

```bash
curl -s http://localhost:9090/health
curl -s http://localhost:8080/health
```

## 4. Create Label Studio Tasks

Generate tasks from year folders under [data/car/chm_2m_label](data/car/chm_2m_label):

```bash
bash scripts/run_python_in_docker_conda.sh scripts/labelstudio/generate_labelstudio_tasks.py \
  --workspace-root . \
  --input-root data/car/chm_2m_label \
  --glob "*/tiles/*.png" \
   --output output/labelstudio_pipeline/tasks/car_chm_tasks.json
```

Quick dry run with subset:

```bash
bash scripts/run_python_in_docker_conda.sh scripts/labelstudio/generate_labelstudio_tasks.py --limit 500
```

## 5. UI Setup (Step-by-Step)

1. Open `http://localhost:8080` and create/login account.
2. Get API token:
   1. Click profile icon.
   2. Open Account & Settings.
   3. Copy Access Token.
3. Bootstrap project:

```bash
bash scripts/run_python_in_docker_conda.sh scripts/labelstudio/bootstrap_labelstudio_project.py \
  --api-key "<PASTE_TOKEN>" \
  --tasks-json output/labelstudio_pipeline/tasks/car_chm_tasks.json \
   --label-config configs/label_studio/car_polygon_label_config.xml
```

4. In UI project settings:
   1. Verify labeling config includes `sam_point`, `sam_box`, `label`, and `sam_polygon` controls.
   2. Open Settings -> Model and confirm backend URL.
   3. In model `Extra Params`, set:

```json
{
  "from_name": "label",
  "to_name": "image",
  "label": "car",
  "sam_from_name": "sam_polygon",
  "sam_label": "car",
  "sam_point_name": "sam_point",
  "sam_box_name": "sam_box"
}
```

   3. Trigger model setup once by opening first task and requesting prediction.
5. Annotation page shortcuts for speed:
   1. Enable auto-next task after submit.
   2. Use prediction first, then adjust polygon vertices only when needed.
   3. Keep zoom high for 32x32 tiles and snap to visible boundaries.

### Click-to-mask with SAM prompts

Use the labeling config [configs/label_studio/car_polygon_label_config.xml](configs/label_studio/car_polygon_label_config.xml), which includes:
- Point prompt tool (`sam_point`)
- Box prompt tool (`sam_box`)
- Manual/YOLO polygon output tool (`label`)
- SAM click-to-mask polygon output tool (`sam_polygon`)

Workflow:
1. Place one point prompt on a car (or draw one tight box prompt around a car).
2. Request prediction from the model.
3. Backend runs SAM with your prompt and returns a polygon mask for class `car` in `sam_polygon`.
4. Adjust vertices only if needed, then submit.

Notes:
- Box prompts are usually more stable than point prompts.
- If no prompt is present, backend falls back to YOLO proposals + SAM refinement.

## 6. Balanced Automation Policy (F1-Calibrated)

Use this operational policy instead of fixed confidence only:

1. Export validation predictions and compute confidence bins.
2. Select confidence thresholds that satisfy desired precision/F1 constraints.
3. Route tasks by threshold bands:
   - `auto-accept`: very high calibrated quality
   - `review queue`: moderate confidence
   - `manual first`: low confidence
4. Recompute thresholds after each daily mini-retrain.

Note: current backend returns per-polygon `score`. You should calibrate this score against a validation set before enabling auto-accept in production.

## 7. Export Labels -> YOLO Dataset

In Label Studio, export project annotations as JSON, then convert:

```bash
bash scripts/run_python_in_docker_conda.sh scripts/labelstudio/export_labelstudio_to_yolo.py \
  --workspace-root . \
  --input-json /path/to/labelstudio_export.json \
  --output-dir output/labelstudio_pipeline/yolo_dataset \
   --val-frac 0.15
```

Dataset output:
- [output/labelstudio_pipeline/yolo_dataset/images/train](output/labelstudio_pipeline/yolo_dataset/images/train)
- [output/labelstudio_pipeline/yolo_dataset/images/val](output/labelstudio_pipeline/yolo_dataset/images/val)
- [output/labelstudio_pipeline/yolo_dataset/labels/train](output/labelstudio_pipeline/yolo_dataset/labels/train)
- [output/labelstudio_pipeline/yolo_dataset/labels/val](output/labelstudio_pipeline/yolo_dataset/labels/val)
- [output/labelstudio_pipeline/yolo_dataset/dataset.yaml](output/labelstudio_pipeline/yolo_dataset/dataset.yaml)

## 8. Daily Mini-Retrain

```bash
export LAMAPUIT_DOCKER_GPU=1
bash scripts/run_python_in_docker_conda.sh scripts/labelstudio/daily_mini_retrain.py \
  --dataset-yaml output/labelstudio_pipeline/yolo_dataset/dataset.yaml \
  --model-candidates output/labelstudio_pipeline/models/yolo26s-seg.pt,output/labelstudio_pipeline/models/yolo11s-seg.pt,yolo11n-seg.pt \
  --epochs 20 \
   --device 0
```

## 9. Fast Labeling Tactics

1. Uncertainty-first queue: annotate the most uncertain predictions first.
2. Disagreement queue: prioritize tasks where YOLO and SAM polygons differ strongly.
3. Spot-check auto-accepted labels (2-5%) to detect drift.
4. Keep class policy strict: one class (`car`) and no class switching overhead.
5. Use short daily retrains over long weekly retrains for faster quality feedback.

## 10. Troubleshooting

1. ML backend unhealthy:
   - Check logs: `docker compose -f docker-compose.labelstudio.yml logs ml-backend`
   - Ensure model files exist in [output/labelstudio_pipeline/models](output/labelstudio_pipeline/models)
2. GPU not used:
   - Verify NVIDIA Container Toolkit installation
   - Set `DEVICE=cpu` temporarily in [docker-compose.labelstudio.yml](docker-compose.labelstudio.yml)
3. Tasks show broken image paths:
   - Confirm generated JSON uses `/data/local-files/?d=...`
   - Confirm `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/workspace`
4. YOLO26 compatibility issue:
   - Keep fallback models in model candidates
   - Backend auto-falls back to YOLO11 family

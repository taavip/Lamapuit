from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


class PredictPayload(BaseModel):
    tasks: list[dict[str, Any]]
    project: int | None = None
    label_config: str | None = None
    params: dict[str, Any] | None = None


class ModelBackend:
    def __init__(self) -> None:
        self.workspace_root = Path(os.getenv("WORKSPACE_ROOT", "/workspace"))
        self.task_image_key = os.getenv("TASK_IMAGE_KEY", "image")
        self.yolo_candidates = [
            Path(p.strip())
            for p in os.getenv(
                "YOLO_MODEL_CANDIDATES",
                "/models/yolo26s-seg.pt,/models/yolo11s-seg.pt,/models/yolo11n-seg.pt",
            ).split(",")
            if p.strip()
        ]
        self.sam_candidates = [
            Path(p.strip())
            for p in os.getenv("SAM_MODEL_CANDIDATES", "/models/sam2_b.pt,/models/mobile_sam.pt").split(
                ","
            )
            if p.strip()
        ]
        self.yolo_conf = _env_float("YOLO_CONFIDENCE", 0.25)
        self.yolo_imgsz = _env_int("YOLO_IMGSZ", 640)
        self.return_topk = _env_int("RETURN_TOPK", 6)
        self.enable_sam_refinement = _env_bool("ENABLE_SAM_REFINEMENT", True)
        self.device = os.getenv("DEVICE", "cpu")

        self._yolo = None
        self._yolo_name = "uninitialized"
        self._sam = None
        self._sam_name = "disabled"

    def health(self) -> dict[str, str]:
        return {
            "status": "ok",
            "yolo": self._yolo_name,
            "sam": self._sam_name,
            "device": self.device,
        }

    def _load_yolo(self):
        if self._yolo is not None:
            return self._yolo

        from ultralytics import YOLO

        errors: list[str] = []
        for candidate in self.yolo_candidates:
            try:
                if not candidate.exists():
                    errors.append(f"missing:{candidate}")
                    continue
                model = YOLO(str(candidate))
                self._yolo = model
                self._yolo_name = candidate.name
                return self._yolo
            except Exception as exc:
                errors.append(f"{candidate}:{exc}")

        raise RuntimeError(
            "Unable to load YOLO model. Checked candidates: " + "; ".join(errors)
        )

    def _load_sam(self):
        if self._sam is not None or not self.enable_sam_refinement:
            return self._sam

        try:
            from ultralytics import SAM
        except Exception:
            self._sam_name = "unavailable"
            return None

        for candidate in self.sam_candidates:
            try:
                if not candidate.exists():
                    continue
                self._sam = SAM(str(candidate))
                self._sam_name = candidate.name
                return self._sam
            except Exception:
                continue

        self._sam_name = "not-loaded"
        return None

    def _resolve_task_path(self, task: dict[str, Any]) -> Path:
        data = task.get("data", {})
        image_value = data.get(self.task_image_key)
        if not image_value:
            raise ValueError(f"Task is missing data.{self.task_image_key}")

        if image_value.startswith("/data/local-files/"):
            parsed = urlparse(image_value)
            query = parse_qs(parsed.query)
            rel = query.get("d", [""])[0]
            rel = unquote(rel).lstrip("/")
            return self.workspace_root / rel

        if image_value.startswith("/"):
            return Path(image_value)

        return self.workspace_root / image_value

    @staticmethod
    def _contour_from_mask(mask: np.ndarray) -> np.ndarray | None:
        m = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            return None
        return contour[:, 0, :]

    @staticmethod
    def _contour_to_polygon_points(contour: np.ndarray, width: int, height: int) -> list[list[float]]:
        pts: list[list[float]] = []
        for x, y in contour.tolist():
            pts.append([100.0 * float(x) / width, 100.0 * float(y) / height])
        return pts

    @staticmethod
    def _iter_result_nodes(node: Any):
        if isinstance(node, dict):
            if "type" in node and "value" in node and isinstance(node.get("value"), dict):
                yield node
            for value in node.values():
                yield from ModelBackend._iter_result_nodes(value)
        elif isinstance(node, list):
            for item in node:
                yield from ModelBackend._iter_result_nodes(item)

    @staticmethod
    def _point_from_result(result: dict[str, Any], width: int, height: int) -> list[float] | None:
        value = result.get("value", {})
        if "x" in value and "y" in value:
            x = float(value["x"]) * width / 100.0
            y = float(value["y"]) * height / 100.0
            return [x, y]

        pts = value.get("points")
        if isinstance(pts, list) and pts and isinstance(pts[0], list) and len(pts[0]) >= 2:
            x = float(pts[0][0]) * width / 100.0
            y = float(pts[0][1]) * height / 100.0
            return [x, y]

        return None

    @staticmethod
    def _bbox_from_result(result: dict[str, Any], width: int, height: int) -> list[float] | None:
        value = result.get("value", {})
        required = ("x", "y", "width", "height")
        if not all(k in value for k in required):
            return None

        x = float(value["x"]) * width / 100.0
        y = float(value["y"]) * height / 100.0
        w = float(value["width"]) * width / 100.0
        h = float(value["height"]) * height / 100.0

        return [x, y, x + w, y + h]

    def _extract_sam_prompts(
        self,
        task: dict[str, Any],
        payload: dict[str, Any] | None,
        width: int,
        height: int,
        point_prompt_name: str | None = None,
        box_prompt_name: str | None = None,
    ) -> tuple[list[list[float]], list[list[float]]]:
        points: list[list[float]] = []
        bboxes: list[list[float]] = []
        point_seen: set[tuple[int, int]] = set()
        bbox_seen: set[tuple[int, int, int, int]] = set()

        sources: list[Any] = [task]
        if payload is not None:
            sources.append(payload)

        for source in sources:
            for result in self._iter_result_nodes(source):
                rtype = str(result.get("type", "")).lower()
                result_from_name = str(result.get("from_name", ""))
                if rtype == "keypointlabels":
                    if point_prompt_name and result_from_name and result_from_name != point_prompt_name:
                        continue
                    pt = self._point_from_result(result, width, height)
                    if pt is not None:
                        key = (int(round(pt[0])), int(round(pt[1])))
                        if key not in point_seen:
                            point_seen.add(key)
                            points.append(pt)
                elif rtype == "rectanglelabels":
                    if box_prompt_name and result_from_name and result_from_name != box_prompt_name:
                        continue
                    bbox = self._bbox_from_result(result, width, height)
                    if bbox is not None:
                        key = (
                            int(round(bbox[0])),
                            int(round(bbox[1])),
                            int(round(bbox[2])),
                            int(round(bbox[3])),
                        )
                        if key not in bbox_seen:
                            bbox_seen.add(key)
                            bboxes.append(bbox)

        return points, bboxes

    def _sam_predict_with_point(self, sam: Any, image: np.ndarray, point: list[float]):
        try:
            return sam.predict(source=image, points=[point], labels=[1], verbose=False)
        except Exception:
            return sam.predict(source=image, points=[[point[0], point[1]]], labels=[1], verbose=False)

    def _predict_from_sam_prompts(
        self,
        image: np.ndarray,
        from_name: str,
        to_name: str,
        label_name: str,
        width: int,
        height: int,
        points: list[list[float]],
        bboxes: list[list[float]],
    ) -> dict[str, Any]:
        sam = self._load_sam()
        if sam is None:
            return {
                "model_version": self._sam_name,
                "result": [],
                "score": 0.0,
            }

        result_items: list[dict[str, Any]] = []

        # Prefer box prompts when available; boxes are usually more stable for object masks.
        for idx, bbox in enumerate(bboxes[: self.return_topk]):
            try:
                res = sam.predict(source=image, bboxes=[bbox], verbose=False)
            except Exception:
                continue
            if not res or res[0].masks is None:
                continue
            mask = res[0].masks.data[0].detach().cpu().numpy()
            contour = self._contour_from_mask(mask)
            if contour is None:
                continue
            points_pct = self._contour_to_polygon_points(contour, width, height)
            if len(points_pct) < 3:
                continue
            result_items.append(
                {
                    "id": f"sam-box-{idx}",
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "polygonlabels",
                    "score": 0.95,
                    "value": {
                        "points": points_pct,
                        "polygonlabels": [label_name],
                    },
                }
            )

        if not result_items:
            for idx, point in enumerate(points[: self.return_topk]):
                try:
                    res = self._sam_predict_with_point(sam, image, point)
                except Exception:
                    continue
                if not res or res[0].masks is None:
                    continue
                mask = res[0].masks.data[0].detach().cpu().numpy()
                contour = self._contour_from_mask(mask)
                if contour is None:
                    continue
                points_pct = self._contour_to_polygon_points(contour, width, height)
                if len(points_pct) < 3:
                    continue
                result_items.append(
                    {
                        "id": f"sam-point-{idx}",
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "polygonlabels",
                        "score": 0.9,
                        "value": {
                            "points": points_pct,
                            "polygonlabels": [label_name],
                        },
                    }
                )

        score = 0.9 if result_items else 0.0
        return {
            "model_version": f"{self._sam_name}-prompt",
            "result": result_items,
            "score": score,
        }

    def _run_yolo(self, image: np.ndarray):
        model = self._load_yolo()
        return model.predict(
            source=image,
            conf=self.yolo_conf,
            imgsz=self.yolo_imgsz,
            device=self.device,
            verbose=False,
        )

    def _run_sam_refine(
        self,
        image: np.ndarray,
        bbox_xyxy: list[float],
        fallback_contour: np.ndarray | None,
    ) -> np.ndarray | None:
        sam = self._load_sam()
        if sam is None:
            return fallback_contour

        try:
            res = sam.predict(source=image, bboxes=[bbox_xyxy], verbose=False)
            if not res or res[0].masks is None:
                return fallback_contour
            mask = res[0].masks.data[0].detach().cpu().numpy()
            contour = self._contour_from_mask(mask)
            return contour if contour is not None else fallback_contour
        except Exception:
            return fallback_contour

    def predict_task(
        self,
        task: dict[str, Any],
        from_name: str,
        to_name: str,
        label_name: str,
        sam_from_name: str | None = None,
        sam_label_name: str | None = None,
        sam_point_name: str | None = None,
        sam_box_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        image_path = self._resolve_task_path(task)
        if not image_path.exists():
            return {
                "model_version": self._yolo_name,
                "result": [],
                "score": 0.0,
            }

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return {
                "model_version": self._yolo_name,
                "result": [],
                "score": 0.0,
            }

        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image_rgb.shape[:2]

        sam_points, sam_bboxes = self._extract_sam_prompts(
            task,
            payload,
            w,
            h,
            point_prompt_name=sam_point_name,
            box_prompt_name=sam_box_name,
        )
        if sam_points or sam_bboxes:
            prompted = self._predict_from_sam_prompts(
                image=image_rgb,
                from_name=sam_from_name or from_name,
                to_name=to_name,
                label_name=sam_label_name or label_name,
                width=w,
                height=h,
                points=sam_points,
                bboxes=sam_bboxes,
            )
            if prompted.get("result"):
                return prompted

        yolo_results = self._run_yolo(image_rgb)
        if not yolo_results:
            return {
                "model_version": self._yolo_name,
                "result": [],
                "score": 0.0,
            }

        r0 = yolo_results[0]
        masks = r0.masks
        boxes = r0.boxes

        if masks is None or boxes is None:
            return {
                "model_version": self._yolo_name,
                "result": [],
                "score": 0.0,
            }

        result_items: list[dict[str, Any]] = []
        conf_values: list[float] = []

        n = min(len(masks.data), self.return_topk)
        for idx in range(n):
            mask = masks.data[idx].detach().cpu().numpy()
            contour = self._contour_from_mask(mask)
            bbox = boxes.xyxy[idx].detach().cpu().tolist()
            conf = float(boxes.conf[idx].detach().cpu().item())

            contour = self._run_sam_refine(image_rgb, bbox, contour)
            if contour is None:
                continue

            points = self._contour_to_polygon_points(contour, w, h)
            if len(points) < 3:
                continue

            conf_values.append(conf)
            result_items.append(
                {
                    "id": f"pred-{idx}",
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "polygonlabels",
                    "score": conf,
                    "value": {
                        "points": points,
                        "polygonlabels": [label_name],
                    },
                }
            )

        score = float(np.mean(conf_values)) if conf_values else 0.0
        return {
            "model_version": f"{self._yolo_name}+{self._sam_name}",
            "result": result_items,
            "score": score,
        }


app = FastAPI(title="Lamapuit Label Studio ML Backend")
backend = ModelBackend()


@app.get("/health")
def health() -> dict[str, str]:
    return backend.health()


@app.post("/setup")
def setup() -> dict[str, Any]:
    # Warm model loading once to reduce first prediction latency.
    try:
        backend._load_yolo()
        backend._load_sam()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    return {"status": "ok", **backend.health()}


@app.post("/predict")
def predict(payload: PredictPayload) -> dict[str, Any]:
    params = payload.params or {}

    from_name = params.get("from_name", "label")
    to_name = params.get("to_name", backend.task_image_key)
    label_name = params.get("label", "car")
    sam_from_name = params.get("sam_from_name", from_name)
    sam_label_name = params.get("sam_label", label_name)
    sam_point_name = params.get("sam_point_name", "sam_point")
    sam_box_name = params.get("sam_box_name", "sam_box")

    payload_raw = payload.model_dump()
    results: list[dict[str, Any]] = []
    for task in payload.tasks:
        pred = backend.predict_task(
            task,
            from_name,
            to_name,
            label_name,
            sam_from_name=sam_from_name,
            sam_label_name=sam_label_name,
            sam_point_name=sam_point_name,
            sam_box_name=sam_box_name,
            payload=payload_raw,
        )
        results.append(pred)

    return {"results": results}

#!/usr/bin/env python3
"""Daily mini-retrain entrypoint for Label Studio iterative loop."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from cdw_detect.train import train


def resolve_model(candidate_csv: str) -> str:
    for item in candidate_csv.split(","):
        model = item.strip()
        if not model:
            continue
        path = Path(model)
        if path.exists() or model.endswith(".pt"):
            return model
    raise FileNotFoundError(f"No valid model candidate from: {candidate_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily mini-retrain on latest Label Studio exports")
    parser.add_argument(
        "--dataset-yaml",
        default="output/labelstudio_pipeline/yolo_dataset/dataset.yaml",
        help="Path to YOLO dataset.yaml produced from Label Studio export",
    )
    parser.add_argument(
        "--model-candidates",
        default="/models/yolo26s-seg.pt,yolo11s-seg.pt,yolo11n-seg.pt",
        help="Comma-separated model candidates (first available will be used)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Short retrain epochs")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Training device, e.g. 0 or cpu")
    parser.add_argument(
        "--project",
        default="output/labelstudio_pipeline/runs",
        help="Output train runs directory",
    )
    args = parser.parse_args()

    dataset_yaml = Path(args.dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML missing: {dataset_yaml}")

    model = resolve_model(args.model_candidates)
    run_name = f"daily_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    best_model = train(
        dataset_yaml=str(dataset_yaml),
        model=model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=run_name,
        device=args.device,
    )

    print(f"Daily mini-retrain complete. Best model: {best_model}")


if __name__ == "__main__":
    main()

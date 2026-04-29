#!/usr/bin/env python3
"""Convert Label Studio polygon exports to YOLO segmentation dataset format."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


def resolve_image_path(workspace_root: Path, image_value: str) -> Path:
    if image_value.startswith("/data/local-files/"):
        parsed = urlparse(image_value)
        rel = parse_qs(parsed.query).get("d", [""])[0]
        return workspace_root / unquote(rel).lstrip("/")
    if image_value.startswith("/"):
        return Path(image_value)
    return workspace_root / image_value


def yolo_line_from_polygon(points_percent: list[list[float]], class_id: int = 0) -> str:
    flat: list[str] = []
    for x_pct, y_pct in points_percent:
        x = max(0.0, min(100.0, float(x_pct))) / 100.0
        y = max(0.0, min(100.0, float(y_pct))) / 100.0
        flat.append(f"{x:.6f}")
        flat.append(f"{y:.6f}")
    return f"{class_id} " + " ".join(flat)


def collect_polygons(task: dict, label_name: str) -> list[str]:
    lines: list[str] = []
    annotations = task.get("annotations", [])
    if not annotations:
        return lines

    # Use latest annotation for deterministic export.
    latest = annotations[-1]
    for result in latest.get("result", []):
        if result.get("type") != "polygonlabels":
            continue
        value = result.get("value", {})
        labels = value.get("polygonlabels", [])
        if label_name not in labels:
            continue
        points = value.get("points", [])
        if len(points) < 3:
            continue
        lines.append(yolo_line_from_polygon(points, class_id=0))
    return lines


def write_dataset_yaml(output_dir: Path) -> None:
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        """
path: .
train: images/train
val: images/val
names:
  0: car
""".strip()
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Label Studio JSON export into YOLO segmentation dataset")
    parser.add_argument("--workspace-root", default=".", help="Workspace root path")
    parser.add_argument("--input-json", required=True, help="Label Studio JSON export path")
    parser.add_argument(
        "--output-dir",
        default="output/labelstudio_pipeline/yolo_dataset",
        help="Output dataset directory",
    )
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--label", default="car", help="Label name to export")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images (default is symlink for speed)",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    input_json = Path(args.input_json).resolve()
    output_dir = (workspace_root / args.output_dir).resolve()

    data = json.loads(input_json.read_text())
    if not isinstance(data, list):
        raise ValueError("Expected list of tasks in Label Studio JSON export")

    random.seed(args.seed)

    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"

    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0

    for task in data:
        image_value = task.get("data", {}).get("image")
        if not image_value:
            skipped += 1
            continue

        image_path = resolve_image_path(workspace_root, image_value)
        if not image_path.exists():
            skipped += 1
            continue

        polygons = collect_polygons(task, label_name=args.label)
        if not polygons:
            # Keep negatives as empty label files for segmentation training.
            polygons = []

        stem = image_path.stem
        ext = image_path.suffix
        is_val = random.random() < args.val_frac

        img_out = (images_val if is_val else images_train) / f"{stem}{ext}"
        lbl_out = (labels_val if is_val else labels_train) / f"{stem}.txt"

        if args.copy_images:
            shutil.copy2(image_path, img_out)
        else:
            if img_out.exists() or img_out.is_symlink():
                img_out.unlink()
            img_out.symlink_to(image_path)

        lbl_out.write_text("\n".join(polygons) + ("\n" if polygons else ""))
        exported += 1

    write_dataset_yaml(output_dir)

    print(f"Exported {exported} tasks to YOLO dataset: {output_dir}")
    print(f"Skipped {skipped} tasks (missing image path or image file)")


if __name__ == "__main__":
    main()

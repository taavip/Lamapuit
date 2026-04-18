#!/usr/bin/env python3

"""Generate Label Studio tasks JSON for project_bbox including data and data_ood folders."""

from __future__ import annotations

import json
from pathlib import Path

WORKSPACE = Path(".").resolve()
INPUT_DIRS = [
    Path("data/labeling_projects/project_bbox/data"),
    Path("data/labeling_projects/project_bbox/data_ood"),
    Path("data/labeling_projects/project_bbox/images"),
]
OUTPUT = Path("output/labelstudio_pipeline/tasks/project_bbox_tasks.json")


def build_task_url(workspace_root: Path, image_path: Path) -> str:
    rel = image_path.relative_to(workspace_root).as_posix()
    return f"/data/local-files/?d={rel}"


def main() -> None:
    workspace_root = WORKSPACE
    image_paths = []
    for d in INPUT_DIRS:
        root = (workspace_root / d).resolve()
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.png")):
            image_paths.append(p)
        for p in sorted(root.rglob("*.jpg")):
            image_paths.append(p)
        for p in sorted(root.rglob("*.jpeg")):
            image_paths.append(p)
        for p in sorted(root.rglob("*.tif")):
            image_paths.append(p)
    tasks = []
    for image_path in image_paths:
        name = image_path.name
        tasks.append({"data": {"image": build_task_url(workspace_root, image_path), "filename": name, "source": str(image_path.parent.name)}})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(tasks, indent=2))
    print(f"Generated {len(tasks)} tasks -> {OUTPUT}")


if __name__ == "__main__":
    main()

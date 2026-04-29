#!/usr/bin/env python3
"""Generate Label Studio task JSON from CHM car tile folders.

Creates tasks with local-files URLs for images under data/car/chm_2m_label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_task_url(workspace_root: Path, image_path: Path) -> str:
    rel = image_path.relative_to(workspace_root).as_posix()
    return f"/data/local-files/?d={rel}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Label Studio task JSON for CHM car tiles")
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root mounted in Label Studio local-files settings",
    )
    parser.add_argument(
        "--input-root",
        default="data/car/chm_2m_label",
        help="Folder with year subfolders and tiles directories",
    )
    parser.add_argument(
        "--glob",
        default="*/tiles/*.png",
        help="Glob under input-root to include as tasks",
    )
    parser.add_argument(
        "--output",
        default="output/labelstudio_pipeline/tasks/car_chm_tasks.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of tasks (0 means all)",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    input_root = (workspace_root / args.input_root).resolve()
    output_path = (workspace_root / args.output).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    image_paths = sorted(input_root.glob(args.glob))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    tasks: list[dict] = []
    for image_path in image_paths:
        year = image_path.parts[-3] if len(image_path.parts) >= 3 else "unknown"
        tasks.append(
            {
                "data": {
                    "image": build_task_url(workspace_root, image_path),
                    "filename": image_path.name,
                    "year": year,
                }
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2))

    print(f"Generated {len(tasks)} tasks -> {output_path}")


if __name__ == "__main__":
    main()

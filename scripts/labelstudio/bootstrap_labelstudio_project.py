#!/usr/bin/env python3
"""Bootstrap Label Studio project for CHM car polygon segmentation."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import requests


def _decode_jwt_payload(token: str) -> dict[str, str]:
    parts = token.split(".")
    if len(parts) != 3:
        return {}

    try:
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8"))
        data = json.loads(decoded.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_auth_token(base_url: str, api_key: str) -> tuple[str, str]:
    payload = _decode_jwt_payload(api_key)
    token_type = str(payload.get("token_type", "")).lower()

    if token_type == "refresh":
        resp = requests.post(
            f"{base_url}/api/token/refresh",
            json={"refresh": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        access = str(resp.json().get("access", "")).strip()
        if not access:
            raise RuntimeError("Refresh token exchange succeeded but no access token was returned")
        return ("Bearer", access)

    if token_type == "access":
        return ("Bearer", api_key)

    return ("Token", api_key)


def _headers(auth_scheme: str, token: str) -> dict[str, str]:
    return {
        "Authorization": f"{auth_scheme} {token}",
        "Content-Type": "application/json",
    }


def create_project(base_url: str, auth_scheme: str, token: str, title: str, label_config: str) -> int:
    payload = {
        "title": title,
        "label_config": label_config,
        "description": "Semi-automated CHM car polygon annotation with YOLO+SAM prelabels",
    }
    resp = requests.post(
        f"{base_url}/api/projects",
        headers=_headers(auth_scheme, token),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return int(resp.json()["id"])


def import_tasks(base_url: str, auth_scheme: str, token: str, project_id: int, tasks: list[dict]) -> int:
    resp = requests.post(
        f"{base_url}/api/projects/{project_id}/import",
        headers=_headers(auth_scheme, token),
        json=tasks,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return int(data.get("task_count", len(tasks)))


def attach_ml_backend(
    base_url: str,
    auth_scheme: str,
    token: str,
    project_id: int,
    ml_url: str,
    title: str,
) -> None:
    payload = {
        "url": ml_url,
        "title": title,
        "is_interactive": True,
        "project": project_id,
    }
    resp = requests.post(
        f"{base_url}/api/ml",
        headers=_headers(auth_scheme, token),
        json=payload,
        timeout=30,
    )
    if resp.status_code >= 400:
        print(f"[warn] Could not attach ML backend automatically ({resp.status_code}): {resp.text}")
        print("[warn] Attach manually in UI: Settings -> Model -> Connect Model")
        return

    print("[ok] ML backend attached")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and populate Label Studio project")
    parser.add_argument("--url", default="http://localhost:8080", help="Label Studio base URL")
    parser.add_argument("--api-key", required=True, help="Label Studio API token")
    parser.add_argument(
        "--title",
        default="Lamapuit Cars CHM Polygons",
        help="Project title",
    )
    parser.add_argument(
        "--label-config",
        default="configs/label_studio/car_polygon_label_config.xml",
        help="Path to Label Studio XML labeling config",
    )
    parser.add_argument(
        "--tasks-json",
        default="output/labelstudio_pipeline/tasks/car_chm_tasks.json",
        help="Path to generated task JSON",
    )
    parser.add_argument(
        "--ml-url",
        default="http://ml-backend:9090",
        help="ML backend URL as seen by Label Studio container",
    )
    parser.add_argument(
        "--skip-ml-attach",
        action="store_true",
        help="Skip automatic ML backend attachment",
    )
    args = parser.parse_args()

    label_config_path = Path(args.label_config)
    tasks_json_path = Path(args.tasks_json)

    if not label_config_path.exists():
        raise FileNotFoundError(f"Label config not found: {label_config_path}")
    if not tasks_json_path.exists():
        raise FileNotFoundError(f"Tasks JSON not found: {tasks_json_path}")

    label_config = label_config_path.read_text()
    tasks = json.loads(tasks_json_path.read_text())

    auth_scheme, token = _resolve_auth_token(args.url, args.api_key)
    project_id = create_project(args.url, auth_scheme, token, args.title, label_config)
    imported = import_tasks(args.url, auth_scheme, token, project_id, tasks)

    print(f"[ok] Project created id={project_id}")
    print(f"[ok] Imported tasks={imported}")

    if not args.skip_ml_attach:
        attach_ml_backend(args.url, auth_scheme, token, project_id, args.ml_url, "Lamapuit YOLO+SAM")

    print(
        "[next] Open project in UI -> Settings -> Annotation -> enable 'Show predictions' and pre-annotations"
    )


if __name__ == "__main__":
    main()

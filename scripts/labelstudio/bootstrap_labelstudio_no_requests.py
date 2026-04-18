#!/usr/bin/env python3
"""Bootstrap Label Studio project using only stdlib (no requests required)."""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def post_json(url: str, data: dict, headers: dict | None = None) -> dict:
    payload = json.dumps(data).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = Request(url, data=payload, headers=hdrs, method="POST")
    try:
        with urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except HTTPError as e:
        try:
            body = e.read().decode()
            print("HTTPError", e.code, body, file=sys.stderr)
        except Exception:
            print("HTTPError", e.code, file=sys.stderr)
        raise


def post_json_bytes(url: str, raw: bytes, headers: dict | None = None) -> dict:
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = Request(url, data=raw, headers=hdrs, method="POST")
    try:
        with urlopen(req, timeout=120) as resp:
            return json.load(resp)
    except HTTPError as e:
        try:
            body = e.read().decode()
            print("HTTPError", e.code, body, file=sys.stderr)
        except Exception:
            print("HTTPError", e.code, file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--label-config", required=True)
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--title", default="Project BBox")
    parser.add_argument("--ml-url", default="http://ml-backend:9090")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    token = args.api_key

    # Try exchange if refresh token
    auth_scheme = "Token"
    access_token = token
    try:
        payload_part = token.split(".")[1]
        payload_part += "=" * (-len(payload_part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_part.encode()).decode())
        if payload.get("token_type", "").lower() == "refresh":
            r = post_json(f"{base}/api/token/refresh", {"refresh": token})
            access_token = r.get("access", "")
            auth_scheme = "Bearer"
        elif payload.get("token_type", "").lower() == "access":
            auth_scheme = "Bearer"
    except Exception:
        # fallback: use Token scheme
        auth_scheme = "Token"

    headers = {"Authorization": f"{auth_scheme} {access_token}"}

    lc_path = Path(args.label_config)
    tasks_path = Path(args.tasks_json)
    if not lc_path.exists():
        print("Label config not found", lc_path, file=sys.stderr)
        sys.exit(2)
    if not tasks_path.exists():
        print("Tasks JSON not found", tasks_path, file=sys.stderr)
        sys.exit(2)

    label_config = lc_path.read_text()
    tasks_raw = tasks_path.read_bytes()

    # Create project
    payload = {"title": args.title, "label_config": label_config, "description": "Bounding-box image labeling"}
    proj = post_json(f"{base}/api/projects", payload, headers)
    project_id = int(proj.get("id"))
    print("PROJECT_ID", project_id)

    # Import tasks
    imported = post_json_bytes(f"{base}/api/projects/{project_id}/import", tasks_raw, headers)
    print("IMPORT_RESULT", imported)

    # Attach ML backend
    try:
        attach_payload = {"url": args.ml_url, "title": "Lamapuit YOLO+SAM", "is_interactive": True, "project": project_id}
        ml = post_json(f"{base}/api/ml", attach_payload, headers)
        print("ML_ATTACH", ml)
    except Exception:
        print("ML backend attach failed; attach manually in UI", file=sys.stderr)

    print(f"DONE project={project_id}")


if __name__ == "__main__":
    main()

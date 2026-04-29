#!/usr/bin/env python3
"""Request JWT token and run bootstrap script inside container context."""
from __future__ import annotations

import json
import subprocess
import sys
from urllib.request import Request, urlopen


def get_access_token(username: str, password: str) -> str:
    payload = json.dumps({"username": username, "password": password}).encode()
    req = Request("http://localhost:8080/api/token", data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=15) as resp:
        data = json.load(resp)
    return data.get("access", "")


def main() -> None:
    username = "taavi.pipar+lablestudio"
    password = "TempPass123!"
    try:
        access = get_access_token(username, password)
    except Exception as e:
        print("TOKEN_ERROR", e, file=sys.stderr)
        sys.exit(2)

    if not access:
        print("NO_ACCESS_TOKEN", file=sys.stderr)
        sys.exit(2)

    print("GOT_ACCESS", access)

    # Call the bootstrap script using the obtained access token
    cmd = ["python3", "/workspace/scripts/labelstudio/bootstrap_labelstudio_no_requests.py", "--api-key", access, "--label-config", "/workspace/configs/label_studio/bbox_label_config.xml", "--tasks-json", "/workspace/output/labelstudio_pipeline/tasks/project_bbox_tasks.json", "--title", "Project BBox", "--ml-url", "http://ml-backend:9090"]
    print("RUN_CMD:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

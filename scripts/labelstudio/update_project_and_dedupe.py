#!/usr/bin/env python3
"""Update a Label Studio project's label config and remove duplicate tasks.

Usage: run inside container or host where Label Studio is reachable.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def request_json(url: str, data: bytes | None = None, headers: dict | None = None, method: str = "GET"):
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = Request(url, data=data, headers=hdrs, method=method)
    try:
        with urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8")
            if not text:
                return {}
            return json.loads(text)
    except HTTPError as e:
        body = None
        try:
            body = e.read().decode()
        except Exception:
            pass
        print(f"HTTPError {e.code}: {body}", file=sys.stderr)
        raise


def resolve_access(base: str, token: str) -> tuple[str, str]:
    # token may be refresh or access; if refresh, exchange
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload.encode()).decode())
        ttype = data.get("token_type", "").lower()
        if ttype == "refresh":
            r = request_json(f"{base}/api/token/refresh", data=json.dumps({"refresh": token}).encode(), method="POST")
            access = r.get("access", "")
            return ("Bearer", access)
        if ttype == "access":
            return ("Bearer", token)
    except Exception:
        pass
    return ("Token", token)


def patch_project_label_config(base: str, auth_header: str, project_id: int, label_config: str):
    url = f"{base}/api/projects/{project_id}"
    payload = json.dumps({"label_config": label_config}).encode()
    return request_json(url, data=payload, headers={"Authorization": auth_header}, method="PATCH")


def list_project_tasks(base: str, auth_header: str, project_id: int) -> list[dict]:
    # try to fetch many tasks in one call
    url = f"{base}/api/tasks?project={project_id}&limit=10000"
    data = request_json(url, headers={"Authorization": auth_header})
    # API may return {'results': [...]} or a flat list
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    return []


def delete_task(base: str, auth_header: str, task_id: int) -> None:
    url = f"{base}/api/tasks/{task_id}"
    # urllib doesn't support DELETE directly in Request constructor on some Pythons, but method param works
    request_json(url, headers={"Authorization": auth_header}, method="DELETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--label-config", required=True)
    parser.add_argument("--auto-delete-duplicates", action="store_true")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    auth_scheme, token = resolve_access(base, args.api_key)
    auth_header = f"{auth_scheme} {token}"

    lc = open(args.label_config).read()
    print("Patching project label config...")
    patch_project_label_config(base, auth_header, args.project_id, lc)
    print("Patched label config.")

    print("Listing project tasks...")
    tasks = list_project_tasks(base, auth_header, args.project_id)
    print(f"Found {len(tasks)} tasks in project {args.project_id}")

    # find duplicates by image url (data.image)
    image_map: dict[str, list[int]] = {}
    for t in tasks:
        tid = int(t.get("id") or t.get("pk"))
        data = t.get("data") or {}
        img = data.get("image") or json.dumps(data, sort_keys=True)
        image_map.setdefault(img, []).append(tid)

    duplicates = {k: v for k, v in image_map.items() if len(v) > 1}
    if not duplicates:
        print("No duplicate tasks found.")
        return

    print(f"Found {len(duplicates)} duplicated image entries.")
    for img, ids in duplicates.items():
        print(f"Image: {img} -> task ids: {ids}")

    if args.auto_delete_duplicates:
        print("Deleting duplicate tasks (keeping first id for each image)...")
        for img, ids in duplicates.items():
            keep = ids[0]
            for tid in ids[1:]:
                try:
                    delete_task(base, auth_header, tid)
                    print(f"Deleted task {tid}")
                except Exception as e:
                    print(f"Failed to delete task {tid}: {e}")
        print("Duplicate deletion complete.")
    else:
        print("Run this script with --auto-delete-duplicates to remove them.")


if __name__ == "__main__":
    main()

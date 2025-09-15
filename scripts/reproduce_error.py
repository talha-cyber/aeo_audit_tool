#!/usr/bin/env python3
"""Incident Reproducer

Reads an incident JSON (schema: reports/schemas/incident_v1.schema.json) and attempts to
replay the failing operation to aid debugging. Supports HTTP replay; prints Celery guidance.

Examples:
  python scripts/reproduce_error.py --incident reports/incidents/incident_2025-09-09T10-00-00Z.json \
      --mode http --base http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

import httpx


def load_incident(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


async def replay_http(incident: Dict[str, Any], base: Optional[str]) -> None:
    http = incident.get("http", {})
    method = (http.get("method") or "GET").upper()
    url = http.get("url")
    path = http.get("path")
    headers = http.get("headers") or {}
    body = http.get("body")

    if base and path:
        target = base.rstrip("/") + "/" + path.lstrip("/")
    else:
        target = url

    if not target:
        print("No URL or path to replay.")
        return

    print(f"Replaying HTTP {method} {target}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.request(method, target, headers=headers, content=body)
        print(f"Status: {resp.status_code}")
        try:
            print("JSON:", resp.json())
        except Exception:
            print("Body:", resp.text[:2000])


def print_task_guidance(incident: Dict[str, Any]) -> None:
    task = incident.get("task", {})
    name = task.get("name")
    args = task.get("args")
    kwargs = task.get("kwargs")
    print("Celery reproduction guidance:")
    print("- Ensure redis and worker are running: docker-compose up -d redis worker")
    if name:
        print(
            f"- From a shell with app context: python -c \"from celery_worker import app; app.send_task('{name}', args={args}, kwargs={kwargs})\""
        )
    else:
        print("- No task name found in incident. Check incident JSON for 'task.name'.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Incident reproducer")
    parser.add_argument("--incident", required=True, help="path to incident JSON file")
    parser.add_argument("--mode", choices=["http", "task"], default="http")
    parser.add_argument(
        "--base", help="base URL for HTTP replay, e.g., http://localhost:8000"
    )
    args = parser.parse_args()

    incident_path = Path(args.incident)
    incident = load_incident(incident_path)

    meta = {
        "incident_id": incident.get("incident_id"),
        "request_id": incident.get("request_id"),
        "environment": incident.get("environment"),
        "kind": incident.get("kind"),
        "timestamp": incident.get("timestamp"),
    }
    print("Incident summary:", json.dumps(meta, indent=2))

    if args.mode == "http":
        await replay_http(incident, args.base)
    else:
        print_task_guidance(incident)


if __name__ == "__main__":
    asyncio.run(main())

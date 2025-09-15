from __future__ import annotations

from typing import Any, Dict

import psutil
from fastapi import APIRouter

from app.utils.resilience.monitoring.health import ResilienceHealthChecker

router = APIRouter(tags=["monitoring"], prefix="/monitoring")


@router.get("/snapshot")
async def monitoring_snapshot() -> Dict[str, Any]:
    """
    Lightweight monitoring snapshot for dashboards and runbooks.

    Returns:
      - dlq depths for primary queues
      - process CPU/memory
      - hints to Prometheus/Grafana endpoints (if known by deployment)
    """
    dlq_queues = ["dlq:audit:tasks"]
    checker = ResilienceHealthChecker()
    try:
        dlq = await checker.snapshot(dlq_queues=dlq_queues)
    except Exception:
        dlq = {"dlq": {q: None for q in dlq_queues}}

    # Process metrics
    p = psutil.Process()
    try:
        cpu_ratio = (p.cpu_percent(interval=0.0) or 0.0) / 100.0
    except Exception:
        cpu_ratio = 0.0
    try:
        mem_mb = p.memory_info().rss / (1024 * 1024)
    except Exception:
        mem_mb = None

    return {
        "dlq": dlq.get("dlq", {}),
        "process": {
            "cpu_ratio": cpu_ratio,
            "mem_mb": mem_mb,
        },
        "hints": {
            "prometheus": "http://localhost:9090",
            "grafana": "http://localhost:3000",
        },
    }

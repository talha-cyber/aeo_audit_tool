from __future__ import annotations

import os
from typing import Any, Dict

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - fallback metrics below
    psutil = None  # type: ignore
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
    cpu_ratio = 0.0
    mem_mb = None

    if psutil:
        p = psutil.Process()
        try:
            cpu_ratio = (p.cpu_percent(interval=0.0) or 0.0) / 100.0
        except Exception:
            cpu_ratio = 0.0
        try:
            mem_mb = p.memory_info().rss / (1024 * 1024)
        except Exception:
            mem_mb = None
    else:
        try:
            load, _, _ = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            cpu_ratio = max(0.0, min(1.0, load / cpu_count))
        except (AttributeError, OSError):
            cpu_ratio = 0.0

        try:
            import resource  # type: ignore

            usage = resource.getrusage(resource.RUSAGE_SELF)
            mem_mb = usage.ru_maxrss / 1024 if usage.ru_maxrss else None
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

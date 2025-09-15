from __future__ import annotations

from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utils.logger import get_logger

logger = get_logger(__name__)


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Scrub potentially sensitive data
        client_ip = request.client.host if request.client else None
        ua = request.headers.get("user-agent")
        path = request.url.path
        method = request.method
        logger.info(
            "access_start", method=method, path=path, client_ip=client_ip, ua=ua
        )
        resp = await call_next(request)
        logger.info(
            "access_end",
            method=method,
            path=path,
            status_code=resp.status_code,
            client_ip=client_ip,
        )
        return resp

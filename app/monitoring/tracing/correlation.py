from __future__ import annotations

import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Ensures every request gets a correlation/request ID.
    Propagates existing header `X-Request-ID` if present; otherwise generates one.
    Adds `X-Request-ID` to the response and sets `request.state.request_id`.
    """

    header_name = "X-Request-ID"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get(self.header_name) or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers.setdefault(self.header_name, request_id)
        return response

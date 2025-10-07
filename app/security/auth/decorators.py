"""Authorization helpers for capability-based access control."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, Optional

from fastapi import HTTPException, status


def _extract_user_arg(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Optional[dict[str, Any]]:
    signature = inspect.signature(func)
    bound = signature.bind_partial(*args, **kwargs)
    if "user" in bound.arguments:
        return bound.arguments["user"]
    return kwargs.get("user")


def _ensure_capability(user: Optional[dict[str, Any]], capability: str) -> None:
    capabilities = []
    if isinstance(user, dict):
        raw_capabilities = user.get("capabilities")
        if isinstance(raw_capabilities, (list, tuple, set)):
            capabilities = [str(item) for item in raw_capabilities]
    if capability not in capabilities:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient capability"
        )


def require_capability(
    capability: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator enforcing that the request context includes a capability."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                user = _extract_user_arg(func, args, kwargs)
                _ensure_capability(user, capability)
                return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            user = _extract_user_arg(func, args, kwargs)
            _ensure_capability(user, capability)
            return func(*args, **kwargs)

        return sync_wrapper

    return decorator


__all__ = ["require_capability"]

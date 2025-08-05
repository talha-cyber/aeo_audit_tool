"""
Structured logging configuration using structlog.

This module provides centralized logging configuration for both FastAPI and Celery.
It supports both development (human-readable) and production (JSON) formats.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.types import Processor

from app.core.config import settings


def configure_logging() -> None:
    """
    Configure structured logging for the entire application.

    This function sets up:
    - Development: Human-readable logs with colors
    - Production: JSON structured logs for machine processing
    """
    # Determine if we're in development mode
    is_development = settings.APP_ENV.lower() in ("development", "dev", "local")

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    # Build processor chain
    processors: list[Processor] = [
        # Add timestamp
        structlog.processors.TimeStamper(fmt="ISO"),
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Add log level
        structlog.stdlib.add_log_level,
        # Add caller info (useful for debugging)
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
    ]

    if is_development:
        # Development: Human-readable colored output
        processors.extend(
            [
                # Pretty print stack traces
                structlog.dev.set_exc_info,
                # Add colors and formatting
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        )
    else:
        # Production: JSON structured logs
        processors.extend(
            [
                # Format exception info
                structlog.processors.format_exc_info,
                # Render as JSON
                structlog.processors.JSONRenderer(),
            ]
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Standard context processors for common scenarios
def add_request_context(request: Any) -> Dict[str, Any]:
    """Add HTTP request context to logs."""
    try:
        return {
            "request_id": getattr(request.state, "request_id", None),
            "method": request.method,
            "url": str(request.url),
            "user_agent": request.headers.get("user-agent"),
        }
    except AttributeError:
        return {}


def add_audit_context(audit_run_id: str, client_id: str = None) -> Dict[str, Any]:
    """Add audit-specific context to logs."""
    context = {"audit_run_id": audit_run_id}
    if client_id:
        context["client_id"] = client_id
    return context


def add_platform_context(platform_name: str) -> Dict[str, Any]:
    """Add AI platform context to logs."""
    return {"ai_platform": platform_name}

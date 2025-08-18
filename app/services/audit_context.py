"""
Audit context helpers for structured logging and observability.

This module provides context managers and utilities for adding audit-specific
context to log entries, making it easier to trace and debug audit operations.
"""

import contextlib
import functools
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Thread-local storage for audit context
_context_storage = threading.local()


class AuditContext:
    """Container for audit execution context"""

    def __init__(self, audit_run_id: str, **extra_context):
        self.audit_run_id = audit_run_id
        self.extra_context = extra_context
        self.start_time = datetime.now(timezone.utc)
        self.context_id = str(uuid.uuid4())[:8]  # Short ID for this context

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging"""
        base_context = {
            "audit_run_id": self.audit_run_id,
            "context_id": self.context_id,
            "context_start_time": self.start_time.isoformat(),
        }
        base_context.update(self.extra_context)
        return base_context

    def __str__(self) -> str:
        return f"AuditContext(audit_run_id={self.audit_run_id}, context_id={self.context_id})"


def get_current_audit_context() -> Optional[AuditContext]:
    """Get the current audit context from thread-local storage"""
    return getattr(_context_storage, "audit_context", None)


def set_current_audit_context(context: Optional[AuditContext]) -> None:
    """Set the current audit context in thread-local storage"""
    _context_storage.audit_context = context


@contextlib.contextmanager
def add_audit_context(audit_run_id: str, **extra_context):
    """
    Context manager to add audit-specific context to all log entries.

    Usage:
        with add_audit_context(audit_run_id="123", client_id="456"):
            logger.info("This will include audit context")

    Args:
        audit_run_id: The ID of the audit run
        **extra_context: Additional context fields
    """
    # Create new audit context
    context = AuditContext(audit_run_id, **extra_context)

    # Store previous context to restore later
    previous_context = get_current_audit_context()

    try:
        # Set new context
        set_current_audit_context(context)

        # Bind context to logger
        bound_logger = logger.bind(**context.to_dict())

        # Replace logger temporarily
        import app.services.audit_context

        original_logger = app.services.audit_context.logger
        app.services.audit_context.logger = bound_logger

        bound_logger.info("Audit context established", context=str(context))
        yield bound_logger

    finally:
        # Restore previous state
        set_current_audit_context(previous_context)
        app.services.audit_context.logger = original_logger


@contextlib.contextmanager
def add_stage_context(stage_name: str, **extra_context):
    """
    Add stage-specific context to current audit context.

    Usage:
        with add_stage_context("question_generation", provider="openai"):
            logger.info("Processing stage")

    Args:
        stage_name: Name of the current processing stage
        **extra_context: Additional stage-specific context
    """
    current_context = get_current_audit_context()
    if not current_context:
        # No audit context available, just use regular logging
        yield logger
        return

    # Create stage context
    stage_context = {
        "stage": stage_name,
        "stage_id": str(uuid.uuid4())[:8],
        "stage_start_time": datetime.now(timezone.utc).isoformat(),
        **extra_context,
    }

    # Merge with existing context
    full_context = {**current_context.to_dict(), **stage_context}

    # Bind to logger
    bound_logger = logger.bind(**full_context)

    bound_logger.info("Stage started", stage=stage_name)
    try:
        yield bound_logger
    finally:
        bound_logger.info("Stage completed", stage=stage_name)


def add_platform_context(platform_name: str, **extra_context) -> Dict[str, Any]:
    """
    Add platform-specific context to logs.

    Args:
        platform_name: Name of the AI platform (e.g., "openai", "anthropic")
        **extra_context: Additional platform-specific context

    Returns:
        Dictionary of context fields
    """
    platform_context = {
        "platform": platform_name,
        "platform_request_id": str(uuid.uuid4())[:8],
        **extra_context,
    }

    # Add current audit context if available
    current_context = get_current_audit_context()
    if current_context:
        platform_context.update(current_context.to_dict())

    return platform_context


def add_question_context(
    question_id: str, question_preview: str, **extra_context
) -> Dict[str, Any]:
    """
    Add question-specific context to logs.

    Args:
        question_id: ID of the question being processed
        question_preview: Preview of question text (truncated)
        **extra_context: Additional question-specific context

    Returns:
        Dictionary of context fields
    """
    question_context = {
        "question_id": question_id,
        "question_preview": question_preview[:100] + "..."
        if len(question_preview) > 100
        else question_preview,
        **extra_context,
    }

    # Add current audit context if available
    current_context = get_current_audit_context()
    if current_context:
        question_context.update(current_context.to_dict())

    return question_context


def add_batch_context(
    batch_index: int, total_batches: int, **extra_context
) -> Dict[str, Any]:
    """
    Add batch processing context to logs.

    Args:
        batch_index: Current batch number (1-based)
        total_batches: Total number of batches
        **extra_context: Additional batch-specific context

    Returns:
        Dictionary of context fields
    """
    batch_context = {
        "batch_index": batch_index,
        "total_batches": total_batches,
        "batch_progress": f"{batch_index}/{total_batches}",
        "batch_id": str(uuid.uuid4())[:8],
        **extra_context,
    }

    # Add current audit context if available
    current_context = get_current_audit_context()
    if current_context:
        batch_context.update(current_context.to_dict())

    return batch_context


def add_brand_detection_context(target_brands: list, **extra_context) -> Dict[str, Any]:
    """
    Add brand detection context to logs.

    Args:
        target_brands: List of brands being detected
        **extra_context: Additional brand detection context

    Returns:
        Dictionary of context fields
    """
    brand_context = {
        "target_brands": target_brands,
        "brand_count": len(target_brands),
        "detection_id": str(uuid.uuid4())[:8],
        **extra_context,
    }

    # Add current audit context if available
    current_context = get_current_audit_context()
    if current_context:
        brand_context.update(current_context.to_dict())

    return brand_context


def audit_context_decorator(func: Callable) -> Callable:
    """
    Decorator to automatically add audit context to function calls.

    This decorator looks for 'audit_run_id' in the function arguments
    and automatically adds audit context for the duration of the function.

    Usage:
        @audit_context_decorator
        def process_audit(audit_run_id: str, other_param: str):
            logger.info("This will have audit context")
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Look for audit_run_id in arguments
        audit_run_id = None

        # Check kwargs first
        if "audit_run_id" in kwargs:
            audit_run_id = kwargs["audit_run_id"]
        else:
            # Check if function has audit_run_id parameter
            import inspect

            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            if "audit_run_id" in param_names:
                param_index = param_names.index("audit_run_id")
                if param_index < len(args):
                    audit_run_id = args[param_index]

        if audit_run_id:
            # Add audit context
            with add_audit_context(
                audit_run_id=str(audit_run_id), function_name=func.__name__
            ):
                return func(*args, **kwargs)
        else:
            # No audit context available, call function normally
            return func(*args, **kwargs)

    return wrapper


async def async_audit_context_decorator(func: Callable) -> Callable:
    """
    Async version of audit_context_decorator.

    Usage:
        @async_audit_context_decorator
        async def process_audit_async(audit_run_id: str):
            logger.info("This will have audit context")
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Look for audit_run_id in arguments (same logic as sync version)
        audit_run_id = None

        if "audit_run_id" in kwargs:
            audit_run_id = kwargs["audit_run_id"]
        else:
            import inspect

            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            if "audit_run_id" in param_names:
                param_index = param_names.index("audit_run_id")
                if param_index < len(args):
                    audit_run_id = args[param_index]

        if audit_run_id:
            with add_audit_context(
                audit_run_id=str(audit_run_id), function_name=func.__name__
            ):
                return await func(*args, **kwargs)
        else:
            return await func(*args, **kwargs)

    return wrapper


class ContextualLogger:
    """
    Logger wrapper that automatically includes current audit context.

    This provides a convenient way to log with context without manually
    binding context to every log call.
    """

    def __init__(self, base_logger=None):
        self.base_logger = base_logger or logger

    def _get_contextual_logger(self):
        """Get logger with current context bound"""
        current_context = get_current_audit_context()
        if current_context:
            return self.base_logger.bind(**current_context.to_dict())
        return self.base_logger

    def info(self, message: str, **kwargs):
        """Log info message with current context"""
        self._get_contextual_logger().info(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with current context"""
        self._get_contextual_logger().error(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with current context"""
        self._get_contextual_logger().warning(message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message with current context"""
        self._get_contextual_logger().debug(message, **kwargs)


# Global contextual logger instance
contextual_logger = ContextualLogger()


def get_audit_context_summary() -> Dict[str, Any]:
    """
    Get a summary of the current audit context for debugging.

    Returns:
        Dictionary with current context information
    """
    current_context = get_current_audit_context()
    if current_context:
        return {
            "has_context": True,
            "audit_run_id": current_context.audit_run_id,
            "context_id": current_context.context_id,
            "start_time": current_context.start_time.isoformat(),
            "extra_fields": list(current_context.extra_context.keys()),
        }
    else:
        return {
            "has_context": False,
            "message": "No audit context currently active",
        }

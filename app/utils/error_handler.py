"""
Comprehensive error handling system for the AEO Audit Tool.

This module provides standardized error handling, custom exception classes,
error categorization, recovery strategies, and integration with monitoring.
"""

import asyncio
import functools
import traceback
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.audit_metrics import get_audit_metrics
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels for categorization and alerting"""

    LOW = "low"  # Minor issues, logging only
    MEDIUM = "medium"  # Important issues, monitoring alerts
    HIGH = "high"  # Critical issues, immediate attention
    CRITICAL = "critical"  # System-threatening issues, emergency response


class ErrorCategory(str, Enum):
    """Error categories for systematic handling"""

    CONFIGURATION = "configuration"  # Config/setup errors
    AUTHENTICATION = "authentication"  # Auth/authorization errors
    PLATFORM = "platform"  # AI platform errors
    NETWORK = "network"  # Network/connectivity errors
    DATABASE = "database"  # Database errors
    VALIDATION = "validation"  # Input validation errors
    BUSINESS_LOGIC = "business_logic"  # Business rule violations
    SYSTEM = "system"  # System/infrastructure errors
    RATE_LIMIT = "rate_limit"  # Rate limiting errors
    TIMEOUT = "timeout"  # Timeout errors


class ErrorContext:
    """Container for error context information"""

    def __init__(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        user_message: Optional[str] = None,
        technical_details: Optional[Dict[str, Any]] = None,
        audit_run_id: Optional[str] = None,
        platform_name: Optional[str] = None,
        recovery_suggestions: Optional[list] = None,
    ):
        self.error = error
        self.severity = severity
        self.category = category
        self.user_message = user_message or self._generate_user_message()
        self.technical_details = technical_details or {}
        self.audit_run_id = audit_run_id
        self.platform_name = platform_name
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = datetime.now(timezone.utc)
        self.error_id = f"err_{int(self.timestamp.timestamp())}"

    def _generate_user_message(self) -> str:
        """Generate user-friendly error message based on error type"""
        error_type = type(self.error).__name__

        user_messages = {
            "PlatformAuthenticationError": "There was an issue with platform authentication. Please check your API credentials.",
            "PlatformRateLimitError": "The platform is currently rate limiting requests. Please try again in a few minutes.",
            "PlatformTimeoutError": "The platform request timed out. This may be a temporary issue.",
            "AuditConfigurationError": "There's an issue with the audit configuration. Please check your settings.",
            "DatabaseConnectionError": "Unable to connect to the database. Please try again.",
            "ValidationError": "The provided data is invalid. Please check your input.",
            "NetworkError": "Network connectivity issue. Please check your connection.",
        }

        return user_messages.get(
            error_type,
            "An unexpected error occurred. Please try again or contact support.",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for logging/serialization"""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "severity": self.severity.value,
            "category": self.category.value,
            "user_message": self.user_message,
            "technical_details": self.technical_details,
            "audit_run_id": self.audit_run_id,
            "platform_name": self.platform_name,
            "recovery_suggestions": self.recovery_suggestions,
            "traceback": traceback.format_exc()
            if self.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
            else None,
        }


# === Custom Exception Classes ===


class AuditProcessorError(Exception):
    """Base exception for audit processor errors"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        technical_details: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[list] = None,
    ):
        super().__init__(message)
        self.severity = severity
        self.category = category
        self.technical_details = technical_details or {}
        self.recovery_suggestions = recovery_suggestions or []


class AuditConfigurationError(AuditProcessorError):
    """Raised when audit configuration is invalid"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            **kwargs,
        )


class PlatformError(AuditProcessorError):
    """Base class for AI platform errors"""

    def __init__(self, platform_name: str, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.PLATFORM, **kwargs)
        self.platform_name = platform_name


class PlatformAuthenticationError(PlatformError):
    """Raised when platform authentication fails"""

    def __init__(self, platform_name: str, message: str = None, **kwargs):
        message = message or f"Authentication failed for platform: {platform_name}"
        super().__init__(
            platform_name,
            message,
            severity=ErrorSeverity.HIGH,
            recovery_suggestions=[
                "Check API credentials",
                "Verify account permissions",
                "Check platform service status",
            ],
            **kwargs,
        )


class PlatformRateLimitError(PlatformError):
    """Raised when platform rate limits are exceeded"""

    def __init__(self, platform_name: str, retry_after: Optional[int] = None, **kwargs):
        message = f"Rate limit exceeded for platform: {platform_name}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"

        super().__init__(
            platform_name,
            message,
            severity=ErrorSeverity.MEDIUM,
            recovery_suggestions=[
                "Wait before retrying",
                "Check platform usage limits",
                "Consider upgrading platform plan",
            ],
            technical_details={"retry_after": retry_after},
            **kwargs,
        )


class PlatformTimeoutError(PlatformError):
    """Raised when platform requests timeout"""

    def __init__(
        self, platform_name: str, timeout_seconds: Optional[float] = None, **kwargs
    ):
        message = f"Timeout error for platform: {platform_name}"
        if timeout_seconds:
            message += f" after {timeout_seconds} seconds"

        super().__init__(
            platform_name,
            message,
            severity=ErrorSeverity.MEDIUM,
            recovery_suggestions=[
                "Retry the request",
                "Check platform service status",
                "Consider increasing timeout settings",
            ],
            technical_details={"timeout_seconds": timeout_seconds},
            **kwargs,
        )


class PlatformUnavailableError(PlatformError):
    """Raised when platform is unavailable"""

    def __init__(self, platform_name: str, **kwargs):
        super().__init__(
            platform_name,
            f"Platform unavailable: {platform_name}",
            severity=ErrorSeverity.HIGH,
            recovery_suggestions=[
                "Check platform service status",
                "Try alternative platforms",
                "Contact platform support",
            ],
            **kwargs,
        )


class QuestionGenerationError(AuditProcessorError):
    """Raised when question generation fails"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.BUSINESS_LOGIC,
            recovery_suggestions=[
                "Check client configuration",
                "Verify question templates",
                "Review audit parameters",
            ],
            **kwargs,
        )


class BrandDetectionError(AuditProcessorError):
    """Raised when brand detection fails"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.BUSINESS_LOGIC,
            recovery_suggestions=[
                "Check brand detection configuration",
                "Verify response text quality",
                "Review detection parameters",
            ],
            **kwargs,
        )


class DatabaseConnectionError(AuditProcessorError):
    """Raised when database connection fails"""

    def __init__(self, message: str = "Database connection failed", **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATABASE,
            recovery_suggestions=[
                "Check database connectivity",
                "Verify database credentials",
                "Check database service status",
            ],
            **kwargs,
        )


# === Error Handler Class ===


class ErrorHandler:
    """Centralized error handling and recovery system"""

    def __init__(self):
        self.metrics = get_audit_metrics()
        self._error_counts = {}

    def handle_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """
        Handle an error with appropriate categorization, logging, and recovery suggestions.

        Args:
            error: The exception that occurred
            context: Additional context information

        Returns:
            ErrorContext with detailed error information
        """
        context = context or {}

        # Categorize error
        error_context = self._categorize_error(error, context)

        # Log error with appropriate level
        self._log_error(error_context)

        # Record metrics
        self._record_error_metrics(error_context)

        # Check for recovery opportunities
        self._suggest_recovery(error_context)

        return error_context

    def _categorize_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> ErrorContext:
        """Categorize error and determine severity"""

        # Handle custom audit processor errors
        if isinstance(error, AuditProcessorError):
            return ErrorContext(
                error=error,
                severity=error.severity,
                category=error.category,
                technical_details=error.technical_details,
                audit_run_id=context.get("audit_run_id"),
                platform_name=getattr(error, "platform_name", None),
                recovery_suggestions=error.recovery_suggestions,
            )

        # Handle standard Python exceptions
        error_mappings = {
            ConnectionError: (ErrorSeverity.MEDIUM, ErrorCategory.NETWORK),
            TimeoutError: (ErrorSeverity.MEDIUM, ErrorCategory.TIMEOUT),
            ValueError: (ErrorSeverity.LOW, ErrorCategory.VALIDATION),
            KeyError: (ErrorSeverity.LOW, ErrorCategory.VALIDATION),
            FileNotFoundError: (ErrorSeverity.MEDIUM, ErrorCategory.CONFIGURATION),
            PermissionError: (ErrorSeverity.HIGH, ErrorCategory.AUTHENTICATION),
        }

        severity, category = error_mappings.get(
            type(error), (ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
        )

        return ErrorContext(
            error=error,
            severity=severity,
            category=category,
            audit_run_id=context.get("audit_run_id"),
            platform_name=context.get("platform_name"),
            technical_details=context,
        )

    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level and context"""

        log_data = {
            "error_id": error_context.error_id,
            "error_type": type(error_context.error).__name__,
            "category": error_context.category.value,
            "severity": error_context.severity.value,
            "audit_run_id": error_context.audit_run_id,
            "platform_name": error_context.platform_name,
            "technical_details": error_context.technical_details,
        }

        # Log with appropriate level based on severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(error_context.user_message, **log_data, exc_info=True)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(error_context.user_message, **log_data, exc_info=True)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(error_context.user_message, **log_data)
        else:  # LOW
            logger.info(error_context.user_message, **log_data)

    def _record_error_metrics(self, error_context: ErrorContext):
        """Record error metrics for monitoring"""

        # Record general error metrics
        self.metrics.increment_error_count(
            error_type=type(error_context.error).__name__,
            category=error_context.category.value,
            severity=error_context.severity.value,
        )

        # Record platform-specific metrics if applicable
        if error_context.platform_name:
            self.metrics.increment_platform_error(error_context.platform_name)

        # Track error frequency for pattern detection
        error_key = (
            f"{type(error_context.error).__name__}:{error_context.category.value}"
        )
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1

        # Alert on error patterns
        if self._error_counts[error_key] > 5:  # Configurable threshold
            logger.warning(
                "High frequency error detected",
                error_key=error_key,
                count=self._error_counts[error_key],
                error_id=error_context.error_id,
            )

    def _suggest_recovery(self, error_context: ErrorContext):
        """Add intelligent recovery suggestions based on error patterns"""

        # Add category-specific recovery suggestions
        category_suggestions = {
            ErrorCategory.PLATFORM: [
                "Check platform service status",
                "Verify API credentials",
                "Try alternative platforms",
            ],
            ErrorCategory.NETWORK: [
                "Check internet connectivity",
                "Verify firewall settings",
                "Retry after network stabilizes",
            ],
            ErrorCategory.DATABASE: [
                "Check database connectivity",
                "Verify database credentials",
                "Check available disk space",
            ],
            ErrorCategory.RATE_LIMIT: [
                "Wait before retrying",
                "Consider platform upgrade",
                "Implement request batching",
            ],
        }

        if error_context.category in category_suggestions:
            error_context.recovery_suggestions.extend(
                category_suggestions[error_context.category]
            )


# === Error Handling Decorators ===


def handle_audit_errors(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
):
    """Decorator for handling audit-specific errors in functions"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handler = ErrorHandler()
                context = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs),
                }
                handler.handle_error(e, context)

                # Re-raise as appropriate error type
                if isinstance(e, AuditProcessorError):
                    raise
                else:
                    raise AuditProcessorError(
                        str(e),
                        severity=severity,
                        category=category,
                        technical_details=context,
                    )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = ErrorHandler()
                context = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs),
                }
                handler.handle_error(e, context)

                # Re-raise as appropriate error type
                if isinstance(e, AuditProcessorError):
                    raise
                else:
                    raise AuditProcessorError(
                        str(e),
                        severity=severity,
                        category=category,
                        technical_details=context,
                    )

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# === FastAPI Error Handling Middleware ===


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for centralized error handling"""

    def __init__(self, app, error_handler: Optional[ErrorHandler] = None):
        super().__init__(app)
        self.error_handler = error_handler or ErrorHandler()

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Handle error
            context = {
                "request_url": str(request.url),
                "request_method": request.method,
                "client_ip": request.client.host if request.client else None,
            }

            error_context = self.error_handler.handle_error(e, context)

            # Convert to HTTP response
            if isinstance(e, HTTPException):
                raise  # Let FastAPI handle it

            # Map severity to HTTP status codes
            status_code_map = {
                ErrorSeverity.LOW: 400,
                ErrorSeverity.MEDIUM: 500,
                ErrorSeverity.HIGH: 500,
                ErrorSeverity.CRITICAL: 503,
            }

            status_code = status_code_map.get(error_context.severity, 500)

            return JSONResponse(
                status_code=status_code,
                content={
                    "error": {
                        "id": error_context.error_id,
                        "message": error_context.user_message,
                        "category": error_context.category.value,
                        "severity": error_context.severity.value,
                        "recovery_suggestions": error_context.recovery_suggestions,
                        "timestamp": error_context.timestamp.isoformat(),
                    }
                },
            )


# === Global Error Handler Instance ===

error_handler = ErrorHandler()


# === Utility Functions ===


def create_error_response(
    error: Exception, context: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Create standardized error response for APIs"""

    error_context = error_handler.handle_error(error, context)

    status_code_map = {
        ErrorSeverity.LOW: 400,
        ErrorSeverity.MEDIUM: 500,
        ErrorSeverity.HIGH: 500,
        ErrorSeverity.CRITICAL: 503,
    }

    status_code = status_code_map.get(error_context.severity, 500)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "id": error_context.error_id,
                "message": error_context.user_message,
                "category": error_context.category.value,
                "severity": error_context.severity.value,
                "recovery_suggestions": error_context.recovery_suggestions,
                "timestamp": error_context.timestamp.isoformat(),
            }
        },
    )


def log_audit_error(
    error: Exception,
    audit_run_id: str,
    platform_name: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Log an audit-specific error and return error ID for tracking.

    Args:
        error: The exception that occurred
        audit_run_id: ID of the audit run
        platform_name: Name of the platform if applicable
        additional_context: Additional context information

    Returns:
        Error ID for tracking and correlation
    """
    context = {"audit_run_id": audit_run_id, "platform_name": platform_name}

    if additional_context:
        context.update(additional_context)

    error_context = error_handler.handle_error(error, context)
    return error_context.error_id

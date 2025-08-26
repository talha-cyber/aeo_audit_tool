"""
Custom exceptions for AI platform clients.

Provides a hierarchy of exceptions for different types of platform errors
to enable proper error handling and retry logic.
"""


class PlatformError(Exception):
    """Base exception for all platform errors."""

    pass


class TransientError(PlatformError):
    """
    Temporary error that should be retried.

    Used for network timeouts, server errors (5xx), and other temporary issues
    that may resolve with retry attempts.
    """

    pass


class RateLimitError(PlatformError):
    """
    Rate limit exceeded error.

    Includes optional retry_after information for implementing proper backoff.
    """

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(PlatformError):
    """
    Authentication/API key error.

    Used for 401/403 errors that indicate invalid credentials.
    These errors should not be retried.
    """

    pass


class QuotaExceededError(PlatformError):
    """
    Monthly/usage quota exceeded.

    Indicates the API quota has been exceeded and requests will fail
    until quota resets. Should not be retried immediately.
    """

    pass


class MalformedResponseError(PlatformError):
    """
    Response format is invalid or unexpected.

    Used when the platform returns a response that doesn't match
    the expected format for text extraction.
    """

    pass


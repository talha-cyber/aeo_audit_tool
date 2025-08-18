"""
Tests for the comprehensive error handling system.

This module tests the error handling functionality including:
- Custom exception classes
- Error categorization and severity
- Error context and logging
- Recovery suggestions
- Metrics integration
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

from app.utils.error_handler import (
    AuditConfigurationError,
    AuditProcessorError,
    BrandDetectionError,
    DatabaseConnectionError,
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorHandlingMiddleware,
    ErrorSeverity,
    PlatformAuthenticationError,
    PlatformRateLimitError,
    PlatformTimeoutError,
    PlatformUnavailableError,
    QuestionGenerationError,
    create_error_response,
    handle_audit_errors,
    log_audit_error,
)


class TestErrorContext:
    """Test cases for ErrorContext class"""

    def test_error_context_creation(self):
        """Test basic error context creation"""
        error = ValueError("Test error")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            audit_run_id="audit_123",
            platform_name="openai",
        )

        assert context.error == error
        assert context.severity == ErrorSeverity.HIGH
        assert context.category == ErrorCategory.VALIDATION
        assert context.audit_run_id == "audit_123"
        assert context.platform_name == "openai"
        assert context.error_id.startswith("err_")
        assert isinstance(context.timestamp, datetime)

    def test_error_context_user_message_generation(self):
        """Test automatic user message generation"""
        # Test with custom error type
        error = PlatformAuthenticationError("openai", "Invalid API key")
        context = ErrorContext(error=error)

        assert "authentication" in context.user_message.lower()

        # Test with generic error
        error = ValueError("Some validation error")
        context = ErrorContext(error=error)

        assert "unexpected error" in context.user_message.lower()

    def test_error_context_serialization(self):
        """Test error context serialization to dict"""
        error = ValueError("Test error")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            technical_details={"key": "value"},
            recovery_suggestions=["Check input", "Try again"],
        )

        context_dict = context.to_dict()

        assert context_dict["error_type"] == "ValueError"
        assert context_dict["error_message"] == "Test error"
        assert context_dict["severity"] == "medium"
        assert context_dict["category"] == "validation"
        assert context_dict["technical_details"]["key"] == "value"
        assert "Check input" in context_dict["recovery_suggestions"]


class TestCustomExceptions:
    """Test cases for custom exception classes"""

    def test_audit_processor_error(self):
        """Test base AuditProcessorError"""
        error = AuditProcessorError(
            "Test error",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            technical_details={"detail": "value"},
            recovery_suggestions=["Suggestion 1"],
        )

        assert str(error) == "Test error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.SYSTEM
        assert error.technical_details["detail"] == "value"
        assert "Suggestion 1" in error.recovery_suggestions

    def test_audit_configuration_error(self):
        """Test AuditConfigurationError"""
        error = AuditConfigurationError("Invalid configuration")

        assert str(error) == "Invalid configuration"
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.CONFIGURATION

    def test_platform_authentication_error(self):
        """Test PlatformAuthenticationError"""
        error = PlatformAuthenticationError("openai", "Invalid API key")

        assert str(error) == "Invalid API key"
        assert error.platform_name == "openai"
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.PLATFORM
        assert "Check API credentials" in error.recovery_suggestions

    def test_platform_rate_limit_error(self):
        """Test PlatformRateLimitError"""
        error = PlatformRateLimitError("anthropic", retry_after=60)

        assert "Rate limit exceeded" in str(error)
        assert "60 seconds" in str(error)
        assert error.platform_name == "anthropic"
        assert error.technical_details["retry_after"] == 60
        assert "Wait before retrying" in error.recovery_suggestions

    def test_platform_timeout_error(self):
        """Test PlatformTimeoutError"""
        error = PlatformTimeoutError("openai", timeout_seconds=30.0)

        assert "Timeout error" in str(error)
        assert "30.0 seconds" in str(error)
        assert error.technical_details["timeout_seconds"] == 30.0
        assert "Retry the request" in error.recovery_suggestions

    def test_platform_unavailable_error(self):
        """Test PlatformUnavailableError"""
        error = PlatformUnavailableError("google")

        assert "Platform unavailable: google" in str(error)
        assert error.severity == ErrorSeverity.HIGH
        assert "Check platform service status" in error.recovery_suggestions

    def test_question_generation_error(self):
        """Test QuestionGenerationError"""
        error = QuestionGenerationError("Failed to generate questions")

        assert error.category == ErrorCategory.BUSINESS_LOGIC
        assert "Check client configuration" in error.recovery_suggestions

    def test_brand_detection_error(self):
        """Test BrandDetectionError"""
        error = BrandDetectionError("Brand detection failed")

        assert error.category == ErrorCategory.BUSINESS_LOGIC
        assert "Check brand detection configuration" in error.recovery_suggestions

    def test_database_connection_error(self):
        """Test DatabaseConnectionError"""
        error = DatabaseConnectionError()

        assert "Database connection failed" in str(error)
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.DATABASE


class TestErrorHandler:
    """Test cases for ErrorHandler class"""

    @pytest.fixture
    def error_handler(self):
        """Create ErrorHandler instance with mocked dependencies"""
        with patch("app.utils.error_handler.get_audit_metrics") as mock_metrics:
            mock_metrics.return_value = Mock()
            return ErrorHandler()

    def test_error_handler_initialization(self, error_handler):
        """Test error handler initialization"""
        assert error_handler.metrics is not None
        assert error_handler._error_counts == {}

    def test_handle_custom_error(self, error_handler):
        """Test handling custom audit processor errors"""
        error = PlatformAuthenticationError("openai", "Invalid key")
        context = {"audit_run_id": "audit_123"}

        error_context = error_handler.handle_error(error, context)

        assert error_context.error == error
        assert error_context.severity == ErrorSeverity.HIGH
        assert error_context.category == ErrorCategory.PLATFORM
        assert error_context.platform_name == "openai"
        assert error_context.audit_run_id == "audit_123"

    def test_handle_standard_exception(self, error_handler):
        """Test handling standard Python exceptions"""
        error = ConnectionError("Network connection failed")
        context = {"audit_run_id": "audit_456"}

        error_context = error_handler.handle_error(error, context)

        assert error_context.error == error
        assert error_context.severity == ErrorSeverity.MEDIUM
        assert error_context.category == ErrorCategory.NETWORK
        assert error_context.audit_run_id == "audit_456"

    def test_error_categorization(self, error_handler):
        """Test error categorization for different exception types"""
        test_cases = [
            (TimeoutError("Timeout"), ErrorSeverity.MEDIUM, ErrorCategory.TIMEOUT),
            (ValueError("Invalid value"), ErrorSeverity.LOW, ErrorCategory.VALIDATION),
            (KeyError("Missing key"), ErrorSeverity.LOW, ErrorCategory.VALIDATION),
            (
                FileNotFoundError("File not found"),
                ErrorSeverity.MEDIUM,
                ErrorCategory.CONFIGURATION,
            ),
            (
                PermissionError("Access denied"),
                ErrorSeverity.HIGH,
                ErrorCategory.AUTHENTICATION,
            ),
        ]

        for error, expected_severity, expected_category in test_cases:
            error_context = error_handler.handle_error(error)
            assert error_context.severity == expected_severity
            assert error_context.category == expected_category

    def test_error_metrics_recording(self, error_handler):
        """Test that error metrics are recorded"""
        error = ValueError("Test error")

        error_handler.handle_error(error)

        # Verify metrics methods are called
        assert error_handler.metrics.increment_error_count.called

    def test_error_frequency_tracking(self, error_handler):
        """Test error frequency tracking and alerting"""
        error = ValueError("Repeated error")

        # Generate multiple occurrences of the same error
        for _ in range(6):  # Above threshold of 5
            error_handler.handle_error(error)

        # Verify error count tracking
        error_key = "ValueError:validation"
        assert error_handler._error_counts[error_key] == 6

    def test_recovery_suggestions(self, error_handler):
        """Test recovery suggestion generation"""
        error = ConnectionError("Network error")

        error_context = error_handler.handle_error(error)

        # Should have network-specific recovery suggestions
        suggestions = error_context.recovery_suggestions
        assert any("connectivity" in suggestion.lower() for suggestion in suggestions)

    @patch("app.utils.error_handler.logger")
    def test_error_logging(self, mock_logger, error_handler):
        """Test error logging at appropriate levels"""
        # Test critical error logging
        critical_error = DatabaseConnectionError("Critical DB error")
        critical_error.severity = ErrorSeverity.CRITICAL
        error_handler.handle_error(critical_error)

        # Verify critical logging was called
        assert mock_logger.critical.called

        # Test medium error logging
        medium_error = PlatformTimeoutError("openai", 30)
        error_handler.handle_error(medium_error)

        # Verify warning logging was called
        assert mock_logger.warning.called


class TestErrorDecorator:
    """Test cases for error handling decorator"""

    @pytest.mark.asyncio
    async def test_async_function_decorator_success(self):
        """Test decorator with successful async function"""

        @handle_audit_errors(category=ErrorCategory.PLATFORM)
        async def test_function():
            return "success"

        result = await test_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_function_decorator_with_error(self):
        """Test decorator with async function that raises error"""

        @handle_audit_errors(
            category=ErrorCategory.PLATFORM, severity=ErrorSeverity.HIGH
        )
        async def test_function():
            raise ValueError("Test error")

        with pytest.raises(AuditProcessorError) as exc_info:
            await test_function()

        assert exc_info.value.severity == ErrorSeverity.HIGH
        assert exc_info.value.category == ErrorCategory.PLATFORM

    def test_sync_function_decorator_success(self):
        """Test decorator with successful sync function"""

        @handle_audit_errors(category=ErrorCategory.SYSTEM)
        def test_function():
            return "success"

        result = test_function()
        assert result == "success"

    def test_sync_function_decorator_with_error(self):
        """Test decorator with sync function that raises error"""

        @handle_audit_errors(
            category=ErrorCategory.SYSTEM, severity=ErrorSeverity.MEDIUM
        )
        def test_function():
            raise KeyError("Missing key")

        with pytest.raises(AuditProcessorError) as exc_info:
            test_function()

        assert exc_info.value.severity == ErrorSeverity.MEDIUM
        assert exc_info.value.category == ErrorCategory.SYSTEM


class TestErrorHandlingMiddleware:
    """Test cases for FastAPI error handling middleware"""

    @pytest.fixture
    def mock_app(self):
        """Mock FastAPI app"""
        return Mock()

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request"""
        request = Mock(spec=Request)
        request.url = "http://localhost/test"
        request.method = "GET"
        request.client.host = "127.0.0.1"
        return request

    @pytest.fixture
    def error_middleware(self, mock_app):
        """Create error handling middleware"""
        with patch("app.utils.error_handler.ErrorHandler") as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            return ErrorHandlingMiddleware(mock_app, mock_handler)

    @pytest.mark.asyncio
    async def test_middleware_success_response(self, error_middleware, mock_request):
        """Test middleware with successful response"""

        async def call_next(request):
            return "success_response"

        response = await error_middleware.dispatch(mock_request, call_next)
        assert response == "success_response"

    @pytest.mark.asyncio
    async def test_middleware_with_exception(self, error_middleware, mock_request):
        """Test middleware handling exceptions"""

        async def call_next(request):
            raise ValueError("Test error")

        # Mock error handler
        mock_error_context = Mock()
        mock_error_context.error_id = "err_123"
        mock_error_context.user_message = "Test error message"
        mock_error_context.category = ErrorCategory.VALIDATION
        mock_error_context.severity = ErrorSeverity.MEDIUM
        mock_error_context.recovery_suggestions = ["Try again"]
        mock_error_context.timestamp = datetime.now(timezone.utc)

        error_middleware.error_handler.handle_error.return_value = mock_error_context

        response = await error_middleware.dispatch(mock_request, call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500  # Medium severity maps to 500


class TestUtilityFunctions:
    """Test cases for utility functions"""

    def test_create_error_response(self):
        """Test create_error_response utility function"""
        error = ValueError("Test error")

        with patch("app.utils.error_handler.error_handler") as mock_handler:
            mock_context = Mock()
            mock_context.error_id = "err_123"
            mock_context.user_message = "User friendly message"
            mock_context.category = ErrorCategory.VALIDATION
            mock_context.severity = ErrorSeverity.LOW
            mock_context.recovery_suggestions = ["Check input"]
            mock_context.timestamp = datetime.now(timezone.utc)

            mock_handler.handle_error.return_value = mock_context

            response = create_error_response(error)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 400  # Low severity maps to 400

    def test_log_audit_error(self):
        """Test log_audit_error utility function"""
        error = PlatformTimeoutError("openai", 30)

        with patch("app.utils.error_handler.error_handler") as mock_handler:
            mock_context = Mock()
            mock_context.error_id = "err_456"
            mock_handler.handle_error.return_value = mock_context

            error_id = log_audit_error(
                error=error,
                audit_run_id="audit_123",
                platform_name="openai",
                additional_context={"request_id": "req_789"},
            )

            assert error_id == "err_456"

            # Verify handler was called with correct context
            call_args = mock_handler.handle_error.call_args
            assert call_args[0][0] == error  # First argument is the error

            context = call_args[0][1]  # Second argument is the context
            assert context["audit_run_id"] == "audit_123"
            assert context["platform_name"] == "openai"
            assert context["request_id"] == "req_789"


class TestErrorHandlerIntegration:
    """Integration tests for error handling system"""

    @pytest.mark.asyncio
    async def test_full_error_handling_flow(self):
        """Test complete error handling flow from exception to response"""
        with patch("app.utils.error_handler.get_audit_metrics") as mock_metrics:
            mock_metrics.return_value = Mock()

            error_handler = ErrorHandler()

            # Create a complex error scenario
            original_error = PlatformRateLimitError("openai", retry_after=120)
            context = {
                "audit_run_id": "audit_123",
                "platform_name": "openai",
                "request_id": "req_456",
                "user_id": "user_789",
            }

            # Handle the error
            error_context = error_handler.handle_error(original_error, context)

            # Verify error context is properly populated
            assert error_context.error == original_error
            assert error_context.platform_name == "openai"
            assert error_context.audit_run_id == "audit_123"
            assert error_context.severity == ErrorSeverity.MEDIUM
            assert error_context.category == ErrorCategory.PLATFORM

            # Verify recovery suggestions are included
            assert len(error_context.recovery_suggestions) > 0
            assert any(
                "wait" in suggestion.lower()
                for suggestion in error_context.recovery_suggestions
            )

            # Verify technical details are preserved
            assert error_context.technical_details["retry_after"] == 120

            # Create API response
            response = create_error_response(original_error, context)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 500  # Medium severity

    def test_error_pattern_detection(self):
        """Test error pattern detection and alerting"""
        with patch("app.utils.error_handler.get_audit_metrics") as mock_metrics:
            mock_metrics.return_value = Mock()

            with patch("app.utils.error_handler.logger") as mock_logger:
                error_handler = ErrorHandler()

                # Generate pattern of similar errors
                for i in range(7):  # Above threshold
                    error = ConnectionError(f"Network error {i}")
                    error_handler.handle_error(error, {"instance": i})

                # Verify pattern detection triggered warning
                warning_calls = [
                    call
                    for call in mock_logger.warning.call_args_list
                    if "High frequency error detected" in str(call)
                ]
                assert len(warning_calls) > 0


if __name__ == "__main__":
    pytest.main([__file__])

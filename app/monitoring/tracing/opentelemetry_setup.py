from __future__ import annotations

from typing import Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def setup_tracing(app=None) -> Optional[object]:
    """
    Initialize OpenTelemetry tracing if enabled via settings.

    Returns tracer provider (or None) to keep a handle for shutdown if needed.
    """
    if not getattr(settings, "TRACING_ENABLED", False):
        logger.info("Tracing disabled")
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": getattr(settings, "SERVICE_NAME", settings.APP_NAME),
                "service.version": "1.0.0",
                "deployment.environment": settings.APP_ENV,
            }
        )
        provider = TracerProvider(resource=resource)

        endpoint = getattr(
            settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"
        )
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        # Instrument FastAPI and Requests
        if app is not None:
            FastAPIInstrumentor.instrument_app(app)
        RequestsInstrumentor().instrument()

        logger.info("OpenTelemetry tracing initialized", endpoint=endpoint)
        return provider
    except Exception as e:
        logger.error("Failed to initialize tracing", error=str(e))
        return None

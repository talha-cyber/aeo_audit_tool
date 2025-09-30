import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

from app.api.v1 import audits
from app.api.v1 import monitoring as monitoring_routes
from app.api.v1 import personas as personas_routes
from app.api.v1 import security as security_routes
from app.api.v1.providers import health as provider_health
from app.core.config import settings
from app.monitoring.tracing.correlation import CorrelationIdMiddleware
from app.monitoring.tracing.opentelemetry_setup import setup_tracing
from app.security.audit.access_logger import AccessLogMiddleware
from app.security.validation.security_headers import SecurityHeadersMiddleware
from app.utils.logger import configure_logging, get_logger

# Configure logging before anything else
configure_logging()
logger = get_logger(__name__)

# Initialize Sentry if DSN is provided
app = FastAPI(
    title="AEO Competitive Intelligence Tool",
    description=(
        "Multi-platform AEO audit tool that simulates user questions "
        "across AI platforms"
    ),
    version="1.0.0",
)

# Initialize Sentry if DSN is provided
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.APP_ENV,
        traces_sample_rate=0.1,
    )
    app.add_middleware(SentryAsgiMiddleware)
    logger.info(
        "Sentry initialized with SentryAsgiMiddleware", environment=settings.APP_ENV
    )
else:
    logger.info("Sentry not configured (SENTRY_DSN not set)")

# Add CORS middleware with environment-aware defaults
allowed_origins = settings.CORS_ALLOW_ORIGINS or ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware (headers + access logs)
if settings.ENABLE_SECURITY_HEADERS:
    app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(CorrelationIdMiddleware)

app.include_router(audits.router, prefix="/api/v1")
app.include_router(personas_routes.router, prefix="/api/v1")
app.include_router(provider_health.router, prefix="/api/v1/providers")
app.include_router(security_routes.router, prefix="/api/v1")
app.include_router(monitoring_routes.router, prefix="/api/v1")

# Initialize Prometheus metrics
instrumentator = Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics"],  # Don't monitor the metrics endpoint itself
)
instrumentator.instrument(app).expose(app)


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info(
        "AEO Audit Tool starting up",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        sentry_enabled=bool(settings.SENTRY_DSN),
    )
    # Initialize OpenTelemetry tracing if enabled
    setup_tracing(app)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "ok"}
if settings.ENABLE_DEBUG_ENDPOINTS:

    @app.get("/debug-sentry")
    async def debug_sentry():
        """Debug endpoint to test Sentry error reporting (development only)."""
        logger.warning("Debug Sentry endpoint called - this will raise an exception")
        raise Exception("This is a test exception for Sentry")

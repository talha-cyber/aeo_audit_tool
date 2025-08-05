import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

from app.api.v1 import audits
from app.api.v1.providers import health as provider_health
from app.core.config import settings
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audits.router, prefix="/api/v1")
app.include_router(provider_health.router, prefix="/api/v1/providers")

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


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "ok"}


@app.get("/debug-sentry")
async def debug_sentry():
    """Debug endpoint to test Sentry error reporting (remove in production)"""
    logger.warning("Debug Sentry endpoint called - this will raise an exception")
    raise Exception("This is a test exception for Sentry")

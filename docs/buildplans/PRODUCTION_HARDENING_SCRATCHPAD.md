# Production Hardening Implementation Plan
**AEO Audit Tool - Modular & Scalable Production Readiness**

> **Status:** Planning Phase
> **Target:** Production-hardened system addressing identified gaps
> **Architecture:** Modular, scalable, maintainable

---

## Executive Summary

This plan addresses the 5 key production readiness gaps identified in the feedback analysis:

1. **Sentiment Analysis** - Replace placeholder heuristics with robust ML models
2. **Audit Scheduling** - Implement comprehensive scheduling system with persistence
3. **Error Handling Resilience** - Add DLQ, backoff strategies, circuit breakers
4. **CI/CD Automation** - Complete deployment pipeline with blue/green deployments
5. **Embedded Security** - Bake security practices into architecture design

**Implementation Approach:** Phased rollout with backward compatibility, feature flags, and gradual migration.

---

## Phase 1: Advanced Sentiment Analysis System

### 1.1 Sentiment Analysis Engine Module
**Location:** `app/services/sentiment/`

```
app/services/sentiment/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── engine.py              # Main SentimentEngine class
│   ├── models.py              # Model definitions & abstractions
│   └── config.py              # Sentiment-specific configuration
├── providers/
│   ├── __init__.py
│   ├── base.py                # Abstract base provider
│   ├── transformer_provider.py # HuggingFace transformers
│   ├── vader_provider.py      # Current VADER (fallback)
│   ├── openai_provider.py     # GPT-based sentiment
│   └── anthropic_provider.py  # Claude sentiment analysis
├── models/
│   ├── __init__.py
│   ├── local_models.py        # Local ML model management
│   └── cached_models.py       # Model caching & versioning
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py       # Text preprocessing
│   ├── postprocessing.py      # Result aggregation
│   └── validation.py          # Sentiment result validation
└── tests/
    ├── __init__.py
    ├── test_engine.py
    ├── test_providers.py
    └── test_integration.py
```

**Key Features:**
- Multi-provider sentiment analysis (local transformers, API-based)
- Model caching and versioning
- Confidence scoring and ensemble methods
- Async processing with batching
- Performance monitoring and A/B testing

### 1.2 Implementation Strategy
```python
# Core abstraction
class SentimentProvider(ABC):
    @abstractmethod
    async def analyze_sentiment(self, text: str) -> SentimentResult

    @abstractmethod
    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]

# Main engine with provider orchestration
class SentimentEngine:
    def __init__(self, providers: List[SentimentProvider], strategy: str = "ensemble"):
        self.providers = providers
        self.strategy = strategy  # "primary", "ensemble", "consensus"

    async def analyze(self, text: str, confidence_threshold: float = 0.7) -> SentimentResult:
        # Implement strategy-based analysis
```

---

## Phase 2: Comprehensive Audit Scheduling System

### 2.1 Scheduling Infrastructure
**Location:** `app/services/scheduling/`

```
app/services/scheduling/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── scheduler.py           # Main scheduler engine
│   ├── job_manager.py         # Job lifecycle management
│   └── triggers.py            # Scheduling triggers (cron, interval, etc.)
├── persistence/
│   ├── __init__.py
│   ├── models.py              # Database models for scheduled jobs
│   ├── repository.py          # Data access layer
│   └── migrations/            # Alembic migrations for scheduling
├── executors/
│   ├── __init__.py
│   ├── celery_executor.py     # Celery-based execution
│   ├── direct_executor.py     # Direct async execution
│   └── distributed_executor.py # Future: distributed execution
├── policies/
│   ├── __init__.py
│   ├── retry_policy.py        # Retry strategies
│   ├── concurrency_policy.py  # Concurrency limits
│   └── priority_policy.py     # Job prioritization
└── monitoring/
    ├── __init__.py
    ├── metrics.py             # Scheduling metrics
    └── health_check.py        # Scheduler health monitoring
```

**Database Schema:**
```sql
-- Scheduled jobs table
CREATE TABLE scheduled_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    job_type VARCHAR(100) NOT NULL,
    trigger_config JSONB NOT NULL,
    job_config JSONB NOT NULL,
    status scheduled_job_status NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_run_at TIMESTAMP WITH TIME ZONE,
    next_run_at TIMESTAMP WITH TIME ZONE,
    run_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    max_failures INTEGER DEFAULT 3
);

-- Job execution history
CREATE TABLE job_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES scheduled_jobs(id),
    execution_id VARCHAR(255) UNIQUE,
    status execution_status NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    result JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0
);
```

### 2.2 Features
- Persistent job scheduling with database storage
- Multiple trigger types (cron, interval, one-time)
- Job dependency management
- Retry policies with exponential backoff
- Job monitoring and alerting
- Dynamic job modification without restarts

---

## Phase 3: Error Handling Resilience

### 3.1 Enhanced Error Handling System
**Location:** `app/utils/resilience/`

```
app/utils/resilience/
├── __init__.py
├── circuit_breaker/
│   ├── __init__.py
│   ├── breaker.py             # Circuit breaker implementation
│   ├── policies.py            # Failure policies
│   └── monitoring.py          # Breaker state monitoring
├── retry/
│   ├── __init__.py
│   ├── strategies.py          # Retry strategies (exponential, linear, etc.)
│   ├── backoff.py             # Backoff algorithms
│   └── decorators.py          # Retry decorators
├── dead_letter/
│   ├── __init__.py
│   ├── queue.py               # Dead letter queue implementation
│   ├── processor.py           # DLQ message processing
│   └── recovery.py            # Message recovery strategies
├── bulkhead/
│   ├── __init__.py
│   ├── isolator.py            # Resource isolation
│   └── pools.py               # Connection/thread pools
└── monitoring/
    ├── __init__.py
    ├── health.py              # System health monitoring
    └── alerts.py              # Alerting system
```

### 3.2 Key Components

**Circuit Breaker Pattern:**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None

    async def call(self, func: Callable, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError()

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

**Dead Letter Queue:**
```python
class DeadLetterQueue:
    def __init__(self, redis_client: Redis, max_retries: int = 3):
        self.redis = redis_client
        self.max_retries = max_retries

    async def send_to_dlq(self, message: dict, original_queue: str, error: Exception):
        dlq_message = {
            "original_message": message,
            "original_queue": original_queue,
            "error": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "retry_count": message.get("retry_count", 0) + 1
        }

        await self.redis.lpush(f"dlq:{original_queue}", json.dumps(dlq_message))

    async def process_dlq_messages(self, queue: str, processor: Callable):
        # Implementation for processing DLQ messages
```

### 3.3 Implementation Status & Usage

Status: Implemented modular resilience suite under `app/utils/resilience/` with async-first APIs, structlog, and Prometheus metrics. Configurable via `app/core/config.py`.

Implemented Modules:
- Circuit Breaker: `app/utils/resilience/circuit_breaker/breaker.py`
- Retry + Backoff: `app/utils/resilience/retry/{decorators.py,strategies.py,backoff.py}`
- Dead Letter Queue: `app/utils/resilience/dead_letter/{queue.py,processor.py,recovery.py}`
- Bulkhead Isolation: `app/utils/resilience/bulkhead/{isolator.py,pools.py}`
- Resilience Monitoring: `app/utils/resilience/circuit_breaker/monitoring.py`, `app/utils/resilience/monitoring/{health.py,alerts.py}`

Config Defaults (env-overridable): see `app/core/config.py`
- `CIRCUIT_BREAKER_FAILURE_THRESHOLD`, `CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS`, `CIRCUIT_BREAKER_HALF_OPEN_SUCCESS_THRESHOLD`
- `RETRY_MAX_ATTEMPTS`, `RETRY_BACKOFF_BASE_SECONDS`, `RETRY_BACKOFF_MULTIPLIER`, `RETRY_BACKOFF_MAX_SECONDS`, `RETRY_USE_JITTER`
- `DLQ_ENABLED`, `DLQ_MAX_RETRIES`, `DLQ_RETENTION_SECONDS`
- `BULKHEAD_DEFAULT_MAX_CONCURRENCY`

Quick Usage Examples:
```python
# Circuit breaker + retry + bulkhead around an async IO call
from app.utils.resilience import CircuitBreaker, retry, ExponentialBackoffStrategy, Bulkhead
from app.core.config import settings

cb = CircuitBreaker(
    name="platform_openai",
    failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS,
    half_open_success_threshold=settings.CIRCUIT_BREAKER_HALF_OPEN_SUCCESS_THRESHOLD,
)

bulkhead = Bulkhead(name="platform_openai", max_concurrency=settings.BULKHEAD_DEFAULT_MAX_CONCURRENCY)

@retry(
    max_attempts=settings.RETRY_MAX_ATTEMPTS,
    backoff_strategy=ExponentialBackoffStrategy(
        base_delay_seconds=settings.RETRY_BACKOFF_BASE_SECONDS,
        multiplier=settings.RETRY_BACKOFF_MULTIPLIER,
        max_delay_seconds=settings.RETRY_BACKOFF_MAX_SECONDS,
    ),
    use_jitter=settings.RETRY_USE_JITTER,
)
async def call_platform(client, prompt: str):
    async def _do():
        return await client.query(prompt)
    # Bulkhead + Circuit breaker
    return await bulkhead.run(cb.decorate(_do))
```

DLQ Integration Example:
```python
from app.utils.resilience.dead_letter import DeadLetterQueue

dlq = DeadLetterQueue(max_retries=settings.DLQ_MAX_RETRIES)

async def handle_message(msg: dict):
    try:
        # ... process ...
        return True
    except Exception as e:
        await dlq.send_to_dlq(msg, original_queue="audit:tasks", error=e)
        return False

# Recovery pass (e.g. cron or admin endpoint)
await dlq.process_dlq_messages("audit:tasks", processor=handle_message, max_messages=200)
```

Metrics:
- Circuit state, successes, failures, blocked calls are exported via Prometheus counters/gauges in `circuit_breaker/monitoring.py`.
- Bulkhead inflight gauge `bulkhead_inflight{bulkhead=...}` tracks live concurrency.

Next Integration Targets:
- Wrap external provider calls with `CircuitBreaker + retry + Bulkhead`.
- Route failed task payloads to DLQ where enabled and add a periodic DLQ recovery task.
- Add alert policies using `ResilienceHealthChecker` snapshot and `AlertManager`.

---

## Phase 4: CI/CD Automation & Deployment

### 4.1 Enhanced CI/CD Pipeline
**Location:** `.github/workflows/`

**Pipeline Structure:**
```
.github/workflows/
├── ci.yml                     # Current CI (enhanced)
├── security-scan.yml          # Dedicated security pipeline
├── deploy-staging.yml         # Staging deployment
├── deploy-production.yml      # Production deployment
├── rollback.yml               # Automated rollback
└── performance-test.yml       # Performance testing
```

**Key Features:**
- Blue/Green deployments with health checks
- Database migration automation
- Rollback capabilities
- Performance regression testing
- Security scanning integration
- Multi-environment promotion

### 4.2 Deployment Infrastructure
**Location:** `deployment/`

```
deployment/
├── docker/
│   ├── Dockerfile.prod        # Production-optimized container
│   ├── Dockerfile.staging     # Staging container
│   └── docker-compose.prod.yml # Production compose
├── k8s/                       # Kubernetes manifests (future)
│   ├── namespace.yml
│   ├── deployment.yml
│   ├── service.yml
│   ├── ingress.yml
│   └── configmap.yml
├── terraform/                 # Infrastructure as Code
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── modules/
└── scripts/
    ├── deploy.sh              # Deployment scripts
    ├── rollback.sh
    ├── health-check.sh
    └── migrate.sh
```

### 4.3 Implementation Status & Usage

Status: Implemented CI/CD workflows and deployment scaffolding.

Added Workflows:
- Staging deploy: `.github/workflows/deploy-staging.yml`
- Production deploy: `.github/workflows/deploy-production.yml`
- Rollback: `.github/workflows/rollback.yml`
- Performance tests: `.github/workflows/performance-test.yml`

Deployment Scripts:
- Deploy: `deployment/scripts/deploy.sh` (supports blue/green or rolling)
- Health: `deployment/scripts/health-check.sh`
- Migrate: `deployment/scripts/migrate.sh`
- Rollback: `deployment/scripts/rollback.sh`

Kubernetes/IaC:
- K8s placeholders under `deployment/k8s` for future migration
- Terraform placeholders under `deployment/terraform`

Required Secrets:
- Staging: `SSH_HOST_STAGING`, `SSH_USER_STAGING`, `SSH_KEY_STAGING`
- Production: `SSH_HOST_PROD`, `SSH_USER_PROD`, `SSH_KEY_PROD`

Container Registry:
- Using GHCR (`ghcr.io/<owner>/<repo>`). Build tags: `staging`, `staging-<sha>`, `prod`, `prod-<sha>`

Usage:
- Staging: push to `staging` or run workflow dispatch.
- Production: run Deploy Production workflow; defaults to blue/green.
- Rollback: run Rollback workflow with `target_image` (e.g., `ghcr.io/owner/repo:prod-<sha>`).

---

## Phase 5: Embedded Security Architecture

### 5.1 Security-First Design
**Location:** `app/security/`

```
app/security/
├── __init__.py
├── auth/
│   ├── __init__.py
│   ├── jwt_handler.py         # JWT token management
│   └── session_manager.py     # Session management (Redis)
├── encryption/
│   ├── __init__.py
│   ├── field_encryption.py    # Database field encryption
│   ├── api_key_manager.py     # API key encryption/rotation
│   └── secrets_manager.py     # External secrets integration
├── audit/
│   ├── __init__.py
│   ├── access_logger.py       # Access audit logging
│   ├── change_tracker.py      # Data change tracking
│   └── compliance.py          # Compliance reporting
├── validation/
│   ├── __init__.py
│   ├── input_sanitizer.py     # Input sanitization
│   ├── schema_validator.py    # Enhanced schema validation
│   └── security_headers.py    # Security headers middleware
└── monitoring/
    ├── __init__.py
    ├── threat_detection.py     # Threat detection
    └── security_metrics.py     # Security metrics
```

### 5.2 Security Features
- API key rotation and secure storage
- Field-level encryption for sensitive data
- Comprehensive audit logging
- Threat detection and alerting
- Input validation and sanitization
- Security headers and CORS policies
- Regular security assessments

### 5.3 Implementation Status & Notes

Status: Core security modules implemented and wired.

Implemented modules:
- JWT auth: `app/security/auth/jwt_handler.py`
- Sessions (Redis): `app/security/auth/session_manager.py`
- Field encryption (Fernet): `app/security/encryption/field_encryption.py` (requires `cryptography`)
- API key manager: `app/security/encryption/api_key_manager.py`
- Access logger: `app/security/audit/access_logger.py`
- Change tracker: `app/security/audit/change_tracker.py`
- Compliance snapshot: `app/security/audit/compliance.py`
- Input sanitizer: `app/security/validation/input_sanitizer.py`
- Schema validation helper: `app/security/validation/schema_validator.py`
- Security headers middleware: `app/security/validation/security_headers.py` (enabled in `app/main.py`)
- Threat detection heuristics: `app/security/monitoring/threat_detection.py`
- Security metrics: `app/security/monitoring/security_metrics.py`

Config additions (env-overridable) in `app/core/config.py`:
- `ENABLE_SECURITY_HEADERS`, `CSP_POLICY`, `JWT_ALGORITHM`, `JWT_EXPIRES_MINUTES`, `JWT_ISSUER`, `SESSION_TTL_SECONDS`, `FIELD_ENCRYPTION_KEY`, `API_KEY_PEPPER`

Usage:
- JWT: `from app.security.auth.jwt_handler import get_jwt_handler`
- Encrypt: `FieldEncryptor().encrypt(value)` / `.decrypt(token)`
- Sessions: `SessionManager(ttl).create/get_user/revoke`
- Security headers: auto-enabled; adjust CSP via `CSP_POLICY`

---

## Phase 6: Monitoring & Observability Enhancement

### 6.1 Advanced Monitoring
**Location:** `app/monitoring/`

```
app/monitoring/
├── __init__.py
├── metrics/
│   ├── __init__.py
│   ├── business_metrics.py    # Business KPIs
│   ├── technical_metrics.py   # Technical metrics
│   └── custom_metrics.py      # Custom metric definitions
├── tracing/
│   ├── __init__.py
│   ├── opentelemetry_setup.py # Distributed tracing
│   └── correlation.py         # Request correlation
├── alerting/
│   ├── __init__.py
│   ├── alert_manager.py       # Alert management
│   ├── notification.py        # Notification handlers
│   └── escalation.py          # Alert escalation
└── dashboards/
    ├── __init__.py
    ├── grafana_templates/      # Grafana dashboard templates
    └── prometheus_rules/       # Prometheus alerting rules
```

---

### 6.1 Implementation Status & Usage

Status: Implemented monitoring modules and integrated middleware.

Added Modules:
- Metrics: `business_metrics.py`, `technical_metrics.py`, `custom_metrics.py`
- Tracing: `opentelemetry_setup.py` (OTLP exporter), `correlation.py` (X-Request-ID)
- Alerting: `alert_manager.py`, `notification.py` (webhook), `escalation.py`
- Dashboards: Grafana `app/monitoring/dashboards/grafana_templates/aeo_overview.json`
- Prometheus rules: `app/monitoring/dashboards/prometheus_rules/aeo_alerts.yml`

App Integration:
- CorrelationIdMiddleware added; response includes `X-Request-ID`.
- Optional OpenTelemetry tracing emits spans when `TRACING_ENABLED=true`.

Config Additions (env-overridable in `app/core/config.py`):
- `TRACING_ENABLED`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `TRACING_SAMPLE_RATIO`, `SERVICE_NAME`, `ALERT_WEBHOOK_URL`

Examples:
```python
from app.monitoring.metrics.business_metrics import set_success_rate, inc_client_audits
from app.monitoring.metrics.technical_metrics import update_process_metrics, update_redis_queue_depth

set_success_rate(0.92)
inc_client_audits(client_id="acme")
update_process_metrics()
await update_redis_queue_depth("dlq:audit:tasks")

from app.monitoring.alerting.alert_manager import AlertManager, ThresholdRule
from app.monitoring.metrics.business_metrics import AUDIT_SUCCESS_RATE

mgr = AlertManager()
mgr.add_rule(ThresholdRule(
    name="LowSuccessRate",
    metric_getter=lambda: AUDIT_SUCCESS_RATE._value.get(),
    threshold=0.8,
    comparison="<",
    severity="warning",
    description="Audit success rate below 80%",
))
fired = mgr.evaluate()
```

Grafana/Prometheus:
- Import dashboard JSON and point Prometheus to `prometheus_rules/aeo_alerts.yml`.
- Compose stack already includes Prometheus/Grafana containers (see `docker-compose.prod.yml`).

Monitoring API:
- GET `/api/v1/monitoring/snapshot` provides DLQ depths and process CPU/memory.

Prometheus Config (Compose):
- `monitoring/prometheus/prometheus.prod.yml` is mounted as Prometheus config.
- Alert rules mounted at `/etc/prometheus/aeo_alerts.yml` via compose.

Testing:
- Correlation ID middleware test added under `tests/monitoring/test_correlation_id.py`.
 - Alert rule evaluation test in `tests/monitoring/test_alert_manager.py`.
 - Monitoring snapshot endpoint test in `tests/monitoring/test_snapshot_endpoint.py`.

## Implementation Timeline

### Phase 1: Foundation (Weeks 1-2)
- [ ] Sentiment analysis engine core
- [ ] Basic scheduling infrastructure
- [ ] Enhanced error handling base

### Phase 2: Core Features (Weeks 3-4)
- [ ] Multi-provider sentiment analysis
- [ ] Persistent job scheduling
- [ ] Circuit breakers and retry logic

### Phase 3: Advanced Features (Weeks 5-6)
- [ ] Dead letter queue implementation
- [ ] Security enhancements
- [ ] CI/CD pipeline improvements

### Phase 4: Production Readiness (Weeks 7-8)
- [ ] Monitoring and alerting
- [ ] Performance optimization
- [ ] Documentation and runbooks

### Phase 5: Testing & Validation (Week 9)
- [ ] Load testing
- [ ] Security penetration testing
- [ ] Disaster recovery testing

### Phase 6: Go-Live (Week 10)
- [ ] Production deployment
- [ ] Monitoring validation
- [ ] Post-deployment optimization

---

## Success Metrics

### Technical Metrics
- **Uptime:** > 99.9%
- **Response Time:** < 200ms p95
- **Error Rate:** < 0.1%
- **Recovery Time:** < 5 minutes

### Business Metrics
- **Audit Success Rate:** > 95%
- **Sentiment Analysis Accuracy:** > 90%
- **Scheduled Job Success:** > 99%
- **Security Incidents:** 0

### Quality Metrics
- **Test Coverage:** > 90%
- **Security Scan Pass Rate:** 100%
- **Documentation Coverage:** > 95%

---

## Risk Mitigation

### Technical Risks
- **Database Migration Issues:** Blue/green deployment with rollback
- **Performance Degradation:** Gradual rollout with monitoring
- **Third-party Dependencies:** Fallback providers and circuit breakers

### Business Risks
- **Downtime:** Zero-downtime deployment strategy
- **Data Loss:** Comprehensive backup and recovery procedures
- **Security Breaches:** Defense in depth security architecture

---

## Next Steps

1. **Review and Approve Plan:** Stakeholder review and sign-off
2. **Set Up Development Environment:** Feature branches and testing infrastructure
3. **Begin Phase 1 Implementation:** Start with sentiment analysis engine
4. **Establish Monitoring:** Set up metrics and alerting for implementation tracking

---

**Document Version:** 1.0
**Last Updated:** 2025-09-09
**Next Review:** Weekly during implementation

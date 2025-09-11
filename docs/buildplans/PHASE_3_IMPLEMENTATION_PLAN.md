# Phase 3 Implementation Plan: Observability & Security

This document provides a battle-hardened, step-by-step guide for implementing Phase 3 of the AEO Audit Tool build plan. The goal is to establish a robust foundation for telemetry, monitoring, and security. Adhering to this sequence will minimize integration issues and ensure each layer is testable and reliable before the next is added.

---

## **Part 1: Foundational Tooling & Configuration**

**Objective:** Update the project with all necessary dependencies and configurations *before* writing implementation code. This prevents churn and ensures a stable base.

1.  **Update Dependencies:**
    *   **Action:** Add the following packages to `requirements.txt`. These specific packages are chosen for their mature integrations with FastAPI and Celery.
        ```
        # requirements.txt
        ...
        # Observability & Security
        structlog
        sentry-sdk[fastapi,celery]
        prometheus-fastapi-instrumentator
        bandit
        ```
    *   **Rationale:** We are adding `structlog` for structured logging, `sentry-sdk` for error tracking, `prometheus-fastapi-instrumentator` for FastAPI metrics, and `bandit` for security analysis. The `[fastapi,celery]` extras for Sentry are crucial for auto-instrumentation.

2.  **Update Environment & Configuration:**
    *   **Action:** Add the Sentry DSN to the Pydantic settings in `app/core/config.py`.
        ```python
        // ... existing code in app/core/config.py ...
        class Settings(BaseSettings):
            # ... existing settings ...
            SENTRY_DSN: Optional[str] = None
        // ... existing code ...
        ```
    *   **Action:** Add a `SENTRY_DSN` variable to your local `.env` file. Leave it empty for now.
        ```
        # .env
        SENTRY_DSN=""
        ```
    *   **Rationale:** Centralizing configuration in Pydantic and sourcing from `.env` is critical. It prevents secret leakage and allows for different configurations per environment (local, staging, prod) without code changes. A missing or empty `SENTRY_DSN` will serve as a feature flag to disable Sentry locally.

---

## **Part 2: Telemetry Implementation (Logging & Error Tracking)**

**Objective:** Instrument the application to produce structured, machine-readable logs and automatically capture all exceptions for centralized analysis.

1.  **Implement Structured Logging with `structlog`:**
    *   **Action:** Create a new file `app/core/logging.py` to house the entire logging configuration. This is the single source of truth for how logs are formatted and processed.
    *   **Rationale:** Isolating logging configuration prevents circular dependencies and makes it easy to initialize consistently across both the FastAPI app and the Celery worker. The configuration should be environment-aware: human-readable logs for development, JSON logs for production.
    *   **Action:** Integrate the logger into FastAPI by calling the configuration function in `app/main.py` during application startup.
    *   **Action:** Integrate the logger into Celery by using the `@setup_logging.connect` signal in `app/core/celery_app.py`. This is the correct, official way to override Celery's default logging setup.

2.  **Integrate Sentry for Error Tracking:**
    *   **Action:** In `app/main.py`, read the `SENTRY_DSN` from settings. If it's present, initialize `sentry_sdk` with the `FastAPIIntegration` and `CeleryIntegration`.
    *   **Action:** In `celery_worker.py`, do the same. It's crucial to initialize the SDK in both the web and worker processes.
    *   **Rationale:** This setup ensures that any unhandled exception in either the API or a background task is automatically captured and sent to Sentry with rich context (request data, task ID, etc.).
    *   **Testability:** Create a temporary test endpoint `/debug-sentry` in FastAPI that deliberately raises an exception to confirm that errors appear in the Sentry dashboard.

---

## **Part 3: CI/CD Hardening (Automation & Security Gates)**

**Objective:** Automate security and quality checks within the CI/CD pipeline to catch issues before they reach production.

1.  **Configure `dependabot`:**
    *   **Action:** Create a `.github/dependabot.yml` file. Configure it to check for Python dependency updates daily.
    *   **Rationale:** This is a low-effort, high-reward task that automates dependency management, patching security vulnerabilities as they are discovered.

2.  **Add SAST and Vulnerability Scanning to CI:**
    *   **Action:** In your GitHub Actions workflow file (e.g., `.github/workflows/main.yml`), add two new steps to the main job:
        1.  **Bandit:** Add a step to run `bandit -r app -c pyproject.toml --exit-zero-on-warn`. We use `--exit-zero-on-warn` initially to avoid breaking the build, allowing us to triage existing issues.
        2.  **Trivy:** After building the Docker image, add a step to run `trivy image --severity HIGH,CRITICAL --exit-code 1 YOUR_IMAGE_NAME:latest`.
    *   **Rationale:** `Bandit` scans for common security issues in the Python code (SAST). `Trivy` scans the final Docker image for known OS and library vulnerabilities (SCA). Failing the build on `HIGH` or `CRITICAL` severity is a non-negotiable security gate.

3.  **Enforce Test Coverage:**
    *   **Action:** Modify the `pytest` command in the CI workflow.
    *   **From:** `pytest`
    *   **To:** `pytest --cov=app --cov-report=xml --cov-fail-under=80`
    *   **Rationale:** This enforces development discipline. Pull requests that decrease test coverage below the 80% threshold will fail, preventing a gradual erosion of code quality.

---

## **Part 4: Application Monitoring (Prometheus & Grafana)**

**Objective:** Expose key performance indicators from the application and create a basic dashboard for real-time visualization.

1.  **Expose FastAPI Metrics:**
    *   **Action:** In `app/main.py`, use `prometheus-fastapi-instrumentator` to "instrument" the FastAPI app. This will automatically create a `/metrics` endpoint.
        ```python
        # app/main.py
        from prometheus_fastapi_instrumentator import Instrumentator

        # ... after app creation
        Instrumentator().instrument(app).expose(app)
        ```
    *   **Rationale:** This provides essential "RED" metrics (Rate, Errors, Duration) for all API endpoints out-of-the-box.

2.  **Add `celery-exporter` for Worker Metrics:**
    *   **Action:** Add a new service for `celery-exporter` to `docker-compose.yml`. This container connects to Redis and exposes metrics about task states (received, started, succeeded, failed) and latencies.
    *   **Rationale:** Celery does not expose Prometheus metrics natively. A sidecar exporter is the standard, most reliable pattern for monitoring Celery clusters.

3.  **Configure Prometheus to Scrape Metrics:**
    *   **Action:** Add a `prometheus` service to `docker-compose.yml`.
    *   **Action:** Create a `prometheus/prometheus.yml` configuration file. Define two scrape jobs: one for the `fastapi_app:8000` target and another for the `celery_exporter:9808` target.
    *   **Rationale:** This tells Prometheus where to find the metrics being exposed by the application and the worker exporter.

4.  **Provision a Basic Grafana Dashboard:**
    *   **Action:** Add a `grafana` service to `docker-compose.yml`.
    *   **Action:** Create a `grafana/provisioning/dashboards/default.yml` file to tell Grafana to load dashboards from a specific folder.
    *   **Action:** Create `grafana/provisioning/datasources/default.yml` to pre-configure the Prometheus datasource.
    *   **Action:** Outline the structure of a `grafana/dashboards/aeo-overview.json`. I will provide a basic JSON model for this dashboard when we reach this step.
    *   **Rationale:** Provisioning Grafana with files (`.yml`, `.json`) makes the monitoring setup repeatable and version-controlled. The initial dashboard should focus on the most critical KPIs: API Latency (95th percentile), API Error Rate (5xx), and Celery Task Failure Rate.

# AEO Audit Tool: Chronological Build Plan (Hardened)

This document outlines a clear, step-by-step roadmap for building the AEO Audit Tool. It integrates the original `docs/ARCHITECTURE.md` goals with a disciplined, vertical-slice approach to mitigate risk and deliver value iteratively.

---

## 1. Lock the Ground Floor (This Week)

*Goal: Create a thin but solid horizontal infrastructure slice. If this foundation shifts, every subsequent feature accrues technical debt.*

- [x] **① Finalize DB Schema & Create Alembic Migration v1**
  - [x] Define all tables (`clients`, `audit_configs`, `questions`, `audit_runs`, `responses`, `reports`) in SQLAlchemy models.
  - [x] Generate the initial Alembic migration script (`alembic revision --autogenerate`).
- [x] **② Implement `settings.py` via Pydantic**
  - [x] Create a typed, environment-driven configuration file that loads from `.env`.
  - [x] Ensure secrets are handled properly, preparing for vault integration.
- [x] **③ Setup Linters, Formatting, and Initial Tests**
  - [x] Configure `black` (formatting), `ruff`, and `mypy` (linting) with project standards.
  - [x] Implement pre-commit hooks to automate checks.
  - [x] Seed `tests/` with a basic Pydantic settings test and a DB migration test.
- [x] **④ Initial CI Pipeline**
  - [x] Create a GitHub Actions workflow that runs `pytest` on every push.
  - [x] Configure `pytest-cov` to generate and upload a `coverage.xml` report.
  - [x] Add a secret scanner (e.g., `trufflehog`) to the pipeline to fail builds with exposed secrets.

---

## 2. Ship a "Walking-Skeleton" Vertical Slice (Sprint 2, ≤ 2 weeks)

*Goal: Validate that all core components (API, Worker, DB, AI Client, Secrets) work together in a real, end-to-end flow.*

- [x] **API Endpoint:** Create a `POST /audit` endpoint in FastAPI that accepts basic audit parameters.
- [x] **Background Task:** The endpoint enqueues a Celery task to perform the audit.
- [x] **AI Integration:** The Celery task calls a minimal `OpenAIPlatform` client.
- [x] **Persistence:** The task persists the raw AI response into the `responses` table.
- [x] **Status Check:** Implement a `GET /audit/{run_id}/status` endpoint.
- [ ] **Secrets Management:** Integrate Doppler, Vault, or GitHub OIDC with AWS Secrets Manager.
- [ ] **API Documentation:** Publish the auto-generated `/openapi.json` to a documentation platform (e.g., Stoplight, SwaggerHub).

> **Done When:** The walking skeleton successfully deploys to a staging environment via GitHub Actions, with **secrets pulled securely from a vault**, and the API documentation is visible and versioned.

---

## 3. Wire in Observability & Security Guard-rails (Sprint 3)

*Goal: Build the necessary rails to prevent the codebase from degrading in quality, security, or stability as it grows.*

| Layer | Must-Have This Sprint |
| --- | --- |
| **CI/CD** | - [ ] Enhance CI to fail builds if test coverage drops below 80%. <br> - [ ] Add `trivy` scan for vulnerabilities in the Docker image. <br> - [ ] Add `dependabot` or `renovate` for automated dependency updates. <br> - [ ] Configure `ruff` and/or `bandit` for static application security testing (SAST). |
| **Telemetry** | - [ ] Implement structured logging (JSON format) using `structlog` in both FastAPI and Celery. <br> - [ ] Wire Sentry SDK for centralized error tracking and alerting. |
| **Monitoring**| - [ ] Expose Prometheus metrics for API (latency, request rate, error rate) and Celery (task failures, duration). <br> - [ ] Create a minimal Grafana dashboard to visualize these KPIs. <br> - [ ] Configure basic Grafana alerts for high 5xx error rates and Celery task failure rates. |

---

## 4. Re-baseline the Roadmap: Vertical Increments

*Goal: Break the original waterfall plan into manageable, value-driven milestones, all built on the frozen, tested foundation.*

| Increment | New Capability | Horizontal “Done” Criteria |
| --- | --- | --- |
| **M1** (now) | **Walking Skeleton** | Schema v1 locked; env/config complete; CI/CD basics & security scans in place. |
| **M2** | **BrandDetector MVP** | - Implement `BrandDetector` service & integrate into the audit task. <br> - Persist detected brands in the `responses` table and surface the result via `GET /audit/{run_id}`. |
| **M3** | **AuditProcessor Skeleton** | - Implement `AuditProcessor` to orchestrate question generation and execution. <br> - Prove the task queue is robust with retries and idempotent task design. |
| **M4** | **Multi-Provider AI Clients** | - Add `AnthropicPlatform` client. <br> - Implement a basic feature flag or configuration to select the AI provider for an audit. |
| **M4.5**|**Hardening: Perf, Scale & DR**| - Implement a Locust/Gatling load test simulating 50 concurrent audits. <br> - Configure DB/Redis connection pooling. <br> - Set up nightly DB backups (`pg_dump`) to versioned S3. <br> - Add a basic load balancer (nginx/Traefik) in `docker-compose`. |
| **M5** | **ReportGenerator (PDF/HTML)** | - Implement `ReportGenerator` service in a Celery task. <br> - Set up a pipeline to upload the generated report to an S3 bucket fronted by a CDN (e.g., CloudFront). |
| **M6** | **Auth & Rate Limiting** | - Add JWT-based authentication to API endpoints. <br> - Implement basic rate-limiting middleware. <br> - Ensure database queries are scoped to the authenticated agency/tenant. |
| **M7** | **Frontend Alpha (Optional)**| - Create a read-only React dashboard that calls existing endpoints to visualize audit status and results. |

---

## 5. Put Guard-rails on Vertical-Slice Freedom

*Goal: Maintain discipline and quality through process.*

- **Data Migrations:** Any PR that touches a database model **must** include the corresponding Alembic migration script.
- **Demo-Specific Code:** Use a dedicated `demo/*` branch for any "hacky" code needed for a demonstration. This code must be hardened and tested before being merged into `main`.
- **Weekly Retrospectives:** Hold a weekly meeting to review progress, velocity, and technical debt. Keep an eye on DORA metrics (Lead Time for Changes, Change Failure Rate) to ensure we stay on track.

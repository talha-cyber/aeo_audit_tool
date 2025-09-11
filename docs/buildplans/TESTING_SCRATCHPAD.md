# Testing Scratchpad — AEO Audit Tool

> AI-friendly, end-to-end oriented notes to run, triage, and extend tests quickly.

---

## Quick Commands
- Unit (fast): `pytest -q -m "unit and not slow"`
- Integration: `pytest -q -m integration`
- E2E/API: `pytest -q -m e2e`
- Full suite (local): `pytest -m "not slow"` for PRs; nightly: `pytest -m "all"` (no filter)
- Coverage: `pytest --cov=app --cov-report=xml --cov-report=json:coverage.json --junitxml=reports/junit.xml`
- Lint/Types: `ruff check . && mypy app`

---

## Markers (Add In pytest.ini)
Define markers to enable selective runs and avoid warnings:

```
[pytest]
markers =
    unit: fast unit tests
    integration: integration tests (db/redis/celery)
    e2e: end-to-end tests exercising API flows
    slow: long-running tests; excluded on PRs
    chaos: chaos/resilience scenarios
    security: security scans/tests
```

Run policy:
- PRs: `-m "not slow and not chaos and not security"`
- Nightly: run all markers

---

## Module Test Matrix (Condensed)
- app/services/ai_platforms: unit for adapters/rate limits; integration with sandbox; property on response schema.
- app/services/scheduling: unit for triggers/policies; integration for repo/beat/worker; chaos for Redis outage.
- app/services/report_generator.py: unit for templates/layout; e2e generate after audit; snapshot metadata.
- app/tasks/*: unit idempotency; integration Celery flow; DLQ processing tests.
- app/api/v1/*: unit request models; integration DB session; e2e full audit run; OpenAPI conformance.
- app/utils/resilience/*: unit breaker/retry; integration DLQ ops; chaos jitter/failures.
- app/utils/error_handler.py: unit mapping; integration sentry hook; incident schema tests.

---

## Data/Env Prereqs
- DB/Redis up: `docker-compose up -d db redis`
- Celery worker: `docker-compose up -d worker`
- Web/API: `docker-compose up -d web`
- Test env vars: ensure `.env` has DB/REDIS credentials; optional `SENTRY_DSN` and OTEL exporter.

---

## Artifacts (AI-Readable)
- JUnit: `reports/junit.xml`
- Coverage: `coverage.xml`, `coverage.json`
- Incidents: `reports/incidents/incident_*.json` (validate with `reports/schemas/incident_v1.schema.json`)
- Security: `bandit-report.json`, `trivy-results.sarif`, `zap-report.xml`

---

## Reproduce An Error
1) Find `request_id` in logs or incident JSON.
2) Locate the incident file under `reports/incidents/` (latest timestamp).
3) Replay:
   - API: `python scripts/reproduce_error.py --incident reports/incidents/incident_YYYY.json --mode http`
   - Celery: `python scripts/reproduce_error.py --incident ... --mode task`
4) Compare outputs vs expected; attach `request_id` in any new logs.

---

## DLQ Ops
- Inspect: `python scripts/requeue_dlq.py --queue audit:tasks --action stats`
- Process with handler: `python scripts/requeue_dlq.py --queue audit:tasks --action process --max 100`
- Requeue to original: `python scripts/requeue_dlq.py --queue audit:tasks --action requeue --max 50`

---

## Chaos Recipes
- Network jitter: run toxiproxy between web↔redis; add 200–400ms jitter; run integration suite; assert SLO ≤ 5 min.
- Redis outage: stop `redis` for 60s during audit; verify retries, DLQ parks > max_retries, and recovery.

---

## Security Checks
- Bandit: `bandit -r app -f json -o bandit-report.json`
- Container: `docker build -t aeo:scan . && trivy image --severity HIGH,CRITICAL --exit-code 1 aeo:scan`
- OWASP ZAP: baseline against staging URL; authenticated scan via context file; export XML.

---

## Coverage Expectations
- Core packages ≥ 90% statements: `app/services`, `app/utils`, `app/tasks`, `app/api`.
- Diff coverage ≥ 90% in PRs (changed lines/files).
- Gating: CI fails if thresholds unmet.

---

## Common Gotchas
- Mark slow tests or they will break PR speed gates.
- Do not add high-cardinality labels (request_id) to Prom metrics; use logs/traces for correlation.
- Snapshot tests: remember to update snapshots only when intentional changes occur.

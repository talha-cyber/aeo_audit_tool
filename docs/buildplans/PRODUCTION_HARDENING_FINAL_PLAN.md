# Production Hardening — Final Plan (Integrated)

> Backbone: Plan #3 (environments, canary, replay, AI artifacts) augmented with #2’s Module Test Matrix and #1’s Execution Matrix, Crash Dumps, DLQ, and OWASP ZAP gate.

---

## Goals
- Ship safely through CI → Staging → Prod-Canary with clear gates and fast rollback.
- Tie logs, traces, and metrics by the same correlation IDs for quick root cause.
- Provide reproducible incident artifacts and replay tools for humans and AI agents.
- Enforce coverage, security, and performance quality bars prior to release.

---

## Environments & Promotion
- Local: Rapid development with Docker Compose (web, db, redis, worker, monitoring).
- CI: Fast checks on PRs; full matrix on main; nightly deep runs.
- Staging: Mirror of prod; receive full E2E, ZAP, load smoke; synthetic checks.
- Prod-Canary: 5–10% traffic with auto-rollback on health SLO breach.
- Prod: Full rollout; continuous monitoring; incident capture + replay enabled.

Promotion rules:
- PR: Lint + types + unit + integration (no slow); diff coverage ≥ 90% on changed files.
- Main: All tests including slow/integration; package build succeeds; artifacts saved.
- Pre-release: Security gates (Bandit, Trivy, ZAP) pass; chaos smoke; coverage ≥ 90% for core packages.
- Staging: Green synthetic checks; no P1/P2 alerts for 30 min; operator approval.
- Canary: 30–60 min bake with SLOs; auto-rollback if violated.

---

## Observability Model
- Correlation IDs: HTTP via `X-Request-ID` middleware; propagate into logs and traces.
- Tracing: OpenTelemetry configured with OTLP exporter (optional); instrument FastAPI + requests.
- Logging: `structlog` JSON in prod, includes `request_id`, audit context, and platform context.
- Metrics: Prometheus for business, health, and DLQ depth; Grafana dashboards and alerts.
- Incident Artifacts: Emit structured JSON on unhandled exceptions and alert triggers under `reports/incidents/` following `reports/schemas/incident_v1.schema.json`.

ID alignment:
- Logs: include `request_id` and, when available, `trace_id`/`span_id` (via OTEL context).
- Traces: carry `request_id` as attribute; follow propagation across services.
- Metrics: do NOT label by high-cardinality IDs; use exemplars linking to trace IDs where supported.

---

## Testing Strategy (Pyramid)
- Unit: Pure logic, contracts, property tests; deterministic and fast (< 1s/test).
- Integration: DB (SQLAlchemy), Redis, Celery task flow, providers behind fakes.
- E2E/API: FastAPI routes + Celery + DB + Redis; golden-report snapshots.
- Performance/Load: Key flows (audit run, report generate) under realistic data.
- Chaos/Resilience: Network jitter (toxiproxy), Redis outage, rate-limit spikes; verify recovery SLO ≤ 5 min.
- Security: Bandit SAST; Docker image scan (Trivy); OWASP ZAP baseline + authenticated scan pre-release.

Markers and policy:
- Markers: `unit`, `integration`, `e2e`, `slow`, `chaos`, `security`.
- PRs: `-m "not slow and not chaos and not security"`.
- Nightly: run all markers.

---

## Module Test Matrix (from #2)
| Module | Unit | Integration | E2E | Property/Contract | Load/Chaos |
|---|---|---|---|---|---|
| `app/services/ai_platforms/*` | adapters, rate limiters | real HTTP via sandbox/fakes | end-to-end prompt pipelines | provider response schema | burst rate + timeouts |
| `app/services/scheduling/*` | trigger math, policies | DB repo, Celery beat/worker | schedule→execute→status | cron/interval parsing | Redis outage recovery |
| `app/services/report_generator.py` | layout utils, templates | DB read/write paths | generate PDF after audit | snapshot PDFs metadata | large reports perf |
| `app/tasks/*` | idempotency helpers | Celery task flow | API→task→result | payload schema | DLQ processing |
| `app/api/v1/*` | request validation | DB session + auth | full audit run | OpenAPI conformance | load on critical routes |
| `app/utils/resilience/*` | retry/backoff, breaker state | Redis DLQ ops | DLQ + recovery | DLQ message schema | jitter + failures |
| `app/utils/error_handler.py` | mapping, normalization | sentry hook | 500 path capture | incident schema | crash flood handling |

Note: Favor snapshot tests for API/JSON schemas to aid AI diffing.

---

## Execution Matrix (from #1)
- PR (fast): ruff, mypy, unit, integration (not slow), coverage diff ≥ 90% on changed lines; JUnit + coverage JSON artifacts.
- Main: full test suite including slow; build docker; upload artifacts; generate SBOM.
- Nightly: E2E, load smoke, chaos (toxiproxy), security (Bandit) — alerts on regression.
- Pre-release: Bandit, Trivy, OWASP ZAP baseline + full authenticated scan; block on HIGH/CRITICAL.
- Staging: deploy, run synthetic transactions, execute incident replay batch; verify dashboards and alert silence.
- Prod-Canary: 5–10% traffic; automated rollback if error rate/latency SLO breached.

---

## Crash Dumps (from #1)
- Trigger: unhandled exception in API worker or Celery worker.
- Contents: last 100 in-memory log lines, sanitized env (whitelist), stack trace, process info, optional request payload metadata.
- Location: `/var/log/aeo/` or `./reports/incidents/` in containers without writable `/var/log`.
- Naming: `crash_{timestamp}_{process}_{pid}_{request_id?}.jsonl` plus `stack_{...}.txt`.
- Privacy: redact secrets; include `request_id`, `audit_run_id`.

Implementation hook:
- API: exception handler middleware writes crash artifact and incident JSON.
- Celery: `task_failure` signal writes crash artifact with task name and args summary.

---

## DLQ Policy (from #1)
- Persist failed Celery payloads with: original message, original queue, exception string, timestamp, retry_count, `request_id`.
- Queue naming: `dlq:{original_queue}` (already implemented under `app/utils/resilience/dead_letter`).
- Recovery: scheduled processor re-tries messages; park after `max_retries`.
- Scripts: `scripts/requeue_dlq.py` supports moving DLQ payloads back to source queues.
- Metrics: export DLQ depth and recovery counts to Prometheus; alerts when depth exceeds thresholds.

---

## Security Gate (add OWASP ZAP)
- Add ZAP baseline scan (unauthenticated) against staging URL.
- Add ZAP full scan (authenticated) using a seeded user and context; block on medium+ alerts.
- Keep Bandit (SAST), Trivy (container), and Trufflehog (secrets) already configured.

---

## Immediate Fixes To Ship Green
- Dependencies: ensure Celery present in dev/CI; include in docker-compose (done; verify in prod compose too — present).
- Pre-commit/proxy: pin pre-commit or run `ruff`, `mypy`, `pytest` directly in CI (current CI uses direct tools).
- Pytest markers: add `pytest.ini` registering `slow`, `integration`, `e2e`, `chaos`, `security`; default PR runs exclude slow.
- Coverage: fail under 90% for core packages; report diff coverage in PRs; upload coverage XML + JSON.
- Security gate: add Bandit + ZAP scans to pre-release job; block on HIGH/CRITICAL (Trivy already blocking).
- Chaos: weekly toxiproxy tests adding network jitter + Redis outage; assert recovery SLO ≤ 5 min.
- Crash dumps: write last logs + env whitelist + stack to `/var/log/aeo/` (or `reports/incidents/` if not writable).
- DLQ: persist exception + trace info; add `scripts/requeue_dlq.py` for ops.
- Runbooks: one markdown per alert under `docs/runbooks/` with exact curl/docker commands.
- AI artifacts: always emit JUnit XML, coverage XML+JSON, and incident JSON per schema.

---

## How-To Debug (Playbooks)
- Given `request_id`:
  - Search logs for `request_id` in Loki/Cloud logs.
  - Find linked trace in OTEL backend; inspect spans around provider calls.
  - Check metrics dashboards for error/latency spikes; use exemplars to open traces.
  - If crash artifact exists under `reports/incidents/`, open JSON and replay with `scripts/reproduce_error.py`.
- Given DLQ depth alert:
  - Run `python scripts/requeue_dlq.py --queue audit:tasks --action stats` to inspect.
  - Reprocess with `--action process --max 50` or requeue to original `--action requeue`.
- Given security gate failure:
  - Open ZAP report artifact, confirm alert type and path; patch and rerun pre-release job.

---

## Required Artifacts (AI-Friendly)
- Test: `reports/junit.xml`, `coverage.xml`, `coverage.json`.
- Incidents: `reports/incidents/incident_*.json` with schema `reports/schemas/incident_v1.schema.json`.
- Security: Bandit JSON, Trivy SARIF, ZAP XML/HTML.
- Chaos: snapshot JSON of SLO measurements per scenario.

---

## Backlog & Nice-To-Haves
- OTEL propagators into Celery tasks (carry `trace_id`/`request_id`).
- Prometheus exemplars for key counters/histograms with trace IDs.
- Golden snapshots for PDF layout metadata to detect visual regressions quickly.

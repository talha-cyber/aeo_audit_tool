# Agent Playbook for AEO Audit Tool

## Why this doc
- Give AI coding agents the minimum context needed to work safely and effectively in this repository.
- Highlight project conventions so changes land cleanly in PRs and integrate with the existing automation stack.

## Project orientation
- FastAPI backend backed by PostgreSQL, Redis, and Celery workers; entrypoint at `app/main.py`.
- Domain logic is grouped under `app/services`, persistence lives in `app/db` and `app/models`, and async tasks live in `app/tasks`.
- Reporting pipelines render PDFs under `reports/`; keep outputs deterministic for the tests in `tests/` and the integration scripts under `examples/`.
- Monitoring and security layers (`app/monitoring`, `app/security`) are first-class; preserve middlewares, logging hooks, and headers.
- Documentation canon lives in `docs/` (architecture, build plans, runbooks). Always scan relevant guides before large edits.

## Standard agent workflow
1. **Collect context**: read the root `README.md`, `docs/ARCHITECTURE*.md`, and any build plan or runbook related to the target module before touching code.
2. **Shape a plan**: break work into verifiable steps. Respect existing partial changes; never undo user modifications outside your scope.
3. **Work in place**: prefer targeted edits with `rg`, `sed -n`, or `python -m` scripts. Do not restructure modules unless explicitly asked.
4. **Keep it observable**: when adding logic, wire it into existing logging (`app.utils.logger`), tracing, and access controls.
5. **Validate early**: run focused tests or scripts that exercise the change before concluding.
6. **Document decisions**: update `docs/` or add inline comments only when it clarifies non-obvious logic.

## Implementation guidelines
- Match the async/await style used in the module you are changing; FastAPI endpoints and Celery tasks should remain async-safe.
- Reuse service-layer helpers instead of talking to models or external APIs directly from routers.
- Database changes go through SQLAlchemy models and migrations (`alembic/`). Do not touch `.db` artifacts (`organic_memory.db`, `test.db`).
- When extending Celery tasks, make them idempotent and encode retries/backoff via Celery options.
- Follow the security defaults: keep security headers, auth checks, and PII scrubbing intact.
- For report generation, keep layout logic separate from data transforms (see `docs/buildplans/Report_Generator_v2_Plan.md`).
- Logging should use the project logger; avoid bare `print` statements.

## Testing and quality gates
- Unit/integration tests: `pytest` (use `pytest tests/path::TestClass::test_case` for focus).
- Type and lint: `ruff check .` and `mypy .` (mirrors the health check make target).
- Full pipeline (dockerized): `make healthcheck` runs migrations, lint, type check, and tests inside containers.
- For new features, add or update tests in `tests/` or targeted modules; keep fixtures deterministic.
- Validate report outputs or long-running tasks via the example scripts in the repository when relevant.

## Data, secrets, and environment
- Never commit secrets; use `.env` locally and reference settings through `app/core/config.py`.
- Assume Postgres, Redis, and external LLM APIs are mocked during tests; gate network calls behind feature flags or settings.
- Generated artifacts (reports, logs) should stay out of version control unless explicitly needed for fixtures.

## Hand-off checklist
- Code formatted, linted, and type-checked.
- Tests covering the change pass locally.
- Relevant documentation (`docs/` or README snippets) updated when behavior or workflows change.
- Provide a concise change note pointing to touched files and any follow-up actions for humans.

Stick to this playbook to keep contributions predictable and production-safe.

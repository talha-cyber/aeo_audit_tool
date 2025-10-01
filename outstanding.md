# Outstanding Tasks – Backend/Frontend Alignment

Refer to `scratchpad_backend_frontend_alignment.md` for full context, status history, and detailed notes on the dashboard integration work.

## Backend Phase 1 follow-ups
- Replace static fallbacks (widgets, settings, comparison matrix, insights) with live data sources once upstream services are available (comparison matrix now derives from `Response.brand_mentions`).
- Finalize audit run naming and health-score weighting; capture decision in scratchpad + implementation.
- Add focused API tests (`tests/api/v1/test_dashboard.py`) and service-level tests for reports, personas, widgets, comparison matrix, and settings endpoints (new smoke test added for `/audits/test-run`; comparison matrix now covered by `tests/services/dashboard/test_comparison_service.py`).
- `/api/v1/dashboard/audits/test-run` now returns a simulated run payload; replace with real execution path when backend orchestration is ready.

## Frontend Phase 2
- Point frontend client (`frontend/src/lib/api/client.ts`) at the new backend endpoints; keep mock toggle documented.
- Review React Query polling/staleness once backend exposes refresh hints.
- Run frontend lint/tests (`npm run lint`, `npm run test`) after wiring to live API.

## Documentation & QA Phase 3
- Update docs/runbooks with dashboard API usage + required env vars.
- Extend `docs/buildplans` with the backend/ frontend data contract summary.
- Plan full test passes (`pytest`, `ruff`, `mypy`, frontend test suites) before shipping.

## Dependencies / Environment
- Monitor for PyPI wheels supporting `pydantic` on Python 3.13; rerun skipped insight tests when available.
- Capture seeding strategy (e.g., `scripts/seed_dashboard_jobs.py`) in operational docs.

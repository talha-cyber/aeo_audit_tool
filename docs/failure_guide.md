# Persona Composition Failure Guide

## Symptom
- Creating a persona from the dashboard fails with the toast message **"Unable to compose persona. Check your selections and try again."**
- API responses show HTTP 400 errors originating from `_resolve_client_context` in `app/api/v1/dashboard.py`.

## Root Cause
- Persona endpoints require an active tenant (`client_id`).
- After recent hardening, `_resolve_client_context` no longer falls back silently; it expects either an impersonation token (`act_as`) or an existing tenant record.
- On fresh dev setups, the database may not contain the “Internal Preview – AEO” tenant yet, so the fallback query returns `None`. The request is rejected even though the UI selections are valid.

## Immediate Fix
1. Ensure the preview tenant exists:
   ```bash
   python scripts/seed_internal_client.py
   python scripts/seed_dashboard_jobs.py  # optional but keeps dashboard populated
   ```
2. Restart the backend (`uvicorn`, or the docker-compose service) so it picks up model/service changes.
3. Refresh the dashboard and retry persona composition.

## Prevention / Workflow Notes
- Always impersonate a client (via the Admin overlay) or seed tenants before testing persona flows.
- For automated tests, the security layer now detects the pytest environment and injects a deterministic `test-client` context when no Authorization header is provided (`app/api/v1/security.py`).
- If you skim seed scripts, look for log lines confirming `Internal Preview – AEO` was inserted; without that record the fallback will continue to fail.

## Related Changes
- `frontend/src/lib/auth/tokenManager.ts` now decodes the base JWT to recover tenant defaults when no act-as session is active.
- `app/api/v1/dashboard.py` resolves missing client context by querying for an internal tenant instead of returning 400 immediately.
- `app/services/dashboard/audit_run_creation_service.py` sanitises persona IDs and logs failed Celery enqueue attempts to keep local runs resilient.

Keep this sequence handy whenever persona composition fails right after pulling or setting up a new environment.

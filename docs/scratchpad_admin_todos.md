# Admin Overlay / Impersonation Remaining TODOs

- [x] Backend: finish wiring admin-only endpoints (`POST /api/v1/audits/{id}` with `action` support) and ensure structured logging plus tenant-scoped safeguards.
- [x] Backend: thread the `exclude_internal` analytics flag through dashboard summary endpoints and service layer usage.
- [x] Frontend: implement Admin Overlay provider, toolbar trigger, keyboard shortcut handling, and impersonation banner wiring via the shared layout.
- [x] Frontend: surface impersonation session controls (banner with exit) and ensure client context propagates through persona requests.
- [x] Seeds & fixtures: expand internal preview seed scripts to include admin settings + mixed audit runs for preview tenant.
- [x] Testing: cover impersonation token minting, admin endpoints, analytics exclusion and overlay UX (PyTest + Vitest).

# Backend ↔ Frontend Integration Scratchpad

_Last updated: 2025-09-30 by Codex_

## 1. Context Recap — Backend Surface
- **Primary entrypoint:** `app/main.py` with routers under `/api/v1`.
- **Audit data models:** `AuditRun`, `Client`, `Report`, `Response`, `Question` (see `app/models`). Key fields we can surface now:
  - `AuditRun`: `status`, `started_at`, `completed_at`, `total_questions`, `processed_questions`, `progress_data`, `platform_stats`, `error_log`.
  - `ScheduledJob` (`app/models/scheduling.py`): holds recurring audit definitions (`name`, `trigger_config`, `job_config`, `created_by`, `last_run_at`).
  - `Report`: `report_type`, `generated_at`, `file_path`, `template_version`, `theme_key`.
  - `Response`: `brand_mentions`, `response_metadata`, `processing_time_ms`, `satisfaction_score`.
- **Current routers:**
  - `/audits`: trigger run, poll status, generate report, fetch latest report.
  - `/audit-status`: detailed run info, progress snapshots, system metrics.
  - `/personas`: expose catalog + modes (no flattened persona list yet).
  - `/monitoring`, `/providers/health`, `/secure` for support tooling.
- **Supporting services worth reusing:**1
  - `AuditProcessor`, `ProgressTracker`, `ReportGenerator`, `PersonaExtractor`, `PlatformManager`.
  - Scheduling repository & models for cadence/ownership metadata.
  - `CentralIntelligence` (insight engine) can produce structured `SystemInsight` records.

## 2. Context Recap — Frontend Contracts (Next.js)
All remote fetches are defined in `frontend/src/lib/api/client.ts` + `schemas.ts`. With `NEXT_PUBLIC_USE_MOCKS !== 'false'` the UI pulls from the mock client; integration requires the real API to match these payloads:

| Query | Method | Path | Expected Shape |
| --- | --- | --- | --- |
| `auditSummaries` | GET | `/audits` | `AuditSummary` → `{id,name,cadence,owner,platforms[],lastRun,healthScore}` |
| `auditRuns` | GET | `/audits/runs` | `AuditRun` → `{id,name,status,startedAt?,completedAt?,progress{done,total},issues[]}` |
| `auditRun` | GET | `/audits/run/{id}` | `{ run: AuditRun, questions: [{id,prompt,platform,sentiment,mentions[{brand,frequency,sentiment}]}] }` |
| `launchTestRun` | POST | `/audits/test-run` | Body `{scenarioId,questionCount,platforms[]}` → returns `AuditRun` |
| `reportSummaries` | GET | `/reports` | `ReportSummary` → `{id,title,generatedAt,auditId,coverage{completed,total}}` |
| `insights` | GET | `/insights` | `Insight` → `{id,title,kind,summary,detectedAt,impact}` |
| `personas` | GET | `/personas` | `Persona` → `{id,name,segment,priority,keyNeed,journeyStage[{stage,question,coverage}]}` |
| `widgets` | GET | `/embeds/widgets` | `Widget` → `{id,name,preview,status}` |
| `comparison` | GET | `/comparisons/matrix` | `{competitors[], signals[{label,weights[]}]}` |
| `settings` | GET | `/settings` | `{branding, members[], billing, integrations[]}` |

The UI expects fairly opinionated derived fields (health scores, coverage %, insight impact labels) and refresh intervals (React Query refetches running audits every 5s, run details every 4s).

## 3. Gap Map (Backend reality → Frontend needs)

| Domain | Backend Source Today | Missing / Transform Work |
| --- | --- | --- |
| Audit library (programs) | `ScheduledJob` + latest `AuditRun` stats | Need API to assemble `AuditSummary` (cadence from `trigger_config`, owner from `created_by`/`job_config`, platforms from `job_config`, healthScore derived from rolling success/failure + latency).
| Live runs | `AuditRun` + `progress_data` + `platform_stats` | Need projection to `AuditRun` schema (generate friendly `name`, map `processed_questions` → `progress.done`, fold `progress_data.platform_errors`/`error_log` into `issues`).
| Run detail | `AuditRun.questions` + `Response` rows | Need join & serializer to emit question list with sentiment + mentions (brand detection output). Fill sentiment via `Response.response_metadata` or `sentiment` service fallback.
| Launch test run | No current analogue; closest is `/audits/configs/{client_id}/run` | Need lightweight “guided simulation” endpoint honoring scenario/platform parameters, optionally flag to use mock providers; update Celery entry to support synchronous fast path when mocks requested.
| Reports | `Report` table + linked `AuditRun` totals | Need list endpoint sorted desc, convert `report_type` to title, build `coverage` from `AuditRun.processed_questions/total_questions` or stash in `Report`
| Insights | `CentralIntelligence.get_active_insights()` + audit analytics | Need REST wrapper + fallback generator if intelligence disabled; map to `kind/impact` enums and human-friendly titles.
| Personas | `PersonaExtractor.export_catalog` | Need flattener to produce array of personas (derive `segment` from role desc, `keyNeed` from driver anchor, `journeyStage` from contexts with coverage heuristics).
| Widgets | None persisted | Need config source (could live in `app/services/portal/widgets.py` with static definitions or DB table) + API.
| Comparison matrix | Brand detection stats (from `Response.brand_mentions`), platform stats | Need aggregation job to compute share-of-voice weights per competitor; consider cached materialized view for UI.
| Settings | Scattered: env settings, clients, integrations table (none) | Need settings service to stitch: branding tokens (new config model), members (maybe from `Client` contacts or scheduled jobs), billing (static placeholder), integrations (based on provider health / flags).

## 4. Integration Strategy

### Phase 0 — Data audit & enablement
### Phase 0 Findings (2025-09-30)
- Rewired scheduling model/engine to treat `job_config` as the canonical payload wrapper (with backward-compatible `job_data` alias) so future audit summaries can source cadence/owner/platform metadata.
- `ScheduledJob` model persists `job_config` JSON, but the scheduler engine still attempts to write a `job_data` field. We need to reconcile this by either:
  - Updating the engine/repository to populate `job_config` (with a top-level `payload` key for handler inputs) and adding an alias property for legacy `job_data` usage, or
  - Introducing a backward-compatible hybrid property on `ScheduledJob` so existing scheduler code consumes `job_config` transparently.
- No concrete job records are defined in repo fixtures; the shape must cover at least `client_id`, `owner`, `platforms`, `question_pack`, and cadence metadata. Recommend standardizing on:
  ```json
  {
    "client_id": "uuid",
    "owner": {"name": "", "email": ""},
    "cadence": {"label": "Monthly", "cron": "0 9 1 * *"},
    "platforms": ["openai", "anthropic"],
    "question_bank_id": "bank-123",
    "alerting": {"channels": ["email"], "severity_threshold": "high"}
  }
  ```
  Store additional execution hints (batch size overrides, SLA minutes) under `job_config["execution"]` so view-model logic can surface health KPIs.
- `trigger_config` is consistently expected to hold `{ "trigger_type": "cron" | "interval" | ... }`; we can rely on this for cadence display but should add helper to map to human-readable labels.

- Catalogue existing `ScheduledJob.job_config` shape; ensure audit definitions include `platforms`, `owner`, `cadence`. Add migration to capture missing metadata if required.
- Decide source of truth for insight, widget, settings data (DB vs static config). Seed fixtures to unblock API responses.
- Align environment config: expose `NEXT_PUBLIC_API_BASE_URL` + default `NEXT_PUBLIC_USE_MOCKS=false` in deployed env; document in `docs/commands`.

### Phase 1 — Backend API alignment

#### Phase 1 Draft — View Models & Services
- [2025-09-30] Implemented persona, widget, comparison, and settings services with `/api/v1/dashboard` endpoints backed by static fallbacks.
- [2025-09-30] Implemented insight service fallback with `/api/v1/dashboard/insights` (skips to static fixtures when Central Intelligence is unavailable).
- [2025-09-30] Implemented `list_audit_programs` in `app/services/dashboard/audit_summary_service.py` to surface cadence/owner/platforms and a preliminary health score from scheduled jobs.
- [2025-09-30] Scaffolded `app/api/v1/dashboard/schemas.py` with alias-aware models and created service skeletons under `app/services/dashboard/` (raising NotImplementedError until populated).
**Pydantic view models (`app/api/v1/dashboard/schemas.py`):**
  - `AuditSummaryView`: id, name, cadence (label|string), owner (name/email), platforms[], last_run (datetime | None), health_score (float 0-100).
  - `AuditIssueView`: id (f"{platform}:{code}"), label, severity (`low`/`medium`/`high`).
  - `AuditRunProgressView`: done, total, updated_at.
  - `AuditRunView`: id, name, status, started_at, completed_at, progress (`AuditRunProgressView`), issues[List[`AuditIssueView`]].
  - `AuditRunDetailView`: run (`AuditRunView`), questions[List[`AuditQuestionView`]] where question has id, prompt, platform, sentiment, mentions[List[`BrandMentionView`]].
  - `ReportSummaryView`: id, title, generated_at, audit_id, coverage (`CoverageView` with completed/total ints).
  - `InsightView`: id, title, kind (`opportunity`/`risk`/`signal`), summary, detected_at, impact (`low`/`medium`/`high`).
  - `PersonaView`: id, name, segment, priority, key_need, journey_stage[List[`PersonaStageView` stage/question/coverage float]].
  - `WidgetView`, `ComparisonMatrixView`, `SettingsView` mirroring frontend zod schemas but with explicit typing.
- **Service skeletons (`app/services/dashboard/`):**
  - `audit_summary_service.py`: `list_audit_programs(db: Session) -> List[AuditSummaryView]`. Pull active `ScheduledJob` (type `AUDIT`) with latest `AuditRun` (JOIN on `JobExecution.audit_run_id` + `AuditRun`). Compute health_score using success/failure counts + SLA thresholds from `job_config`.
  - `audit_run_service.py`: `list_runs(db) -> List[AuditRunView]`, `get_run(db, run_id) -> AuditRunView`. Normalize progress from `AuditRun.processed_questions`, fallback to `progress_data`. `collect_issues(audit_run)` helper turning `platform_stats`, `error_log` into view issues.
  - `run_detail_service.py`: `get_run_detail(db, run_id) -> AuditRunDetailView`. Preload `AuditRun.questions` + `Response` (selectinload), call `sentiment.analyze` when metadata missing.
  - `report_service.py`: `list_reports(db) -> List[ReportSummaryView]`. Compose coverage from stored snapshot or `AuditRun` metrics.
  - `insight_service.py`: `list_insights(db) -> List[InsightView]`. Use `CentralIntelligence`; fallback to deterministic fixtures if disabled or empty.
  - `persona_service.py`: `list_personas(mode: PersonaMode) -> List[PersonaView]`. Flatten persona catalog via `PersonaExtractor` + coverage heuristics.
  - `widget_service.py`: `list_widgets() -> List[WidgetView]`. For now reference static config in `app/services/dashboard/static_data.py`.
  - `comparison_service.py`: `get_comparison_matrix(db) -> ComparisonMatrixView`. Aggregate `Response.brand_mentions` by competitor with share-of-voice normalization and caching hook.
  - `settings_service.py`: `get_settings() -> SettingsView`. Pull branding defaults from `settings`, membership info from static fixtures or future table.
- **Router wiring (`app/api/v1/dashboard.py`):** Each endpoint injects `Session` dependency, invokes corresponding service, and returns Pydantic response. Use `APIRouter(prefix="", tags=["dashboard"] )` or keep grouped by domain.
- **Constants/Helpers:** Add `app/services/dashboard/utils.py` for cadence formatting (`cron -> label`), health scoring weights, brand mention normalization, plus `app/services/dashboard/types.py` for enums shared across services.

1. **View-layer models**: introduce `app/api/v1/dashboard/schemas.py` with pydantic serializers mirroring frontend schemas (AuditSummaryView, AuditRunView, etc.) but typed + datetimes as ISO8601.
2. **Service orchestrators**: under `app/services/dashboard/` add facades:
   - `audit_summary_service.py`: query `ScheduledJob` + `AuditRun` stats, compute health score (`success_rate`, `avg_duration` normalized to 0–100). Ensure queries are batched to avoid N+1.
   - `audit_run_service.py`: fetch runs, compute progress (use `processed_questions`/`total_questions`, fallback to `progress_data`). Extract issues from `platform_stats` + `error_log` (map severity based on counts > thresholds).
   - `run_detail_service.py`: join `Question` + `Response` in one pass, run sentiment (call `app/services/sentiment/analyzer.py` if metadata missing), normalize brand mentions.
   - `insight_service.py`: wrap `CentralIntelligence`; if disabled, synthesize insights from latest report analytics (rely on `ReportGenerator` helper summarizing competitor momentum).
   - `persona_service.py`: flatten persona catalog; compute coverage as `ContextCatalogItem.priority` normalized, or attach `0.0` default.
   - `report_service.py`: list reports with computed coverage (lookup linked run, or store coverage snapshot on report creation via `ReportGenerator`).
   - `widget_service.py`, `comparison_service.py`, `settings_service.py`: provide structured data; start with deterministic static config modules so tests are stable, plan later DB roadmap.
3. **Routers**: create `app/api/v1/dashboard.py` (or `app/api/v1/frontend.py`) with endpoints matching frontend paths but namespaced under `/api/v1`. e.g.:
   - `GET /api/v1/audits` → `AuditSummaryListResponse`
   - `GET /api/v1/audits/runs` & `GET /api/v1/audits/run/{id}` (note singular path)
   - `POST /api/v1/audits/test-run` (input validation, kicks off `AuditProcessor` with mock flag)
   - `GET /api/v1/reports`, `/insights`, `/personas`, `/embeds/widgets`, `/comparisons/matrix`, `/settings`
   Ensure responses leverage new schemas and include CORS/perf instrumentation.
4. **Test harness**:
   - Add focused API tests under `tests/api/v1/test_dashboard.py` with fixtures for `ScheduledJob`, `AuditRun`, `Response`, `Report`. Use factory helpers to seed DB (existing tests may have similar patterns).
   - Unit-test services (e.g., `tests/services/dashboard/test_audit_summary_service.py`) to keep logic deterministic.
5. **Observability**: log fetch durations via `app.utils.logger`, emit Prometheus counters for dashboard endpoints (requests/sec) via `Instrumentator`, respect security middlewares.

### Phase 2 — Frontend wiring
1. Update `frontend/src/lib/api/client.ts` base paths to match new endpoints (no shape changes if backend aligns fully).
2. Gate mocks via env: set default `NEXT_PUBLIC_USE_MOCKS=false`, keep ability to opt-in for local dev. Provide fallback UI messaging if API returns empty arrays.
3. Adjust React Query stale times/intervals if backend surfaces `nextRefresh` hints (optional).
4. Ensure type alignment: run `npm run lint` / `npm run test` to confirm zod schemas still match responses (update if backend adds optional fields like `healthScore` decimals).

**Status notes**
- [2025-09-30] API client now calls the `/api/v1/dashboard/*` endpoints and parses updated schemas (owner object, optional `lastRun`/`healthScore`, progress timestamps). `LaunchTestRun` currently returns the hydrated `run` view from the backend placeholder.
- [2025-09-30] Mock toggle now requires `NEXT_PUBLIC_USE_MOCKS=true`; drawer flow blocks launches when real API mode is active and mocks mirror backend field optionality.
- [2025-09-30] Dashboard UI components handle missing run totals/timestamps gracefully (`formatRelative`/`formatScore` fallbacks) and navigation links comply with typed routes.
- Remaining: tighten React Query polling once backend exposes refresh hints; surface API error states in the tables.

**Phase 1 follow-up — test-run orchestration**
- [2025-09-30] `/api/v1/dashboard/audits/test-run` now returns a deterministic simulated run via `test_run_service.launch_simulated_run`; replace the mock response with real orchestration once Celery wiring is ready.
- [2025-09-30] `comparison_service.get_comparison_matrix` now aggregates real `Response.brand_mentions` data when available, with a fallback for empty datasets; service covered by `tests/services/dashboard/test_comparison_service.py`.

### Phase 3 — Documentation, DX, & QA
- Update `docs/commands` and `docs/runbooks` with instructions for enabling dashboard API + required env vars.
- Extend `docs/buildplans` with the data contract alignment summary so future agents keep backend/frontend parity.
- Validate end-to-end via `pytest`, `ruff`, `mypy`, plus frontend `npm run test` / `vitest` and Playwright smoke (if configured).
- Capture follow-up tasks (e.g., move widget/settings data from static config to DB) in `docs/missing_features.md`.

## 5. Domain-specific Notes & Open Questions

### Audits
- **Name source:** Currently `AuditRun` lacks a user-facing name. Options:
  1. Persist `display_name` on `AuditRun` via migration.
  2. Derive from `ScheduledJob.name` + run timestamp.
- **Health score formula:** Combine (success rate %, average duration vs SLA, error counts). Need product sign-off on weighting.
- **Issues projection:** Map `platform_stats[platform]['error_count']` > threshold to severity (`high` if failing, `medium` for throttling, etc.).

### Run Detail
- Ensure `Question` ↔ `Response` relationships are eagerly loaded to avoid per-row queries.
- Sentiment: verify if `Response.satisfaction_score` or `emotional_satisfaction` already holds value; otherwise call sentiment analyzer with caching.
- Mentions: `Response.brand_mentions` already fits needed schema (counts, sentiment) but confirm shape; add normalization helper.

### Persona Catalog
- Flattening rule proposal: treat each voice preset as persona, `segment` = role description, `journeyStage` coverage from context priority (fallback 0.5). Document heuristics.
- Provide mode filter support (`mode=b2b|b2c`) to mirror existing query param.

### Insights & Widgets
- If `CentralIntelligence` is disabled, create `app/services/insights/dummy.py` with deterministic sample insights seeded by latest audit metrics so UI stays populated.
- Widgets can start from static definitions stored under `app/services/portal/widgets.py` (align with frontend catalog). Plan DB migration later.

### Settings
- Introduce new config class (`app/core/branding.py`) or extend `Settings` to hold branding defaults (color hex, tone string) so API can serve canonical values.
- Members: leverage `Client` contacts or create `team_members` table. For first pass, allow static config with TODO for dynamic management.

## 6. Testing & Tooling Checklist
- Backend: `pytest tests/api/v1/test_dashboard.py`, `pytest tests/services/dashboard`, `ruff check app/api/v1/dashboard.py`, `mypy app/api/v1/dashboard.py app/services/dashboard`.
- Frontend: `npm run lint`, `npm run test`, `npm run test -- --runInBand` for Vitest, optional Playwright smoke once endpoints live.
- Integration: spin up docker-compose with Postgres/Redis, run `uvicorn app.main:app --reload` + `npm run dev` to validate real data flow.

## 7. Follow-ups
- Align Celery audit completion hooks to update `ScheduledJob.last_run_at` + health KPI.
- Consider GraphQL facade? Probably not now; REST suffices.
- Move mocked insight/widget/settings data into persistent store once product requirements firm.
- Document data contracts in `docs/ARCHITECTURE_FRONTEND.md` stub for future agents.

---
Use this scratchpad as the source of truth while implementing the alignment. Update sections as backend schemas evolve.

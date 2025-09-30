# Question Engine Overhaul Scratchpad

_Last updated: 2025-09-25 16:27:40Z_

## Run Strategy
- Work through phases in order; stop around request budget and prompt user to reply with `continue`.
- Keep commits atomic per phase; ensure tests/lint per feature gate.

## Phase 0 — Context & Baseline
- [x] Review architecture docs, question engine reference, existing providers and models.
- [x] Document current data model gaps (Question/Response) vs v2 fields; capture required migrations.
  - `questions`: add columns `persona`, `role`, `driver`, `emotional_anchor`, `context_stage`, `seed_type`, `provider_version`; ensure JSON metadata keeps compatibility.
  - `responses`: add columns `emotional_satisfaction` (enum/string), `satisfaction_score` (float), `satisfaction_model` (string).
  - Indexes: (`audit_run_id`, `persona`) and (`driver`, `context_stage`) on `questions`; confirm Alembic migration updates ORM + factories.

## Phase 1 — V2 Package Scaffold
- [x] Create `app/services/question_engine_v2/` package with module stubs:
  - `__init__.py`, `engine.py`, `schemas.py`, `persona_extractor.py`, `constraints.py`, `scoring.py`, `cache.py`.
  - Subpackages: `catalogs/`, `providers/`, `prompts/`, `evaluator/`.
- [x] Ensure modules import project logger/config only; no heavy logic yet.
- [x] Decide placement for shared enums/constants (likely `schemas.py`).
  - Use `schemas.py` to centralize enums/constants for persona/driver definitions.

## Phase 2 — Catalogs & Persona Extraction
- [x] Author YAML catalogs for B2C/B2B roles, drivers, contexts, voices (placeholders acceptable initially).
  - Placeholder catalogs added for B2C/B2B roles, drivers, contexts, voices under `app/services/question_engine_v2/catalogs/`.
- [x] Implement `persona_extractor.py` to load catalogs with caching + validation.
  - Loader validates catalog coherence, caches per-mode bundles, resolves presets/explicit selections with emotional anchors.
- [x] Add unit tests under `tests/services/question_engine_v2/` for catalog loading, preset voice resolution, custom persona mixes.
  - Added coverage for catalog loading, preset + custom persona resolution, and error handling.
- [x] Wire logging + error handling (fallback to defaults when missing entries).
  - Persona extractor now skips unknown voices/roles with warnings and falls back to role/default contexts when needed.

## Phase 3 — Schema & Persistence Updates
- [x] Extend Pydantic request/response models to carry persona/seed mix metadata.
  - Added v2 request envelopes, persona configuration, seed mix, quota, and provider option schemas.
- [x] Update SQLAlchemy `Question`/`Response` models with new columns + defaults.
  - Added persona metadata columns and satisfaction fields with indexes for persona/driver-context lookups.
- [x] Create Alembic migration adding columns and indexes (`questions`, `responses`).
  - Migration `d2bb41f94837_add_question_engine_v2_fields` introduces new columns and composite indexes.
- [x] Refresh fixtures/factories and adjust any tests referencing models.
  - Existing factories unaffected; will revisit once v2 generation tests require extended attributes.

## Phase 4 — Providers & LLM Router
- [x] Design provider interface abstractions for v2 (likely building on existing `QuestionProvider`).
  - Introduced `BaseProviderV2` and `ProviderExecutionContext` foundation with logging hooks.
- [x] Implement `providers/template_provider.py` with deterministic quota-aware generation (scaffold).
  - Added persona-aware rendering against per-persona quotas with metadata tagging (placeholder templates to follow).
  - LLMRouter `complete_json` supports schema validation/retries; ready for dynamic provider wiring.
- [x] Implement `providers/dynamic_provider.py` using shared `LLMRouter` (new or reuse existing LLM clients).
  - Added prompt/schema scaffolding with router invocation and structured question parsing.
- [x] Build JSON schema validation + error surfacing for dynamic outputs.
  - Prompt/schema scaffold includes seed mix & persona context with retries via LLMRouter.
  - Tests: mock router to raise validation errors + ensure retries use schema reminder prompt.
- [x] Create provider unit tests (template determinism, dynamic schema, router selection).
  - Added tests for template quotas and dynamic router schema retries using stub router.
  - Subtasks: design shared `BaseProviderV2` with persona-aware context; reuse legacy `QuestionProvider` but add normalization helpers.
  - Implement `LLMRouter` adapter mapping provider key -> existing ai_platform clients, contract `await router.complete_json(prompt, schema, options)`; lazy init per model.
  - Template provider to load deterministic question templates keyed by persona/driver/context with quota enforcement hooks.
  - Dynamic provider to assemble prompts via Jinja, invoke router with strict JSON schema + retry on validation failure.
  - Tests: mocks for router responses, ensure quota splits + schema validation + persona metadata propagation.

## Phase 5 — Constraints & Scoring
  - Pending: implement seed mix validation, quota enforcement, and scoring weights with property tests.
  - TODO: implement `constraints.enforce_seed_mix`, `constraints.dedupe_questions`, and `scoring.score_questions` with property tests.
- [x] Encode seed mix policy, dedupe, max length, quotas in `constraints.py`.
  - Implemented seed mix allocation, dedupe, and length filtering with redistribution fallback.
- [x] Implement scoring priorities in `scoring.py` (driver/context/seed weights, provider bonus, dup penalty).
  - Added default driver/context/seed/provider weights to compute `priority_score`.
- [x] Property/unit tests verifying quotas + scoring behavior.
  - Added tests for constraint enforcement and scoring differentiation using stubbed structlog.

## Phase 6 — Engine Orchestration
- [x] Implement `engine.py` orchestrator: persona resolution → quota planning → provider fan-out → merge → constraint enforcement → scoring.
  - Engine now resolves personas, runs providers, and applies constraint+scoring passes before returning results.
- [x] Integrate Redis caching (`cache.py`) with feature flag toggles.
  - Engine v2 now resolves personas, filters providers by config, and gathers results via async orchestration scaffold.
- [x] Golden snapshot & async tests ensuring deterministic template outputs and concurrency behavior.
  - Added async engine test verifying persona resolution, constraint, and scoring pipeline with stub providers.

## Phase 7 — Satisfaction Evaluator
- [x] Implement `evaluator/answer_eval.py` with prompt templates and LLMRouter usage.
  - Satisfaction evaluator implemented with Jinja prompt + LLMRouter JSON schema.
- [x] Hook optional satisfaction scoring into engine pipeline.
  - Engine exposes evaluator hook and records satisfaction metrics when enabled.
- [x] Tests for evaluator prompt assembly + scoring interpretation.
  - Evaluator tests cover schema invocation and error scenarios.

## Phase 8 — Integration & API Wiring
- [x] Feature flag gating (`QUESTION_ENGINE_V2`).
  - Legacy engine, audit processor, and API respect the v2 feature flag.
- [x] Update FastAPI routes + schemas for persona catalog endpoint + enhanced question generation payload.
  - Added `/api/v1/personas` catalog/modes endpoints.
- [x] Ensure backward compatibility; add integration tests for flag on/off.
  - Integration test verifies v1 fallback and v2 delegation.

## Phase 9 — Observability & Metrics
- [x] Register Prometheus metrics for generation volume, seed mix, satisfaction.
  - QE v2 emits question volume, seed mix, and satisfaction metrics.
- [x] Verify emitted labels align with monitoring spec; add smoke tests if feasible.
  - Metrics include provider/role/driver/context/seed labels validated via tests.

## Phase 10 — Documentation & Rollout
- [x] Update build plan progress, runbooks, and `.env.example` for new settings.
  - Updated `.env.example`, docs/README_setup.md, and buildplan observability/testing sections.
- [x] Draft rollout notes (shadow mode → GA).
  - Documented rollout phases in build plan with status (Phase 3 pending product sign-off).

## Phase 11 — Validation Sweep
- [x] Targeted pytest suite for v2 modules.
  - Ran targeted pytest suites covering API, providers, engine, evaluator, constraints.
- [x] `ruff check`, `mypy`, and relevant integration tests.
  - Ruff/mypy executed on v2 scope; legacy issues noted but unchanged.
- [x] Prepare summary + follow-up checklist for maintainers.
  - Final summary delivered; outstanding legacy lint/type fixes noted for future work.

## Open Questions / Assumptions
- Do we reuse existing LLM client wrappers or build router anew? (Investigate before Phase 4.)
- Confirm Redis availability in tests; may need mocks.
- Determine source for driver/context weights (config vs hardcoded).

-> Answers to Quesions: 
Open Questions / Assumptions — Decisions
LLM client wrappers: Reuse via adapter, don’t rebuild.
Implement a thin LLMRouter that adapts to your existing wrappers (OpenAIClient, AnthropicClient, etc.).
Contract: complete_json(prompt, schema, model, **opts) -> dict.
Benefit: zero churn, faster delivery, and you can still swap providers.
Only build new code where wrappers are missing (e.g., JSON-mode validation or strict schema retry).
Redis in tests: Use in-memory/mocked Redis by default, real Redis in one integration job.
Unit tests: fakeredis (Python) / ioredis-mock (Node) to avoid external deps.
Integration pipeline (nightly or PR tag): spin ephemeral Redis (Docker) to catch config/auth issues.
Add an env toggle: QE_V2_FAKE_REDIS=1 for local/dev tests.
Driver/context weights: Config-first with sane defaults, hot-reloadable.
Source of truth: catalogs/*_weights.yaml (per mode B2C/B2B).
Boot defaults (hardcoded constants) only as fallback if file missing.
Expose overrides via env (e.g., QE_V2_WEIGHT_ROI=10) → merged atop YAML → logged at startup for auditability.

## Parking Lot
- Consider migrating legacy template assets into catalogs once v2 stable.
- Evaluate need for feature gating in Celery tasks consuming question engine.

-> Parking lot recommendations: 
Parking Lot — Recommendations
Migrate legacy templates: Plan a one-way migrator once v2 stabilizes KPIs.
Write a small migration script that:
parses legacy template files → emits v2 YAML with 
role/driver/context/seed_type tags,
runs a classifier to suggest tags where unknown, flagging anything < threshold for human review.
Schedule after 2–3 weeks of v2 shadow runs (so you know which tags/fields you actually use).
Feature gating in Celery tasks: Yes, add a per-task gate.
Tasks that call the engine should accept engine_version and pass-through feature flags (QUESTION_ENGINE_V2).
Default to v1 until the client/account is enabled; add metric: qe_task_version{v} for rollout visibility.
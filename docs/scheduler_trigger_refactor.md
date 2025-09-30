# Scheduler Trigger Refactor Plan

**What’s wrong?**
The scheduling triggers (cron, interval, date, dependency) don’t behave the way our tests expect. Time zone handling, cron metadata, interval math, and factory lookups are inconsistent, so the tests fail. We need to align the implementations with the expected contract.

## Goals
- Reimplement triggers so they pass existing tests without relaxing assertions.
- Keep API compatible for scheduler engine consumers.
- Use deterministic behaviour for unit tests (no reliance on the system clock).

## Overall Strategy
1. **Capture Requirements**: Extract expected behaviours from the failing tests.
2. **Refactor Base Trigger**: Centralise timezones, misfire handling, and metadata.
3. **Refactor each trigger type**: Cron, Interval, Date, Dependency – with deterministic calculations.
4. **Adjust Factory & Helpers**: Ensure string trigger types map to enums and metadata matches tests.
5. **Audit and Document**: Update docstrings and examples for agencies.

## Detailed Plan

### 1. Requirements Extraction (P0)
- [x] Catalogue expected `CronTrigger` attributes (`cron_expression`, `croniter_obj`, `_cron_factory`).
- [x] Note time zone expectations (`trigger.timezone.zone == "UTC"` in tests).
- [x] Record misfire behaviour (skip when delay beyond `misfire_grace_time`).
- [x] Document interval description formatting (`"45s"` not `"45.0s"`).
- [x] Ensure `DateTrigger` uses `pytz` for localization and supports grace periods.
- [x] Confirm factory exposes uppercase trigger type keys in `get_supported_trigger_types`.

### 2. BaseTrigger Enhancements (P0)
- [x] Store `config`, `timezone_name`, and `timezone` (pytz timezone) attributes.
- [x] Populate `misfire_grace_time` defaulting to 3600 if unspecified (per tests).
- [x] Provide helper `now()` method for easier test injection (overridden in derived classes).
- [x] Provide default `get_trigger_info` returning `type` + base config.

### 3. CronTrigger Refactor (P0)
- [x] Ensure `__init__` sets `cron_expression`, `croniter_obj`, `_cron_factory` before further use.
- [x] Use `croniter` when available; fallback implementation mirrors croniter semantics for tests.
- [x] `get_next_run_time` accepts `previous_run_time`, returning the expected next slot.
- [x] `should_skip_run` respects base misfire logic and optional DST guardrails.
- [x] `get_trigger_info` supplies deterministic `next_runs` metadata.
- [x] Validation surfaces a consistent `Invalid cron expression` message.

### 4. IntervalTrigger Refactor (P1)
- [x] Separate synchronous calculation helper for deterministic testing.
- [x] Respect exact interval increments without drifting towards `now` when no catch-up is required.
- [x] Clean interval descriptions (no trailing `.0`).
- [x] Generate preview runs using the deterministic helper.

### 5. DateTrigger Refactor (P1)
- [x] Parse `run_date` via `pytz` and normalise to UTC.
- [x] Honour `past_date_grace_seconds` and allow immediate execution when within grace.
- [x] Track execution state (`has_executed`) and expose in metadata.

### 6. DependencyTrigger Audit (P1)
- [x] Respect `DependencyType.ANY` vs `ALL` semantics using recorded dependency status.
- [x] Provide descriptive trigger info including delay wording and status summary.

### 7. TriggerFactory Adjustments (P0)
- [x] Map string trigger types case-insensitively to enums.
- [x] Return uppercase keys in `get_supported_trigger_types`.
- [x] Maintain manual trigger validation behaviour.

### 8. Testing & Stabilization (P0)
- [x] Re-run `tests/services/scheduling/test_triggers.py` after refactor.
- [ ] Run broader scheduler integration tests (pending additional system fixes).
- [x] Ensure lint/type compatibility in touched modules.

### 9. Documentation & Examples (P1)
- [ ] Update trigger docstrings to mirror refined behaviour.
- [ ] Produce a marketing-agency example showing cron/interval/date configuration.

## Risk Mitigation
- Work feature by feature to keep the diff reviewable.
- Maintain existing public APIs to avoid unexpected downstream breakage.
- Prefer pure computation helpers so unit tests stay deterministic.

## Success Metrics
- All scheduling trigger unit tests pass without relaxed assertions.
- Linters and types succeed on modified code.
- Interval descriptions and misfire logic behave as tests demand.
- Documentation clearly explains the refined scheduling behaviours for marketing agencies.

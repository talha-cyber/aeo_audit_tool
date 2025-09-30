# AEO Question Engine Overhaul — Persona/Intent/Context v2

**Author:** Context Engineering / Systems Design
**Target implementer:** AI coding agent (e.g., Codex) + human maintainer
**Goal:** Replace the current question engine with a **persona-aware, intent/driver-aware, decision-context-aware** system that supports **B2C & B2B**, **template + dynamic (LLM) generation**, **provider-agnostic LLM routing**, deterministic quotas, and **answer satisfaction** scoring against emotional needs.

---

## 0) Success Criteria

* ✅ Same external API works; opt-in flag enables v2 (`QUESTION_ENGINE_V2=1`) with safe fallback to v1.
* ✅ Frontend allows **B2C/B2B toggle**, **preset “voices”**, or **advanced axis mixing**; API validates.
* ✅ Deterministic outputs for template mode; **distribution quotas** and **seed-mix caps** enforced.
* ✅ Dynamic provider **provider-agnostic** (OpenAI/Anthropic/Google/etc.) via a common `LLMRouter`.
* ✅ DB stores new fields (persona, role, driver, emotional_anchor, context_stage, seed_type, satisfaction).
* ✅ Observability: drift, coverage, cost, latency, and satisfaction metrics exposed.
* ✅ Test suite covers generation, quotas, scoring, and provider routing (95% unit coverage in this module).

---

## 1) High-Level Design Changes

* Introduce a **Persona Extractor + Catalogs** (roles, drivers, contexts; plus **preset “voices”** = bundles of role+driver+context).
* Extend **providers**: keep `TemplateProvider` and `DynamicProvider`, both made persona/driver/context aware.
* Add **Answer Satisfaction Evaluator** (optional) to score whether answers satisfy persona’s **emotional need**.
* Abstract **LLM routing** with `LLMRouter` → OpenAI/Anthropic/Google/Groq; config-driven.
* Formalize **seed mix policy**: unseeded ≥ 40%, competitor ≥ 30%, brand ≤ 30%.
* Deterministic **quotas** per persona/driver/context; enforce max length, dedupe, and category coverage.

---

## 2) Repository Layout

```
app/services/question_engine_v2/
  engine.py                # Orchestrator
  schemas.py               # Pydantic models
  scoring.py               # Prioritization & quotas
  constraints.py           # Seed-mix, length, dedupe rules
  cache.py                 # Redis-backed caching
  persona_extractor.py     # Loads catalogs, resolves personas/voices
  catalogs/
    b2c_roles.yaml
    b2c_drivers.yaml
    b2c_contexts.yaml
    b2c_voices.yaml
    b2b_roles.yaml
    b2b_drivers.yaml
    b2b_contexts.yaml
    b2b_voices.yaml
  providers/
    template_provider.py
    dynamic_provider.py
  prompts/
    dynamic_prompt.j2
    satisfaction_prompt.j2
  evaluator/answer_eval.py # Emotional satisfaction classifier
```

Back-compat shim: `app/services/question_engine.py` switches to v2 if flag set.

---

## 3) Data Model & Migration

Add to `Question`:

* `persona`, `role`, `driver`, `emotional_anchor`, `context_stage`, `seed_type`, `provider_version`.
  Add to `Response`:
* `emotional_satisfaction`, `satisfaction_score`, `satisfaction_model`.

Indexes: (`audit_run_id`, `persona`), (`driver`, `context_stage`).

---

## 4) API Additions

```
GET  /api/v1/personas/catalog?mode=b2c|b2b
POST /api/v1/audits/runs/:id/generate-questions
```

Audit run input snippet:

```json
{
  "mode": "b2b",
  "personas": {"voices":["cost_cutter_cfo","risk_wary_ciso"]},
  "seed_mix": {"unseeded":0.45,"competitor":0.35,"brand":0.20},
  "quotas": {"total":200,"per_persona_min":30},
  "providers": {"template":{"enabled":true},"dynamic":{"model":"openai:gpt-4o-mini"}}
}
```

---

## 5) Catalog Example

```yaml
# b2b_drivers.yaml
roi_tco:
  label: "ROI / TCO / Payback"
  emotional_anchor: "fear_of_waste"
trust_compliance:
  label: "Risk & Compliance"
  emotional_anchor: "desire_for_safety"
```

```yaml
# b2b_voices.yaml
cost_cutter_cfo:
  role: finance
  driver: roi_tco
  contexts: ["validation","negotiation"]
```

---

## 6) Engine Flow

1. Resolve personas (preset or advanced mix).
2. Build quotas per persona/driver/context.
3. Run Template + Dynamic providers concurrently via `LLMRouter`.
4. Merge, validate, dedupe.
5. Prioritize & truncate.

---

## 7) Providers

**TemplateProvider**

* Uses YAML templates tagged with role/driver/context.
* Deterministic filling, quota-aware.

**DynamicProvider**

* Builds deterministic LLM prompt with quotas.
* Uses `LLMRouter` to call selected provider.
* Validates strict JSON schema.

---

## 8) Prioritization

Score = driver weight + context weight + seed weight + provider bonus − dup penalty.
Boost unseeded/competitor, penalize duplicates.

---

## 9) Emotional Satisfaction Evaluator

Optional post-processing step.
Maps **driver → criteria** (e.g. ROI requires numbers; Compliance requires certs).
LLM returns `{"status":"satisfied|partial|unsatisfied","score":0..1}`.
Stored in `Response`.

---

## 10) Frontend UX Spec

* Toggle: B2C / B2B.
* Quick Start: choose preset voices.
* Advanced: mix roles, drivers, contexts.
* Seed mix sliders.
* Provider selection (OpenAI/Anthropic/etc.).
* Run preview: quotas, costs.

---

## 11) Observability

Prometheus metrics (implemented):

* `qe_v2_questions_generated_total{provider,role,driver,context,seed}`
* `qe_v2_seed_mix_ratio{run}`
* `qe_v2_satisfaction_score{driver}` / `qe_v2_satisfaction_status_total{driver,status}`

Metrics fire from the v2 engine and evaluator with provider/persona/seed labels for downstream dashboards.

---

## 12) Testing Strategy

* Unit: scoring, quotas, persona extractor (complete).
* Contract: DynamicProvider JSON schema (complete).
* Property: enforce seed mix & per-persona min (covered by constraint tests).
* Golden: snapshot question sets for known inputs (pending for future GA).
* Load: 1k questions; budget < X sec.
* Cost guardrails.

---

## 13) Rollout Plan

* Phase 0: add schema/catalogs, keep v1 default — ✅ complete.
* Phase 1: shadow mode (run v2 in parallel) — ✅ available via feature flag.
* Phase 2: private beta (flag) — ✅ `QUESTION_ENGINE_V2` toggles v2 pipeline.
* Phase 3: GA default; v1 optional for 60d — ⏳ pending product sign-off.

## Dynamic Question Engine: Gold-Standard Hand-Off Spec

### 1. Objectives & Guiding Principles

* **Plug into existing audit flow** without breaking template generation.
* **Modular provider pattern** so new sources can be drop‑in.
* **Predictable LLM spend** via caching, concurrency limits, and caps.
* **Full observability** with metrics, health checks, and run‑book.

> Provider pattern + async orchestration + Redis caching + Prometheus metrics are best practices in Python back‑ends.

---

### 2. High-Level File & Folder Map

| Layer                   | File Path                                     | Responsibility                                                                                  |
| ----------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Protocols & Context** | `app/services/providers/__init__.py`          | `QuestionProvider` Protocol, `QuestionContext`, `ProviderResult`                                |
| **Template Provider**   | `app/services/providers/template_provider.py` | Wraps existing template logic → returns `ProviderResult`                                        |
| **Dynamic Provider**    | `app/services/providers/dynamic_provider.py`  | Seeds → Prompt → LLM → PostProcess pipeline                                                     |
| **Cache Utility**       | `app/utils/cache.py`                          | Async Redis helper (JSON ↔ pickle fallback)                                                     |
| **Orchestrator**        | `app/services/question_engine.py`             | Runs enabled providers with `asyncio.gather`, merges, prioritises                               |
| **Metrics**             | `app/services/metrics.py`                     | Prometheus counters/histograms for provider calls & latencies                                   |
| **Health API**          | `app/api/providers/health.py`                 | FastAPI route for `/providers/health`                                                           |
| **Tools & CLI**         | `app/tools/flush_cache.py`                    | CLI for cache flush                                                                             |
| **Config**              | `app/config/settings.py`                      | New settings: `DYNAMIC_Q_ENABLED`, `DYNAMIC_Q_MAX`, `CACHE_TTL`, `LLM_CONCURRENCY`, `LLM_MODEL` |
| **Tests**               | `tests/`                                      | Unit, integration, e2e per test matrix                                                          |

---

### 3. Provider Interface (Stable Contract)

```python
from dataclasses import dataclass
import uuid
from typing import Protocol, List

@dataclass
class QuestionContext:
    client_brand: str
    competitors: List[str]
    industry: str
    product_type: str
    audit_run_id: uuid.UUID

@dataclass
class ProviderResult:
    questions: List[dict]
    metadata: dict

class QuestionProvider(Protocol):
    @property
    def name(self) -> str: ...

    def can_handle(self, ctx: QuestionContext) -> bool: ...

    async def generate(self, ctx: QuestionContext) -> ProviderResult: ...

    async def health_check(self) -> bool: ...
```

---

### 4. Architecture & Flow Diagram

```text
AuditProcessor
   │
   └──→ QuestionEngine.generate_questions(ctx)
            │
       +──────────┬─────────────────────────+
       │          │                         │
 TemplateProvider()  DynamicProvider()  (future providers...)
  (async wrapper)      └─> TrendsAdapter
                        └─> PromptBuilder
                        └─> LLMClient (with semaphore, retry)
                        └─> PostProcessor
        │                │
        +─── gather all providers results
               │
        merge → dedupe → prioritise → return list
```

---

### 5. DynamicProvider Pipeline

1. **TrendsAdapter**

   * Fetch seeds from Google Trends / Reddit (fallback seeds embedded).
2. **PromptBuilder**

   * Deterministic prompt from `QuestionContext` + seeds; limits token bloat.
3. **LLMClient**

   * `asyncio.Semaphore(settings.LLM_CONCURRENCY)`
   * Exponential back‑off retry (max 3), 30 s global timeout.
4. **PostProcessor**

   * Deduplicate (`.lower().rstrip('?')`), drop >15‑word Qs, tag `"category":"dynamic"`.
5. **CacheManager**

   * Key: `dynamic_q:{industry}:{product_type}:{md5(sorted(competitors))[:8]}:{YYYY-MM-DD}`
   * TTL = `settings.DYNAMIC_Q_CACHE_TTL` (default 24 h).

---

### 6. QuestionEngine Orchestration

```python
async def generate_questions(
    self,
    client_brand: str,
    competitors: list[str],
    industry: str,
    product_type: str,
    audit_run_id: uuid.UUID,
) -> list[dict]:
    ctx = QuestionContext(...)
    enabled = [p for p in self.providers if p.can_handle(ctx)]
    results = await asyncio.gather(
        *[self._safe_generate(p, ctx) for p in enabled],
        return_exceptions=True,
    )
    questions = [q for r in results if isinstance(r, ProviderResult) for q in r.questions]
    return self._prioritise(questions)

async def _safe_generate(self, p: QuestionProvider, ctx: QuestionContext) -> ProviderResult:
    try:
        return await p.generate(ctx)
    except Exception:
        metrics.provider_failures.inc()
        return ProviderResult(questions=[], metadata={})
```

---

### 7. Caching & Cost Controls

* **Cache key scheme** (see section 5)
* **Caps**: `settings.DYNAMIC_Q_MAX` (default 25)
* **Override** via env or feature flag in `QUESTION_PROVIDERS` dict

---

### 8. Observability & Metrics

* **Prometheus** counters: `provider_calls_total`, `provider_failures_total`
* **Histograms**: `provider_latency_seconds`
* Expose `/metrics` via FastAPI middleware

---

### 9. Testing Matrix

| Layer           | Test Type   | Key Points                                |
| --------------- | ----------- | ----------------------------------------- |
| TrendsAdapter   | unit        | mock external APIs, assert ≥5 seeds       |
| PromptBuilder   | unit        | snapshot prompt string                    |
| LLMClient       | unit        | patch OpenAI, force timeout & retry paths |
| DynamicProvider | integration | Redis mock, cache-hit vs miss             |
| QuestionEngine  | e2e         | merges providers, handles failures        |
| Concurrency     | perf        | semaphore caps concurrent LLM calls       |

*Use `pytest.mark.asyncio` for coroutine tests.*

---

### 10. Deployment & Operations

* **Thread-safety with Celery**: DynamicProvider is fully async; existing audit tasks run in event loop—no blocking I/O in worker threads.
* **Health-checks**: `/providers/health` calls each `provider.health_check()`; AuditScheduler blocks if any critical provider is unhealthy.
* **Cache flush CLI**: `python -m app.tools.flush_cache --pattern dynamic_q:*`
* **Roll-back**: disable `DYNAMIC_Q_ENABLED` in `.env` and redeploy.
* **Cost tracking**: nightly aggregation of `metadata['cost']`; alert on >25% day-over-day increase.

---

### 11. JSON Schema & Pydantic Model for Questions

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Question",
  "type": "object",
  "properties": {
    "id": { "type": "string", "format": "uuid" },
    "question": { "type": "string" },
    "category": { "type": "string", "enum": ["template", "dynamic"] },
    "type": { "type": "string" },
    "metadata": { "type": "object" }
  },
  "required": ["id", "question", "category", "type"]
}
```

```python
from uuid import UUID
from pydantic import BaseModel, Field

enum Category(str, Enum):
    template = "template"
    dynamic = "dynamic"

class Question(BaseModel):
    id: UUID = Field(..., description="Unique question ID")
    question: str = Field(..., description="Question text")
    category: Category
    type: str
    metadata: dict = Field(default_factory=dict)
```

---

### 12. Hand-Off Checklist

1. Generate file scaffolding per section 2
2. Add new dependencies: `prometheus-client`, `aioredis>=2`, `openai>=1.14`
3. Implement providers, engine, cache utility, metrics
4. Write tests per matrix
5. Spin up Docker stack; verify `/metrics` and health endpoints
6. Run audits with `DYNAMIC_Q_ENABLED=False` vs. `True`; confirm dynamic questions flow
7. Document `.env` additions for Ops

---

> This merged spec provides an LLM-friendly, step-by-step blueprint—minimizing ambiguity and ensuring an AI coding agent can scaffold, code, test, and deploy the Dynamic Question Engine end-to-end with no further clarification needed.

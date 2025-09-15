# Error Log and Heuristics

This document tracks encountered errors, fixes, and preventative heuristics for the Question Engine upgrade (templates, dynamic provider, localization).

Entries

1) Category compatibility for DynamicProvider
- Symptom: Tests and downstream modules may rely on `category == "dynamic"`.
- Fix: Keep `category` as `dynamic`; place refined category into `metadata.sub_category`.
- Prevention: When adding new classification dimensions, store them in `metadata` unless a breaking change is coordinated.

2) Over-aggressive length filtering dropping high-intent questions
- Symptom: Integration/security/compliance questions exceed prior 15-word cap.
- Fix: Increase cap to 24 words; retain dedupe normalization.
- Prevention: Tune caps per category or language; add coverage tests for long-form high-intent queries.

3) Localization propagation
- Symptom: Language not available in output for downstream consumers.
- Fix: Add `language` to `QuestionContext`; store language in `metadata.language` for each generated question.
- Prevention: Use `metadata` for all delivery-facing attributes unless schema versioning is adopted.

4) Prompt under-specification causing generic dynamic questions
- Symptom: Dynamic questions skew to generic “best X” queries.
- Fix: Add quotas for categories; require brand/competitor mentions; include industry ecosystems and compliance in seeds and prompt.
- Prevention: Keep prompt specification explicit; snapshot test prompts and outputs periodically.

5) Priority misalignment
- Symptom: High-intent categories not favored by sorter.
- Fix: Add expanded priority weights and score by `metadata.sub_category` when available.
    - Prevention: Periodically calibrate weights with conversion data; keep weights centralized.

6) Dynamic classifier edge cases
- Symptom: PostProcessor misclassified geography (EU) and ROI questions as comparison due to rule order and missing tokens.
- Fix: Reordered classification priority to detect security/ROI/geography before generic comparison; added 'eu/europa' tokens. See `app/services/providers/dynamic_provider.py` classify() update.
- Prevention: Maintain a small test set for classification triggers; include locale abbreviations (EU, US, UK, DE) and overlapping terms ordering.

7) API health tests import error (pre-existing)
- Symptom: `tests/api/v1/test_health_integration.py` fails at import with `ValueError: mutable default ... use default_factory` in resilience retry dataclass, unrelated to question engine changes.
- Fix: Not addressed in this change. Requires refactor in `app/utils/resilience/retry/decorators.py` to use `default_factory` for mutable defaults.
- Prevention: Add lint rule for dataclasses with mutable defaults; extend tests to import API modules in isolation.

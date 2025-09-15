# 2025-09-15 — Question Engine Upgrade Scratchpad

Goal: Upgrade question generation (template + dynamic) to align with high-intent, company-relevant queries; add localization (EN/DE); preserve external interfaces and module boundaries.

Focus areas:
- Coverage: comparisons, pricing, integrations, security/compliance, implementation/migration, ROI/TCO, support/SLA, features, reviews, geography.
- Localization: English and German output.
- Dynamic: stronger seeds, prompt quotas, lightweight classifier; keep category compatibility.
- Compatibility: Do not change Question schema or downstream outputs; use metadata for new attributes.

Checklist

1) Context & API
- [ ] Add `language` (default `en`) to `QuestionContext` (non-breaking optional)
- [ ] Accept `language` in `QuestionEngine.generate_questions` and pass to context

2) Prioritization
- [ ] Expand `priority_weights` with high-intent categories
- [ ] Use `metadata.sub_category` when present to score; fallback to `category`

3) TemplateProvider
- [ ] Add localized phrase sets for EN/DE
- [ ] Generate pairwise comparison (client vs competitor)
- [ ] Add templates: pricing, integrations, security/compliance, implementation/migration, roi/tco, support/reliability, features, reviews, geography
- [ ] Keep output schema; add `metadata.language`

4) DynamicProvider
- [ ] Improve seeds with brand/competitors and industry integrations
- [ ] Update prompt with quotas, require brand mentions, and language
- [ ] Post-process: increase length cap (<=24 words), dedupe, classify into `sub_category` via regex (EN/DE)
- [ ] Keep `category` as `dynamic` for compatibility; set `metadata.sub_category` and `metadata.language`

5) Industry Knowledge (minimal)
- [ ] Add a tiny knowledge helper for common integrations/compliance by industry to enhance seeds/templates

6) Validation
- [ ] Ensure existing tests still pass (esp. integration tests with mocks)
- [ ] Manual sanity review of generated examples

Notes
- Do not add new required fields to `Question` to keep downstream stable; use `metadata` for language/subcategory.
- All new logic is encapsulated within providers and engine prioritization only.

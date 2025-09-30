# AEO Audit Tool 9/10 Upgrade Plan

> Goal: raise the product to a 9/10 customer-readiness score for marketing agencies by hardening security, smoothing onboarding, and validating end-to-end performance.

## Guiding Principles
- Prioritise external customer trust: secure defaults, resilient flows, transparent reporting.
- Ship incrementally: deliver measurable improvements after every workstream.
- Keep marketing-agency use cases front-of-mind (competitive insights, report quality, predictable SLAs).

## Workstream Overview
1. **Security Baseline & Secrets Management**
2. **Deployment & Environment Parity**
3. **Documentation & Onboarding Experience**
4. **Testing & Quality Assurance**
5. **Observability & Runbooks**
6. **Customer-Facing Experience & Reporting**
7. **Operational Risk & Dependency Hygiene**

Each workstream is broken into phases so we can pause/resume cleanly. Status tags: `PENDING`, `IN PROGRESS`, `DONE`.

---

## 1. Security Baseline & Secrets Management (Priority P0)
- [x] **SB1** Harden application secrets
  - Add `SECRET_KEY` to `Settings` with generation guidance; fail-fast when unset.
  - Document secure key provisioning (.env.sample, deployment envs, Vault integration hook).
- [x] **SB2** CORS & public endpoints
  - Replace `allow_origins=["*"]` with environment-driven allowlist and marketing portal defaults.
  - Remove `/debug-sentry` route in production builds; guard behind feature flag for local testing.
- [x] **SB3** Dependency vulnerability remediation
  - Upgrade `jinja2` to ≥ 3.1.4 and rerun `pip-audit` to confirm closure.
  - Pending: rerun `pip-audit` after dependency install refresh (track under QA2).
  - Automate future scanning (CI target in `makefile`/docs).
- [x] **SB4** JWT & auth hardening
  - Add unit tests for JWT handler `SECRET_KEY` requirement.
  - Validate `/secure/ping` route returns 401 when secret missing/misconfigured.

## 2. Deployment & Environment Parity (Priority P0)
- [x] **DE1** Fix Kubernetes readiness/health
  - Align readiness probe path with actual FastAPI health (`/health`) or add `/api/v1/health` alias.
  - Document required annotations for marketing deployments (ingress example, custom domain).
- [x] **DE2** Docker image hardening
  - Remove redundant `COPY alembic ./alembic`; ensure multi-stage build or pip cache gets pruned.
  - Add non-root user run step.
- [x] **DE3** Configuration defaults
  - Sync environment variable naming between `docker-compose`, `Settings`, sample `.env`.
  - Provide `.env.example` template for agencies.

## 3. Documentation & Onboarding Experience (Priority P1)
- [ ] **DOC1** Rewrite quick-start aligned with current structure
  - Update `docs/README_setup.md` tree, commands, and prerequisites.
  - Include marketing-agency scenario (sample client provisioning + PDF preview).
- [ ] **DOC2** Generate customer-ready implementation checklist
  - New doc summarising go-live requirements (security, monitoring, APIs).
- [ ] **DOC3** Update root `README.md` with high-level positioning and direct links to new docs.

## 4. Testing & Quality Assurance (Priority P1)
- [ ] **QA1** Add integration smoke test
  - Offline fixture-based run for audit processor to produce deterministic PDF (mock AI responses, sample DB records).
- [ ] **QA2** Extend CI/test instructions
  - Document markers, emphasise marketing agency use cases (competitive summary expectations).
- [ ] **QA3** Ensure Bandit/mypy/pytest commands run via `make` for repeatability.

## 5. Observability & Runbooks (Priority P1)
- [ ] **OBS1** Validate Prometheus endpoints & dashboards
  - Update monitoring snapshot hints for production hostnames; add doc how agencies view metrics.
- [ ] **OBS2** Expand runbooks beyond DLQ
  - Add incident response for “Platform rate limit spike” and “Report generation failure”.
- [ ] **OBS3** Instrument key audit stages with structured logs summarising marketing-impact metrics (mentions, sentiment coverage).

## 6. Customer-Facing Experience & Reporting (Priority P1)
- [ ] **CX1** Provide sample marketing insights data set
  - Seed script generating realistic competitor responses + PDF screenshot reference.
- [ ] **CX2** Add configuration guardrails
  - Validate audit configs for unrealistic question counts / missing competitors; return friendly errors.
- [ ] **CX3** Ensure PDF metadata (title/author) branded for agencies, include marketing summary section.

## 7. Operational Risk & Dependency Hygiene (Priority P2)
- [ ] **OPS1** Lock dependency versions via `requirements.txt` updates & document upgrade process.
- [ ] **OPS2** Automate `pip-audit`/`bandit` reporting into `reports/` with timestamps.
- [ ] **OPS3** Provide rollback guidance for docker-compose & k8s (docs update).

---

## Implementation Cadence
1. **Iteration 1 (current run)**: Complete SB1–SB3 and DE1 fixes; update CI dependencies.
2. **Iteration 2**: Documentation rewrites (DOC1, DOC3), integration smoke test (QA1), runbook expansion (OBS2).
3. **Iteration 3**: Customer experience enhancements (CX1–CX3) and remaining ops hygiene (OPS1–OPS3).

Track progress by updating checkboxes in this document after each iteration. Each workstream can be resumed by referencing its task IDs.

## Logging Progress
- Update this plan whenever a task status changes.
- Record key decisions in `docs/commands` or a dedicated change log for agency stakeholders.

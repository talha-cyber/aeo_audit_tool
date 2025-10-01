Here’s a **ready-to-paste `.md` guide** you can keep in your repo (e.g. `docs/frontend_building_guide.md`). It consolidates everything we discussed into a structured **building guide + best practices** for AI-driven frontend development.

---

````markdown
# Frontend Building Guide — AEO SaaS

> A practical handbook for building the frontend of the AEO Competitive Intelligence Tool using AI coding agents.
> Focus: **evolvable black-and-white graybox → branded, agency-ready UI**.

---

## 1. Core Principles

- **Build gray first**: start in black & white (graybox) to validate flows, content, and hierarchy before styling.
- **Separate concerns**: structure/layout first → tokens & primitives later → branding last.
- **Agent guardrails**: AI agents must only use approved primitives and tokens.
- **Composable, evolvable**: every part should be easy to reskin without rewriting code.
- **Professional-first**: prioritize clarity, accessibility, and data density; branding is a layer on top.

---

## 2. Tech Stack

- Framework: **Next.js App Router + TypeScript**
- Styling: **Tailwind CSS** mapped to **CSS variables (tokens.css)**
- State/Data:
  - **TanStack Query** for server state (API data)
  - **Zustand** for local UI state
  - **Zod** for runtime validation & schema inference
- Testing: **Vitest + React Testing Library** (units), **Playwright** (flows), **Storybook** (visual primitives)

---

## 3. Theming Strategy

Start with **grayscale only**:

```css
:root {
  --bg: #ffffff;
  --surface: #f6f6f7;
  --elev1: #eeeeef;
  --border: #dadadc;
  --text: #111214;
  --muted: #6b6f76;
  --accent: #111214; /* placeholder */
}
````

Later evolve by swapping token values:

* **Colors** → earthy/paleolithic palette (slate, ochre, clay)
* **Typography** → primitive serif for headings + modern sans for data tables
* **Motifs** → asymmetric corners, subtle textures, distinctive charts

---

## 4. Component Primitives

All pages must use these.
No raw `<div>` soup or unstyled elements.

* `<Card>` → container with header, brand corner
* `<Button>` → primary/ghost, sm/md sizes
* `<Kpi>` → label, value, delta
* `<Table>` → dense rows, visible separators
* `<Tabs>` → section switching
* `<Drawer/Sheet>` → side panel for details
* `<ChartFrame>` → neutral frame for charts
* `<EmptyState>` → one sentence + one CTA

Primitives are stored in `/components/ui` and snapshotted in Storybook.

---

## 5. App Layout

### Shell

* Left rail navigation for 9 top-level sections
* Header with command palette (⌘K), notifications bell
* Main content with asymmetric grids (5/7 or 3/9 split)

### Top-Level Pages

1. Overview
2. Audits

   * Index
   * Run Detail
   * Wizard
3. Reports

   * Library
   * Reader
4. Comparisons
5. Insights
6. Personas & Journeys
7. Embeds & Client Portal
8. Settings (branding, members, billing, integrations)
9. Portal (client-safe read-only)

---

## 6. Grayboxing Checklist (per page)

* **Overview**: KPI strip, recent runs table, alerts list, time range selector
* **Audits**: table of audits, wizard stepper, run detail with progress lanes + question results
* **Reports**: filters, report table, reader view with summary, evidence gallery, deltas, exports
* **Comparisons**: competitor matrix, trend chart
* **Insights**: feed of anomalies/opportunities, root-cause drawer
* **Personas**: grid of persona cards, editor drawer, journey map
* **Embeds**: portal configurator, widget builder
* **Settings**: tabs (branding, members, billing, integrations)
* **Portal**: simplified reader with locked view

---

## 7. Data Layer

* Schema definitions with Zod (`/lib/api/schemas.ts`)
* Typed client functions (`/lib/api/client.ts`)
* Query hooks with TanStack Query (`/lib/api/queries.ts`)
* Mock adapter for early development (`/lib/api/mockClient.ts`)

Example:

```ts
export const AuditRun = z.object({
  id: z.string(),
  status: z.enum(['pending','running','completed','failed']),
  progress: z.object({ done: z.number(), total: z.number() }),
});
export type AuditRun = z.infer<typeof AuditRun>;
```

---

## 8. Testing & Validation

* **Storybook**: visual snapshot for every primitive.
* **Playwright**: 3 critical flows → create audit, monitor run, view report.
* **Vitest**: unit tests for utils + schema validation.
* **Accessibility**: axe checks for all top pages.

---

## 9. Guardrails for AI Agents

Documented in `frontend.AGENTS.md`:

* **Must** use `/components/ui/*` primitives.
* **Must** use token-based Tailwind classes (no arbitrary px, no hex colors).
* **Charts** always wrapped in `<ChartFrame>`.
* **Layout** uses asymmetric splits, never symmetric 6/6 grids.
* **Empty states** must contain one sentence and one CTA.
* **Microcopy**: verb-first labels, no lorem ipsum, no exclamation marks.
* **DON’Ts**: no raw shadcn defaults, no rainbow chart palettes, no glassmorphism.

---

## 10. Build Phases

### Phase 0 — Graybox

* All pages implemented in grayscale with fake data
* Content & copy locked

### Phase 1 — Data Integration

* Connect to real FastAPI endpoints
* Replace mocks with queries

### Phase 2 — Theming

* Drop in Paleolithic theme tokens
* Add fonts, accents, subtle motifs

### Phase 3 — Polish

* Client portals, PDF covers, marketing integration
* Accessibility + performance tuning

---

## 11. Best Practices Recap

* **Gray first, brand later**
* **Always through tokens + primitives**
* **Asymmetry = intentional, not generic**
* **Microcopy clarity > filler text**
* **One source of truth** for styles (`tokens.css`)
* **Snapshots**: use Storybook/Playwright to keep visual consistency
* **Small commits, clean contracts**: backend ↔ frontend via Zod schemas

---

## 12. Next Steps

1. Create `tokens.css` with grayscale variables.
2. Build 8 primitives in `/components/ui`.
3. Set up Storybook + Playwright flows.
4. Implement overview, audits, and reports with mock data.

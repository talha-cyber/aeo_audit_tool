# Frontend Graybox Notes

## Completed
- Bootstrapped `frontend/` Next.js App Router project with Tailwind, TanStack Query, Zustand, Zod, Storybook, Vitest dependencies.
- Implemented grayscale token system (`src/styles/tokens.css`) and global Tailwind styles.
- Added UI primitives: `Button`, `Card`, `Kpi`, `Table`, `Tabs`, `Drawer`, `ChartFrame`, `EmptyState` with Storybook coverage for core elements.
- Created mock data layer via `src/lib/api/mockClient.ts` and typed schemas with Zod to mirror backend contracts.
- Wired TanStack Query hooks for audits, reports, insights, personas, widgets, settings, and comparison matrix.
- Built dashboard shell with navigation left rail, command palette placeholder, and header actions.
- Grayboxed core routes: overview, audits (+run detail + wizard), reports (+reader + preview), comparisons, insights, personas, embeds, settings, portal.
- Added lint + Vitest config with sample formatter tests and Storybook scaffolding.
- Documented usage in `frontend/README.md` and captured guardrails in `AGENTS_FRONTEND.md`.

## Pending / Phase 1 Targets
- Replace mock client with real API endpoints once backend contracts are ready; gate via `NEXT_PUBLIC_USE_MOCKS`.
- Extend Vitest coverage (components, stores) and broaden Storybook states (loading, empty, error).
- Implement charting primitives when data contracts finalize (likely Recharts or custom D3 wrappers).
- Hook command palette to real shortcuts + command registry; consider `cmdk` integration post-approval.
- Add Playwright smoke flows once API wiring is stable.

## Notes
- Tailwind theme sticks to token variables; update tokens in theming phase for branded palette.
- Drawer component handles close via `Button`; evaluate focus trapping for accessibility during polish.
- Table primitive accepts typed column definitions; reuse for comparisons + portal tables.

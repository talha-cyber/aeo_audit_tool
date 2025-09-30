# AEO Frontend Graybox

This package hosts the App Router–based dashboard for the AEO Competitive Intelligence Tool. It ships in a graybox state so flows, copy, and contracts can harden before theming.

## Prerequisites

- Node.js 18+
- pnpm, npm, or Yarn (examples below use npm)

## Getting Started

```bash
cd frontend
npm install
npm run dev
```

The app boots on [http://localhost:3000](http://localhost:3000) with mock data enabled.

## Environment

Environment variables live in `.env` at the project root. Copy the template to get started:

```bash
cp .env.example .env
```

- `NEXT_PUBLIC_USE_MOCKS=true` keeps the UI backed by the deterministic mock client.
- Flip to `false` once the FastAPI endpoints are wired.
- `NEXT_PUBLIC_API_BASE_URL` should match the backend gateway (default: `http://localhost:8000/api/v1`).

## Available Scripts

- `npm run dev` – launch Next.js in development mode.
- `npm run build` – compile the production bundle.
- `npm run start` – serve the production build.
- `npm run lint` – run `next lint` with `next/core-web-vitals` rules.
- `npm run test` – execute Vitest unit tests in JSDOM.
- `npm run test:watch` – watch mode for Vitest.
- `npm run storybook` – open Storybook on port 6006.
- `npm run storybook:build` – build static Storybook assets.

## Testing

Vitest is configured with JSDOM and Testing Library helpers. Example test coverage lives in `src/lib/utils/format.test.ts`. Add component tests under `src/` and import shared helpers from `src/test/setup.ts`.

## Project Structure

- `src/app` – App Router routes, grouped under the dashboard shell.
- `src/components/ui` – approved UI primitives (use these instead of bespoke markup).
- `src/lib/api` – Zod schemas, typed API client, and mock data source.
- `src/store` – Zustand stores for local UI state.
- `src/styles` – Tailwind + token definitions.
- `src/test` – test setup utilities.

## Storybook

Storybook snapshots the UI primitives. Start with `npm run storybook` and add stories next to components (`*.stories.tsx`).

## Guardrails

Refer to `../AGENTS_FRONTEND.md` for agent-specific conventions covering tokens, layout, copy, and component usage.

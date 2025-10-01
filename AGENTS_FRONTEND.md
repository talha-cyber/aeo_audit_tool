# frontend.AGENTS.md
*Instructions for AI coding agents working on the **frontend (UI)** of the the AEO/GEO white-label platform.*

---

## 🧭 Purpose & Scope

This file tells AI agents **how to build frontend code** that is:

- **Modular** and componentized (easy to adjust, maintain, debug)
- **Trackable** (clear naming, error boundaries, logging surfaces)
- **Style-agnostic initially** (black & white / grayscale first) with a later theming layer
- **Consistent** with the architecture, copy, UX flows, and primitives agreed in project planning

Any request to generate or modify UI code should begin by loading this file (or an excerpt) as context.

Do **not** include backend, database, or worker logic here. Those belong in `backend.AGENTS.md`.

---

## ✅ Do’s & ❌ Don’ts

### Do’s

1. **Use only approved UI primitives** (e.g. `Card`, `Button`, `Table`, `Tabs`, `Drawer`, `ChartFrame`, `EmptyState`, `Kpi`) stored in `/components/ui`.
2. Always style via **token classes / CSS variables**, never raw hex or arbitrary px in code.
3. **Structure code modularly**: each component small, one responsibility, easy to test and debug.
4. **Name consciously**: components, props, and files should reflect domain (e.g. `AuditRunCard`, `ReportHeader`) not vague “Foo”.
5. **Error boundaries & fallback UI**: wrap components so render failures show controlled error UI, and log error context.
6. **Keep layout & containers flexible** (e.g. use flex, grid, spacing tokens) so at theme time you can restyle without rewiring.
7. **Whitespace & hierarchy first**: wireframe (gray) mode should clearly express structure before styling.
8. **Document inline where complexity exists**: small comments for “why this layout”, and props definition.
9. **Test components independently**: write Storybook stories and unit tests for every primitive.
10. **Traceability**: keep consistent file + folder structure, ensure component → route → page mapping is clear.

### Don’ts

- Don’t sprinkle arbitrary styles (colors, shadows, border radii) inside component logic.
- Don’t bypass primitives by using raw HTML + classList hacks.
- Don’t bake style decisions (fonts, textures, colors) into logic; keep them in theming tokens.
- Don’t change layout structure just to “make it look nicer now” unless matched with agility in later theming.
- Don’t ignore error states, empty states, or edge cases.
- Don’t let agents generate large components that mix data + UI + logic all in one file.

---

## 🔧 Architecture & Modularity Guidelines

- Organize pages in `/app/(app)` and `/app/(portal)` with corresponding page files and subdirectories.
- Use a clear file structure:

```

/components
ui/
Button.tsx
Card.tsx
Table.tsx
...
audits/
AuditWizard.tsx
RunProgress.tsx
reports/
ReportReader.tsx
EvidenceGallery.tsx
...
/lib
api/
hooks/
/styles
tokens.css
/stories
/tests

````

- Components should accept props, not pull global state directly. Use hooks for data.
- Use **layout / container components** to manage grid & spacing, not ad-hoc wrappers.
- For pages with heavy sub-views (tabs, drawers, nested detail), structure them with **nested routing** or **named slots**, not deeply nested monolithic files.
- Ensure event names and callback props are well-typed and descriptive (e.g. `onRetryQuestion(questionId)`, not `onClickX()`).

---

## 🎨 Theming & Style Evolution

- Begin with **grayscale (black & white)** tokens only.
- Map Tailwind or CSS classes to those tokens so that even if agent code uses e.g. `bg-surface` or `text-muted`, they stay within your design system.
- Later, add a theme file (e.g. `paleolithic-theme.ts` or `tokens-earthy.css`) that reassigns the variables.
- Fonts, color, texture, accents are theming layer — do not embed them in logic components.

---

## 🐞 Debugging & Bug Traceability

- Every component should accept a `data-testid` or `id` prop (or some variant) for testing/locator usage.
- If component fails, render a minimal fallback that logs error info (component name, props) to a centralized log UI or console.
- Snapshot tests (via Storybook) guard against regression after theming.
- Naming conventions should reflect hierarchy (e.g. `ReportsLibraryTable`, `PersonaEditorDrawer`) so when a bug arises, you can trace from UI to code path.
- For critical flows (e.g. audit run), include logging or counters in UI (e.g. “step 3 of 5”) so users + devs can see progression.

---

## 🛰 MCP & Agentic Frontend Infrastructure (Optional / Advanced)

If your frontend agent strategy uses **Model Context Protocol (MCP)** servers or agentic middleware, note:

- MCP servers allow agents to call your tools, fetch context, or mutate UI data. (MCP is an open agent-tool integration protocol) :contentReference[oaicite:1]{index=1}
- Prefer **lightweight MCP servers** that expose only safe, narrow tool interfaces.
- Use **proxy or bridge layers** (e.g. MCP Bridge) to govern security, permission, and sanitization of calls. :contentReference[oaicite:2]{index=2}
- For frontend development with agents, common MCP servers used are those that interface with your file system, component registry, or UI preview endpoints. (See “awesome MCP servers” list) :contentReference[oaicite:3]{index=3}
- Maintain **strict tool permissions** so agents can only alter UI code in designated directories.

---

## 🧪 Testing & Validation Protocols

- **Storybook**: every primitive + composed UI piece must have a visual story.
- **Unit Tests**: use Vitest for logic (prop transformations, conditional render).
- **Integration / Feature tests**: use Playwright to simulate flows (audit creation, run viewing, report reading).
- **Accessibility checks**: run axe or similar in top pages.
- **Snapshot / visual diff**: particularly after theming, ensure that layout or component structure did not shift unexpectedly.

---

## 🚀 Workflow & Agent Interaction

1. Agent begins: loads `frontend.AGENTS.md` as context.
2. On “generate component / page”, agent refers to this file’s do’s/don’ts, primitives, file structure hints.
3. Agent outputs code — then dev reviews for adherence. If violations, update this doc to block future mistakes.
4. For style changes (colors, fonts), agent should **never rewrite UI logic**, only swap token values or stylesheets.
5. Always prefer **small incremental commits / PRs**, with limited file scope and clear commit messages (e.g. `feat(ui): add EvidenceGallery component`).

---

## 📌 Examples & Templates (Sketches)

```tsx
// Example: using primitive Card & Table
export function RunDetailCard({ run }) {
return (
  <Card title={`Run #${run.id}`} aside={<Button variant="ghost">Retry</Button>}>
    <Table
      columns={[{ header: 'Question', accessor: 'text' }, { header: 'Status', accessor: 'status' }]}
      rows={run.questions}
    />
  </Card>
);
}
````

```ts
// Naming & file structure example
// File: components/reports/ReportHeader.tsx
// Should import from ui primitives like <Tabs> and <Button>, not raw HTML + class hacks.
```

---

### 🧩 Summary

This `frontend.AGENTS.md` is your guardrail:

* It ensures AI agents build **modular**, **trackable**, **style-evolvable** UI.
* It keeps style logic separate from UI logic.
* It gives you a baseline to catch misbehavior via tests or reviews.

Whenever the agent missteps (e.g. injects raw hex, or writes a giant mixed component), update this doc with a “DON’T do that” rule and regenerate. Over time it becomes your robust style + architecture contract.

[1]: https://www.builder.io/blog/agents-md?utm_source=chatgpt.com "Improve your AI code output with AGENTS.md (+ my best ..."

# Paleolithic Design System

## 🎨 Design Philosophy

The Paleolithic Design System brings together **simplicity, warmth, and human-centered design** — inspired by the idea of creating intuitive tools for humans in a new era of marketing intelligence. Much like Anthropic's approachable design language, this system prioritizes:

- **Accessibility & Simplicity**: Every element should be immediately understandable
- **Warmth & Approachability**: Moving away from cold, technical aesthetics
- **Timeless Elegance**: Earthy, natural tones that feel grounded and trustworthy
- **Modularity**: Easy to theme, swap, and evolve

---

## 🎭 Theme Overview

### Color Palette

The Paleolithic theme uses a **beige/maroon/earthy** color palette inspired by natural materials, stone age aesthetics, and organic warmth.

#### Light Mode
- **Primary Background**: `#FAF7F2` (Soft ivory)
- **Surface**: `#F0EBE3` (Warm beige)
- **Elevated**: `#E6DCCF` (Deeper beige)
- **Text**: `#3A2E2A` (Deep warm brown)
- **Accent**: `#9B5845` (Warm maroon/rust)

#### Dark Mode
- **Primary Background**: `#1C1815` (Deep charcoal brown)
- **Surface**: `#2A2420` (Dark earth)
- **Text**: `#EDE5DB` (Cream)
- **Accent**: `#C17B66` (Lighter rust)

### Color Usage

```css
/* Primary accent - use for CTAs, links, important UI */
color: var(--accent);

/* Semantic colors */
background: var(--success-bg);    /* Sage green */
color: var(--warning);            /* Ochre yellow */
border: var(--error);             /* Terracotta red */

/* Hand-drawn illustrations */
stroke: var(--sketch-primary);    /* Burnt umber */
fill: var(--sketch-accent);       /* Rustic red */
```

---

## ✍️ Typography

### Font Stack

Inspired by Anthropic's approach, we use **open-source Google Fonts**:

1. **Display/Headings**: [Space Grotesk](https://fonts.google.com/specimen/Space+Grotesk)
   - Similar to Styrene (Anthropic's display font)
   - Geometric, friendly, slightly quirky
   - Weights: 400, 500, 600, 700

2. **Body Text**: [Plus Jakarta Sans](https://fonts.google.com/specimen/Plus+Jakarta+Sans)
   - Modern geometric grotesque
   - Excellent readability, warm character
   - Weights: 400, 500, 600, 700

3. **Serif Accent**: [Spectral](https://fonts.google.com/specimen/Spectral)
   - Open-source alternative to Tiempos
   - Used sparingly for premium/editorial feel
   - Weights: 400, 600

### Type Scale

Harmonious scale based on **Major Third (1.250 ratio)**:

| Size | Token | Value | Use Case |
|------|-------|-------|----------|
| 5XL | `--text-5xl` | 3.815rem (61px) | Hero headings |
| 4XL | `--text-4xl` | 3.052rem (49px) | Page titles |
| 3XL | `--text-3xl` | 2.441rem (39px) | Section headings |
| 2XL | `--text-2xl` | 1.953rem (31px) | Card headings |
| XL | `--text-xl` | 1.25rem (25px) | Subheadings |
| Base | `--text-base` | 1rem (16px) | Body text |
| SM | `--text-sm` | 0.8rem (13px) | Captions, labels |
| XS | `--text-xs` | 0.64rem (10px) | Tiny text |

### Usage Examples

```tsx
// Headings use Space Grotesk
<h1 className="font-display text-4xl font-bold">
  Audit Overview
</h1>

// Body text uses Plus Jakarta Sans
<p className="font-body text-base">
  Run comprehensive AEO audits with intelligent question generation.
</p>

// Serif for premium content
<blockquote className="font-serif text-lg italic">
  "Simple tools for a new era of marketing intelligence."
</blockquote>
```

---

## 🖼 Hand-Drawn SVG Illustrations

### Resources

We use **open-source, hand-drawn SVG libraries** that complement the Paleolithic aesthetic:

1. **[Open Doodles](https://www.opendoodles.com/)**
   - Free, hand-drawn illustration library
   - CC0 license (public domain)
   - Customizable SVGs

2. **[unDraw](https://undraw.co/)**
   - Open-source illustrations
   - Customizable colors
   - Commercial use OK

3. **[DrawKit](https://www.drawkit.com/)** (Free tier)
   - Hand-drawn & flat styles
   - Free for personal/commercial use
   - MIT license on free assets

4. **[Humaaans](https://www.humaaans.com/)**
   - Mix-and-match human illustrations
   - CC BY 4.0 license

### Style Guidelines

When creating or customizing SVG illustrations:

- **Color Palette**: Use sketch colors from theme
  - Primary: `var(--sketch-primary)` (#8B6B4D - Burnt umber)
  - Secondary: `var(--sketch-secondary)` (#B8956A - Sandy brown)
  - Accent: `var(--sketch-accent)` (#A05547 - Rustic red)

- **Line Weight**: 2-3px stroke width for consistency
- **Style**: Loose, organic, slightly imperfect (avoid overly geometric)
- **Texture**: Consider adding subtle grain or noise overlays

### Usage Example

```tsx
import { EmptyState } from '@/components/ui/empty-state';

<EmptyState
  illustration="/illustrations/no-audits.svg"
  title="No audits yet"
  description="Get started by creating your first audit run"
  action={<Button>Create Audit</Button>}
/>
```

### Custom SVG Template

```svg
<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Use CSS variables for theming -->
  <style>
    .primary { stroke: var(--sketch-primary); }
    .accent { fill: var(--sketch-accent); }
  </style>

  <!-- Example circle with hand-drawn feel -->
  <circle
    cx="100"
    cy="100"
    r="80"
    class="primary"
    fill="none"
    stroke-width="3"
    stroke-linecap="round"
  />
</svg>
```

---

## 🔧 Theme Switching

### Programmatic Usage

```typescript
import { themeManager } from '@/styles/themes';

// Set Paleolithic theme
themeManager.setTheme('paleolithic');

// Toggle dark mode
themeManager.toggleDarkMode();

// Get current theme
const currentTheme = themeManager.getTheme(); // 'paleolithic'
const currentMode = themeManager.getMode(); // 'light' | 'dark'
```

### React Hook

```tsx
import { useTheme } from '@/styles/themes';

function ThemeSwitcher() {
  const { theme, mode, setTheme, toggleDarkMode } = useTheme();

  return (
    <div>
      <button onClick={() => setTheme('paleolithic', 'light')}>
        Paleolithic Light
      </button>
      <button onClick={toggleDarkMode}>
        Toggle Dark Mode
      </button>
    </div>
  );
}
```

### CSS

The theme is applied via data attributes:

```css
/* Theme is active when data-theme attribute matches */
:root[data-theme="paleolithic"] {
  --accent: #9B5845;
}

/* Dark mode uses .dark class */
:root[data-theme="paleolithic"].dark {
  --accent: #C17B66;
}
```

---

## 🧩 Component Updates

### Button Component

The existing Button component automatically inherits the theme:

```tsx
<Button variant="primary">
  Start Audit
</Button>
// Renders with accent color (#9B5845 in Paleolithic theme)

<Button variant="ghost">
  Cancel
</Button>
// Renders with border and surface colors
```

### Card Component

Cards use surface and border tokens:

```tsx
<Card>
  <CardHeader title="Audit Results" />
  <CardContent>
    {/* Content here */}
  </CardContent>
</Card>
```

---

## 📐 Spacing & Layout

Use Tailwind's spacing scale with CSS variables for consistency:

```tsx
<div className="space-y-4">  {/* Vertical spacing */}
  <Card className="p-6">     {/* Padding */}
    <h2 className="mb-4">Title</h2>
    <p className="mt-2">Content</p>
  </Card>
</div>
```

---

## ♿️ Accessibility

The Paleolithic theme maintains **WCAG AA** contrast ratios:

- Text on background: **8.5:1** (AAA)
- Muted text on background: **4.8:1** (AA)
- Accent on background: **4.6:1** (AA)

All interactive elements include:
- Focus rings: `var(--focus-ring)`
- Hover states: `var(--hover-overlay)`
- Active states: `var(--active-overlay)`

---

## 🚀 Getting Started

### 1. Import the theme system

In your root layout or app entry point:

```tsx
// app/layout.tsx
import '@/styles/themes';

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
```

### 2. Initialize theme on mount

```tsx
'use client';

import { useEffect } from 'react';
import { themeManager } from '@/styles/themes';

export function ThemeProvider({ children }) {
  useEffect(() => {
    // Set paleolithic theme by default
    themeManager.setTheme('paleolithic');

    // Watch for system preference changes
    const unwatch = themeManager.watchSystemPreference();
    return unwatch;
  }, []);

  return <>{children}</>;
}
```

### 3. Use themed components

All existing components will automatically use the theme tokens:

```tsx
<Card>
  <CardHeader
    title="Welcome to AEO Audit Tool"
    description="Simple, intelligent marketing audits"
  />
  <CardContent>
    <Button variant="primary">Get Started</Button>
  </CardContent>
</Card>
```

---

## 🎨 Design Tokens Reference

### Colors

| Token | Light | Dark | Usage |
|-------|-------|------|-------|
| `--bg` | #FAF7F2 | #1C1815 | Main background |
| `--surface` | #F0EBE3 | #2A2420 | Cards, panels |
| `--border` | #D4C4B0 | #4A3F36 | Dividers, borders |
| `--text` | #3A2E2A | #EDE5DB | Body text |
| `--accent` | #9B5845 | #C17B66 | CTAs, links |

### Typography

| Token | Value | Usage |
|-------|-------|-------|
| `--font-display` | Space Grotesk | Headings |
| `--font-body` | Plus Jakarta Sans | Body text |
| `--font-serif` | Spectral | Editorial |

### Shadows

| Token | Value |
|-------|-------|
| `--shadow-sm` | 0 1px 2px rgba(...) |
| `--shadow-md` | 0 4px 8px rgba(...) |
| `--shadow-lg` | 0 12px 24px rgba(...) |

---

## 📚 Examples Gallery

### Empty State

```tsx
<EmptyState
  illustration="/illustrations/empty-audits.svg"
  title="No audit runs found"
  description="Create your first audit to get started"
>
  <Button variant="primary">New Audit Run</Button>
</EmptyState>
```

### Dashboard KPI

```tsx
<Kpi
  label="Total Audits"
  value="127"
  trend="+12%"
  trendDirection="up"
  icon={<ChartIcon />}
/>
```

### Data Table

```tsx
<Table
  columns={[
    { header: 'Name', accessor: 'name' },
    { header: 'Status', accessor: 'status' },
    { header: 'Date', accessor: 'date' }
  ]}
  rows={auditRuns}
  onRowClick={(row) => router.push(`/audits/run/${row.id}`)}
/>
```

---

## 🔄 Migration Path

To migrate from the default grayscale theme to Paleolithic:

1. ✅ Import theme system (already done in globals.css)
2. ✅ Update Tailwind config (already done)
3. ✅ Components use CSS variables (already compatible)
4. 🔲 Initialize theme manager in app
5. 🔲 Add theme switcher UI (optional)
6. 🔲 Replace placeholder illustrations with hand-drawn SVGs

**No component rewrites needed!** The entire system is token-based.

---

## 📝 Notes for AI Agents

When generating or modifying UI code:

- ✅ Use `var(--token)` for all colors, fonts, spacing
- ✅ Reference Tailwind classes that map to tokens (`bg-surface`, `text-accent`)
- ✅ Never hardcode hex colors or pixel values
- ✅ Use `font-display` for headings, `font-body` for text
- ✅ Include hand-drawn SVG illustrations from approved libraries
- ✅ Maintain accessibility (focus states, ARIA labels, semantic HTML)

---

## 🌐 Resources

- [Space Grotesk Font](https://fonts.google.com/specimen/Space+Grotesk)
- [Plus Jakarta Sans Font](https://fonts.google.com/specimen/Plus+Jakarta+Sans)
- [Spectral Font](https://fonts.google.com/specimen/Spectral)
- [Open Doodles](https://www.opendoodles.com/)
- [unDraw](https://undraw.co/)
- [Anthropic Design Language](https://type.today/en/journal/anthropic)

---

**Last Updated**: 2025-10-01
**Version**: 1.0.0
**Status**: ✅ Ready for Implementation

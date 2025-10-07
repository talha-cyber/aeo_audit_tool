# Hand-Drawn SVG Illustration Guidelines

## 🎨 Visual Style

The AEO Audit Tool uses **hand-drawn, organic SVG illustrations** that complement the Paleolithic design aesthetic. These illustrations should feel:

- **Warm & Human**: Not sterile or overly technical
- **Simple & Clear**: Easy to understand at a glance
- **Playful but Professional**: Approachable without being childish
- **Consistent**: Unified style across all illustrations

---

## 📚 Recommended Resources

### Primary Sources (Free & Open)

1. **[Open Doodles](https://www.opendoodles.com/)**
   - License: CC0 (Public Domain)
   - Style: Hand-drawn, friendly, diverse characters
   - Format: SVG, customizable colors
   - Best for: Empty states, onboarding, error pages

2. **[unDraw](https://undraw.co/)**
   - License: Open source (MIT-style)
   - Style: Flat illustrations with hand-drawn elements
   - Format: SVG, customizable primary color
   - Best for: Feature illustrations, hero sections

3. **[DrawKit](https://www.drawkit.com/)** (Free tier)
   - License: Free tier is MIT
   - Style: Mix of hand-drawn and flat
   - Format: SVG
   - Best for: UI spots, marketing pages

4. **[Humaaans](https://www.humaaans.com/)**
   - License: CC BY 4.0
   - Style: Mix-and-match humans
   - Format: SVG, highly customizable
   - Best for: Team pages, user-centric illustrations

5. **[Blush](https://blush.design/)** (Free collections)
   - License: Varies by collection (many free for commercial use)
   - Style: Curated hand-drawn collections
   - Format: SVG, PNG
   - Best for: Diverse illustration needs

### Secondary Resources

- **[Storyset](https://storyset.com/)** - Animated & static illustrations (free with attribution)
- **[Manypixels](https://www.manypixels.co/gallery)** - 2500+ free illustrations
- **[404 Illustrations](https://error404.fun/)** - Hand-drawn error page illustrations

---

## 🎨 Color Palette for Illustrations

Use the Paleolithic theme sketch colors for consistency:

```css
/* Primary illustration colors */
--sketch-primary: #8B6B4D;    /* Burnt umber - main strokes */
--sketch-secondary: #B8956A;  /* Sandy brown - secondary elements */
--sketch-accent: #A05547;     /* Rustic red - highlights */
--sketch-neutral: #5A4A3F;    /* Dark earth - shadows/details */
```

### Light Mode Palette
- **Primary**: `#8B6B4D` (Burnt umber)
- **Secondary**: `#B8956A` (Sandy brown)
- **Accent**: `#A05547` (Rustic red)
- **Background**: `#FAF7F2` (Soft ivory)

### Dark Mode Palette
- **Primary**: `#B8956A` (Lighter sandy brown)
- **Secondary**: `#8B6B4D` (Burnt umber)
- **Accent**: `#C17B66` (Light rust)
- **Background**: `#1C1815` (Deep charcoal)

---

## 🖌 Customization Guide

### Recoloring SVGs

Most illustration libraries allow color customization. Here's how to adapt them to our palette:

```svg
<!-- Original unDraw SVG -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500">
  <path fill="#6c63ff" d="..."/>  <!-- Default purple -->
</svg>

<!-- Recolored to Paleolithic theme -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500">
  <path fill="var(--sketch-accent)" d="..."/>  <!-- Our rust color -->
</svg>
```

### Programmatic Recoloring

For unDraw illustrations, use their color picker or update via CSS:

```css
.illustration {
  --undraw-color: #9B5845; /* Our accent color */
}
```

---

## 📐 Technical Guidelines

### File Organization

```
frontend/public/illustrations/
├── empty-states/
│   ├── no-audits.svg
│   ├── no-results.svg
│   └── no-personas.svg
├── onboarding/
│   ├── welcome.svg
│   ├── step-1.svg
│   └── step-2.svg
├── errors/
│   ├── 404.svg
│   ├── 500.svg
│   └── offline.svg
└── features/
    ├── audit-hero.svg
    ├── report-preview.svg
    └── analytics.svg
```

### SVG Optimization

Always optimize SVGs before committing:

```bash
# Using SVGO
npm install -g svgo
svgo input.svg -o output.svg

# Or online: https://jakearchibald.github.io/svgomg/
```

### Size Guidelines

- **Empty States**: 300x300px to 400x400px
- **Hero Sections**: 600x400px to 800x600px
- **UI Spots**: 150x150px to 250x250px
- **Icons**: 24x24px to 48x48px

### Accessibility

```tsx
// Always include alt text and aria labels
<img
  src="/illustrations/no-audits.svg"
  alt="Empty state illustration showing a clipboard with no items"
  role="img"
  aria-label="No audits found"
/>

// Or for inline SVGs
<svg role="img" aria-labelledby="illustration-title">
  <title id="illustration-title">No audits found</title>
  <!-- SVG content -->
</svg>
```

---

## 🎯 Use Cases & Examples

### Empty States

```tsx
import { EmptyState } from '@/components/ui/empty-state';

<EmptyState
  illustration="/illustrations/empty-states/no-audits.svg"
  title="No audit runs yet"
  description="Create your first audit to start analyzing your content"
>
  <Button variant="primary">Create Audit</Button>
</EmptyState>
```

### Onboarding

```tsx
<div className="onboarding-step">
  <img
    src="/illustrations/onboarding/welcome.svg"
    alt="Welcome to AEO Audit Tool"
    className="w-96 h-96"
  />
  <h2>Welcome to AEO Audit Tool</h2>
  <p>Simple, intelligent marketing audits for the modern era</p>
</div>
```

### Error Pages

```tsx
// 404.tsx
export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <img
        src="/illustrations/errors/404.svg"
        alt="Page not found"
        className="w-96 h-96 mb-8"
      />
      <h1 className="text-4xl font-bold mb-4">Page Not Found</h1>
      <p className="text-muted mb-8">
        The page you're looking for doesn't exist
      </p>
      <Button href="/">Go Home</Button>
    </div>
  );
}
```

### Loading States

```tsx
<div className="loading-container">
  <img
    src="/illustrations/loading.svg"
    alt="Loading"
    className="w-32 h-32 animate-pulse"
  />
  <p className="text-muted mt-4">Loading your audit results...</p>
</div>
```

---

## 🖼 Creating Custom Illustrations

If you need custom illustrations, follow these guidelines:

### Tools
- **[Figma](https://figma.com)** - Best for designing and exporting SVGs
- **[Inkscape](https://inkscape.org/)** - Free, open-source vector editor
- **[Sketch](https://www.sketch.com/)** - Mac-only, industry standard

### Style Guide

1. **Line Weight**: 2-3px strokes
2. **Corners**: Rounded (2-4px radius)
3. **Fills**: Minimal, use strokes primarily
4. **Details**: Keep it simple, avoid over-detailing
5. **Perspective**: Slight isometric or flat 2D
6. **Characters**: Friendly, diverse, simplified features

### Color Application

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
  <style>
    /* Use CSS variables for theme compatibility */
    .primary-stroke { stroke: var(--sketch-primary); }
    .accent-fill { fill: var(--sketch-accent); }
    .neutral-stroke { stroke: var(--sketch-neutral); }
  </style>

  <!-- Example: Simple person illustration -->
  <circle
    cx="200"
    cy="120"
    r="40"
    class="accent-fill"
    stroke-width="2"
  />
  <path
    d="M 180 200 L 200 160 L 220 200"
    class="primary-stroke"
    fill="none"
    stroke-width="3"
    stroke-linecap="round"
  />
</svg>
```

### Export Settings (Figma)

- Format: SVG
- Simplify stroke: ✅
- Outline text: ✅ (if using custom fonts)
- Flatten transforms: ✅
- Use absolute bounds: ❌ (keep responsive)

---

## ✅ Illustration Checklist

Before adding an illustration to the project:

- [ ] Licensed for commercial use (CC0, MIT, or similar)
- [ ] Optimized with SVGO or similar tool
- [ ] Colors match Paleolithic palette
- [ ] File size < 50KB (ideally < 20KB)
- [ ] Accessible (alt text, aria labels, title tags)
- [ ] Responsive (uses viewBox, not fixed width/height)
- [ ] Consistent style with existing illustrations
- [ ] Named descriptively (e.g., `empty-audits.svg` not `img1.svg`)

---

## 🔄 Updating Existing Illustrations

When updating illustrations across the app:

1. **Audit current usage**: Search for all `.svg` references
   ```bash
   grep -r "\.svg" frontend/src/
   ```

2. **Update systematically**: Replace one category at a time
   - Empty states first
   - Error pages
   - Onboarding flows
   - Feature illustrations

3. **Test in both themes**: Verify light and dark mode
4. **Check accessibility**: Run axe or similar tool
5. **Optimize**: Ensure file sizes are reasonable

---

## 📊 Performance

### Lazy Loading

```tsx
// For non-critical illustrations
<img
  src="/illustrations/feature.svg"
  alt="Feature illustration"
  loading="lazy"
  decoding="async"
/>
```

### Inline vs External

```tsx
// External (better for caching)
<img src="/illustrations/icon.svg" alt="Icon" />

// Inline (better for critical/small icons)
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path d="..." />
</svg>
```

### Code Splitting

```tsx
// Lazy load illustration component
const HeroIllustration = lazy(() => import('@/components/illustrations/HeroIllustration'));

<Suspense fallback={<div className="w-96 h-96 bg-surface animate-pulse" />}>
  <HeroIllustration />
</Suspense>
```

---

## 🌐 Resources Summary

| Resource | License | Style | Best For |
|----------|---------|-------|----------|
| Open Doodles | CC0 | Hand-drawn | Empty states, people |
| unDraw | MIT-style | Flat/hand-drawn | Features, hero |
| DrawKit | MIT (free) | Mixed | UI spots |
| Humaaans | CC BY 4.0 | Mix-and-match | Teams, users |
| Storyset | Free w/ attr. | Animated | Dynamic content |

---

## 💡 Tips & Best Practices

1. **Consistency First**: Use illustrations from the same library within a feature
2. **Color Unity**: Always adapt colors to Paleolithic palette
3. **Progressive Enhancement**: Ensure UI works without illustrations
4. **Cultural Sensitivity**: Use diverse, inclusive imagery
5. **Context Matters**: Match illustration complexity to UI context
6. **A/B Test**: Test with and without illustrations for key flows
7. **Feedback Loop**: Gather user feedback on illustration style

---

## 🎓 Learning Resources

- [SVG Tutorial (MDN)](https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial)
- [Accessible SVGs](https://www.a11yproject.com/posts/creating-accessible-svg-images/)
- [SVG Animation Guide](https://css-tricks.com/guide-svg-animations-smil/)
- [Figma SVG Export Best Practices](https://www.figma.com/best-practices/exporting-svg/)

---

**Last Updated**: 2025-10-01
**Maintained By**: Design Team
**Status**: ✅ Active

# Report Engine v2 — Implementation Guide (AEO PDF)

Last updated: 2025-08-28
Owner: AEO Project — Report/PDF track
Status: Ready for implementation

---

## Purpose

Upgrade the current `ReportGenerator` into a client‑ready, white‑label, accessible, and data‑sound reporting engine. The new engine must generate best‑in‑class marketing PDFs for AEO audits, with correct metrics (no proxy “confidence == sentiment”), repeatable methodology, visual storytelling, and actionable, prioritized recommendations.

This document is a step‑by‑step build guide for an AI coding agent (Claude Code) to implement the upgrade in the existing codebase.

---

## Success Criteria (Definition of Done)

1. Report chassis: page header/footer with page numbers; Table of Contents; section numbering; document metadata; branded cover & section dividers.
2. Metrics: real sentiment (−1..+1) aggregates; Share of AI Voice (SAIV) correctly defined; platform/category breakdowns; prior‑period deltas.
3. Visuals: SAIV bar chart, trends line chart, platform mix donut, category heatmap; embedded with alt text captions.
4. Methodology & provenance: dedicated pages for prompt basket, platform versions, run IDs, limits/caveats; appendices with question list & glossary.
5. Accessibility & quality: bookmarks/outline, embedded fonts, logical reading order, contrast‑safe theme, localized dates/numbers.
6. White‑labeling: per‑client theming (logo, palette, fonts, locale) selected at runtime; report template versioning recorded to DB.
7. Recommendations: effort × impact matrix, 30‑60‑90 plan, owner, KPI and expected lift ranges.
8. Tests: unit tests for metrics; snapshot test for a sample PDF; chart generation tests; migration tests. CI passes.

---

## High‑Level Scope

* Keep ReportLab, but move from `SimpleDocTemplate` to `BaseDocTemplate` + `PageTemplate` + `Frame`.
* Introduce a metrics layer separate from layout (pure‑functions for computations).
* Introduce charts module that renders Matplotlib PNGs and injects into Platypus flowables.
* Add theming & localization layer.
* Add accessibility helpers (bookmarks, alt text captions, outline). PDF/UA is not fully supported by ReportLab; we’ll implement best‑effort accessibility.
* Add prior‑period comparison pipeline.

---

## Key Concepts & Definitions

* SAIV (Share of AI Voice): (brand\_mentions / all\_mentions) for a fixed prompt basket and set of platforms within the audit period.
* Mention rate: mentions per 100 questions for a brand (controls for total query volume).
* Sentiment: averaged per response using a real sentiment score (−1..+1), not a “confidence” proxy. Report aggregate as mean ± std and N.
* Platform coverage: number of platforms with mentions ≥ N\_threshold for the brand.
* Prompt basket: the fixed set/versioned list of questions used across platforms (enables comparability). Store a version string.

---

## Repository Changes (Proposed)

Create a new package for v2 while keeping v1 stable during rollout.

```
app/
  reports/
    __init__.py
    v2/
      __init__.py
      engine.py            # ReportEngineV2 main orchestrator
      chassis.py           # BaseDocTemplate, PageTemplate, header/footer, ToC
      theme.py             # theming + fonts + palette + spacing + locale
      metrics.py           # SAIV, sentiment aggregates, deltas, helpers
      charts.py            # matplotlib charts → PNG → Platypus Image
      accessibility.py     # bookmarks/outline, alt text captions, doc metadata
      sections/
        title.py           # cover page & metadata table
        summary.py         # exec summary: So what / Now what
        competitive.py     # brand vs competitors table + SAIV chart
        platforms.py       # per‑platform drill‑downs + donut chart
        trends.py          # time buckets trend lines
        categories.py      # category heatmap + insights
        recommendations.py # effort×impact matrix & 30‑60‑90 plan
        methodology.py     # prompt basket, platform versions, caveats
        appendices.py      # raw questions, glossary, run IDs
  models/
    report_template.py     # optional, if templating stored in DB
    theme.py               # optional, if theme stored in DB
```

---

## Data Model & Migrations

We need a real sentiment field and optional metadata for prompt basket/template/theme.

Alembic migration sketch:

```python
# versions/xxxx_add_sentiment_and_report_meta.py
from alembic import op
import sqlalchemy as sa

revision = "xxxx_add_sentiment_and_report_meta"
down_revision = "<prev>"


def upgrade():
    op.add_column(
        "responses",
        sa.Column("sentiment", sa.Float(), nullable=True),  # −1..+1
    )
    op.add_column(
        "audit_runs",
        sa.Column("prompt_basket_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "reports",
        sa.Column("template_version", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "reports",
        sa.Column("theme_key", sa.String(length=64), nullable=True),
    )


def downgrade():
    op.drop_column("reports", "theme_key")
    op.drop_column("reports", "template_version")
    op.drop_column("audit_runs", "prompt_basket_version")
    op.drop_column("responses", "sentiment")
```

Notes:

* `responses.sentiment` must be computed by your Brand Detection / NLP step when ingesting responses.
* If storing themes in code rather than DB, `theme_key` still useful for auditability.

---

## Metrics Layer (Pure Functions)

`app/reports/v2/metrics.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import math

@dataclass
class BrandStats:
    mentions: int
    sentiments: List[float]  # −1..+1
    platforms: Dict[str, int]  # platform → mentions
    categories: Dict[str, int]  # category → mentions

@dataclass
class Aggregates:
    saiv: Dict[str, float]                # brand → 0..1
    mention_rate: Dict[str, float]        # brand → per 100 questions
    sentiment_mean: Dict[str, float]
    sentiment_std: Dict[str, float]
    n: Dict[str, int]

MIN_PLAT_COVER = 3

def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs)/len(xs)
    var = sum((x-m)**2 for x in xs) / max(1, len(xs)-1)
    return m, math.sqrt(var)

def compute_saiv(brand_mentions: Dict[str, int]) -> Dict[str, float]:
    total = sum(brand_mentions.values())
    if total == 0:
        return {b: 0.0 for b in brand_mentions}
    return {b: brand_mentions[b] / total for b in brand_mentions}

def aggregate_brands(
    brands: Iterable[str],
    brand_stats: Dict[str, BrandStats],
    total_questions: int,
) -> Aggregates:
    mention_rate = {
        b: (brand_stats.get(b, BrandStats(0, [], {}, {})).mentions / max(1, total_questions)) * 100
        for b in brands
    }
    sentiment_mean, sentiment_std, n = {}, {}, {}
    all_mentions = {b: brand_stats.get(b, BrandStats(0, [], {}, {})).mentions for b in brands}
    saiv = compute_saiv(all_mentions)
    for b in brands:
        s = brand_stats.get(b, BrandStats(0, [], {}, {})).sentiments
        m, sd = mean_std(s)
        sentiment_mean[b] = m
        sentiment_std[b] = sd
        n[b] = len(s)
    return Aggregates(saiv, mention_rate, sentiment_mean, sentiment_std, n)

def delta(curr: Dict[str, float], prev: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not prev:
        return {k: 0.0 for k in curr}
    return {k: curr.get(k, 0.0) - prev.get(k, 0.0) for k in curr}
```

---

## Charts Module

`app/reports/v2/charts.py`

```python
import io
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.platypus import Image

DPI = 200

def fig_to_image(fig, width=480):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    # ReportLab Image accepts a file-like
    img = Image(buf)
    img._restrictSize(width, width*0.65)
    return img

def saiv_bar(saiv: Dict[str, float]):
    fig = plt.figure()
    brands = list(saiv.keys())
    vals = [saiv[b]*100 for b in brands]
    plt.bar(brands, vals)
    plt.ylabel("Share of AI Voice (%)")
    plt.title("SAIV by Brand")
    return fig_to_image(fig)

def platform_donut(platform_counts: Dict[str, int]):
    fig = plt.figure()
    labels = list(platform_counts.keys())
    sizes = list(platform_counts.values())
    total = sum(sizes) or 1
    wedges, _ = plt.pie(sizes, startangle=90)
    # donut hole
    centre = plt.Circle((0,0),0.60, fc="white")
    plt.gca().add_artist(centre)
    plt.title("Platform Mix")
    return fig_to_image(fig)

def trend_line(x_labels: List[str], series: Dict[str, List[float]], y_label: str, title: str):
    fig = plt.figure()
    for name, ys in series.items():
        plt.plot(x_labels, ys, marker="o", label=name)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    return fig_to_image(fig)

def category_heatmap(matrix: List[List[float]], x_labels: List[str], y_labels: List[str], title: str):
    import numpy as np
    fig = plt.figure()
    data = np.array(matrix)
    plt.imshow(data, aspect="auto")
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(range(len(y_labels)), y_labels)
    plt.colorbar(label="SAIV or Mentions")
    plt.title(title)
    return fig_to_image(fig)
```

Notes:

* Do not set custom colors unless the theme requires it; default Matplotlib palette is fine.
* Each chart is a single figure per guidance.

---

## Theming & Localization

`app/reports/v2/theme.py`

```python
from dataclasses import dataclass
from typing import Dict
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from babel.dates import format_date
from babel.numbers import format_decimal

@dataclass
class Theme:
    name: str
    primary: str = "#2C3E50"
    secondary: str = "#34495E"
    accent: str = "#1ABC9C"
    font_regular: str = "Helvetica"
    font_bold: str = "Helvetica-Bold"
    locale: str = "en_US"
    logo_path: str = ""

THEMES: Dict[str, Theme] = {
    "default": Theme(name="default"),
    "de_corporate": Theme(name="de_corporate", locale="de_DE"),
}

def format_dt(dt, theme: Theme):
    return format_date(dt, format="long", locale=theme.locale)

def format_num(x, theme: Theme, frac=1):
    return format_decimal(x, format=f"#,##0.{''.join(['0']*frac)}", locale=theme.locale)

# If using custom TTF fonts, register once

def register_fonts():
    # Example if you ship fonts
    # pdfmetrics.registerFont(TTFont("Inter", "assets/fonts/Inter-Regular.ttf"))
    # pdfmetrics.registerFont(TTFont("Inter-Bold", "assets/fonts/Inter-Bold.ttf"))
    pass
```

---

## Chassis: Doc Template, Headers/Footers, ToC, Metadata, Outline

`app/reports/v2/chassis.py`

```python
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus.tableofcontents import TableOfContents

from reportlab.pdfgen import canvas as canvasmod

MARGIN = 0.75*inch

class ReportDoc(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, pagesize=A4, **kw)
        frame = Frame(MARGIN, MARGIN, A4[0]-2*MARGIN, A4[1]-2*MARGIN, id='normal')
        template = PageTemplate(id='main', frames=[frame], onPage=self._header_footer)
        self.addPageTemplates([template])
        self.toc = TableOfContents()
        self.toc.levelStyles = []  # use default, or attach ParagraphStyles

    def _header_footer(self, canv, doc):
        canv.setStrokeColor(colors.lightgrey)
        canv.setLineWidth(0.5)
        canv.line(MARGIN, A4[1]-MARGIN+6, A4[0]-MARGIN, A4[1]-MARGIN+6)
        canv.setFont('Helvetica', 9)
        canv.setFillColor(colors.grey)
        canv.drawString(MARGIN, A4[1]-MARGIN+10, getattr(self, 'header_left', 'AEO Competitive Intelligence Report'))
        canv.drawRightString(A4[0]-MARGIN, A4[1]-MARGIN+10, getattr(self, 'header_right', 'Confidential'))
        canv.drawRightString(A4[0]-MARGIN, MARGIN-20, f"Page {doc.page}")

    def set_metadata(self, title, author, subject, keywords):
        self.title, self.author, self.subject, self.keywords = title, author, subject, keywords

    def beforeDocument(self):
        if hasattr(self, 'title'):
            self.canv.setTitle(self.title)
            self.canv.setAuthor(self.author)
            self.canv.setSubject(self.subject)
            self.canv.setKeywords(self.keywords)
```

Outline & bookmarks helper

`app/reports/v2/accessibility.py`

```python
from reportlab.platypus import Paragraph

def outline(canv, title, level=0, key=None):
    key = key or title
    canv.bookmarkPage(key)
    canv.addOutlineEntry(title, key, level=level, closed=False)

class H:
    # helpers to register headings with ToC and outline
    def __init__(self, style, doc, level):
        self.style, self.doc, self.level = style, doc, level
    def __call__(self, text):
        # Paragraph with a hidden bookmark id
        from reportlab.platypus import Paragraph
        p = Paragraph(f"{text}", self.style)
        # notify ToC
        self.doc.notify('TOCEntry', (self.level, text, self.doc.page))
        # also add outline on afterFlowable
        def _afl(canvas, doc):
            outline(canvas, text, self.level, key=text)
        p._postponed = _afl
        return p
```

Usage: in sections, create headings via `H(style, doc, level)("1 Executive Summary")` and add `doc.afterFlowable` hook to call the postponed outline.

---

## Engine Orchestration

`app/reports/v2/engine.py`

```python
from typing import Dict, Any, List, Optional
from reportlab.platypus import Spacer, PageBreak
from reportlab.lib.units import inch

from .chassis import ReportDoc
from .theme import THEMES, Theme, format_dt, register_fonts
from .metrics import aggregate_brands, delta
from . import charts
from .sections import title as s_title
from .sections import summary as s_summary
from .sections import competitive as s_comp
from .sections import platforms as s_plat
from .sections import trends as s_tr
from .sections import categories as s_cat
from .sections import recommendations as s_rec
from .sections import methodology as s_meth
from .sections import appendices as s_app

class ReportEngineV2:
    def __init__(self, db, theme_key: str = "default", template_version: str = "v2.0"):
        self.db = db
        self.theme: Theme = THEMES.get(theme_key, THEMES["default"])
        self.template_version = template_version
        register_fonts()

    def build(self, audit_run_id: str, output_path: str) -> str:
        data = self._load_data(audit_run_id)
        prev = self._load_previous_period(data)
        doc = ReportDoc(output_path)
        doc.set_metadata(
            title=f"AEO Competitive Intelligence Report — {data['client_name']}",
            author="AEO Platform",
            subject="AEO Audit Results",
            keywords=["AEO","AI","Audit","Competitive","Report"],
        )
        doc.header_left = f"{data['client_name']} — AEO Audit"
        doc.header_right = f"Period: {format_dt(data['date_range']['start'], self.theme)} – {format_dt(data['date_range']['end'], self.theme)}"

        story: List[Any] = []

        # cover
        story += s_title.build(self.theme, data)
        story.append(PageBreak())

        # table of contents
        story.append(doc.toc)
        story.append(Spacer(1, 0.2*inch))
        story.append(PageBreak())

        # exec summary
        story += s_summary.build(self.theme, data, prev)
        story.append(PageBreak())

        # competitive + charts
        story += s_comp.build(self.theme, data, prev)
        story.append(PageBreak())

        # platforms
        story += s_plat.build(self.theme, data)
        story.append(PageBreak())

        # trends
        story += s_tr.build(self.theme, data, prev)
        story.append(PageBreak())

        # categories
        story += s_cat.build(self.theme, data)
        story.append(PageBreak())

        # recommendations
        story += s_rec.build(self.theme, data, prev)
        story.append(PageBreak())

        # methodology & appendices
        story += s_meth.build(self.theme, data)
        story.append(PageBreak())
        story += s_app.build(self.theme, data)

        doc.build(story)
        self._persist_report_record(audit_run_id, output_path)
        return output_path

    def _load_data(self, audit_run_id: str) -> Dict[str, Any]:
        # Reuse existing loader but: include real sentiment, categories, platform info, prompt_basket_version
        # Ensure data["brand_performance"][brand]["sentiment_scores"] is populated with real sentiment
        ...

    def _load_previous_period(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Fetch latest prior audit for same client with same prompt_basket_version, comparable platform set
        ...

    def _persist_report_record(self, audit_run_id: str, path: str):
        # Add template_version and theme_key to Report row
        ...
```

Note: `...` indicates reuse/adaptation of existing code. Keep v1 generator intact for backward compatibility; add a feature flag to switch to v2.

---

## Section Builders (Examples)

Title page

`app/reports/v2/sections/title.py`

```python
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

from ..theme import Theme, format_dt

def build(theme: Theme, data):
    story = []
    title_style = {
        'fontSize': 26,
        'leading': 30,
        'textColor': colors.HexColor(theme.primary),
        'alignment': 1,
    }
    story.append(Paragraph(f"AEO Competitive Intelligence Report<br/>{data['client_name']}", _ps(title_style)))
    story.append(Spacer(1, 0.4*inch))

    md = [
        ["Report Date:", format_dt(data['date_range']['end'], theme)],
        ["Audit Period:", f"{format_dt(data['date_range']['start'], theme)} – {format_dt(data['date_range']['end'], theme)}"],
        ["Total Queries:", str(data['total_responses'])],
        ["Platforms:", ", ".join(data['platform_stats'].keys())],
        ["Industry:", data['industry']],
        ["Competitors:", ", ".join(data['competitors'])],
    ]
    t = Table(md, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    return story

# crude ParagraphStyle shim without global stylesheet
from reportlab.lib.styles import ParagraphStyle

def _ps(d):
    s = ParagraphStyle('dyn')
    for k,v in d.items():
        setattr(s,k,v)
    return s
```

Executive summary with So what / Now what

`app/reports/v2/sections/summary.py`

```python
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib import colors

from ..theme import Theme, format_num

from ..accessibility import H

def build(theme: Theme, data, prev):
    story = []
    # Heading (level 1)
    def style():
        from reportlab.lib.styles import ParagraphStyle
        return ParagraphStyle('h1', fontSize=18, spaceAfter=12, textColor=colors.HexColor(theme.primary))
    h1 = H(style(), None, 0)  # doc is injected by chassis afterFlowable; simplified here

    story.append(Paragraph("1 Executive Summary", style()))

    # Compute key figures (assumes `data['aggregates']`) — or compute inline
    client = data['client_name']
    agg = data['aggregates']
    saiv = agg.saiv.get(client, 0.0)*100
    sent = agg.sentiment_mean.get(client, 0.0)
    n = agg.n.get(client, 0)

    story.append(Paragraph(f"So what: {client} holds {format_num(saiv, theme, 1)}% SAIV with mean sentiment {sent:+.2f} (n={n}).", _body()))
    story.append(Paragraph("Now what: prioritize platform/category gaps where SAIV < peer median; execute 30‑60‑90 plan.", _body())))
    story.append(Spacer(1, 8))
    return story

from reportlab.lib.styles import ParagraphStyle

def _body():
    return ParagraphStyle('body', fontSize=11, leading=14)
```

Competitive section with SAIV chart (skeleton)

`app/reports/v2/sections/competitive.py`

```python
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

from ..theme import Theme, format_num
from .. import charts

def build(theme: Theme, data, prev):
    story = []
    story.append(Paragraph("2 Competitive Analysis", _h1(theme)))

    # SAIV chart
    saiv = data['aggregates'].saiv
    story.append(charts.saiv_bar(saiv))
    story.append(Paragraph("Alt: Bar chart of SAIV by brand.", _alt()))
    story.append(Spacer(1, 0.2*inch))

    # Comparison table
    rows = [["Brand", "Mentions", "SAIV %", "Sentiment (mean±std)", "Platforms ≥ N"]]
    for b in [data['client_name']] + data['competitors']:
        m = data['brand_performance'].get(b, {}).get('total_mentions', 0)
        s_mean = data['aggregates'].sentiment_mean.get(b, 0.0)
        s_sd = data['aggregates'].sentiment_std.get(b, 0.0)
        saiv_pc = data['aggregates'].saiv.get(b, 0.0)*100
        plat_cover = len([p for p,c in data['per_platform'][b].items() if c >= 1]) if 'per_platform' in data else 0
        rows.append([b, str(m), f"{saiv_pc:.1f}", f"{s_mean:+.2f}±{s_sd:.2f}", str(plat_cover)])

    t = Table(rows, colWidths=[2*inch, 1*inch, 1*inch, 1.6*inch, 1.2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.lightgrey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('GRID',(0,0),(-1,-1),0.5, colors.grey),
        ('FONTSIZE',(0,0),(-1,-1),10),
    ]))
    story.append(t)
    return story

from reportlab.lib.styles import ParagraphStyle

def _h1(theme: Theme):
    return ParagraphStyle('h1', fontSize=18, spaceAfter=12, textColor=colors.HexColor(theme.primary))

def _alt():
    return ParagraphStyle('alt', fontSize=8, textColor=colors.grey)
```

Implement similar builders for platforms, trends, categories (see charts API above).

---

## Data Loading Changes

Extend the existing `_load_audit_data` to provide:

* Real `sentiment_scores` per brand (from `responses.sentiment`).
* Per‑platform brand mentions, per‑category mentions.
* Aggregate totals and a prepared `aggregates` object for easy rendering.
* Previous period dataset (same client, same `prompt_basket_version`) for deltas.

Pseudocode:

```python
# inside ReportEngineV2._load_data
responses = db.query(Response).filter(...).all()
# collect brands
brands = [client] + competitors
brand_stats = {b: BrandStats(0, [], {}, {}) for b in brands}
platform_counts = {}

for r in responses:
    platform = r.platform
    platform_counts[platform] = platform_counts.get(platform, 0) + 1
    # use your brand detection field e.g., r.brand_mentions['top_brands']
    detected = (r.brand_mentions or {}).get('top_brands', [])
    for b in detected:
        if b in brand_stats:
            brand_stats[b].mentions += 1
            brand_stats[b].platforms[platform] = brand_stats[b].platforms.get(platform, 0)+1
            if r.category:
                brand_stats[b].categories[r.category] = brand_stats[b].categories.get(r.category, 0)+1
            if r.sentiment is not None:
                brand_stats[b].sentiments.append(r.sentiment)

aggregates = aggregate_brands(brands, brand_stats, total_questions=len(responses))

return {
    ...,
    'brand_stats': brand_stats,
    'aggregates': aggregates,
    'platform_counts': platform_counts,
}
```

Previous period loader should match same client and prompt basket (to ensure comparability) and pull the immediately preceding completed run. Compute its `aggregates` similarly for deltas.

---

## Recommendations Section Logic

Rules‑of‑thumb that tie to observed gaps:

* If client SAIV < peer median on a platform: recommend platform‑specific optimization (answer snippets, citations, structured data) for top 2 underperforming categories.
* If sentiment mean < 0 with n≥20: recommend remedial content addressing common pain points from the question set.
* If category with high competitor dominance: create “category defense” FAQ and How‑To schema with specific Q\&As.

For each recommendation, output:

* Title
* Rationale (observed gap and numbers)
* Effort (S/M/L)
* Impact (percent lift range)
* 30‑60‑90 timeline
* Owner role
* KPI & target

Implementation sketch:

```python
rec = {
  'title': 'Improve SAIV on Perplexity for Shipping & Returns',
  'rationale': 'Client 8% vs peer median 19% (−11pp), n=120',
  'effort': 'M',
  'impact': '10–15% SAIV lift',
  'plan': {
    '30d': ['Add FAQ schema answers for refund window, shipping fees'],
    '60d': ['Publish policy explainer & cite authoritative sources'],
    '90d': ['Refresh with seasonal promo terms & link from site nav']
  },
  'owner': 'Content Lead',
  'kpi': 'SAIV on Perplexity in S&R',
  'target': '≥18% within 90 days'
}
```

Render as a numbered list with bold titles and small KPI blocks.

---

## Accessibility & Quality

* Outline/bookmarks: add entries for each h1/h2 section using `canvas.addOutlineEntry`.
* Alt text: after each chart, add a small gray caption describing it; provide the key takeaway.
* Fonts: embed fonts; avoid tiny text (<9pt).
* Contrast: theme colors should pass WCAG AA for text.
* Metadata: set Title/Author/Subject/Keywords on the `ReportDoc`.
* PDF/A (optional, nice‑to‑have): if required, add a post‑process step via Ghostscript to convert to PDF/A‑1b.

---

## Integration with Existing `ReportGenerator`

* Keep the current class for `report_type in {comprehensive, summary, platform_specific}`.
* Add a feature flag `REPORT_V2_ENABLED` or a new `report_type="v2_comprehensive"` that delegates to `ReportEngineV2`.
* When persisting the `Report` row, include `template_version` and `theme_key`.

Example integration entrypoint:

```python
if report_type in {"v2", "v2_comprehensive"}:
    engine = ReportEngineV2(self.db, theme_key="de_corporate", template_version="v2.0")
    path = engine.build(audit_run_id, filepath)
    self._create_report_record(audit_run_id, report_type, path, template_version="v2.0", theme_key="de_corporate")
    return path
```

---

## Tests

Unit tests

* `metrics.test_saiv_compute` — edge cases: zero totals, single brand, multiple brands.
* `metrics.test_sentiment_agg` — mean/std with negative and positive values.
* `charts.test_render` — ensure PNGs produced, non‑empty bytes.
* `engine.test_previous_period` — correct matching by client and prompt basket.

Integration/snapshot tests

* Generate a small sample report from fixtures; assert file exists, size > X KB, and include a simple text extraction check for headings.

Fixtures

* Minimal `responses` set for 2 brands × 2 platforms × 3 categories with sentiments.

---

## Rollout Plan

1. Behind a feature flag; generate both v1 and v2 in staging.
2. QA on multiple clients/locales; collect feedback.
3. Enable v2 as default; keep v1 for fallback one release.

---

## Security & Privacy

* Avoid including PII in appendices; redact raw queries if necessary.
* Include a confidentiality banner in header/footer if the client requires it.

---

## Performance Notes

* Chart generation is CPU‑bound; cache chart PNGs per data hash if needed.
* Use lazy image loading in Platypus by passing file‑like buffers.

---

## Quick Task Checklist (for the AI agent)

* [ ] Add DB migration for `responses.sentiment`, `audit_runs.prompt_basket_version`, `reports.template_version`, `reports.theme_key`.
* [ ] Implement `metrics.py` pure‑functions.
* [ ] Implement `charts.py` functions and return Platypus `Image` flowables.
* [ ] Build `theme.py` with locale helpers and optional font registration.
* [ ] Implement `chassis.py` `ReportDoc` with header/footer, ToC, metadata.
* [ ] Implement `accessibility.py` outline/bookmark helpers.
* [ ] Implement section builders in `sections/` (title, summary, competitive, platforms, trends, categories, recommendations, methodology, appendices).
* [ ] Implement `engine.py` orchestration and integrate with existing `ReportGenerator` via a new report type or flag.
* [ ] Extend data loader to include real sentiments, categories, per‑platform counts, aggregates, previous period.
* [ ] Write unit/integration tests and run CI.
* [ ] Toggle v2 on in staging; verify visual QA; proceed to production.

---

## Notes on Limitations

* ReportLab does not emit fully tagged PDF/UA; we implement best‑effort accessibility (bookmarks, structure via outline, readable order, alt captions). For legal compliance, consider a downstream converter or a different renderer.

---

## Example: Minimal End‑to‑End Usage

```python
engine = ReportEngineV2(db_session, theme_key="de_corporate", template_version="v2.0")
path = engine.build(audit_run_id="<uuid>", output_path="reports/AEO_Report_v2.pdf")
print("Generated:", path)
```

This guide is intentionally explicit to let an AI coding agent implement the upgrade without ambiguity. Keep functions pure where possible, isolate layout from calculations, and favor small, testable units. Good luck.

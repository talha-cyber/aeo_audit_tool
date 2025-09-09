# Chart Implementation Instructions for AEO Audit Reports

## Overview
Add automated chart generation to the existing PDF report system in `app/services/report_generator.py`. Charts should be generated automatically from audit data already processed by the system.

## Current System Context
- Report generator: `app/services/report_generator.py:39-584`
- Data loading: `_load_audit_data()` method processes all necessary metrics
- PDF generation: Uses ReportLab with structured story array
- Dependencies: matplotlib==3.8.2 already installed

## Required Charts

### 1. Market Share Pie Chart
**Purpose**: Show client vs competitor brand mention distribution
**Data Source**: `data["brand_performance"][brand]["total_mentions"]`
**Location**: Insert in Executive Summary section after line 405
```python
# Generate pie chart showing client market share vs all competitors
labels = [client_name] + competitors
sizes = [mention counts for each brand]
```

### 2. Platform Performance Bar Chart
**Purpose**: Compare client brand mentions across AI platforms
**Data Source**: `data["platform_stats"][platform]["brand_mentions"][client_name]`
**Location**: Insert in Platform Analysis section after line 495
```python
# Bar chart of client mentions per platform
platforms = list(platform_stats.keys())
mentions = [stats["brand_mentions"][client_name] for stats in platform_stats.values()]
```

### 3. Competitive Analysis Bar Chart
**Purpose**: Side-by-side comparison of all brands
**Data Source**: `data["brand_performance"]` for all brands
**Location**: Insert in Competitive Analysis section after line 463
```python
# Grouped bar chart: mentions, sentiment, platform coverage per brand
brands = [client_name] + competitors
metrics = ["total_mentions", "avg_sentiment", "platform_count"]
```

### 4. Sentiment Comparison Chart
**Purpose**: Show average sentiment scores by brand
**Data Source**: `data["brand_performance"][brand]["sentiment_scores"]`
**Location**: Additional chart in Competitive Analysis section
```python
# Horizontal bar chart of average sentiment per brand
avg_sentiments = [mean of sentiment_scores for each brand]
```

## Implementation Steps

### Step 1: Create Chart Generation Methods
Add these methods to `ReportGenerator` class:
- `_create_market_share_chart(data)` - Returns ReportLab Image flowable
- `_create_platform_performance_chart(data)` - Returns ReportLab Image flowable
- `_create_competitive_analysis_chart(data)` - Returns ReportLab Image flowable
- `_create_sentiment_comparison_chart(data)` - Returns ReportLab Image flowable

### Step 2: Chart Generation Approach
Use matplotlib to generate charts as images:
```python
import matplotlib.pyplot as plt
import io
from reportlab.platypus import Image

def _create_chart_image(self, fig):
    """Convert matplotlib figure to ReportLab Image."""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    plt.close(fig)
    return Image(img_buffer, width=6*inch, height=4*inch)
```

### Step 3: Integration Points
Insert chart calls in existing report sections:

**Executive Summary** (line 405):
```python
story.append(self._create_market_share_chart(data))
story.append(Spacer(1, 20))
```

**Competitive Analysis** (line 463):
```python
story.append(self._create_competitive_analysis_chart(data))
story.append(Spacer(1, 20))
story.append(self._create_sentiment_comparison_chart(data))
story.append(Spacer(1, 20))
```

**Platform Analysis** (line 495):
```python
story.append(self._create_platform_performance_chart(data))
story.append(Spacer(1, 20))
```

## Data Processing Notes
- All required data is already processed in `_load_audit_data()`
- No additional database queries needed
- Handle cases where data might be empty/missing
- Use existing color schemes and styling consistent with report

## Chart Styling Guidelines
- Use professional color palette (blues, grays, accent colors)
- Include proper titles, labels, and legends
- Ensure readability when printed in PDF format
- Consistent font sizes and styling with report text

## Error Handling
- Gracefully handle missing data (empty charts or fallback text)
- Log chart generation errors but don't fail entire report
- Provide fallback text summaries if chart generation fails

## Testing Approach
- Test with existing audit run IDs that have data
- Verify charts appear correctly in generated PDFs
- Test edge cases: no data, single competitor, single platform
- Ensure PDF file size remains reasonable

## Files to Modify
- `app/services/report_generator.py` - Add chart generation methods and integrate into report sections
- No database models or API changes needed
- No new dependencies required (matplotlib already installed)

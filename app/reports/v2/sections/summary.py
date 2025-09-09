"""
Executive summary section builder for AEO audit reports.

This module creates the executive summary with:
- "So what" and "Now what" structure
- Key performance indicators
- High-level insights and recommendations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer

from ..accessibility import create_section_heading
from ..theme import Theme, create_paragraph_styles, format_number


def build(
    theme: Theme, data: Dict[str, Any], previous_period: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """
    Build executive summary section.

    Args:
        theme: Theme configuration for styling
        data: Current period audit data
        previous_period: Previous period data for comparison (optional)

    Returns:
        List of ReportLab flowables for executive summary
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Section heading
    heading = create_section_heading("Executive Summary", 1, theme)
    story.append(heading)
    story.append(Spacer(1, 0.2 * inch))

    # Key metrics overview
    metrics_overview = build_key_metrics(theme, data)
    story.extend(metrics_overview)

    # "So What" analysis
    so_what = build_so_what_analysis(theme, data, previous_period)
    story.extend(so_what)

    # "Now What" recommendations
    now_what = build_now_what_recommendations(theme, data)
    story.extend(now_what)

    return story


def build_key_metrics(theme: Theme, data: Dict[str, Any]) -> List[Any]:
    """
    Build key metrics overview.

    Args:
        theme: Theme configuration
        data: Audit data dictionary

    Returns:
        List of flowables for metrics overview
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Extract key data
    client_name = data.get("client_name", "Unknown Client")
    aggregates = data.get("aggregates")
    total_responses = data.get("total_responses", 0)

    if not aggregates:
        # Fallback if aggregates not computed
        story.append(
            Paragraph(
                "Key metrics are being processed and will be available in the detailed sections below.",
                styles["body"],
            )
        )
        return story

    # Calculate key figures
    client_saiv = aggregates.saiv.get(client_name, 0.0) * 100
    client_sentiment = aggregates.sentiment_mean.get(client_name, 0.0)
    sentiment_count = aggregates.n.get(client_name, 0)
    mention_rate = aggregates.mention_rate.get(client_name, 0.0)

    # Platform coverage
    platform_count = len(data.get("platform_stats", {}))

    # Create metrics summary
    metrics_text = f"""
    <b>Key Performance Indicators:</b><br/>
    • <b>Share of AI Voice (SAIV):</b> {format_number(client_saiv, theme, 1)}% across all platforms<br/>
    • <b>Brand Mention Rate:</b> {format_number(mention_rate, theme, 1)} mentions per 100 queries<br/>
    • <b>Average Sentiment:</b> {client_sentiment:+.2f} (n={sentiment_count} responses)<br/>
    • <b>Platform Coverage:</b> Active on {platform_count} AI platforms<br/>
    • <b>Analysis Scope:</b> {total_responses:,} total queries analyzed
    """

    story.append(Paragraph(metrics_text, styles["body"]))
    story.append(Spacer(1, 0.2 * inch))

    return story


def build_so_what_analysis(
    theme: Theme, data: Dict[str, Any], previous_period: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """
    Build "So What" analysis section.

    Args:
        theme: Theme configuration
        data: Current audit data
        previous_period: Previous period data for comparison

    Returns:
        List of flowables for analysis
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Section subheading
    story.append(Paragraph("<b>So What: Key Insights</b>", styles["heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    client_name = data.get("client_name", "Unknown Client")
    aggregates = data.get("aggregates")

    if not aggregates:
        story.append(
            Paragraph(
                "Analysis insights will be available once data processing is complete.",
                styles["body"],
            )
        )
        return story

    insights = []

    # SAIV Analysis
    client_saiv = aggregates.saiv.get(client_name, 0.0)
    competitor_saivs = [
        saiv
        for brand, saiv in aggregates.saiv.items()
        if brand != client_name and saiv > 0
    ]

    if competitor_saivs:
        avg_competitor_saiv = sum(competitor_saivs) / len(competitor_saivs)

        if client_saiv > avg_competitor_saiv:
            saiv_insight = f"{client_name} outperforms competitors in AI visibility with {client_saiv*100:.1f}% SAIV vs {avg_competitor_saiv*100:.1f}% competitor average."
        else:
            gap = (avg_competitor_saiv - client_saiv) * 100
            saiv_insight = f"{client_name} trails competitors in AI visibility by {gap:.1f} percentage points, indicating opportunity for optimization."

        insights.append(saiv_insight)

    # Sentiment Analysis
    client_sentiment = aggregates.sentiment_mean.get(client_name, 0.0)
    sentiment_count = aggregates.n.get(client_name, 0)

    if sentiment_count >= 10:  # Only analyze if sufficient data
        if client_sentiment > 0.1:
            sentiment_insight = f"Brand sentiment is positive ({client_sentiment:+.2f}) across {sentiment_count} mentions, indicating strong brand perception in AI responses."
        elif client_sentiment < -0.1:
            sentiment_insight = f"Brand sentiment is negative ({client_sentiment:+.2f}) across {sentiment_count} mentions, suggesting need for reputation management."
        else:
            sentiment_insight = f"Brand sentiment is neutral ({client_sentiment:+.2f}) across {sentiment_count} mentions, presenting opportunity for positive differentiation."

        insights.append(sentiment_insight)

    # Platform Performance
    platform_stats = data.get("platform_stats", {})
    if platform_stats:
        platform_performance = {}
        for platform, stats in platform_stats.items():
            mentions = stats.get("brand_mentions", {}).get(client_name, 0)
            total_questions = stats.get("total_questions", 1)
            performance_rate = (mentions / total_questions) * 100
            platform_performance[platform] = performance_rate

        if platform_performance:
            best_platform = max(
                platform_performance.keys(), key=lambda x: platform_performance[x]
            )
            best_rate = platform_performance[best_platform]
            worst_platform = min(
                platform_performance.keys(), key=lambda x: platform_performance[x]
            )
            worst_rate = platform_performance[worst_platform]

            if len(platform_performance) > 1:
                platform_insight = f"Platform performance varies significantly: strongest on {best_platform} ({best_rate:.1f}% mention rate) vs weakest on {worst_platform} ({worst_rate:.1f}%)."
                insights.append(platform_insight)

    # Period-over-period comparison
    if previous_period:
        prev_aggregates = previous_period.get("aggregates")
        if prev_aggregates:
            prev_saiv = prev_aggregates.saiv.get(client_name, 0.0)
            saiv_change = (client_saiv - prev_saiv) * 100

            if abs(saiv_change) > 1.0:  # Only mention if significant change
                trend_word = "increased" if saiv_change > 0 else "decreased"
                trend_insight = f"SAIV has {trend_word} by {abs(saiv_change):.1f} percentage points compared to the previous period."
                insights.append(trend_insight)

    # Add insights to story
    if insights:
        for insight in insights:
            story.append(Paragraph(f"• {insight}", styles["body"]))
            story.append(Spacer(1, 6))
    else:
        story.append(
            Paragraph(
                "Detailed insights will be provided as more data becomes available through continued auditing.",
                styles["body"],
            )
        )

    story.append(Spacer(1, 0.2 * inch))
    return story


def build_now_what_recommendations(theme: Theme, data: Dict[str, Any]) -> List[Any]:
    """
    Build "Now What" strategic recommendations.

    Args:
        theme: Theme configuration
        data: Audit data dictionary

    Returns:
        List of flowables for recommendations
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Section subheading
    story.append(Paragraph("<b>Now What: Strategic Priorities</b>", styles["heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    client_name = data.get("client_name", "Unknown Client")
    aggregates = data.get("aggregates")

    recommendations = []

    if aggregates:
        # SAIV-based recommendations
        client_saiv = aggregates.saiv.get(client_name, 0.0)
        competitor_saivs = [
            saiv
            for brand, saiv in aggregates.saiv.items()
            if brand != client_name and saiv > 0
        ]

        if competitor_saivs:
            avg_competitor_saiv = sum(competitor_saivs) / len(competitor_saivs)

            if client_saiv < avg_competitor_saiv:
                recommendations.append(
                    "Prioritize AI optimization initiatives to close the competitive SAIV gap through content strategy and structured data implementation."
                )

        # Sentiment-based recommendations
        client_sentiment = aggregates.sentiment_mean.get(client_name, 0.0)
        if client_sentiment < 0:
            recommendations.append(
                "Address negative sentiment through proactive content creation that highlights positive brand attributes and customer success stories."
            )
        elif client_sentiment < 0.2:
            recommendations.append(
                "Enhance positive brand perception by developing content that emphasizes unique value propositions and competitive advantages."
            )

    # Platform-specific recommendations
    platform_stats = data.get("platform_stats", {})
    if len(platform_stats) > 1:
        recommendations.append(
            "Implement platform-specific optimization strategies based on performance variations identified in the detailed analysis."
        )

    # Generic recommendations if no specific insights available
    if not recommendations:
        recommendations = [
            "Monitor AI platform performance regularly to identify optimization opportunities.",
            "Develop comprehensive content strategy addressing key industry questions and use cases.",
            "Implement structured data markup to improve AI platform citation rates.",
        ]

    # Add recommendations to story
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", styles["body"]))
        story.append(Spacer(1, 8))

    # Call to action
    story.append(Spacer(1, 0.1 * inch))
    cta_text = "Detailed recommendations with specific implementation timelines are provided in the Strategic Recommendations section of this report."
    story.append(Paragraph(f"<i>{cta_text}</i>", styles["body"]))

    return story

"""
Competitive analysis section builder for AEO audit reports.

This module creates competitive analysis with:
- SAIV comparison charts
- Brand performance tables
- Competitive positioning insights
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table

from .. import charts
from ..accessibility import create_alt_text_caption, create_section_heading
from ..theme import (
    Theme,
    create_paragraph_styles,
    create_table_styles,
    wrap_table_cell,
)


def build(
    theme: Theme, data: Dict[str, Any], previous_period: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """
    Build competitive analysis section.

    Args:
        theme: Theme configuration for styling
        data: Current period audit data
        previous_period: Previous period data for comparison (optional)

    Returns:
        List of ReportLab flowables for competitive analysis
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Section heading
    heading = create_section_heading("Competitive Analysis", 1, theme)
    story.append(heading)
    story.append(Spacer(1, 0.2 * inch))

    # SAIV visualization
    saiv_chart_section = build_saiv_chart_section(theme, data)
    story.extend(saiv_chart_section)

    # Competitive comparison table
    comparison_table_section = build_comparison_table(theme, data)
    story.extend(comparison_table_section)

    # Sentiment comparison
    sentiment_section = build_sentiment_comparison(theme, data)
    story.extend(sentiment_section)

    # Market share analysis
    market_share_section = build_market_share_analysis(theme, data)
    story.extend(market_share_section)

    # Performance insights
    insights_section = build_competitive_insights(theme, data, previous_period)
    story.extend(insights_section)

    return story


def build_saiv_chart_section(theme: Theme, data: Dict[str, Any]) -> List[Any]:
    """
    Build SAIV comparison chart section.

    Args:
        theme: Theme configuration
        data: Audit data dictionary

    Returns:
        List of flowables for SAIV chart
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Subheading
    story.append(Paragraph("Share of AI Voice (SAIV) Comparison", styles["heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    # Get SAIV data
    aggregates = data.get("aggregates")
    if aggregates and aggregates.saiv:
        try:
            # Create SAIV bar chart
            saiv_chart = charts.create_saiv_bar_chart(
                aggregates.saiv, title="Share of AI Voice by Brand"
            )
            story.append(Spacer(1, 0.1 * inch))  # Space before chart
            story.append(saiv_chart)
            story.append(Spacer(1, 0.1 * inch))  # Space after chart

            # Add alt text caption
            client_name = data.get("client_name", "Client")
            client_saiv = aggregates.saiv.get(client_name, 0.0) * 100

            competitors_text = (
                "competitors" if len(aggregates.saiv) > 2 else "competitor"
            )
            alt_text = f"Bar chart showing {client_name} with {client_saiv:.1f}% SAIV compared to {competitors_text}"

            alt_caption = create_alt_text_caption(alt_text)
            story.append(alt_caption)
            story.append(Spacer(1, 0.2 * inch))

        except Exception:
            # Fallback if chart generation fails
            story.append(
                Paragraph(
                    f"SAIV visualization temporarily unavailable. Raw data: {dict(aggregates.saiv)}",
                    styles["body"],
                )
            )
            story.append(Spacer(1, 0.1 * inch))
    else:
        story.append(
            Paragraph(
                "SAIV data is being processed and will be available in future reports.",
                styles["body"],
            )
        )
        story.append(Spacer(1, 0.1 * inch))

    return story


def build_comparison_table(theme: Theme, data: Dict[str, Any]) -> List[Any]:
    """
    Build competitive comparison table.

    Args:
        theme: Theme configuration
        data: Audit data dictionary

    Returns:
        List of flowables for comparison table
    """
    story = []
    styles = create_paragraph_styles(theme)
    table_styles = create_table_styles(theme)

    # Subheading
    story.append(Paragraph("Brand Performance Comparison", styles["heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    # Build table data
    table_data = build_competitive_table_data(data, theme)

    if table_data and len(table_data) > 1:  # Header + at least one data row
        # Create table
        comparison_table = Table(
            table_data,
            colWidths=[
                1.6 * inch,
                1.0 * inch,
                1.0 * inch,
                1.4 * inch,
                1.2 * inch,
            ],  # Total: 6.2 inches
            hAlign="LEFT",
        )

        comparison_table.setStyle(table_styles["standard"])
        story.append(comparison_table)

        # Table caption
        caption_text = f"Competitive performance metrics across {len(table_data)-1} brands analyzed."
        story.append(Paragraph(f"<i>{caption_text}</i>", styles["caption"]))
        story.append(Spacer(1, 0.2 * inch))

    else:
        story.append(
            Paragraph("Competitive comparison data is being processed.", styles["body"])
        )
        story.append(Spacer(1, 0.1 * inch))

    return story


def build_competitive_table_data(data: Dict[str, Any], theme: Theme) -> List[List]:
    """
    Build table data for competitive comparison.

    Args:
        data: Audit data dictionary

    Returns:
        2D list of table data
    """
    # Table headers - wrapped in Paragraphs
    headers = [
        "Brand",
        "Mentions",
        "SAIV %",
        "Sentiment (mean±std)",
        "Platform Coverage",
    ]
    table_data = [
        [wrap_table_cell(header, theme, is_header=True) for header in headers]
    ]

    client_name = data.get("client_name", "")
    competitors = data.get("competitors", [])
    aggregates = data.get("aggregates")
    brand_performance = data.get("brand_performance", {})

    if not aggregates:
        return table_data

    # Prioritize client first, then competitors
    all_brands = [client_name] + [comp for comp in competitors if comp != client_name]

    for brand in all_brands:
        if not brand:
            continue

        # Extract metrics
        mentions = aggregates.mention_rate.get(
            brand, 0.0
        )  # This is already per 100 questions
        saiv_percentage = aggregates.saiv.get(brand, 0.0) * 100

        sentiment_mean = aggregates.sentiment_mean.get(brand, 0.0)
        sentiment_std = aggregates.sentiment_std.get(brand, 0.0)
        sentiment_count = aggregates.n.get(brand, 0)

        # Platform coverage from brand_performance
        brand_data = brand_performance.get(brand, {})
        platforms = brand_data.get("platforms", set())
        platform_count = len(platforms) if platforms else 0

        # Format data
        mentions_str = f"{mentions:.1f}"
        saiv_str = f"{saiv_percentage:.1f}"

        if sentiment_count >= 3:
            sentiment_str = f"{sentiment_mean:+.2f}±{sentiment_std:.2f}"
        elif sentiment_count > 0:
            sentiment_str = f"{sentiment_mean:+.2f} (low n)"
        else:
            sentiment_str = "N/A"

        platform_str = f"{platform_count} platforms"

        # Add row to table - wrapped in Paragraphs for proper text wrapping
        row = [brand, mentions_str, saiv_str, sentiment_str, platform_str]
        table_data.append(
            [wrap_table_cell(cell, theme, is_header=False) for cell in row]
        )

    return table_data


def build_sentiment_comparison(theme: Theme, data: Dict[str, Any]) -> List[Any]:
    """
    Build sentiment comparison visualization.

    Args:
        theme: Theme configuration
        data: Audit data dictionary

    Returns:
        List of flowables for sentiment comparison
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Subheading
    story.append(Paragraph("Brand Sentiment Comparison", styles["heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    aggregates = data.get("aggregates")
    if aggregates and aggregates.sentiment_mean:
        # Filter brands with sufficient data
        sentiment_data = {
            brand: sentiment
            for brand, sentiment in aggregates.sentiment_mean.items()
            if aggregates.n.get(brand, 0) >= 5  # Minimum sample size
        }

        if sentiment_data:
            try:
                # Create sentiment chart
                sentiment_chart = charts.create_sentiment_comparison_chart(
                    sentiment_data, title="Average Sentiment by Brand"
                )
                story.append(Spacer(1, 0.1 * inch))  # Space before chart
                story.append(sentiment_chart)
                story.append(Spacer(1, 0.1 * inch))  # Space after chart

                # Alt text
                client_name = data.get("client_name", "Client")
                client_sentiment = sentiment_data.get(client_name, 0.0)
                alt_text = f"Horizontal bar chart showing {client_name} sentiment at {client_sentiment:+.2f} compared to competitors"

                alt_caption = create_alt_text_caption(alt_text)
                story.append(alt_caption)
                story.append(Spacer(1, 0.2 * inch))

            except Exception:
                story.append(
                    Paragraph(
                        "Sentiment visualization temporarily unavailable. Data available in table above.",
                        styles["body"],
                    )
                )
                story.append(Spacer(1, 0.1 * inch))
        else:
            story.append(
                Paragraph(
                    "Insufficient sentiment data for meaningful comparison. Continue auditing to build sentiment analysis.",
                    styles["body"],
                )
            )
            story.append(Spacer(1, 0.1 * inch))

    return story


def build_market_share_analysis(theme: Theme, data: Dict[str, Any]) -> List[Any]:
    """
    Build market share pie chart and analysis.

    Args:
        theme: Theme configuration
        data: Audit data dictionary

    Returns:
        List of flowables for market share analysis
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Subheading
    story.append(Paragraph("AI Voice Market Share", styles["heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    # Calculate total mentions per brand
    brand_performance = data.get("brand_performance", {})
    if brand_performance:
        brand_mentions = {
            brand: brand_data.get("total_mentions", 0)
            for brand, brand_data in brand_performance.items()
        }

        # Filter out brands with no mentions
        brand_mentions = {k: v for k, v in brand_mentions.items() if v > 0}

        if brand_mentions:
            try:
                client_name = data.get("client_name", "")
                market_share_chart = charts.create_market_share_pie_chart(
                    brand_mentions, client_name, title="Market Share by Total Mentions"
                )
                story.append(Spacer(1, 0.1 * inch))  # Space before chart
                story.append(market_share_chart)
                story.append(Spacer(1, 0.1 * inch))  # Space after chart

                # Alt text
                total_mentions = sum(brand_mentions.values())
                client_mentions = brand_mentions.get(client_name, 0)
                client_percentage = (
                    (client_mentions / total_mentions * 100)
                    if total_mentions > 0
                    else 0
                )

                alt_text = f"Pie chart showing {client_name} captures {client_percentage:.1f}% of total brand mentions ({client_mentions} of {total_mentions} mentions)"

                alt_caption = create_alt_text_caption(alt_text)
                story.append(alt_caption)
                story.append(Spacer(1, 0.2 * inch))

            except Exception:
                story.append(
                    Paragraph(
                        "Market share visualization temporarily unavailable.",
                        styles["body"],
                    )
                )
                story.append(Spacer(1, 0.1 * inch))

    return story


def build_competitive_insights(
    theme: Theme, data: Dict[str, Any], previous_period: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """
    Build competitive insights and analysis.

    Args:
        theme: Theme configuration
        data: Current audit data
        previous_period: Previous period data for comparison

    Returns:
        List of flowables for insights
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Subheading
    story.append(Paragraph("Competitive Insights", styles["heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    client_name = data.get("client_name", "Client")
    aggregates = data.get("aggregates")

    insights = []

    if aggregates:
        # Market leader identification
        saiv_rankings = sorted(
            aggregates.saiv.items(), key=lambda x: x[1], reverse=True
        )
        if saiv_rankings:
            leader_brand, leader_saiv = saiv_rankings[0]
            client_rank = next(
                (
                    i
                    for i, (brand, _) in enumerate(saiv_rankings, 1)
                    if brand == client_name
                ),
                None,
            )

            if leader_brand == client_name:
                insights.append(
                    f"{client_name} leads the competitive landscape with {leader_saiv*100:.1f}% SAIV."
                )
            else:
                insights.append(
                    f"{leader_brand} leads with {leader_saiv*100:.1f}% SAIV. {client_name} ranks #{client_rank} of {len(saiv_rankings)} brands analyzed."
                )

        # Performance gaps
        client_saiv = aggregates.saiv.get(client_name, 0.0)
        competitor_saivs = [
            saiv for brand, saiv in aggregates.saiv.items() if brand != client_name
        ]

        if competitor_saivs:
            max_competitor_saiv = max(competitor_saivs)
            if client_saiv < max_competitor_saiv:
                gap = (max_competitor_saiv - client_saiv) * 100
                insights.append(
                    f"Primary optimization opportunity: close {gap:.1f}pp SAIV gap with leading competitor."
                )

        # Sentiment positioning
        client_sentiment = aggregates.sentiment_mean.get(client_name, 0.0)
        competitor_sentiments = [
            sentiment
            for brand, sentiment in aggregates.sentiment_mean.items()
            if brand != client_name and aggregates.n.get(brand, 0) >= 5
        ]

        if competitor_sentiments:
            avg_competitor_sentiment = sum(competitor_sentiments) / len(
                competitor_sentiments
            )
            sentiment_advantage = client_sentiment - avg_competitor_sentiment

            if sentiment_advantage > 0.1:
                insights.append(
                    f"Sentiment advantage: {client_name} averages {sentiment_advantage:+.2f} points above competitors."
                )
            elif sentiment_advantage < -0.1:
                insights.append(
                    f"Sentiment challenge: {client_name} trails competitors by {abs(sentiment_advantage):.2f} points on average."
                )

    # Add insights to story
    if insights:
        for insight in insights:
            story.append(Paragraph(f"• {insight}", styles["body"]))
            story.append(Spacer(1, 6))
    else:
        story.append(
            Paragraph(
                "Competitive insights will develop as more audit data becomes available.",
                styles["body"],
            )
        )

    return story

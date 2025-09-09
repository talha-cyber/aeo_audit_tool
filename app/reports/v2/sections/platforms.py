"""
Platform analysis section builder for AEO audit reports.

This module creates platform-specific analysis with:
- Per-platform performance breakdown
- Platform distribution charts
- Platform-specific insights and recommendations
"""

from __future__ import annotations

from typing import Any, Dict, List

from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table

from .. import charts
from ..accessibility import create_alt_text_caption, create_section_heading
from ..theme import Theme, create_paragraph_styles, create_table_styles


def build(theme: Theme, data: Dict[str, Any]) -> List[Any]:
    """
    Build platform analysis section.

    Args:
        theme: Theme configuration for styling
        data: Audit data dictionary

    Returns:
        List of ReportLab flowables for platform analysis
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Section heading
    heading = create_section_heading("Platform Performance Analysis", 1, theme)
    story.append(heading)
    story.append(Spacer(1, 0.2 * inch))

    # Platform distribution overview
    distribution_section = build_platform_distribution(theme, data)
    story.extend(distribution_section)

    # Client performance by platform
    client_performance_section = build_client_platform_performance(theme, data)
    story.extend(client_performance_section)

    # Detailed platform breakdown
    detailed_breakdown = build_detailed_platform_breakdown(theme, data)
    story.extend(detailed_breakdown)

    # Platform-specific insights
    insights_section = build_platform_insights(theme, data)
    story.extend(insights_section)

    return story


def build_platform_distribution(theme: Theme, data: Dict[str, Any]) -> List[Any]:
    """
    Build platform distribution overview.

    Args:
        theme: Theme configuration
        data: Audit data dictionary

    Returns:
        List of flowables for platform distribution
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Subheading
    story.append(Paragraph("Platform Coverage Overview", styles["heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    platform_stats = data.get("platform_stats", {})

    if platform_stats:
        # Calculate total questions per platform
        platform_totals = {
            platform: stats.get("total_questions", 0)
            for platform, stats in platform_stats.items()
        }

        try:
            # Create donut chart
            distribution_chart = charts.create_platform_donut_chart(
                platform_totals, title="Query Distribution Across Platforms"
            )
            story.append(Spacer(1, 0.1 * inch))  # Space before chart
            story.append(distribution_chart)
            story.append(Spacer(1, 0.1 * inch))  # Space after chart

            # Alt text
            total_queries = sum(platform_totals.values())
            platform_count = len(platform_totals)
            alt_text = f"Donut chart showing {total_queries:,} total queries distributed across {platform_count} AI platforms"

            alt_caption = create_alt_text_caption(alt_text)
            story.append(alt_caption)
            story.append(Spacer(1, 0.2 * inch))

        except Exception:
            # Fallback text summary
            total_queries = sum(platform_totals.values())
            summary_text = f"Analysis covers {total_queries:,} queries across {len(platform_totals)} platforms: {', '.join(platform_totals.keys())}"
            story.append(Paragraph(summary_text, styles["body"]))
            story.append(Spacer(1, 0.1 * inch))
    else:
        story.append(
            Paragraph("Platform distribution data is being processed.", styles["body"])
        )
        story.append(Spacer(1, 0.1 * inch))

    return story


def build_client_platform_performance(theme: Theme, data: Dict[str, Any]) -> List[Any]:
    """
    Build client performance across platforms.

    Args:
        theme: Theme configuration
        data: Audit data dictionary

    Returns:
        List of flowables for client platform performance
    """
    story = []
    styles = create_paragraph_styles(theme)

    client_name = data.get("client_name", "Client")

    # Subheading
    story.append(
        Paragraph(f"{client_name} Performance by Platform", styles["heading2"])
    )
    story.append(Spacer(1, 0.1 * inch))

    platform_stats = data.get("platform_stats", {})

    if platform_stats:
        # Extract client mentions per platform
        client_platform_data = {}
        for platform, stats in platform_stats.items():
            brand_mentions = stats.get("brand_mentions", {})
            client_mentions = brand_mentions.get(client_name, 0)
            client_platform_data[platform] = client_mentions

        if any(mentions > 0 for mentions in client_platform_data.values()):
            try:
                # Create client platform performance chart
                performance_chart = charts.create_platform_performance_chart(
                    client_platform_data,
                    client_name,
                    title=f"{client_name} Mentions by Platform",
                )
                story.append(Spacer(1, 0.1 * inch))  # Space before chart
                story.append(performance_chart)
                story.append(Spacer(1, 0.1 * inch))  # Space after chart

                # Alt text
                total_client_mentions = sum(client_platform_data.values())
                best_platform = max(
                    client_platform_data.keys(), key=lambda x: client_platform_data[x]
                )
                best_count = client_platform_data[best_platform]

                alt_text = f"Bar chart showing {client_name} with {total_client_mentions} total mentions, performing best on {best_platform} with {best_count} mentions"

                alt_caption = create_alt_text_caption(alt_text)
                story.append(alt_caption)
                story.append(Spacer(1, 0.2 * inch))

            except Exception:
                # Fallback text summary
                best_platform = max(
                    client_platform_data.keys(), key=lambda x: client_platform_data[x]
                )
                summary_text = f"{client_name} performs best on {best_platform} with {client_platform_data[best_platform]} mentions."
                story.append(Paragraph(summary_text, styles["body"]))
                story.append(Spacer(1, 0.1 * inch))
        else:
            story.append(
                Paragraph(
                    f"No {client_name} mentions detected across analyzed platforms.",
                    styles["body"],
                )
            )
            story.append(Spacer(1, 0.1 * inch))

    return story


def build_detailed_platform_breakdown(theme: Theme, data: Dict[str, Any]) -> List[Any]:
    """
    Build detailed platform-by-platform breakdown.

    Args:
        theme: Theme configuration
        data: Audit data dictionary

    Returns:
        List of flowables for detailed breakdown
    """
    story = []
    styles = create_paragraph_styles(theme)
    table_styles = create_table_styles(theme)

    # Subheading
    story.append(Paragraph("Detailed Platform Analysis", styles["heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    platform_stats = data.get("platform_stats", {})
    client_name = data.get("client_name", "Client")

    if not platform_stats:
        story.append(
            Paragraph("Platform analysis data is being processed.", styles["body"])
        )
        return story

    # Build platform performance table
    table_data = [
        [
            "Platform",
            "Total Queries",
            "Client Mentions",
            "Mention Rate %",
            "Avg Sentiment",
        ]
    ]

    for platform, stats in platform_stats.items():
        total_questions = stats.get("total_questions", 0)
        brand_mentions = stats.get("brand_mentions", {})
        client_mentions = brand_mentions.get(client_name, 0)

        # Calculate mention rate
        mention_rate = (
            (client_mentions / total_questions * 100) if total_questions > 0 else 0.0
        )

        # Get average sentiment for this platform
        avg_sentiment = stats.get("avg_sentiment", {}).get(client_name, 0.0)

        table_data.append(
            [
                platform.title(),
                f"{total_questions:,}",
                str(client_mentions),
                f"{mention_rate:.1f}%",
                f"{avg_sentiment:+.2f}" if avg_sentiment != 0 else "N/A",
            ]
        )

    if len(table_data) > 1:  # Has data rows
        platform_table = Table(
            table_data,
            colWidths=[
                1.4 * inch,
                1.15 * inch,
                1.15 * inch,
                1.15 * inch,
                1.15 * inch,
            ],  # Total: 6.0 inches
            hAlign="LEFT",
        )
        platform_table.setStyle(table_styles["standard"])
        story.append(platform_table)

        # Table caption
        story.append(
            Paragraph(
                f"<i>Platform performance metrics for {client_name} across {len(platform_stats)} AI platforms.</i>",
                styles["caption"],
            )
        )
        story.append(Spacer(1, 0.2 * inch))

    return story


def build_platform_insights(theme: Theme, data: Dict[str, Any]) -> List[Any]:
    """
    Build platform-specific insights and recommendations.

    Args:
        theme: Theme configuration
        data: Audit data dictionary

    Returns:
        List of flowables for platform insights
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Subheading
    story.append(Paragraph("Platform Performance Insights", styles["heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    platform_stats = data.get("platform_stats", {})
    client_name = data.get("client_name", "Client")

    insights = []

    if platform_stats:
        # Calculate performance metrics per platform
        platform_performance = {}

        for platform, stats in platform_stats.items():
            total_questions = stats.get("total_questions", 0)
            brand_mentions = stats.get("brand_mentions", {})
            client_mentions = brand_mentions.get(client_name, 0)

            if total_questions > 0:
                mention_rate = (client_mentions / total_questions) * 100
                platform_performance[platform] = {
                    "mention_rate": mention_rate,
                    "mentions": client_mentions,
                    "total_questions": total_questions,
                    "sentiment": stats.get("avg_sentiment", {}).get(client_name, 0.0),
                }

        if platform_performance:
            # Identify best and worst performing platforms
            platforms_by_rate = sorted(
                platform_performance.items(),
                key=lambda x: x[1]["mention_rate"],
                reverse=True,
            )

            if len(platforms_by_rate) > 1:
                best_platform, best_stats = platforms_by_rate[0]
                worst_platform, worst_stats = platforms_by_rate[-1]

                # Performance variation insight
                if best_stats["mention_rate"] > worst_stats["mention_rate"] * 2:
                    insights.append(
                        f"Significant platform variation: {best_platform} outperforms {worst_platform} "
                        f"by {best_stats['mention_rate'] - worst_stats['mention_rate']:.1f} percentage points "
                        f"({best_stats['mention_rate']:.1f}% vs {worst_stats['mention_rate']:.1f}%)."
                    )

                # Optimization opportunities
                if (
                    worst_stats["mention_rate"] < 5.0
                    and worst_stats["total_questions"] >= 20
                ):
                    insights.append(
                        f"Optimization opportunity: {worst_platform} shows low mention rate "
                        f"({worst_stats['mention_rate']:.1f}%) despite significant query volume "
                        f"({worst_stats['total_questions']} questions)."
                    )

                # Strong performance recognition
                if best_stats["mention_rate"] > 15.0:
                    insights.append(
                        f"Strong performance: {best_platform} achieves {best_stats['mention_rate']:.1f}% mention rate, "
                        f"indicating effective AI optimization for this platform."
                    )

            # Sentiment insights by platform
            sentiment_by_platform = {
                platform: stats["sentiment"]
                for platform, stats in platform_performance.items()
                if stats["sentiment"] != 0.0 and stats["mentions"] >= 3
            }

            if sentiment_by_platform:
                best_sentiment_platform = max(
                    sentiment_by_platform.keys(), key=lambda x: sentiment_by_platform[x]
                )
                worst_sentiment_platform = min(
                    sentiment_by_platform.keys(), key=lambda x: sentiment_by_platform[x]
                )

                if (
                    sentiment_by_platform[best_sentiment_platform]
                    > sentiment_by_platform[worst_sentiment_platform] + 0.3
                ):
                    insights.append(
                        f"Sentiment varies by platform: most positive on {best_sentiment_platform} "
                        f"({sentiment_by_platform[best_sentiment_platform]:+.2f}) vs {worst_sentiment_platform} "
                        f"({sentiment_by_platform[worst_sentiment_platform]:+.2f})."
                    )

        # Coverage insights
        total_platforms = len(platform_stats)
        active_platforms = len(
            [p for p, s in platform_performance.items() if s["mentions"] > 0]
        )

        if active_platforms < total_platforms:
            inactive_count = total_platforms - active_platforms
            insights.append(
                f"Coverage gap: {client_name} has no mentions on {inactive_count} of {total_platforms} platforms analyzed. "
                f"Consider platform-specific optimization strategies."
            )

    # Add insights to story
    if insights:
        for insight in insights:
            story.append(Paragraph(f"• {insight}", styles["body"]))
            story.append(Spacer(1, 6))
    else:
        story.append(
            Paragraph(
                "Platform-specific insights will develop as audit data expands across more platforms and queries.",
                styles["body"],
            )
        )

    story.append(Spacer(1, 0.1 * inch))

    # Platform recommendations
    story.append(
        Paragraph("Platform Optimization Recommendations:", styles["heading3"])
    )
    story.append(Spacer(1, 0.05 * inch))

    recommendations = [
        "Focus optimization efforts on platforms with high query volumes but low mention rates",
        "Analyze top-performing platforms to identify successful content patterns",
        "Implement platform-specific structured data and content strategies",
        "Monitor sentiment variations to address platform-specific reputation issues",
    ]

    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles["body"]))
        story.append(Spacer(1, 4))

    return story

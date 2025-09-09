"""
Recommendations section builder for AEO audit reports.

This module creates actionable recommendations with:
- Effort × Impact matrix analysis
- 30-60-90 day implementation plans
- Specific KPIs and success metrics
- Owner assignment and accountability
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table

from ..accessibility import create_section_heading
from ..theme import (
    Theme,
    clean_html_content,
    create_paragraph_styles,
    create_table_styles,
    wrap_table_cell,
)


def build(
    theme: Theme, data: Dict[str, Any], previous_period: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """
    Build strategic recommendations section.

    Args:
        theme: Theme configuration for styling
        data: Current period audit data
        previous_period: Previous period data for comparison

    Returns:
        List of ReportLab flowables for recommendations
    """
    story = []
    styles = create_paragraph_styles(theme)

    # Section heading
    heading = create_section_heading("Strategic Recommendations", 1, theme)
    story.append(heading)
    story.append(Spacer(1, 0.2 * inch))

    # Generate recommendations based on analysis
    recommendations = generate_recommendations(data, previous_period)

    # Introduction
    intro_section = build_recommendations_intro(theme, data, recommendations)
    story.extend(intro_section)

    # Detailed recommendations
    for i, rec in enumerate(recommendations, 1):
        rec_section = build_recommendation_detail(theme, rec, i)
        story.extend(rec_section)

    # Implementation timeline summary
    timeline_section = build_implementation_timeline(theme, recommendations)
    story.extend(timeline_section)

    return story


def generate_recommendations(
    data: Dict[str, Any], previous_period: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate recommendations based on audit data analysis.

    Args:
        data: Current audit data
        previous_period: Previous period data for comparison

    Returns:
        List of recommendation dictionaries
    """
    recommendations = []
    client_name = data.get("client_name", "Client")
    aggregates = data.get("aggregates")

    if not aggregates:
        return get_default_recommendations(client_name)

    # SAIV-based recommendations
    client_saiv = aggregates.saiv.get(client_name, 0.0)
    competitor_saivs = [
        saiv
        for brand, saiv in aggregates.saiv.items()
        if brand != client_name and saiv > 0
    ]

    if competitor_saivs:
        avg_competitor_saiv = sum(competitor_saivs) / len(competitor_saivs)
        max_competitor_saiv = max(competitor_saivs)

        if client_saiv < avg_competitor_saiv:
            gap_percentage = (avg_competitor_saiv - client_saiv) * 100
            recommendations.append(
                {
                    "title": "Close SAIV Competitive Gap",
                    "rationale": f"{client_name} SAIV ({client_saiv*100:.1f}%) trails competitor average by {gap_percentage:.1f}pp",
                    "effort": "L" if gap_percentage > 20 else "M",
                    "impact": "15-25% SAIV increase",
                    "plan": {
                        "30d": [
                            "Audit current content for AI optimization gaps",
                            "Implement structured data markup",
                        ],
                        "60d": [
                            "Create FAQ content targeting top competitor topics",
                            "Optimize for featured snippet capture",
                        ],
                        "90d": [
                            "Launch comprehensive AI content strategy",
                            "Monitor and iterate based on results",
                        ],
                    },
                    "owner": "Content Strategy Lead",
                    "kpi": "SAIV percentage",
                    "target": f"≥{avg_competitor_saiv*100:.1f}% within 90 days",
                    "priority": "High",
                }
            )

    # Sentiment-based recommendations
    client_sentiment = aggregates.sentiment_mean.get(client_name, 0.0)
    sentiment_count = aggregates.n.get(client_name, 0)

    if sentiment_count >= 10:
        if client_sentiment < -0.1:
            recommendations.append(
                {
                    "title": "Improve Brand Sentiment in AI Responses",
                    "rationale": f"Negative sentiment detected ({client_sentiment:+.2f}) across {sentiment_count} mentions",
                    "effort": "M",
                    "impact": "20-30% sentiment improvement",
                    "plan": {
                        "30d": [
                            "Analyze negative sentiment triggers",
                            "Create positive counter-narrative content",
                        ],
                        "60d": [
                            "Publish authoritative thought leadership content",
                            "Engage with industry discussions",
                        ],
                        "90d": [
                            "Launch reputation management campaign",
                            "Monitor sentiment trend improvements",
                        ],
                    },
                    "owner": "Brand Marketing Manager",
                    "kpi": "Average sentiment score",
                    "target": "Achieve positive sentiment (>+0.1)",
                    "priority": "High",
                }
            )
        elif client_sentiment < 0.2:
            recommendations.append(
                {
                    "title": "Enhance Positive Brand Positioning",
                    "rationale": f"Neutral sentiment ({client_sentiment:+.2f}) presents opportunity for positive differentiation",
                    "effort": "M",
                    "impact": "10-20% sentiment boost",
                    "plan": {
                        "30d": [
                            "Identify positive brand attributes",
                            "Create success story content",
                        ],
                        "60d": [
                            "Optimize content for positive keyword associations",
                            "Publish customer testimonials",
                        ],
                        "90d": [
                            "Scale positive content distribution",
                            "Measure sentiment improvements",
                        ],
                    },
                    "owner": "Content Marketing Team",
                    "kpi": "Average sentiment score",
                    "target": "≥+0.3 within 90 days",
                    "priority": "Medium",
                }
            )

    # Platform-specific recommendations
    platform_stats = data.get("platform_stats", {})
    if platform_stats:
        platform_performance = {}

        for platform, stats in platform_stats.items():
            total_questions = stats.get("total_questions", 0)
            brand_mentions = stats.get("brand_mentions", {})
            client_mentions = brand_mentions.get(client_name, 0)

            if total_questions > 0:
                mention_rate = (client_mentions / total_questions) * 100
                platform_performance[platform] = mention_rate

        # Find underperforming platforms
        if platform_performance:
            avg_performance = sum(platform_performance.values()) / len(
                platform_performance.values()
            )
            underperforming = [
                platform
                for platform, rate in platform_performance.items()
                if rate < avg_performance * 0.5
                and platform_stats[platform]["total_questions"] >= 20
            ]

            if underperforming:
                worst_platform = min(
                    underperforming, key=lambda x: platform_performance[x]
                )
                recommendations.append(
                    {
                        "title": f"Optimize Performance on {worst_platform.title()}",
                        "rationale": f'Low mention rate ({platform_performance[worst_platform]:.1f}%) despite {platform_stats[worst_platform]["total_questions"]} queries',
                        "effort": "M",
                        "impact": f"5-15% mention rate increase on {worst_platform}",
                        "plan": {
                            "30d": [
                                f"Research {worst_platform} ranking factors",
                                "Audit competitor content on platform",
                            ],
                            "60d": [
                                f"Create {worst_platform}-optimized content",
                                "Implement platform-specific SEO",
                            ],
                            "90d": [
                                "Monitor performance improvements",
                                "Scale successful tactics",
                            ],
                        },
                        "owner": "SEO Specialist",
                        "kpi": f"Mention rate on {worst_platform}",
                        "target": f"≥{avg_performance:.1f}% within 90 days",
                        "priority": "Medium",
                    }
                )

    # If no specific recommendations generated, use defaults
    if not recommendations:
        recommendations = get_default_recommendations(client_name)

    # Ensure we have at least 3 recommendations but no more than 5
    if len(recommendations) < 3:
        recommendations.extend(get_supplementary_recommendations(client_name))

    return recommendations[:5]  # Cap at 5 recommendations


def get_default_recommendations(client_name: str) -> List[Dict[str, Any]]:
    """Get default recommendations when data-driven ones aren't available."""
    return [
        {
            "title": "Establish AI Content Optimization Foundation",
            "rationale": "Build systematic approach to AI platform visibility optimization",
            "effort": "L",
            "impact": "10-20% visibility improvement",
            "plan": {
                "30d": [
                    "Audit current content for AI optimization potential",
                    "Research competitor AI strategies",
                ],
                "60d": [
                    "Implement structured data markup",
                    "Create FAQ and How-To content",
                ],
                "90d": [
                    "Launch comprehensive AI content strategy",
                    "Begin performance monitoring",
                ],
            },
            "owner": "Digital Marketing Manager",
            "kpi": "AI platform mentions",
            "target": "Establish baseline and 10% monthly growth",
            "priority": "High",
        },
        {
            "title": "Implement Competitive Intelligence Monitoring",
            "rationale": "Regular monitoring enables proactive optimization and competitive advantage",
            "effort": "S",
            "impact": "Ongoing competitive insights",
            "plan": {
                "30d": [
                    "Set up automated AEO audit reporting",
                    "Define competitive benchmarks",
                ],
                "60d": [
                    "Establish monthly review process",
                    "Create competitive response playbook",
                ],
                "90d": [
                    "Launch competitive differentiation campaigns",
                    "Measure market share gains",
                ],
            },
            "owner": "Marketing Strategy Lead",
            "kpi": "Market share of AI voice",
            "target": "Monthly competitive reporting",
            "priority": "Medium",
        },
        {
            "title": "Develop Platform-Specific Optimization Strategy",
            "rationale": "Each AI platform has unique ranking factors requiring tailored approaches",
            "effort": "M",
            "impact": "15-25% platform performance improvement",
            "plan": {
                "30d": [
                    "Research platform-specific optimization factors",
                    "Prioritize highest-value platforms",
                ],
                "60d": [
                    "Create platform-tailored content strategies",
                    "Implement technical optimizations",
                ],
                "90d": [
                    "Scale successful tactics across platforms",
                    "Measure cross-platform performance",
                ],
            },
            "owner": "Content Strategy Team",
            "kpi": "Platform-specific mention rates",
            "target": "20% improvement per platform",
            "priority": "Medium",
        },
    ]


def get_supplementary_recommendations(client_name: str) -> List[Dict[str, Any]]:
    """Get supplementary recommendations to reach minimum threshold."""
    return [
        {
            "title": "Establish Thought Leadership Content Program",
            "rationale": "Position as industry authority to improve AI platform citations",
            "effort": "M",
            "impact": "Brand authority and mention quality improvement",
            "plan": {
                "30d": [
                    "Identify thought leadership topics",
                    "Plan expert content calendar",
                ],
                "60d": [
                    "Publish authoritative industry content",
                    "Engage in industry discussions",
                ],
                "90d": ["Measure authority improvements", "Scale content production"],
            },
            "owner": "Content Marketing Lead",
            "kpi": "Quality of mentions and citations",
            "target": "Increased authoritative references",
            "priority": "Low",
        }
    ]


def build_recommendations_intro(
    theme: Theme, data: Dict[str, Any], recommendations: List[Dict[str, Any]]
) -> List[Any]:
    """Build introduction to recommendations section."""
    story = []
    styles = create_paragraph_styles(theme)

    client_name = data.get("client_name", "Client")

    intro_text = f"""
    Based on the competitive analysis and platform performance data, we recommend the following strategic initiatives
    for {client_name}. Each recommendation includes effort assessment, expected impact, implementation timeline,
    and success metrics.

    <b>Priority Framework:</b> High priority items address significant competitive gaps or negative trends.
    Medium priority items focus on optimization and growth opportunities. Low priority items support long-term positioning.
    """

    story.append(Paragraph(intro_text, styles["body"]))
    story.append(Spacer(1, 0.2 * inch))

    # Priority summary
    high_priority = len([r for r in recommendations if r.get("priority") == "High"])
    medium_priority = len([r for r in recommendations if r.get("priority") == "Medium"])
    low_priority = len([r for r in recommendations if r.get("priority") == "Low"])

    priority_text = f"<b>Recommendations Summary:</b> {high_priority} High Priority, {medium_priority} Medium Priority, {low_priority} Low Priority"
    story.append(Paragraph(priority_text, styles["emphasis"]))
    story.append(Spacer(1, 0.2 * inch))

    return story


def build_recommendation_detail(
    theme: Theme, rec: Dict[str, Any], index: int
) -> List[Any]:
    """Build detailed recommendation section."""
    story = []
    styles = create_paragraph_styles(theme)
    table_styles = create_table_styles(theme)

    # Recommendation title with priority
    priority_color = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#27AE60"}.get(
        rec.get("priority", "Medium"), "#F39C12"
    )

    title_text = (
        f"<font color='{priority_color}'>●</font> <b>{index}. {rec['title']}</b>"
    )
    story.append(Paragraph(title_text, styles["recommendation_title"]))
    story.append(Spacer(1, 0.05 * inch))

    # Rationale
    story.append(Paragraph(f"<b>Rationale:</b> {rec['rationale']}", styles["body"]))
    story.append(Spacer(1, 0.05 * inch))

    # Effort and Impact
    effort_impact_text = f"<b>Effort:</b> {get_effort_description(rec['effort'])} | <b>Expected Impact:</b> {rec['impact']}"
    story.append(Paragraph(effort_impact_text, styles["body"]))
    story.append(Spacer(1, 0.1 * inch))

    # Implementation plan table
    plan_data = [["Timeline", "Key Activities"]]
    for period, activities in rec["plan"].items():
        activities_text = "<br/>".join([f"• {activity}" for activity in activities])
        plan_data.append([period.upper(), activities_text])

    plan_table = Table(plan_data, colWidths=[1 * inch, 5 * inch], hAlign="LEFT")
    plan_table.setStyle(table_styles["highlight"])
    story.append(plan_table)
    story.append(Spacer(1, 0.1 * inch))

    # Success metrics
    metrics_text = f"<b>Owner:</b> {rec['owner']} | <b>KPI:</b> {rec['kpi']} | <b>Target:</b> {rec['target']}"
    story.append(Paragraph(metrics_text, styles["kpi_highlight"]))
    story.append(Spacer(1, 0.2 * inch))

    return story


def build_implementation_timeline(
    theme: Theme, recommendations: List[Dict[str, Any]]
) -> List[Any]:
    """Build implementation timeline summary."""
    story = []
    styles = create_paragraph_styles(theme)
    table_styles = create_table_styles(theme)

    # Timeline summary heading
    story.append(Paragraph("Implementation Timeline Summary", styles["heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    # Create timeline table with proper column widths for page
    headers = ["30 Days", "60 Days", "90 Days"]
    timeline_data = [
        [wrap_table_cell(header, theme, is_header=True) for header in headers]
    ]

    # Collect activities by timeframe (limit to prevent overflow)
    activities_30d = []
    activities_60d = []
    activities_90d = []

    for i, rec in enumerate(recommendations, 1):
        plan = rec.get("plan", {})

        if "30d" in plan and len(activities_30d) < 2:  # Limit to 2 items per column
            activities_30d.append(f"<b>Rec {i}:</b> {plan['30d'][0]}")

        if "60d" in plan and len(activities_60d) < 2:  # Limit to 2 items per column
            activities_60d.append(f"<b>Rec {i}:</b> {plan['60d'][0]}")

        if "90d" in plan and len(activities_90d) < 2:  # Limit to 2 items per column
            activities_90d.append(f"<b>Rec {i}:</b> {plan['90d'][0]}")

    # Format activities for table with proper wrapping
    cell_30d = "<br/>".join(activities_30d) if activities_30d else "Planning phase"
    cell_60d = (
        "<br/>".join(activities_60d) if activities_60d else "Implementation phase"
    )
    cell_90d = "<br/>".join(activities_90d) if activities_90d else "Optimization phase"

    timeline_data.append(
        [
            wrap_table_cell(clean_html_content(cell_30d), theme, is_header=False),
            wrap_table_cell(clean_html_content(cell_60d), theme, is_header=False),
            wrap_table_cell(clean_html_content(cell_90d), theme, is_header=False),
        ]
    )

    timeline_table = Table(
        timeline_data, colWidths=[2 * inch, 2 * inch, 2 * inch], hAlign="LEFT"
    )
    timeline_table.setStyle(table_styles["standard"])
    story.append(timeline_table)

    story.append(Spacer(1, 0.1 * inch))

    # Next steps
    next_steps_text = """
    <b>Recommended Next Steps:</b><br/>
    1. Review and prioritize recommendations based on business objectives<br/>
    2. Assign ownership and establish success metrics<br/>
    3. Begin 30-day implementation activities for high-priority items<br/>
    4. Schedule monthly review meetings to track progress<br/>
    5. Continue regular AEO auditing to measure improvement
    """

    story.append(Paragraph(next_steps_text, styles["body"]))

    return story


def get_effort_description(effort_code: str) -> str:
    """Convert effort code to description."""
    effort_map = {
        "S": "Small (1-2 weeks)",
        "M": "Medium (1-2 months)",
        "L": "Large (3+ months)",
    }
    return effort_map.get(effort_code, "Medium (1-2 months)")

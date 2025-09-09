"""
Report generation service for AEO audit analysis.

This module generates comprehensive PDF reports containing competitive analysis,
brand performance metrics, platform insights, and strategic recommendations
based on audit results.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy.orm import Session

from app.models import audit as audit_models
from app.models import response as response_models
from app.models.report import Report
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generate comprehensive audit reports in PDF format.

    Provides functionality to create executive summaries, competitive analysis,
    platform performance reports, and strategic recommendations based on
    audit run results.
    """

    def __init__(self, db_session: Session):
        """
        Initialize report generator.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
        self.styles = getSampleStyleSheet()

        # Custom styles
        self.title_style = ParagraphStyle(
            "CustomTitle",
            parent=self.styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor("#2C3E50"),
            alignment=1,  # Center
        )

        self.heading_style = ParagraphStyle(
            "CustomHeading",
            parent=self.styles["Heading2"],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor("#34495E"),
        )

        self.subheading_style = ParagraphStyle(
            "CustomSubheading",
            parent=self.styles["Heading3"],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor("#34495E"),
        )

    def generate_audit_report(
        self,
        audit_run_id: str,
        report_type: str = "comprehensive",
        output_dir: str = "reports",
    ) -> str:
        """
        Generate comprehensive audit report.

        Args:
            audit_run_id: ID of the audit run to report on
            report_type: Type of report (comprehensive, summary, platform_specific)
            output_dir: Directory to save the report

        Returns:
            Path to generated report file

        Raises:
            ValueError: If audit run not found or invalid
        """
        logger.info(f"Generating {report_type} report for audit run {audit_run_id}")

        # Load audit data
        audit_data = self._load_audit_data(audit_run_id)

        if not audit_data:
            raise ValueError(f"Audit run {audit_run_id} not found")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        client_name = audit_data.get("client_name", "Unknown").replace(" ", "_")
        filename = f"AEO_Audit_Report_{client_name}_{timestamp}.pdf"
        filepath = os.path.join(output_dir, filename)

        # Create PDF
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []

        # Check for v2 report types
        if report_type in ["v2", "v2_comprehensive", "v2_enhanced"]:
            return self._generate_v2_report(audit_run_id, report_type, filepath)
        elif report_type == "comprehensive":
            story = self._build_comprehensive_report(audit_data)
        elif report_type == "summary":
            story = self._build_summary_report(audit_data)
        elif report_type == "platform_specific":
            story = self._build_platform_report(audit_data)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

        # Build PDF
        doc.build(story)

        # Create report record in database
        self._create_report_record(audit_run_id, report_type, filepath)

        logger.info(f"Report generated successfully: {filepath}")
        return filepath

    def _create_report_record(self, audit_run_id: str, report_type: str, filepath: str):
        """Create a record for the generated report in the database."""
        try:
            new_report = Report(
                id=str(uuid.uuid4()),
                audit_run_id=audit_run_id,
                report_type=report_type,
                file_path=filepath,
                generated_at=datetime.now(),
            )
            self.db.add(new_report)
            self.db.commit()
            logger.info("Report record created in database.", report_id=new_report.id)
        except Exception as e:
            logger.error(
                "Failed to create report record in database.",
                error=str(e),
                exc_info=True,
            )
            self.db.rollback()

    def _load_audit_data(self, audit_run_id: str) -> Dict[str, Any]:
        """Load and process audit data for reporting."""
        try:
            # Load audit run
            audit_run = (
                self.db.query(audit_models.AuditRun)
                .filter(audit_models.AuditRun.id == audit_run_id)
                .first()
            )

            if not audit_run:
                return {}

            # Load responses
            responses = (
                self.db.query(response_models.Response)
                .filter(response_models.Response.audit_run_id == audit_run_id)
                .all()
            )

            # Extract basic info
            config = audit_run.config or {}
            client_data = config.get("client", {})
            client_name = client_data.get("name", "Unknown Client")
            competitors = client_data.get("competitors", [])
            industry = client_data.get("industry", "Unknown")

            all_brands = [client_name] + competitors

            # Process responses for analysis
            platform_stats = {}
            brand_performance = {}

            for response in responses:
                platform = response.platform
                brand_mentions = response.brand_mentions or {}

                # Initialize platform stats
                if platform not in platform_stats:
                    platform_stats[platform] = {
                        "total_questions": 0,
                        "brand_mentions": {brand: 0 for brand in all_brands},
                        "avg_sentiment": {brand: [] for brand in all_brands},
                    }

                platform_stats[platform]["total_questions"] += 1

                # Process brand mentions (handle actual format from brand detection)
                if brand_mentions and isinstance(brand_mentions, dict):
                    top_brands = brand_mentions.get("top_brands", [])
                    avg_confidence = brand_mentions.get("avg_confidence", 0.8)

                    # Count occurrences of each brand
                    brand_counts = {}
                    for brand in top_brands:
                        brand_counts[brand] = brand_counts.get(brand, 0) + 1

                    # Process each detected brand
                    for brand, count in brand_counts.items():
                        if brand in all_brands:
                            platform_stats[platform]["brand_mentions"][brand] += count
                            platform_stats[platform]["avg_sentiment"][brand].append(
                                avg_confidence
                            )

                            # Overall brand performance
                            if brand not in brand_performance:
                                brand_performance[brand] = {
                                    "total_mentions": 0,
                                    "platforms": set(),
                                    "sentiment_scores": [],
                                    "question_types": {},
                                }

                            brand_performance[brand]["total_mentions"] += count
                            brand_performance[brand]["platforms"].add(platform)
                            brand_performance[brand]["sentiment_scores"].append(
                                avg_confidence
                            )

            # Calculate averages
            for platform in platform_stats:
                for brand in platform_stats[platform]["avg_sentiment"]:
                    scores = platform_stats[platform]["avg_sentiment"][brand]
                    platform_stats[platform]["avg_sentiment"][brand] = (
                        sum(scores) / len(scores) if scores else 0.0
                    )

            # Handle None timestamps by using current time as fallback
            from datetime import datetime

            current_time = datetime.now()
            start_time = audit_run.started_at or current_time
            end_time = audit_run.completed_at or audit_run.started_at or current_time

            return {
                "audit_run": audit_run,
                "client_name": client_name,
                "competitors": competitors,
                "industry": industry,
                "platform_stats": platform_stats,
                "brand_performance": brand_performance,
                "total_responses": len(responses),
                "date_range": {"start": start_time, "end": end_time},
            }

        except Exception as e:
            logger.error(f"Error loading audit data: {e}")
            return {}

    def _build_comprehensive_report(self, data: Dict[str, Any]) -> List:
        """Build comprehensive report story."""
        story = []

        # Title Page
        story.extend(self._create_title_page(data))
        story.append(PageBreak())

        # Executive Summary
        story.extend(self._create_executive_summary(data))
        story.append(PageBreak())

        # Competitive Analysis
        story.extend(self._create_competitive_analysis(data))
        story.append(PageBreak())

        # Platform Performance
        story.extend(self._create_platform_analysis(data))
        story.append(PageBreak())

        # Recommendations
        story.extend(self._create_recommendations(data))

        return story

    def _build_summary_report(self, data: Dict[str, Any]) -> List:
        """Build summary report story."""
        story = []

        # Title Page
        story.extend(self._create_title_page(data))
        story.append(PageBreak())

        # Executive Summary
        story.extend(self._create_executive_summary(data))
        story.append(PageBreak())

        # Key Insights
        story.extend(self._create_key_insights(data))

        return story

    def _build_platform_report(self, data: Dict[str, Any]) -> List:
        """Build platform-specific report story."""
        story = []

        # Title Page
        story.extend(self._create_title_page(data))
        story.append(PageBreak())

        # Platform Performance
        story.extend(self._create_platform_analysis(data))

        return story

    def _create_title_page(self, data: Dict[str, Any]) -> List:
        """Create report title page."""
        story = []

        # Title
        title = f"AEO Competitive Intelligence Report<br/>{data['client_name']}"
        story.append(Paragraph(title, self.title_style))
        story.append(Spacer(1, 0.5 * inch))

        # Report metadata
        end_date = data["date_range"]["end"]
        start_date = data["date_range"]["start"]

        metadata = [
            ["Report Date:", end_date.strftime("%B %d, %Y")],
            [
                "Audit Period:",
                f"{start_date.strftime('%B %d')} - {end_date.strftime('%B %d, %Y')}",
            ],
            ["Total Queries:", str(data["total_responses"])],
            ["Platforms Analyzed:", ", ".join(data["platform_stats"].keys())],
            ["Industry:", data["industry"]],
            ["Competitors:", ", ".join(data["competitors"])],
        ]

        table = Table(metadata, colWidths=[2 * inch, 3 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )

        story.append(table)
        return story

    def _create_executive_summary(self, data: Dict[str, Any]) -> List:
        """Create executive summary section."""
        story = []

        story.append(Paragraph("Executive Summary", self.title_style))

        # Calculate key metrics
        client_mentions = (
            data["brand_performance"]
            .get(data["client_name"], {})
            .get("total_mentions", 0)
        )
        total_competitor_mentions = sum(
            data["brand_performance"].get(comp, {}).get("total_mentions", 0)
            for comp in data["competitors"]
        )

        total_mentions = client_mentions + total_competitor_mentions
        market_share = (
            (client_mentions / total_mentions * 100) if total_mentions > 0 else 0
        )

        # Key findings
        findings = [
            f"<b>AI Visibility Market Share:</b> {data['client_name']} captures {market_share:.1f}% of brand mentions across AI platforms",
            f"<b>Total Brand Mentions:</b> {client_mentions} mentions for {data['client_name']} vs {total_competitor_mentions} for all competitors combined",
            f"<b>Platform Coverage:</b> Analysis across {len(data['platform_stats'])} AI platforms",
            f"<b>Query Volume:</b> {data['total_responses']} total queries analyzed",
        ]

        for finding in findings:
            story.append(Paragraph(finding, self.styles["Normal"]))
            story.append(Spacer(1, 12))

        return story

    def _create_competitive_analysis(self, data: Dict[str, Any]) -> List:
        """Create competitive analysis section."""
        story = []

        story.append(Paragraph("Competitive Analysis", self.title_style))

        # Brand comparison table
        table_data = [["Brand", "Total Mentions", "Avg Sentiment", "Platform Coverage"]]

        for brand in [data["client_name"]] + data["competitors"]:
            brand_data = data["brand_performance"].get(brand, {})
            mentions = brand_data.get("total_mentions", 0)
            sentiment_scores = brand_data.get("sentiment_scores", [])
            avg_sentiment = (
                sum(sentiment_scores) / len(sentiment_scores)
                if sentiment_scores
                else 0.0
            )
            platform_count = len(brand_data.get("platforms", set()))

            sentiment_label = (
                "Positive"
                if avg_sentiment > 0.1
                else "Negative"
                if avg_sentiment < -0.1
                else "Neutral"
            )

            table_data.append(
                [
                    brand,
                    str(mentions),
                    f"{sentiment_label} ({avg_sentiment:.2f})",
                    f"{platform_count} platforms",
                ]
            )

        table = Table(
            table_data, colWidths=[2 * inch, 1 * inch, 1.5 * inch, 1.5 * inch]
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        story.append(table)
        return story

    def _create_platform_analysis(self, data: Dict[str, Any]) -> List:
        """Create platform-specific analysis."""
        story = []

        story.append(Paragraph("Platform Performance Analysis", self.title_style))

        for platform, stats in data["platform_stats"].items():
            story.append(
                Paragraph(f"{platform.title()} Performance", self.heading_style)
            )

            # Platform summary
            client_mentions = stats["brand_mentions"].get(data["client_name"], 0)
            total_questions = stats["total_questions"]
            mention_rate = (
                (client_mentions / total_questions * 100) if total_questions > 0 else 0
            )
            avg_sentiment = stats["avg_sentiment"].get(data["client_name"], 0)

            summary = f"""
            <b>Platform Overview:</b><br/>
            • Total questions analyzed: {total_questions}<br/>
            • {data['client_name']} mentions: {client_mentions}<br/>
            • Mention rate: {mention_rate:.1f}%<br/>
            • Average sentiment: {avg_sentiment:.2f}
            """

            story.append(Paragraph(summary, self.styles["Normal"]))
            story.append(Spacer(1, 20))

        return story

    def _create_key_insights(self, data: Dict[str, Any]) -> List:
        """Create key insights section."""
        story = []

        story.append(Paragraph("Key Insights", self.title_style))

        insights = self._generate_insights(data)

        for insight in insights:
            story.append(Paragraph(f"• {insight}", self.styles["Normal"]))
            story.append(Spacer(1, 12))

        return story

    def _create_recommendations(self, data: Dict[str, Any]) -> List:
        """Create recommendations section."""
        story = []

        story.append(Paragraph("Strategic Recommendations", self.title_style))

        recommendations = self._generate_recommendations(data)

        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. <b>{rec['title']}</b>", self.heading_style))
            story.append(Paragraph(rec["description"], self.styles["Normal"]))
            story.append(
                Paragraph(
                    f"<b>Expected Impact:</b> {rec['impact']}", self.styles["Normal"]
                )
            )
            story.append(Spacer(1, 15))

        return story

    def _generate_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate key insights based on audit data."""
        insights = []

        # Market positioning insight
        brand_mentions = [
            (brand, data["brand_performance"].get(brand, {}).get("total_mentions", 0))
            for brand in [data["client_name"]] + data["competitors"]
        ]

        leader = max(brand_mentions, key=lambda x: x[1])
        if leader[0] != data["client_name"] and leader[1] > 0:
            insights.append(
                f"{leader[0]} leads in AI visibility with {leader[1]} total mentions"
            )

        # Platform performance insight
        best_platform = ""
        max_mentions = 0
        for platform, stats in data["platform_stats"].items():
            mentions = stats["brand_mentions"].get(data["client_name"], 0)
            if mentions > max_mentions:
                max_mentions = mentions
                best_platform = platform

        if best_platform:
            insights.append(
                f"Strongest performance on {best_platform} with {max_mentions} mentions"
            )

        return insights

    def _generate_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate strategic recommendations."""
        recommendations = [
            {
                "title": "Optimize Content for AI Platforms",
                "description": "Create structured content that directly answers industry questions to improve AI citation rates across all platforms.",
                "impact": "15-25% increase in mention rate",
            },
            {
                "title": "Competitive Content Strategy",
                "description": "Develop content positioning your brand as a superior alternative to top-mentioned competitors.",
                "impact": "10-20% improvement in competitive scenarios",
            },
            {
                "title": "Platform-Specific Optimization",
                "description": "Focus optimization efforts on platforms showing strongest performance potential.",
                "impact": "Enhanced visibility on primary platforms",
            },
        ]

        return recommendations

    def _generate_v2_report(
        self, audit_run_id: str, report_type: str, filepath: str
    ) -> str:
        """
        Generate v2 enhanced report using the new report engine.

        Args:
            audit_run_id: ID of the audit run
            report_type: V2 report type
            filepath: Output file path

        Returns:
            Path to generated report
        """
        try:
            from app.reports.v2.engine import ReportEngineV2

            # Determine theme based on report type
            theme_key = "default"
            if report_type == "v2_enhanced":
                theme_key = "corporate"

            # Initialize v2 engine
            engine = ReportEngineV2(
                db_session=self.db, theme_key=theme_key, template_version="v2.0"
            )

            # Generate report
            result_path = engine.generate_report(
                audit_run_id=audit_run_id, output_path=filepath, report_type=report_type
            )

            logger.info(f"Successfully generated v2 report: {result_path}")
            return result_path

        except ImportError as e:
            logger.error(f"V2 report engine not available: {e}")
            raise ValueError(
                "V2 report generation requires the enhanced report engine module"
            )
        except Exception as e:
            logger.error(f"V2 report generation failed: {e}")
            raise

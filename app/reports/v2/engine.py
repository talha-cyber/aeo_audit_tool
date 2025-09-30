"""
Report Engine v2 - Main orchestrator for AEO audit reports.

This module provides the main ReportEngineV2 class that coordinates:
- Data loading and processing with real sentiment analysis
- SAIV and metrics calculations
- Professional PDF generation with charts
- Multi-theme support and accessibility features
- Prior-period comparison capabilities
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib.units import inch
from reportlab.platypus import Spacer
from sqlalchemy.orm import Session

from app.models import audit as audit_models
from app.models import response as response_models
from app.models.report import Report
from app.services.brand_detection.core.sentiment import SentimentAnalyzer
from app.services.report_utils import extract_question_context

from .accessibility import enhance_pdf_accessibility
from .chassis import ReportBuilder, ReportDoc

try:
    from .chassis import create_toc_with_styles
except ImportError:
    # Fallback if ToC not available
    def create_toc_with_styles():
        from reportlab.platypus import Spacer

        return Spacer(1, 0.1 * inch)


from .metrics import BrandStats, aggregate_brands
from .sections import (
    appendix as s_appendix,
    competitive as s_competitive,
    platforms as s_platforms,
    recommendations as s_recommendations,
    summary as s_summary,
    title as s_title,
)
from .theme import format_date, get_theme, register_fonts

logger = logging.getLogger(__name__)


class ReportEngineV2:
    """
    Next-generation AEO audit report engine.

    Features:
    - Professional PDF structure with ToC and navigation
    - Real sentiment analysis (not confidence proxy)
    - SAIV (Share of AI Voice) calculations
    - Interactive charts and visualizations
    - Multi-theme support for white-labeling
    - Prior-period comparison capabilities
    - Accessibility features and bookmarks
    """

    def __init__(
        self,
        db_session: Session,
        theme_key: str = "default",
        template_version: str = "v2.0",
    ):
        """
        Initialize report engine.

        Args:
            db_session: SQLAlchemy database session
            theme_key: Theme identifier for styling
            template_version: Report template version
        """
        self.db = db_session
        self.theme = get_theme(theme_key)
        self.template_version = template_version
        self.sentiment_analyzer = SentimentAnalyzer()

        # Register any custom fonts
        register_fonts()

        logger.info(
            f"Initialized ReportEngineV2 with theme '{theme_key}' and template '{template_version}'"
        )

    def generate_report(
        self, audit_run_id: str, output_path: str, report_type: str = "v2_comprehensive"
    ) -> str:
        """
        Generate comprehensive v2 audit report.

        Args:
            audit_run_id: ID of the audit run to report on
            output_path: Path where PDF will be saved
            report_type: Type of report (for database record)

        Returns:
            Path to generated report file

        Raises:
            ValueError: If audit run not found or invalid data
            Exception: If report generation fails
        """
        try:
            logger.info(f"Generating v2 report for audit run {audit_run_id}")

            # Load current audit data
            current_data = self._load_audit_data(audit_run_id)
            if not current_data:
                raise ValueError(f"Audit run {audit_run_id} not found or has no data")

            # Load previous period data for comparison
            previous_data = self._load_previous_period_data(current_data)

            # Create output directory
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Build the report
            final_path = self._build_report(current_data, previous_data, output_path)

            # Create database record
            self._create_report_record(audit_run_id, report_type, final_path)

            logger.info(f"Successfully generated v2 report: {final_path}")
            return final_path

        except Exception as e:
            logger.error(
                f"Failed to generate v2 report for {audit_run_id}: {e}", exc_info=True
            )
            raise

    def _load_audit_data(self, audit_run_id: str) -> Optional[Dict[str, Any]]:
        """
        Load and process audit data for reporting with real sentiment analysis.

        Args:
            audit_run_id: Audit run identifier

        Returns:
            Processed audit data dictionary
        """
        try:
            # Load audit run
            audit_run = (
                self.db.query(audit_models.AuditRun)
                .filter(audit_models.AuditRun.id == audit_run_id)
                .first()
            )

            if not audit_run:
                logger.warning(f"Audit run {audit_run_id} not found")
                return None

            # Load responses
            responses = (
                self.db.query(response_models.Response)
                .filter(response_models.Response.audit_run_id == audit_run_id)
                .all()
            )

            if not responses:
                logger.warning(f"No responses found for audit run {audit_run_id}")
                return None

            # Extract configuration data
            config = audit_run.config or {}
            client_data = config.get("client", {})
            client_name = client_data.get("name", "Unknown Client")
            competitors = client_data.get("competitors", [])
            industry = client_data.get("industry", "Unknown")
            product_type = client_data.get("product_type", "")

            all_brands = [client_name] + competitors

            # Initialize brand statistics
            brand_stats = {brand: BrandStats(0, [], {}, {}) for brand in all_brands}
            platform_stats = {}
            qa_entries: List[Dict[str, Any]] = []

            logger.info(
                f"Processing {len(responses)} responses for {len(all_brands)} brands"
            )

            # Process responses
            for response in responses:
                platform = response.platform
                brand_mentions = response.brand_mentions or {}
                question_context = extract_question_context(response)

                # Initialize platform stats if needed
                if platform not in platform_stats:
                    platform_stats[platform] = {
                        "total_questions": 0,
                        "brand_mentions": {brand: 0 for brand in all_brands},
                        "avg_sentiment": {brand: [] for brand in all_brands},
                    }

                platform_stats[platform]["total_questions"] += 1

                # Process brand mentions using existing brand detection results
                if brand_mentions and isinstance(brand_mentions, dict):
                    top_brands = brand_mentions.get("top_brands", [])

                    # Count brand occurrences
                    brand_counts = {}
                    for brand in top_brands:
                        brand_counts[brand] = brand_counts.get(brand, 0) + 1

                    # Process detected brands
                    for brand, count in brand_counts.items():
                        if brand in all_brands:
                            # Update brand statistics
                            brand_stats[brand].mentions += count

                            # Platform-specific mentions
                            if platform not in brand_stats[brand].platforms:
                                brand_stats[brand].platforms[platform] = 0
                            brand_stats[brand].platforms[platform] += count

                            # Platform stats
                            platform_stats[platform]["brand_mentions"][brand] += count

                            # Use real sentiment if available, otherwise calculate it
                            if (
                                hasattr(response, "sentiment")
                                and response.sentiment is not None
                            ):
                                sentiment_score = response.sentiment
                            else:
                                # Calculate sentiment using existing analyzer
                                try:
                                    sentiment_result = self.sentiment_analyzer.analyze_business_sentiment(
                                        response.response_text, brand
                                    )
                                    sentiment_score = sentiment_result.score
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to calculate sentiment for response {response.id}: {e}"
                                    )
                                    sentiment_score = 0.0

                            # Store sentiment
                            brand_stats[brand].sentiments.append(sentiment_score)
                            platform_stats[platform]["avg_sentiment"][brand].append(
                                sentiment_score
                            )

                            # Category tracking (if available)
                            if hasattr(response, "category") and response.category:
                                if (
                                    response.category
                                    not in brand_stats[brand].categories
                                ):
                                    brand_stats[brand].categories[response.category] = 0
                                brand_stats[brand].categories[
                                    response.category
                                ] += count

                # Capture Q&A context for appendix generation
                qa_entry = {
                    "question_id": question_context.get("question_id"),
                    "question": question_context.get("question"),
                    "answer": response.response_text,
                    "provider": question_context.get("provider")
                    or response.platform
                    or "Unknown Provider",
                    "category": question_context.get("category"),
                    "question_type": question_context.get("question_type"),
                    "platform": platform,
                }
                qa_entries.append(qa_entry)

            # Calculate platform sentiment averages
            for platform in platform_stats:
                for brand in platform_stats[platform]["avg_sentiment"]:
                    scores = platform_stats[platform]["avg_sentiment"][brand]
                    platform_stats[platform]["avg_sentiment"][brand] = (
                        sum(scores) / len(scores) if scores else 0.0
                    )

            # Generate aggregated metrics
            aggregates = aggregate_brands(all_brands, brand_stats, len(responses))

            # Handle timestamps with fallbacks
            current_time = datetime.now()
            start_time = audit_run.started_at or current_time
            end_time = audit_run.completed_at or audit_run.started_at or current_time

            # Build comprehensive data structure
            audit_data = {
                "audit_run": audit_run,
                "client_name": client_name,
                "competitors": competitors,
                "industry": industry,
                "product_type": product_type,
                "platform_stats": platform_stats,
                "brand_performance": self._build_brand_performance_dict(brand_stats),
                "brand_stats": brand_stats,
                "aggregates": aggregates,
                "total_responses": len(responses),
                "date_range": {"start": start_time, "end": end_time},
                "prompt_basket_version": getattr(
                    audit_run, "prompt_basket_version", None
                ),
                "qa_entries": qa_entries,
                "appendix_group_by": "provider",
            }

            logger.info(
                f"Successfully loaded audit data with {aggregates.saiv.get(client_name, 0)*100:.1f}% client SAIV"
            )
            return audit_data

        except Exception as e:
            logger.error(
                f"Error loading audit data for {audit_run_id}: {e}", exc_info=True
            )
            return None

    def _build_brand_performance_dict(
        self, brand_stats: Dict[str, BrandStats]
    ) -> Dict[str, Dict[str, Any]]:
        """Convert brand stats to brand performance dictionary format."""
        brand_performance = {}

        for brand, stats in brand_stats.items():
            brand_performance[brand] = {
                "total_mentions": stats.mentions,
                "platforms": set(stats.platforms.keys()) if stats.platforms else set(),
                "sentiment_scores": stats.sentiments,
                "categories": stats.categories,
            }

        return brand_performance

    def _load_previous_period_data(
        self, current_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Load previous period data for comparison.

        Args:
            current_data: Current audit data

        Returns:
            Previous period data if available
        """
        try:
            client_name = current_data.get("client_name")
            current_prompt_version = current_data.get("prompt_basket_version")
            current_start_date = current_data.get("date_range", {}).get("start")

            if not client_name or not current_start_date:
                return None

            # Find most recent completed audit for same client before current period
            previous_audit = (
                self.db.query(audit_models.AuditRun)
                .join(audit_models.Client)
                .filter(
                    audit_models.Client.name == client_name,
                    audit_models.AuditRun.completed_at.isnot(None),
                    audit_models.AuditRun.completed_at < current_start_date,
                    audit_models.AuditRun.status == "completed",
                )
                .order_by(audit_models.AuditRun.completed_at.desc())
                .first()
            )

            if previous_audit:
                logger.info(f"Found previous period audit: {previous_audit.id}")
                return self._load_audit_data(previous_audit.id)

            return None

        except Exception as e:
            logger.warning(f"Could not load previous period data: {e}")
            return None

    def _build_report(
        self,
        current_data: Dict[str, Any],
        previous_data: Optional[Dict[str, Any]],
        output_path: str,
    ) -> str:
        """
        Build the complete PDF report.

        Args:
            current_data: Current period audit data
            previous_data: Previous period data (optional)
            output_path: Output file path

        Returns:
            Final report file path
        """
        # Create document
        doc = ReportDoc(output_path)
        builder = ReportBuilder(doc)

        # Set document metadata
        client_name = current_data.get("client_name", "Unknown Client")
        date_range = current_data.get("date_range", {})

        doc.set_metadata(
            title=f"AEO Competitive Intelligence Report — {client_name}",
            author="AEO Audit Platform",
            subject="AI Platform Competitive Analysis",
            keywords=[
                "AEO",
                "AI",
                "Audit",
                "Competitive Intelligence",
                "SAIV",
                "Sentiment Analysis",
            ],
        )

        # Set header context
        if date_range.get("start") and date_range.get("end"):
            date_range_str = f"{format_date(date_range['start'], self.theme)} – {format_date(date_range['end'], self.theme)}"
            doc.set_header_context(client_name, date_range_str)

        # Apply accessibility enhancements
        enhance_pdf_accessibility(
            doc.canv if hasattr(doc, "canv") else None,
            {
                "title": f"AEO Report - {client_name}",
                "author": "AEO Platform",
                "subject": "Competitive Intelligence Analysis",
                "language": self.theme.locale.split("_")[0],
            },
        )

        # Build report sections

        # 1. Title page
        title_content = s_title.build(self.theme, current_data)
        builder.add_title_page(title_content)

        # 2. Table of Contents
        toc = create_toc_with_styles()
        builder.add_standard_content([toc, Spacer(1, 0.2 * inch)])
        builder.add_page_break()

        # 3. Executive Summary
        summary_content = s_summary.build(self.theme, current_data, previous_data)
        builder.add_standard_content(summary_content)
        builder.add_page_break()

        # 4. Competitive Analysis
        competitive_content = s_competitive.build(
            self.theme, current_data, previous_data
        )
        builder.add_standard_content(competitive_content)
        builder.add_page_break()

        # 5. Platform Performance
        platforms_content = s_platforms.build(self.theme, current_data)
        builder.add_standard_content(platforms_content)
        builder.add_page_break()

        # 6. Strategic Recommendations
        recommendations_content = s_recommendations.build(
            self.theme, current_data, previous_data
        )
        builder.add_standard_content(recommendations_content)

        # 7. Appendix (optional)
        if current_data.get("qa_entries"):
            builder.add_page_break()
            appendix_content = s_appendix.build(self.theme, current_data)
            builder.add_standard_content(appendix_content)

        # Build final document
        return builder.build()

    def _create_report_record(
        self, audit_run_id: str, report_type: str, output_path: str
    ):
        """
        Create database record for generated report.

        Args:
            audit_run_id: Audit run identifier
            report_type: Report type classification
            output_path: Generated file path
        """
        try:
            new_report = Report(
                id=str(uuid.uuid4()),
                audit_run_id=audit_run_id,
                report_type=report_type,
                file_path=output_path,
                generated_at=datetime.now(),
                template_version=self.template_version,
                theme_key=self.theme.name,
            )

            self.db.add(new_report)
            self.db.commit()

            logger.info(f"Created report record: {new_report.id}")

        except Exception as e:
            logger.error(f"Failed to create report record: {e}", exc_info=True)
            self.db.rollback()

    def get_available_themes(self) -> List[str]:
        """Get list of available theme keys."""
        from .theme import THEMES

        return list(THEMES.keys())

    def validate_audit_data(self, audit_run_id: str) -> Dict[str, Any]:
        """
        Validate audit data for report generation.

        Args:
            audit_run_id: Audit run identifier

        Returns:
            Validation results with recommendations
        """
        try:
            data = self._load_audit_data(audit_run_id)

            if not data:
                return {
                    "valid": False,
                    "issues": ["Audit run not found or has no data"],
                    "recommendations": ["Ensure audit run completed successfully"],
                }

            issues = []
            recommendations = []

            # Check for minimum data requirements
            if data.get("total_responses", 0) < 10:
                issues.append("Low response count may affect analysis quality")
                recommendations.append("Consider expanding query set or audit duration")

            # Check brand mention coverage
            client_name = data.get("client_name", "")
            if data.get("aggregates"):
                client_mentions = data["aggregates"].mention_rate.get(client_name, 0)
                if client_mentions < 1.0:
                    issues.append("Very low client mention rate detected")
                    recommendations.append(
                        "Review content optimization and brand presence strategies"
                    )

            # Check platform coverage
            platform_count = len(data.get("platform_stats", {}))
            if platform_count < 2:
                issues.append("Limited platform coverage")
                recommendations.append("Expand analysis to include more AI platforms")

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "recommendations": recommendations,
                "data_summary": {
                    "total_responses": data.get("total_responses", 0),
                    "platforms": list(data.get("platform_stats", {}).keys()),
                    "brands_analyzed": len(data.get("aggregates", {}).saiv or {}),
                    "date_range": data.get("date_range", {}),
                },
            }

        except Exception as e:
            logger.error(f"Error validating audit data: {e}")
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "recommendations": ["Check audit run status and data integrity"],
            }

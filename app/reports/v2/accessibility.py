"""
Accessibility helpers for AEO audit reports.

This module provides best-effort accessibility features within ReportLab limitations:
- PDF bookmarks and outline structure
- Alt text captions for charts and images
- Logical reading order support
- Document metadata for screen readers
"""

from __future__ import annotations

import logging
from typing import Optional

from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

logger = logging.getLogger(__name__)


def add_bookmark(
    canvas_obj: Canvas, title: str, level: int = 0, key: Optional[str] = None
):
    """
    Add bookmark entry to PDF outline.

    Args:
        canvas_obj: ReportLab canvas object
        title: Bookmark title text
        level: Outline level (0 = top level, 1 = sub-level, etc.)
        key: Optional unique key for bookmark (defaults to title)
    """
    try:
        bookmark_key = key or title.replace(" ", "_").replace("/", "_")
        canvas_obj.bookmarkPage(bookmark_key)
        canvas_obj.addOutlineEntry(title, bookmark_key, level=level, closed=False)
        logger.debug(f"Added bookmark: {title} at level {level}")
    except Exception as e:
        logger.warning(f"Failed to add bookmark '{title}': {e}")


def create_alt_text_caption(alt_text: str, style_name: str = "caption") -> Paragraph:
    """
    Create alt text caption for charts and images.

    Args:
        alt_text: Descriptive text for the visual element
        style_name: Paragraph style name to use

    Returns:
        Paragraph object with alt text
    """
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()

    # Use provided style or fallback to normal
    if style_name in styles:
        style = styles[style_name]
    else:
        style = styles["Normal"]
        style.fontSize = 9
        style.textColor = "grey"

    # Prefix with accessibility indicator
    formatted_text = f"<i>Chart description: {alt_text}</i>"

    return Paragraph(formatted_text, style)


class AccessibleHeading:
    """
    Helper class to create accessible headings with ToC and bookmark integration.

    This class ensures headings are properly registered in the document outline
    and table of contents for navigation accessibility.
    """

    def __init__(self, style, doc, level: int):
        """
        Initialize accessible heading helper.

        Args:
            style: Paragraph style for the heading
            doc: Document object for ToC integration
            level: Heading level (0=h1, 1=h2, etc.)
        """
        self.style = style
        self.doc = doc
        self.level = level

    def create(self, text: str, bookmark_key: Optional[str] = None) -> Paragraph:
        """
        Create accessible heading paragraph.

        Args:
            text: Heading text content
            bookmark_key: Optional custom bookmark key

        Returns:
            Paragraph object with accessibility hooks
        """
        # Create the paragraph
        paragraph = Paragraph(text, self.style)

        # Add ToC notification
        if self.doc:
            try:
                self.doc.notify(
                    "TOCEntry", (self.level, text, getattr(self.doc, "page", 1))
                )
            except Exception as e:
                logger.warning(f"Failed to add ToC entry for '{text}': {e}")

        # Add bookmark creation as deferred action
        def add_bookmark_deferred(canvas_obj, doc):
            add_bookmark(canvas_obj, text, self.level, bookmark_key)

        # Attach the deferred action to the paragraph
        paragraph._postponed_bookmark = add_bookmark_deferred

        return paragraph


def create_section_heading(text: str, level: int, theme, doc=None) -> Paragraph:
    """
    Create accessible section heading with consistent styling.

    Args:
        text: Heading text
        level: Heading level (1, 2, 3)
        theme: Theme object for styling
        doc: Document object for ToC integration

    Returns:
        Accessible heading paragraph
    """
    from .theme import create_paragraph_styles

    styles = create_paragraph_styles(theme)

    # Select appropriate style based on level
    if level == 1:
        style = styles["heading1"]
    elif level == 2:
        style = styles["heading2"]
    elif level == 3:
        style = styles["heading3"]
    else:
        style = styles["body"]

    # Create accessible heading
    heading_helper = AccessibleHeading(style, doc, level - 1)  # Convert to 0-based
    return heading_helper.create(text)


def add_document_metadata(canvas_obj: Canvas, metadata: dict):
    """
    Add comprehensive metadata to PDF document for accessibility.

    Args:
        canvas_obj: ReportLab canvas object
        metadata: Dictionary with metadata fields
    """
    try:
        # Standard PDF metadata
        if "title" in metadata:
            canvas_obj.setTitle(metadata["title"])

        if "author" in metadata:
            canvas_obj.setAuthor(metadata["author"])

        if "subject" in metadata:
            canvas_obj.setSubject(metadata["subject"])

        if "keywords" in metadata:
            keywords = metadata["keywords"]
            if isinstance(keywords, list):
                keywords = ", ".join(keywords)
            canvas_obj.setKeywords(keywords)

        # Additional metadata for accessibility
        canvas_obj.setCreator(metadata.get("creator", "AEO Audit Platform"))
        canvas_obj.setProducer(metadata.get("producer", "AEO Report Generator v2.0"))

        # Language specification (helps screen readers)
        if "language" in metadata:
            # Note: ReportLab doesn't directly support language metadata
            # but we log it for potential future enhancement
            logger.info(f"Document language: {metadata['language']}")

        logger.debug("Document metadata added successfully")

    except Exception as e:
        logger.warning(f"Failed to add document metadata: {e}")


def create_accessible_table_caption(
    caption_text: str, table_summary: str = ""
) -> Paragraph:
    """
    Create accessible table caption with summary.

    Args:
        caption_text: Main caption text
        table_summary: Optional summary of table structure/purpose

    Returns:
        Formatted paragraph with table accessibility information
    """
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()
    caption_style = styles["Normal"]
    caption_style.fontSize = 10
    caption_style.spaceBefore = 6
    caption_style.spaceAfter = 3

    # Combine caption and summary
    full_text = caption_text
    if table_summary:
        full_text += f" {table_summary}"

    return Paragraph(f"<b>{full_text}</b>", caption_style)


def add_reading_order_hint(paragraph: Paragraph, order_hint: str):
    """
    Add reading order hint to paragraph (for complex layouts).

    Args:
        paragraph: Paragraph object to enhance
        order_hint: Reading order description
    """
    # Store reading order hint as attribute (for potential future processing)
    paragraph._reading_order = order_hint
    logger.debug(f"Added reading order hint: {order_hint}")


def create_skip_link(target_text: str = "Skip to main content") -> Paragraph:
    """
    Create accessibility skip link (conceptual - limited ReportLab support).

    Args:
        target_text: Text for skip link

    Returns:
        Paragraph with skip link styling
    """
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()
    skip_style = styles["Normal"]
    skip_style.fontSize = 8
    skip_style.textColor = "blue"

    # Note: True skip links require interactive PDF features not fully supported
    return Paragraph(f"<i>{target_text}</i>", skip_style)


class AccessibilityChecker:
    """
    Helper class to validate accessibility features in generated reports.

    This provides basic checks for common accessibility issues.
    """

    def __init__(self):
        self.issues = []

    def check_headings_hierarchy(self, headings: list) -> list:
        """
        Check for proper heading hierarchy (h1 -> h2 -> h3).

        Args:
            headings: List of (level, text) tuples

        Returns:
            List of accessibility issues found
        """
        issues = []
        prev_level = 0

        for level, text in headings:
            if level > prev_level + 1:
                issues.append(
                    f"Heading level skip: '{text}' jumps from h{prev_level} to h{level}"
                )
            prev_level = level

        return issues

    def check_alt_text_coverage(self, charts: list, alt_texts: list) -> list:
        """
        Check that all charts have corresponding alt text.

        Args:
            charts: List of chart identifiers
            alt_texts: List of alt text entries

        Returns:
            List of missing alt text issues
        """
        issues = []

        if len(charts) > len(alt_texts):
            missing_count = len(charts) - len(alt_texts)
            issues.append(f"{missing_count} charts missing alt text descriptions")

        return issues

    def generate_report(self) -> str:
        """
        Generate accessibility compliance report.

        Returns:
            Text report of accessibility status
        """
        if not self.issues:
            return "Accessibility check passed - no issues found."

        report = "Accessibility Issues Found:\n"
        for i, issue in enumerate(self.issues, 1):
            report += f"{i}. {issue}\n"

        return report


def enhance_pdf_accessibility(
    canvas_obj: Canvas, doc_metadata: dict, language: str = "en"
):
    """
    Apply comprehensive accessibility enhancements to PDF.

    Args:
        canvas_obj: ReportLab canvas object
        doc_metadata: Document metadata dictionary
        language: Document language code
    """
    # Add comprehensive metadata
    add_document_metadata(
        canvas_obj,
        {
            **doc_metadata,
            "language": language,
            "creator": "AEO Audit Platform",
            "producer": "AEO Report Generator v2.0 (Accessible)",
        },
    )

    # Set up document structure (conceptual - ReportLab limitations)
    logger.info("Applied accessibility enhancements to PDF")
    logger.info("Note: Full PDF/UA compliance requires additional post-processing")

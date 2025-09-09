"""
Theming and localization system for AEO audit reports.

This module provides:
- Multi-theme support for white-labeling
- Color palettes and typography
- Localization helpers for dates and numbers
- Font management and registration
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle


@dataclass
class Theme:
    """
    Theme configuration for report styling and branding.

    Attributes:
        name: Theme identifier
        primary: Primary brand color (hex)
        secondary: Secondary brand color (hex)
        accent: Accent color for highlights (hex)
        font_regular: Regular font family name
        font_bold: Bold font family name
        locale: Locale code for formatting (e.g., 'en_US', 'de_DE')
        logo_path: Optional path to logo image
    """

    name: str
    primary: str = "#2C3E50"  # Dark blue-gray
    secondary: str = "#34495E"  # Medium blue-gray
    accent: str = "#1ABC9C"  # Teal
    font_regular: str = "Helvetica"
    font_bold: str = "Helvetica-Bold"
    locale: str = "en_US"
    logo_path: str = ""


# Predefined themes for different use cases
THEMES: Dict[str, Theme] = {
    "default": Theme(
        name="default",
        primary="#2C3E50",
        secondary="#34495E",
        accent="#1ABC9C",
        locale="en_US",
    ),
    "corporate": Theme(
        name="corporate",
        primary="#1f4e79",  # Corporate blue
        secondary="#5b9bd5",  # Light corporate blue
        accent="#ff6b35",  # Orange accent
        locale="en_US",
    ),
    "german_corporate": Theme(
        name="german_corporate",
        primary="#003366",  # Dark blue
        secondary="#0066cc",  # Medium blue
        accent="#ff9900",  # Orange
        locale="de_DE",
    ),
    "minimalist": Theme(
        name="minimalist",
        primary="#333333",  # Dark gray
        secondary="#666666",  # Medium gray
        accent="#0066ff",  # Blue accent
        locale="en_US",
    ),
}


def get_theme(theme_key: str) -> Theme:
    """
    Get theme by key, fallback to default if not found.

    Args:
        theme_key: Theme identifier

    Returns:
        Theme object
    """
    return THEMES.get(theme_key, THEMES["default"])


def format_date(dt: datetime, theme: Theme, format_type: str = "long") -> str:
    """
    Format date according to theme locale.

    Args:
        dt: DateTime object to format
        theme: Theme with locale information
        format_type: Format type ('short', 'medium', 'long', 'full')

    Returns:
        Formatted date string
    """
    try:
        from babel.dates import format_date as babel_format_date

        return babel_format_date(dt, format=format_type, locale=theme.locale)
    except ImportError:
        # Fallback if Babel not available
        if theme.locale.startswith("de"):
            return dt.strftime("%d.%m.%Y")
        else:
            return dt.strftime("%B %d, %Y")
    except Exception:
        # Fallback for any other formatting errors
        return dt.strftime("%Y-%m-%d")


def format_number(
    value: float, theme: Theme, decimal_places: int = 1, format_type: str = "decimal"
) -> str:
    """
    Format number according to theme locale.

    Args:
        value: Numeric value to format
        theme: Theme with locale information
        decimal_places: Number of decimal places
        format_type: Format type ('decimal', 'percent', 'currency')

    Returns:
        Formatted number string
    """
    try:
        from babel.numbers import format_decimal, format_percent

        if format_type == "percent":
            return format_percent(value, locale=theme.locale)
        else:
            pattern = f"#,##0.{'0' * decimal_places}"
            return format_decimal(value, format=pattern, locale=theme.locale)

    except ImportError:
        # Fallback if Babel not available
        if format_type == "percent":
            return f"{value * 100:.{decimal_places}f}%"
        else:
            return f"{value:,.{decimal_places}f}"
    except Exception:
        # Fallback for any formatting errors
        return f"{value:.{decimal_places}f}"


def create_paragraph_styles(theme: Theme) -> Dict[str, ParagraphStyle]:
    """
    Create paragraph styles based on theme.

    Args:
        theme: Theme configuration

    Returns:
        Dictionary of style name -> ParagraphStyle
    """
    styles = {}

    # Title style
    styles["title"] = ParagraphStyle(
        "Title",
        fontSize=24,
        leading=30,
        textColor=colors.HexColor(theme.primary),
        fontName=theme.font_bold,
        alignment=1,  # Center
        spaceAfter=30,
        spaceBefore=0,
    )

    # Main heading styles
    styles["heading1"] = ParagraphStyle(
        "Heading1",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor(theme.primary),
        fontName=theme.font_bold,
        spaceBefore=20,
        spaceAfter=12,
        keepWithNext=1,
    )

    styles["heading2"] = ParagraphStyle(
        "Heading2",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor(theme.secondary),
        fontName=theme.font_bold,
        spaceBefore=16,
        spaceAfter=10,
        keepWithNext=1,
    )

    styles["heading3"] = ParagraphStyle(
        "Heading3",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor(theme.secondary),
        fontName=theme.font_bold,
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=1,
    )

    # Body text styles
    styles["body"] = ParagraphStyle(
        "Body",
        fontSize=11,
        leading=14,
        textColor=colors.black,
        fontName=theme.font_regular,
        spaceBefore=6,
        spaceAfter=6,
        alignment=0,  # Left
    )

    styles["body_indent"] = ParagraphStyle(
        "BodyIndent", parent=styles["body"], leftIndent=20
    )

    # Special styles
    styles["emphasis"] = ParagraphStyle(
        "Emphasis",
        parent=styles["body"],
        fontName=theme.font_bold,
        textColor=colors.HexColor(theme.accent),
    )

    styles["caption"] = ParagraphStyle(
        "Caption",
        fontSize=9,
        leading=12,
        textColor=colors.grey,
        fontName=theme.font_regular,
        alignment=1,  # Center
        spaceBefore=6,
        spaceAfter=12,
    )

    styles["footer"] = ParagraphStyle(
        "Footer",
        fontSize=8,
        leading=10,
        textColor=colors.grey,
        fontName=theme.font_regular,
        alignment=1,  # Center
    )

    # Recommendation styles
    styles["recommendation_title"] = ParagraphStyle(
        "RecommendationTitle",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor(theme.primary),
        fontName=theme.font_bold,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=1,
    )

    styles["kpi_highlight"] = ParagraphStyle(
        "KPIHighlight",
        fontSize=10,
        leading=12,
        textColor=colors.HexColor(theme.accent),
        fontName=theme.font_bold,
        borderColor=colors.HexColor(theme.accent),
        borderWidth=1,
        borderPadding=4,
    )

    return styles


def get_color_palette(theme: Theme) -> Dict[str, colors.Color]:
    """
    Get color palette from theme.

    Args:
        theme: Theme configuration

    Returns:
        Dictionary of color name -> Color object
    """
    return {
        "primary": colors.HexColor(theme.primary),
        "secondary": colors.HexColor(theme.secondary),
        "accent": colors.HexColor(theme.accent),
        "light_gray": colors.lightgrey,
        "dark_gray": colors.grey,
        "success": colors.HexColor("#27AE60"),
        "warning": colors.HexColor("#F39C12"),
        "danger": colors.HexColor("#E74C3C"),
        "info": colors.HexColor("#3498DB"),
    }


def register_fonts():
    """
    Register custom fonts if available.

    This function attempts to register custom TTF fonts for enhanced typography.
    Falls back gracefully if fonts are not available.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        # Example font registration (uncomment and adjust paths as needed)
        # pdfmetrics.registerFont(TTFont("Inter-Regular", "assets/fonts/Inter-Regular.ttf"))
        # pdfmetrics.registerFont(TTFont("Inter-Bold", "assets/fonts/Inter-Bold.ttf"))

        pass  # No custom fonts available in current setup

    except ImportError:
        pass  # ReportLab not available or fonts not found


def apply_theme_to_chart_colors(theme: Theme) -> list:
    """
    Generate chart color palette based on theme.

    Args:
        theme: Theme configuration

    Returns:
        List of color hex codes for chart styling
    """
    base_colors = [
        theme.primary,
        theme.accent,
        theme.secondary,
        "#95A5A6",  # Light gray
        "#E67E22",  # Orange
        "#8E44AD",  # Purple
        "#16A085",  # Teal
        "#2980B9",  # Blue
    ]

    return base_colors


def clean_html_content(text: str) -> str:
    """
    Clean HTML content for proper ReportLab rendering.

    Args:
        text: Text potentially containing HTML tags

    Returns:
        Cleaned text with proper line breaks
    """
    if not isinstance(text, str):
        return str(text)

    import re

    # Replace <br>, <br/>, <br /> with proper line breaks
    text = re.sub(r"<br\s*/?>", "<br/>", text, flags=re.IGNORECASE)

    # Ensure proper HTML formatting is maintained for ReportLab Paragraph
    return text


def wrap_table_cell(text: str, theme: Theme, is_header: bool = False) -> "Paragraph":
    """
    Wrap table cell text in a Paragraph for proper formatting and wrapping.

    Args:
        text: Cell text content
        theme: Theme configuration
        is_header: Whether this is a header cell

    Returns:
        Paragraph object ready for table cell
    """
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph

    # Clean HTML content
    clean_text = clean_html_content(str(text))

    # Create appropriate style for cell content
    if is_header:
        cell_style = ParagraphStyle(
            "TableHeader",
            fontSize=10,
            leading=12,
            fontName="Helvetica-Bold",
            alignment=1,  # Center
            textColor=colors.whitesmoke,
        )
    else:
        cell_style = ParagraphStyle(
            "TableCell",
            fontSize=10,
            leading=12,
            fontName="Helvetica",
            alignment=0,  # Left alignment for better readability
            textColor=colors.black,
            wordWrap="CJK",  # Enable word wrapping
        )

    return Paragraph(clean_text, cell_style)


def create_table_styles(theme: Theme):
    """
    Create table styles based on theme.

    Args:
        theme: Theme configuration

    Returns:
        Dictionary of table style configurations
    """
    from reportlab.platypus import TableStyle

    colors_palette = get_color_palette(theme)

    styles = {}

    # Standard data table
    styles["standard"] = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors_palette["primary"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),  # Center headers only
            ("ALIGN", (0, 1), (-1, -1), "LEFT"),  # Left-align body cells
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors_palette["light_gray"]),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP",
            ),  # Changed from MIDDLE to TOP for better text wrapping
            ("WORDWRAP", (0, 0), (-1, -1), True),  # Enable word wrapping
        ]
    )

    # Minimal table (for metadata)
    styles["minimal"] = TableStyle(
        [
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (0, -1), colors_palette["primary"]),
        ]
    )

    # Highlight table (for recommendations)
    styles["highlight"] = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors_palette["accent"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8F9FA")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors_palette["light_gray"]),
        ]
    )

    return styles

"""
Chart generation module for AEO audit reports.

This module generates professional charts using matplotlib and converts them
to ReportLab Image flowables for embedding in PDF reports.

Charts include:
- SAIV bar charts showing Share of AI Voice by brand
- Platform performance comparisons
- Competitive analysis visualizations
- Sentiment comparison charts
- Category heatmaps and distributions
"""

from __future__ import annotations

import io
import logging
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.units import inch
from reportlab.platypus import Image

# Use non-interactive backend for server environments
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Chart configuration
DPI = 150
DEFAULT_WIDTH = 5.5 * inch  # Reduced to fit better within page margins
DEFAULT_HEIGHT = 3.5 * inch  # Reduced to fit better within page margins

# Professional color palette
BRAND_COLORS = [
    "#2C3E50",
    "#3498DB",
    "#E74C3C",
    "#F39C12",
    "#27AE60",
    "#9B59B6",
    "#1ABC9C",
    "#34495E",
]


def fig_to_image(
    fig, width: float = DEFAULT_WIDTH, height: float = DEFAULT_HEIGHT
) -> Image:
    """
    Convert matplotlib figure to ReportLab Image flowable.

    Args:
        fig: Matplotlib figure object
        width: Image width in ReportLab units
        height: Image height in ReportLab units

    Returns:
        ReportLab Image object ready for PDF embedding
    """
    try:
        # Create buffer and save figure
        img_buffer = io.BytesIO()
        fig.savefig(
            img_buffer,
            format="png",
            dpi=DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.1,  # Add small padding
        )
        img_buffer.seek(0)

        # Clean up matplotlib figure
        plt.close(fig)

        # Create ReportLab Image
        img = Image(img_buffer, width=width, height=height)
        return img

    except Exception as e:
        logger.error(f"Failed to convert figure to image: {e}")
        plt.close(fig)  # Ensure cleanup even on error
        raise


def create_saiv_bar_chart(
    saiv_data: Dict[str, float], title: str = "Share of AI Voice (SAIV) by Brand"
) -> Image:
    """
    Create bar chart showing SAIV percentages for each brand.

    Args:
        saiv_data: Dictionary of brand -> SAIV value (0.0 to 1.0)
        title: Chart title

    Returns:
        ReportLab Image object
    """
    if not saiv_data:
        logger.warning("No SAIV data provided for chart")
        # Return empty figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No SAIV data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig_to_image(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    brands = list(saiv_data.keys())
    values = [saiv_data[brand] * 100 for brand in brands]  # Convert to percentages

    # Create bars with professional colors
    bars = ax.bar(brands, values, color=BRAND_COLORS[: len(brands)])

    # Formatting
    ax.set_ylabel("Share of AI Voice (%)")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim(0, max(values) * 1.1 if values else 1)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(values) * 0.01,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Rotate x-axis labels if too many brands
    if len(brands) > 4:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    return fig_to_image(fig)


def create_platform_performance_chart(
    platform_data: Dict[str, int],
    client_brand: str,
    title: str = "Platform Performance",
) -> Image:
    """
    Create bar chart showing client brand mentions across platforms.

    Args:
        platform_data: Dictionary of platform -> mention count for client
        client_brand: Name of the client brand
        title: Chart title

    Returns:
        ReportLab Image object
    """
    if not platform_data:
        logger.warning("No platform data provided for chart")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No platform data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig_to_image(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    platforms = list(platform_data.keys())
    mentions = list(platform_data.values())

    # Create bars
    bars = ax.bar(
        platforms, mentions, color=BRAND_COLORS[1]
    )  # Use consistent blue color

    # Formatting
    ax.set_ylabel("Brand Mentions")
    ax.set_title(f"{title} - {client_brand}", fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim(0, max(mentions) * 1.1 if mentions else 1)

    # Add value labels on bars
    for bar, value in zip(bars, mentions):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(mentions) * 0.01,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Rotate x-axis labels if needed
    if len(platforms) > 4:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    return fig_to_image(fig)


def create_competitive_analysis_chart(
    brand_performance: Dict[str, Dict],
    brands: List[str],
    title: str = "Competitive Analysis",
) -> Image:
    """
    Create grouped bar chart comparing brands across multiple metrics.

    Args:
        brand_performance: Dictionary of brand -> performance metrics
        brands: List of brand names to include
        title: Chart title

    Returns:
        ReportLab Image object
    """
    if not brand_performance or not brands:
        logger.warning("No competitive data provided for chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No competitive data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig_to_image(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Extract metrics for comparison
    mentions = []
    platform_counts = []
    avg_sentiments = []

    for brand in brands:
        brand_data = brand_performance.get(brand, {})
        mentions.append(brand_data.get("total_mentions", 0))
        platform_counts.append(len(brand_data.get("platforms", set())))

        # Calculate average sentiment
        sentiment_scores = brand_data.get("sentiment_scores", [])
        avg_sentiment = (
            sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        )
        avg_sentiments.append(avg_sentiment)

    # Create grouped bars
    x = np.arange(len(brands))
    width = 0.25

    # Normalize metrics for comparison (mentions on different scale)
    max_mentions = max(mentions) if mentions else 1
    normalized_mentions = [m / max_mentions * 100 for m in mentions]
    normalized_platforms = [p * 10 for p in platform_counts]  # Scale up platform counts
    normalized_sentiments = [
        (s + 1) * 50 for s in avg_sentiments
    ]  # Scale sentiment from -1,1 to 0,100

    bars1 = ax.bar(
        x - width,
        normalized_mentions,
        width,
        label="Mentions (scaled)",
        color=BRAND_COLORS[0],
    )
    bars2 = ax.bar(
        x,
        normalized_platforms,
        width,
        label="Platform Count (×10)",
        color=BRAND_COLORS[1],
    )
    bars3 = ax.bar(
        x + width,
        normalized_sentiments,
        width,
        label="Sentiment (scaled)",
        color=BRAND_COLORS[2],
    )

    # Formatting
    ax.set_ylabel("Normalized Scores")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(brands)
    ax.legend()

    # Rotate x-axis labels if needed
    if len(brands) > 4:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    return fig_to_image(fig)


def create_sentiment_comparison_chart(
    sentiment_data: Dict[str, float], title: str = "Sentiment Comparison"
) -> Image:
    """
    Create horizontal bar chart comparing average sentiment across brands.

    Args:
        sentiment_data: Dictionary of brand -> average sentiment (-1 to 1)
        title: Chart title

    Returns:
        ReportLab Image object
    """
    if not sentiment_data:
        logger.warning("No sentiment data provided for chart")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No sentiment data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig_to_image(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    brands = list(sentiment_data.keys())
    sentiments = list(sentiment_data.values())

    # Create horizontal bars with color coding
    colors = [
        "#E74C3C" if s < -0.1 else "#F39C12" if s < 0.1 else "#27AE60"
        for s in sentiments
    ]
    bars = ax.barh(brands, sentiments, color=colors)

    # Formatting
    ax.set_xlabel("Average Sentiment")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlim(-1, 1)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)  # Neutral line

    # Add value labels on bars
    for bar, value in zip(bars, sentiments):
        width = bar.get_width()
        ax.text(
            width + (0.05 if width >= 0 else -0.05),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.2f}",
            ha="left" if width >= 0 else "right",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    return fig_to_image(fig)


def create_market_share_pie_chart(
    brand_mentions: Dict[str, int],
    client_brand: str,
    title: str = "Market Share Distribution",
) -> Image:
    """
    Create pie chart showing market share distribution across brands.

    Args:
        brand_mentions: Dictionary of brand -> total mentions
        client_brand: Name of the client brand (will be highlighted)
        title: Chart title

    Returns:
        ReportLab Image object
    """
    if not brand_mentions or sum(brand_mentions.values()) == 0:
        logger.warning("No mention data provided for pie chart")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(
            0.5,
            0.5,
            "No market share data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig_to_image(fig)

    fig, ax = plt.subplots(figsize=(5, 5))

    brands = list(brand_mentions.keys())
    mentions = list(brand_mentions.values())

    # Highlight client brand
    colors = []
    explode = []
    for brand in brands:
        if brand == client_brand:
            colors.append(BRAND_COLORS[0])  # Primary color for client
            explode.append(0.1)  # Slightly separate client slice
        else:
            colors.append(BRAND_COLORS[len(colors) % len(BRAND_COLORS)])
            explode.append(0)

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        mentions,
        labels=brands,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    # Use square dimensions for pie charts to avoid distortion
    return fig_to_image(fig, width=4 * inch, height=4 * inch)


def create_platform_donut_chart(
    platform_counts: Dict[str, int], title: str = "Platform Distribution"
) -> Image:
    """
    Create donut chart showing distribution across platforms.

    Args:
        platform_counts: Dictionary of platform -> total mentions
        title: Chart title

    Returns:
        ReportLab Image object
    """
    if not platform_counts or sum(platform_counts.values()) == 0:
        logger.warning("No platform data provided for donut chart")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(
            0.5,
            0.5,
            "No platform data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig_to_image(fig)

    fig, ax = plt.subplots(figsize=(5, 5))

    platforms = list(platform_counts.keys())
    counts = list(platform_counts.values())

    # Create donut chart
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=platforms,
        colors=BRAND_COLORS[: len(platforms)],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
        wedgeprops=dict(width=0.5),  # Creates donut effect
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add center text
    ax.text(
        0,
        0,
        f"Total\n{sum(counts)}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    # Use square dimensions for donut charts to avoid distortion
    return fig_to_image(fig, width=4 * inch, height=4 * inch)


def create_trend_line_chart(
    x_labels: List[str],
    series_data: Dict[str, List[float]],
    y_label: str,
    title: str = "Trend Analysis",
) -> Image:
    """
    Create line chart showing trends over time.

    Args:
        x_labels: List of x-axis labels (time periods)
        series_data: Dictionary of series_name -> list of values
        y_label: Y-axis label
        title: Chart title

    Returns:
        ReportLab Image object
    """
    if not series_data or not x_labels:
        logger.warning("No trend data provided for chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No trend data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig_to_image(fig)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each series
    for i, (series_name, values) in enumerate(series_data.items()):
        if len(values) == len(x_labels):
            ax.plot(
                x_labels,
                values,
                marker="o",
                label=series_name,
                color=BRAND_COLORS[i % len(BRAND_COLORS)],
                linewidth=2,
            )

    # Formatting
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels if needed
    if len(x_labels) > 6:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    return fig_to_image(fig)


def create_category_heatmap(
    matrix: List[List[float]],
    x_labels: List[str],
    y_labels: List[str],
    title: str = "Category Performance Heatmap",
) -> Image:
    """
    Create heatmap showing performance across categories.

    Args:
        matrix: 2D array of values (y_labels x x_labels)
        x_labels: Column labels
        y_labels: Row labels
        title: Chart title

    Returns:
        ReportLab Image object
    """
    if not matrix or not x_labels or not y_labels:
        logger.warning("No heatmap data provided for chart")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No category data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig_to_image(fig)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to numpy array
    data = np.array(matrix)

    # Create heatmap
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Performance Score", rotation=-90, va="bottom")

    # Add text annotations
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(
                j,
                i,
                f"{data[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    return fig_to_image(fig)

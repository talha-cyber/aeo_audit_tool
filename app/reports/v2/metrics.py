"""
Metrics calculation layer for AEO audit reports.

This module provides pure functions for calculating key metrics including
SAIV (Share of AI Voice), sentiment aggregations, and prior-period deltas.
All functions are stateless and testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class BrandStats:
    """Statistics for a single brand across an audit period."""

    mentions: int
    sentiments: List[float]  # -1..+1 scale
    platforms: Dict[str, int]  # platform → mentions count
    categories: Dict[str, int]  # category → mentions count


@dataclass
class Aggregates:
    """Aggregated metrics across all brands in an audit."""

    saiv: Dict[str, float]  # brand → Share of AI Voice (0..1)
    mention_rate: Dict[str, float]  # brand → mentions per 100 questions
    sentiment_mean: Dict[str, float]  # brand → mean sentiment
    sentiment_std: Dict[str, float]  # brand → sentiment std deviation
    n: Dict[str, int]  # brand → number of sentiment measurements


# Minimum platform coverage threshold
MIN_PLAT_COVER = 3


def mean_std(values: List[float]) -> Tuple[float, float]:
    """
    Calculate mean and standard deviation of a list of values.

    Args:
        values: List of numeric values

    Returns:
        Tuple of (mean, standard_deviation)
    """
    if not values:
        return 0.0, 0.0

    mean = sum(values) / len(values)

    if len(values) == 1:
        return mean, 0.0

    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    std_dev = math.sqrt(variance)

    return mean, std_dev


def compute_saiv(brand_mentions: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate Share of AI Voice (SAIV) for each brand.

    SAIV = (brand_mentions / total_mentions) for all brands in the audit.
    This represents the percentage of total brand mentions that each brand captures.

    Args:
        brand_mentions: Dictionary of brand name -> total mentions

    Returns:
        Dictionary of brand name -> SAIV percentage (0.0 to 1.0)
    """
    total_mentions = sum(brand_mentions.values())

    if total_mentions == 0:
        return {brand: 0.0 for brand in brand_mentions}

    return {
        brand: mentions / total_mentions for brand, mentions in brand_mentions.items()
    }


def compute_mention_rate(
    brand_stats: Dict[str, BrandStats], total_questions: int
) -> Dict[str, float]:
    """
    Calculate mention rate per 100 questions for each brand.

    This metric normalizes mention counts by the total number of questions,
    allowing for fair comparison across different audit sizes.

    Args:
        brand_stats: Dictionary of brand -> BrandStats
        total_questions: Total number of questions in the audit

    Returns:
        Dictionary of brand -> mentions per 100 questions
    """
    if total_questions == 0:
        return {brand: 0.0 for brand in brand_stats}

    return {
        brand: (stats.mentions / total_questions) * 100
        for brand, stats in brand_stats.items()
    }


def compute_sentiment_aggregates(
    brand_stats: Dict[str, BrandStats]
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    """
    Calculate sentiment aggregates (mean, std, count) for each brand.

    Args:
        brand_stats: Dictionary of brand -> BrandStats

    Returns:
        Tuple of (sentiment_means, sentiment_stds, sentiment_counts)
    """
    sentiment_means = {}
    sentiment_stds = {}
    sentiment_counts = {}

    for brand, stats in brand_stats.items():
        mean, std = mean_std(stats.sentiments)
        sentiment_means[brand] = mean
        sentiment_stds[brand] = std
        sentiment_counts[brand] = len(stats.sentiments)

    return sentiment_means, sentiment_stds, sentiment_counts


def aggregate_brands(
    brands: Iterable[str], brand_stats: Dict[str, BrandStats], total_questions: int
) -> Aggregates:
    """
    Generate complete aggregated metrics for all brands.

    This is the main function that calculates all key metrics used in reports:
    - SAIV (Share of AI Voice)
    - Mention rates per 100 questions
    - Sentiment statistics (mean, std, count)

    Args:
        brands: List of all brands to analyze
        brand_stats: Dictionary of brand -> BrandStats
        total_questions: Total questions in the audit

    Returns:
        Aggregates object with all calculated metrics
    """
    # Ensure all brands have stats (even if empty)
    complete_stats = {}
    for brand in brands:
        complete_stats[brand] = brand_stats.get(
            brand, BrandStats(mentions=0, sentiments=[], platforms={}, categories={})
        )

    # Calculate mention rates
    mention_rates = compute_mention_rate(complete_stats, total_questions)

    # Calculate SAIV from total mentions
    total_mentions = {brand: stats.mentions for brand, stats in complete_stats.items()}
    saiv = compute_saiv(total_mentions)

    # Calculate sentiment aggregates
    sentiment_means, sentiment_stds, sentiment_counts = compute_sentiment_aggregates(
        complete_stats
    )

    return Aggregates(
        saiv=saiv,
        mention_rate=mention_rates,
        sentiment_mean=sentiment_means,
        sentiment_std=sentiment_stds,
        n=sentiment_counts,
    )


def calculate_deltas(
    current: Dict[str, float], previous: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """
    Calculate period-over-period changes for metrics.

    Args:
        current: Current period metrics (brand -> value)
        previous: Previous period metrics (brand -> value), or None

    Returns:
        Dictionary of brand -> delta (current - previous)
    """
    if not previous:
        return {brand: 0.0 for brand in current}

    return {
        brand: current.get(brand, 0.0) - previous.get(brand, 0.0) for brand in current
    }


def calculate_platform_coverage(
    brand_stats: Dict[str, BrandStats], min_mentions: int = 1
) -> Dict[str, int]:
    """
    Calculate platform coverage for each brand.

    Platform coverage = number of platforms where brand has >= min_mentions.

    Args:
        brand_stats: Dictionary of brand -> BrandStats
        min_mentions: Minimum mentions required to count platform coverage

    Returns:
        Dictionary of brand -> platform coverage count
    """
    coverage = {}

    for brand, stats in brand_stats.items():
        coverage[brand] = len(
            [
                platform
                for platform, mentions in stats.platforms.items()
                if mentions >= min_mentions
            ]
        )

    return coverage


def calculate_category_distribution(
    brand_stats: Dict[str, BrandStats]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate category distribution percentages for each brand.

    Args:
        brand_stats: Dictionary of brand -> BrandStats

    Returns:
        Dictionary of brand -> {category: percentage}
    """
    distributions = {}

    for brand, stats in brand_stats.items():
        total_category_mentions = sum(stats.categories.values())

        if total_category_mentions == 0:
            distributions[brand] = {}
            continue

        distributions[brand] = {
            category: (mentions / total_category_mentions) * 100
            for category, mentions in stats.categories.items()
        }

    return distributions


def identify_performance_gaps(
    client_brand: str, aggregates: Aggregates, threshold_percentile: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Identify performance gaps for the client brand vs competitors.

    Args:
        client_brand: Name of the client brand
        aggregates: Aggregated metrics for all brands
        threshold_percentile: Percentile to use for gap identification (0.5 = median)

    Returns:
        Dictionary of metric -> gap analysis
    """
    gaps = {}

    # SAIV gaps
    competitor_saivs = [
        saiv
        for brand, saiv in aggregates.saiv.items()
        if brand != client_brand and saiv > 0
    ]

    if competitor_saivs:
        competitor_saivs.sort()
        threshold_idx = int(len(competitor_saivs) * threshold_percentile)
        competitor_threshold = competitor_saivs[threshold_idx]
        client_saiv = aggregates.saiv.get(client_brand, 0.0)

        gaps["saiv"] = {
            "client_value": client_saiv,
            "competitor_threshold": competitor_threshold,
            "gap": client_saiv - competitor_threshold,
            "relative_gap": (
                (client_saiv - competitor_threshold) / competitor_threshold * 100
            )
            if competitor_threshold > 0
            else 0,
        }

    # Sentiment gaps
    competitor_sentiments = [
        sentiment
        for brand, sentiment in aggregates.sentiment_mean.items()
        if brand != client_brand
        and aggregates.n.get(brand, 0) >= 5  # Minimum sample size
    ]

    if competitor_sentiments:
        competitor_sentiments.sort()
        threshold_idx = int(len(competitor_sentiments) * threshold_percentile)
        competitor_threshold = competitor_sentiments[threshold_idx]
        client_sentiment = aggregates.sentiment_mean.get(client_brand, 0.0)

        gaps["sentiment"] = {
            "client_value": client_sentiment,
            "competitor_threshold": competitor_threshold,
            "gap": client_sentiment - competitor_threshold,
            "relative_gap": (
                (client_sentiment - competitor_threshold)
                / abs(competitor_threshold)
                * 100
            )
            if competitor_threshold != 0
            else 0,
        }

    return gaps

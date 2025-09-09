#!/usr/bin/env python3
"""
Simple test script for Report Generator v2 implementation.

This script validates that the v2 report engine is properly integrated
and can be imported without errors.
"""

import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_v2_imports():
    """Test that all v2 modules can be imported successfully."""
    print("Testing v2 report engine imports...")

    try:
        # Test core modules
        from app.reports.v2.engine import ReportEngineV2

        print("✓ ReportEngineV2 imported successfully")

        from app.reports.v2.metrics import Aggregates, BrandStats, aggregate_brands

        print("✓ Metrics module imported successfully")

        from app.reports.v2.charts import (
            create_platform_performance_chart,
            create_saiv_bar_chart,
        )

        print("✓ Charts module imported successfully")

        from app.reports.v2.theme import THEMES, get_theme

        print("✓ Theme module imported successfully")

        from app.reports.v2.chassis import ReportBuilder, ReportDoc

        print("✓ Chassis module imported successfully")

        from app.reports.v2.accessibility import add_bookmark, create_section_heading

        print("✓ Accessibility module imported successfully")

        # Test section modules
        from app.reports.v2.sections import (
            competitive,
            platforms,
            recommendations,
            summary,
            title,
        )

        print("✓ All section modules imported successfully")

        print("\n✅ All v2 modules imported without errors!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_theme_functionality():
    """Test theme functionality."""
    print("\nTesting theme functionality...")

    try:
        from datetime import datetime

        from app.reports.v2.theme import format_date, format_number, get_theme

        # Test theme access
        default_theme = get_theme("default")
        print(f"✓ Default theme loaded: {default_theme.name}")

        corporate_theme = get_theme("corporate")
        print(f"✓ Corporate theme loaded: {corporate_theme.name}")

        # Test formatting
        test_date = datetime(2025, 8, 28)
        formatted_date = format_date(test_date, default_theme)
        print(f"✓ Date formatting works: {formatted_date}")

        formatted_number = format_number(1234.567, default_theme, 2)
        print(f"✓ Number formatting works: {formatted_number}")

        print("✅ Theme functionality working!")
        return True

    except Exception as e:
        print(f"❌ Theme test failed: {e}")
        return False


def test_metrics_functionality():
    """Test metrics calculation functionality."""
    print("\nTesting metrics functionality...")

    try:
        from app.reports.v2.metrics import BrandStats, aggregate_brands, compute_saiv

        # Create test data
        brands = ["ClientBrand", "Competitor1", "Competitor2"]
        brand_stats = {
            "ClientBrand": BrandStats(
                mentions=50,
                sentiments=[0.2, 0.1, 0.3, -0.1, 0.5],
                platforms={"ChatGPT": 20, "Claude": 15, "Perplexity": 15},
                categories={"Product": 30, "Support": 20},
            ),
            "Competitor1": BrandStats(
                mentions=80,
                sentiments=[0.1, 0.0, 0.2, 0.3],
                platforms={"ChatGPT": 30, "Claude": 25, "Perplexity": 25},
                categories={"Product": 50, "Support": 30},
            ),
            "Competitor2": BrandStats(
                mentions=30,
                sentiments=[-0.1, 0.0, 0.1],
                platforms={"ChatGPT": 15, "Claude": 10, "Perplexity": 5},
                categories={"Product": 20, "Support": 10},
            ),
        }

        # Test SAIV calculation
        brand_mentions = {brand: stats.mentions for brand, stats in brand_stats.items()}
        saiv_results = compute_saiv(brand_mentions)
        print(f"✓ SAIV calculation works: {saiv_results}")

        # Test aggregation
        aggregates = aggregate_brands(brands, brand_stats, 100)
        print(
            f"✓ Brand aggregation works - ClientBrand SAIV: {aggregates.saiv['ClientBrand']*100:.1f}%"
        )

        print("✅ Metrics functionality working!")
        return True

    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False


def test_integration():
    """Test integration with existing report generator."""
    print("\nTesting integration...")

    try:
        from app.services.report_generator import ReportGenerator

        print("✓ Can import updated ReportGenerator")

        # Check that the new method exists
        if hasattr(ReportGenerator, "_generate_v2_report"):
            print("✓ V2 report method integrated into ReportGenerator")
        else:
            print("❌ V2 report method not found in ReportGenerator")
            return False

        print("✅ Integration test passed!")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Report Generator v2 Implementation Test")
    print("=" * 50)

    tests = [
        test_v2_imports,
        test_theme_functionality,
        test_metrics_functionality,
        test_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! v2 report engine is ready.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

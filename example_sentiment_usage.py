#!/usr/bin/env python3
"""
Example usage of the new modular sentiment analysis system.

This script demonstrates how to use the new sentiment analysis system
for various use cases in the AEO audit tool.
"""

import asyncio
import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.services.sentiment.core.models import AnalysisContext
from app.services.sentiment.integration import (
    SentimentSystemContext,
    analyze_sentiment,
    analyze_sentiment_batch,
    cleanup_sentiment_system,
    get_sentiment_system_status,
    initialize_sentiment_system,
)


async def basic_sentiment_analysis():
    """Demonstrate basic sentiment analysis"""
    print("\n=== Basic Sentiment Analysis ===")

    # Initialize the system
    engine = await initialize_sentiment_system(config_profile="development")

    # Test texts
    test_texts = [
        "I love this amazing product! It works perfectly.",
        "This service is terrible and disappointing.",
        "The product is okay, nothing special.",
        "I hate dealing with customer support issues.",
        "Outstanding quality and excellent performance!",
    ]

    for text in test_texts:
        result = await analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"  Sentiment: {result.polarity.value}")
        print(f"  Score: {result.score:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Method: {result.method.value}")
        print()


async def brand_context_analysis():
    """Demonstrate brand-aware sentiment analysis"""
    print("\n=== Brand Context Analysis ===")

    # Test texts with brand context
    brand = "TechCorp"
    test_cases = [
        "I highly recommend TechCorp for enterprise solutions.",
        "TechCorp has some serious quality issues.",
        "We chose TechCorp over the competition for reliability.",
        "TechCorp vs CompetitorX shows clear advantages.",
        "Avoid TechCorp for this use case, go with AlternativeY instead.",
    ]

    context = AnalysisContext(brand=brand)

    for text in test_cases:
        result = await analyze_sentiment(text, context=context)
        print(f"Text: {text}")
        print(f"  Sentiment: {result.polarity.value}")
        print(f"  Score: {result.score:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Context: {result.context_type.value}")
        print(f"  Method: {result.method.value}")

        # Show business analysis details if available
        if "business_analysis" in result.metadata:
            ba = result.metadata["business_analysis"]
            print(
                f"  Business Signals: +{ba.get('positive_signals', 0)} -{ba.get('negative_signals', 0)}"
            )
        print()


async def batch_processing_demo():
    """Demonstrate batch processing capabilities"""
    print("\n=== Batch Processing Demo ===")

    # Large batch of texts
    batch_texts = [
        "Great product with excellent features!",
        "Poor customer service experience.",
        "Average quality, meets basic needs.",
        "Fantastic support team, very helpful.",
        "Disappointed with the recent updates.",
        "Highly recommend for small businesses.",
        "Technical issues keep occurring.",
        "Good value for the price point.",
        "Superior to most competitors.",
        "Lacking important functionality.",
    ]

    print(f"Processing batch of {len(batch_texts)} texts...")

    import time

    start_time = time.time()

    batch_result = await analyze_sentiment_batch(batch_texts)

    end_time = time.time()
    processing_time = end_time - start_time

    print("Batch Results:")
    print(f"  Total processed: {batch_result.total_processed}")
    print(f"  Successful: {batch_result.successful_analyses}")
    print(f"  Failed: {batch_result.failed_analyses}")
    print(f"  Average confidence: {batch_result.average_confidence:.3f}")
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  Texts per second: {len(batch_texts)/processing_time:.1f}")

    # Show sentiment distribution
    distribution = batch_result.get_sentiment_distribution()
    print("  Sentiment distribution:")
    for polarity, count in distribution.items():
        print(f"    {polarity.value}: {count}")
    print()


async def system_status_demo():
    """Demonstrate system status monitoring"""
    print("\n=== System Status Demo ===")

    # Get system status
    status = await get_sentiment_system_status()
    print("System Status:")
    print(f"  Initialized: {status['initialized']}")
    print(f"  Status: {status['status']}")
    print(f"  Message: {status['message']}")

    if status["initialized"] and "engine" in status:
        engine_info = status["engine"]
        print(f"  Enabled providers: {engine_info['config']['enabled_providers']}")
        print(f"  Primary provider: {engine_info['config']['primary_provider']}")

        # Show provider health
        print("  Provider health:")
        for provider, health in engine_info["providers"].items():
            print(f"    {provider}: {'✓' if health['available'] else '✗'}")
    print()


async def performance_comparison():
    """Compare different provider configurations"""
    print("\n=== Performance Comparison ===")

    test_text = (
        "I recommend BrandX for this solution. It's excellent quality and reliable."
    )
    context = AnalysisContext(brand="BrandX")

    # Test with different configurations
    configurations = [
        ("VADER Only", "development"),  # Single provider
        ("Production", "production"),  # Multiple providers with ensemble
    ]

    for config_name, profile in configurations:
        print(f"\n{config_name} Configuration:")

        # Clean up previous engine
        await cleanup_sentiment_system()

        # Initialize with new config
        engine = await initialize_sentiment_system(config_profile=profile)

        # Time the analysis
        import time

        start_time = time.time()

        result = await analyze_sentiment(test_text, context=context)

        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms

        print(f"  Result: {result.polarity.value} (score: {result.score:.3f})")
        print(f"  Method: {result.method.value}")
        print(f"  Processing time: {processing_time:.1f}ms")
        print(f"  Confidence: {result.confidence:.3f}")


async def backward_compatibility_demo():
    """Demonstrate backward compatibility with legacy system"""
    print("\n=== Backward Compatibility Demo ===")

    # Initialize with compatibility enabled
    await initialize_sentiment_system(enable_compatibility=True)

    # Import the compatibility analyzer
    from app.services.sentiment.compat import create_sentiment_analyzer

    # Use the legacy interface
    analyzer = create_sentiment_analyzer()

    print("Using legacy SentimentAnalyzer interface:")

    # This should work exactly like the old system
    text = "I highly recommend BrandX for enterprise solutions."
    brand = "BrandX"

    # Test legacy methods
    try:
        # VADER analysis (sync)
        vader_result = analyzer.analyze_sentiment_vader(text)
        print(
            f"VADER (legacy): {vader_result.polarity.value} (confidence: {vader_result.confidence:.3f})"
        )

        # Business sentiment (sync)
        business_result = analyzer.analyze_business_sentiment(text, brand)
        print(
            f"Business (legacy): {business_result.polarity.value} (confidence: {business_result.confidence:.3f})"
        )

        # Hybrid analysis (async)
        hybrid_result = await analyzer.analyze_sentiment_hybrid(text, brand)
        print(
            f"Hybrid (legacy): {hybrid_result.polarity.value} (confidence: {hybrid_result.confidence:.3f})"
        )

        print("✓ Legacy compatibility working correctly!")

    except Exception as e:
        print(f"✗ Legacy compatibility error: {e}")

    print()


async def context_manager_demo():
    """Demonstrate context manager usage"""
    print("\n=== Context Manager Demo ===")

    # Use context manager for automatic cleanup
    async with SentimentSystemContext(config_profile="development") as engine:
        print("Inside context manager - system initialized")

        result = await analyze_sentiment("This is a test message.")
        print(f"Analysis result: {result.polarity.value}")

        status = await get_sentiment_system_status()
        print(f"System status: {status['status']}")

    print("Context manager exited - system cleaned up")

    # Verify cleanup
    status = await get_sentiment_system_status()
    print(f"Post-cleanup status: {status['initialized']}")


async def main():
    """Run all demonstration examples"""
    print("Sentiment Analysis System Demo")
    print("==============================")

    try:
        # Run all demos
        await basic_sentiment_analysis()
        await brand_context_analysis()
        await batch_processing_demo()
        await system_status_demo()
        await performance_comparison()
        await backward_compatibility_demo()
        await context_manager_demo()

    except Exception as e:
        print(f"Demo error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Final cleanup
        await cleanup_sentiment_system()
        print("\n=== Demo Complete ===")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())

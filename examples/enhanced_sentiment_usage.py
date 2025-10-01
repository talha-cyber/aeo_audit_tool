"""
Enhanced Sentiment Analysis Usage Examples.

Demonstrates how to use the new cost-effective ML capabilities
including efficient transformers, domain adaptation, and cost monitoring.
"""

import asyncio

from app.services.sentiment.core.models import AnalysisContext

# Import the enhanced sentiment engine
from app.services.sentiment.enhanced_engine import (
    create_enhanced_sentiment_engine,
)


async def basic_cost_effective_analysis():
    """Example 1: Basic cost-effective sentiment analysis"""
    print("=== Example 1: Basic Cost-Effective Analysis ===")

    # Create enhanced engine with balanced cost preference
    engine = await create_enhanced_sentiment_engine(
        cost_preference="balanced", budget_limit=5.0  # $5 monthly budget
    )

    try:
        # Analyze single text with different cost preferences
        text = "I love this new product! It's amazing and works perfectly."

        # Ultra-low cost analysis (quantized model, minimal resources)
        result_ultra_low = await engine.analyze_cost_effective(
            text, cost_preference="ultra_low"
        )
        print(
            f"Ultra-low cost: {result_ultra_low.polarity.value} "
            f"(confidence: {result_ultra_low.confidence:.3f})"
        )

        # Balanced analysis (good accuracy/cost trade-off)
        result_balanced = await engine.analyze_cost_effective(
            text, cost_preference="balanced"
        )
        print(
            f"Balanced: {result_balanced.polarity.value} "
            f"(confidence: {result_balanced.confidence:.3f})"
        )

        # High accuracy analysis (best model, higher cost)
        result_accurate = await engine.analyze_cost_effective(
            text, cost_preference="high_accuracy"
        )
        print(
            f"High accuracy: {result_accurate.polarity.value} "
            f"(confidence: {result_accurate.confidence:.3f})"
        )

        print("✅ Basic analysis completed successfully")

    finally:
        await engine.cleanup()


async def batch_processing_example():
    """Example 2: Optimized batch processing"""
    print("\n=== Example 2: Optimized Batch Processing ===")

    engine = await create_enhanced_sentiment_engine(
        cost_preference="low",  # Use low-cost models for bulk processing
        budget_limit=10.0,
    )

    try:
        # Sample texts for batch processing
        texts = [
            "This product is fantastic!",
            "Not impressed with the quality.",
            "Average product, nothing special.",
            "Excellent customer service!",
            "Had some issues but got resolved quickly.",
            "Would definitely recommend to others.",
            "Price is too high for what you get.",
            "Perfect for my needs!",
            "Customer support was unhelpful.",
            "Great value for money!",
        ]

        # Process batch with cost optimization
        batch_result = await engine.batch_analyze_optimized(
            texts=texts, cost_preference="low", brand="ExampleBrand"
        )

        print(f"Processed {len(batch_result.results)} texts")
        print(f"Average confidence: {batch_result.average_confidence:.3f}")
        print(f"Processing time: {batch_result.total_processing_time:.2f}s")

        # Show sentiment distribution
        sentiment_dist = batch_result.get_sentiment_distribution()
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment.value}: {count}")

        print("✅ Batch processing completed successfully")

    finally:
        await engine.cleanup()


async def domain_adaptation_example():
    """Example 3: Domain-specific model training"""
    print("\n=== Example 3: Domain Adaptation ===")

    engine = await create_enhanced_sentiment_engine(
        enable_domain_adaptation=True, budget_limit=15.0  # Higher budget for training
    )

    try:
        # Sample training data for a specific domain (e.g., restaurants)
        training_texts = [
            "The food was absolutely delicious!",
            "Service was slow and unfriendly.",
            "Great atmosphere and ambiance.",
            "Overpriced for the portion size.",
            "Fresh ingredients and well prepared.",
            "Long wait times, not worth it.",
            "Excellent wine selection.",
            "The staff was very attentive.",
            "Food arrived cold and tasteless.",
            "Perfect for a romantic dinner.",
            "Noisy environment, couldn't have a conversation.",
            "Best pasta I've ever had!",
            "Poor hygiene in the restroom.",
            "Amazing dessert selection.",
            "Waitress was rude and impatient.",
        ]

        training_labels = [
            "positive",
            "negative",
            "positive",
            "negative",
            "positive",
            "negative",
            "positive",
            "positive",
            "negative",
            "positive",
            "negative",
            "positive",
            "negative",
            "positive",
            "negative",
        ]

        # Check if we have enough budget for training
        cost_estimate = engine.domain_adapter.estimate_training_cost(
            num_samples=len(training_texts), num_epochs=3
        )

        print(f"Estimated training cost: ${cost_estimate['estimated_gpu_cost']:.4f}")
        print(f"Estimated training time: {cost_estimate['estimated_hours']:.2f} hours")

        # Train domain-specific model
        if cost_estimate["estimated_gpu_cost"] < 2.0:  # Only train if under $2
            print("Training restaurant domain model...")

            training_results = await engine.train_domain_model(
                domain_name="restaurant_reviews",
                training_texts=training_texts,
                training_labels=training_labels,
                validation_split=0.3,
                cost_limit=1.0,  # Max 1 hour of training
            )

            if "error" not in training_results:
                print(f"Training completed in {training_results['training_time']:.2f}s")
                print(f"Training cost: ${training_results['estimated_cost']:.4f}")
                print(
                    f"Final training loss: {training_results['final_train_loss']:.4f}"
                )

                # Test the domain model
                test_text = "The service was terrible and the food was cold."
                result = await engine.analyze_with_domain_model(
                    text=test_text, domain_name="restaurant_reviews"
                )

                print(
                    f"Domain model result: {result.polarity.value} "
                    f"(confidence: {result.confidence:.3f})"
                )
                print("✅ Domain adaptation completed successfully")
            else:
                print(f"Training failed: {training_results['error']}")
        else:
            print("Training cost too high, skipping domain adaptation example")

    finally:
        await engine.cleanup()


async def cost_monitoring_example():
    """Example 4: Cost monitoring and optimization"""
    print("\n=== Example 4: Cost Monitoring ===")

    engine = await create_enhanced_sentiment_engine(
        cost_preference="balanced", budget_limit=8.0
    )

    try:
        # Perform various operations to generate usage data
        texts = [
            "Great product, highly recommended!",
            "Poor quality, wouldn't buy again.",
            "Average experience, nothing special.",
            "Exceeded my expectations!",
            "Terrible customer service.",
        ]

        # Mix of different cost preferences
        for i, text in enumerate(texts):
            preferences = ["ultra_low", "low", "balanced", "balanced", "high_accuracy"]
            await engine.analyze_cost_effective(
                text, cost_preference=preferences[i % len(preferences)]
            )

        # Get cost report
        cost_report = engine.get_cost_report("month")

        print("Monthly Cost Summary:")
        summary = cost_report["cost_summary"]
        print(f"  Total cost: ${summary['total_cost']:.4f}")
        print(f"  Budget used: {summary['budget_used_percent']:.1f}%")
        print(f"  Operations: {summary['total_operations']}")
        print(f"  Avg cost per operation: ${summary['avg_cost_per_operation']:.6f}")

        # Show optimization recommendations
        print("\nOptimization Recommendations:")
        for rec in cost_report["optimization_recommendations"]:
            print(f"  • {rec}")

        # Show available capabilities
        capabilities = engine.get_available_capabilities()
        print(f"\nAvailable Models: {len(capabilities['cost_effective_models'])}")
        print(f"Domain Models: {capabilities['domain_models']}")

        # Optimize for lower budget
        optimization = await engine.optimize_for_cost(target_budget=5.0)
        print("\nBudget Optimization:")
        for rec in optimization["recommendations"]:
            print(f"  • {rec}")

        print("✅ Cost monitoring example completed successfully")

    finally:
        await engine.cleanup()


async def business_context_example():
    """Example 5: Business context analysis with cost optimization"""
    print("\n=== Example 5: Business Context Analysis ===")

    engine = await create_enhanced_sentiment_engine(
        cost_preference="balanced", budget_limit=12.0
    )

    try:
        # Business context texts
        business_texts = [
            "We recommend ExampleBrand for all your needs.",
            "ExampleBrand vs CompetitorX - clear winner!",
            "Had issues with ExampleBrand customer support.",
            "ExampleBrand is the industry leader in innovation.",
            "Switched from ExampleBrand to a better alternative.",
            "ExampleBrand's new product launch exceeded expectations.",
        ]

        # Analyze with business context
        context = AnalysisContext(
            brand="ExampleBrand",
            industry="technology",
            source="social_media",
            competitive_context=True,
        )

        print("Business Context Analysis Results:")
        for text in business_texts:
            result = await engine.analyze_cost_effective(
                text=text, cost_preference="balanced", context=context
            )

            print(f"  Text: {text[:50]}...")
            print(
                f"  Sentiment: {result.polarity.value} "
                f"(confidence: {result.confidence:.3f})"
            )
            print(f"  Context: {result.context_type.value}")

            # Show business metadata if available
            if "business_analysis" in result.metadata:
                business_data = result.metadata["business_analysis"]
                print(
                    f"  Business signals: +{business_data.get('positive_signals', 0)} "
                    f"-{business_data.get('negative_signals', 0)}"
                )
            print()

        print("✅ Business context analysis completed successfully")

    finally:
        await engine.cleanup()


async def performance_comparison():
    """Example 6: Performance and cost comparison"""
    print("\n=== Example 6: Performance Comparison ===")

    engine = await create_enhanced_sentiment_engine(
        cost_preference="balanced", budget_limit=10.0
    )

    try:
        test_text = "This is an amazing product that exceeded all my expectations!"

        # Test different providers and measure performance
        providers_to_test = ["ultra_low", "low", "balanced", "high_accuracy"]
        results = {}

        for provider in providers_to_test:
            start_time = asyncio.get_event_loop().time()

            result = await engine.analyze_cost_effective(
                text=test_text, cost_preference=provider
            )

            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time

            results[provider] = {
                "sentiment": result.polarity.value,
                "confidence": result.confidence,
                "processing_time": processing_time,
                "cost_metrics": result.metadata.get("cost_metrics", {}),
            }

        # Display comparison
        print("Provider Performance Comparison:")
        print(
            f"{'Provider':<15} {'Sentiment':<10} {'Confidence':<12} {'Time (s)':<10} {'Memory (MB)':<12}"
        )
        print("-" * 70)

        for provider, data in results.items():
            memory_usage = data["cost_metrics"].get("memory_usage_mb", 0)
            print(
                f"{provider:<15} {data['sentiment']:<10} "
                f"{data['confidence']:<12.3f} {data['processing_time']:<10.3f} "
                f"{memory_usage:<12.1f}"
            )

        # Show cost report
        cost_report = engine.get_cost_report("session")
        print(f"\nSession Cost: ${cost_report['cost_summary']['total_cost']:.6f}")

        print("✅ Performance comparison completed successfully")

    finally:
        await engine.cleanup()


async def main():
    """Run all examples"""
    print("Enhanced Sentiment Analysis - Usage Examples")
    print("=" * 50)

    try:
        await basic_cost_effective_analysis()
        await batch_processing_example()
        await domain_adaptation_example()
        await cost_monitoring_example()
        await business_context_example()
        await performance_comparison()

        print("\n" + "=" * 50)
        print("All examples completed successfully! 🎉")
        print("\nKey Benefits Demonstrated:")
        print("• Cost-effective ML models with quantization")
        print("• Domain-specific fine-tuning capabilities")
        print("• Comprehensive cost monitoring and budgeting")
        print("• Optimized batch processing")
        print("• Business context-aware analysis")
        print("• Performance/cost trade-off comparisons")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

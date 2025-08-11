#!/usr/bin/env python3
"""
Example usage of the Brand Detection Engine
"""

import asyncio
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the brand detection engine
from app.services.brand_detection import (
    DetectionConfig,
    create_detection_engine,
    create_orchestrator,
    initialize_brand_detection,
)


async def main():
    """Main example function"""

    # Initialize the brand detection system
    initialize_brand_detection()

    # Get API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: Please set OPENAI_API_KEY in your .env file")
        return

    # Create detection engine
    print("🚀 Initializing Brand Detection Engine...")
    engine = create_detection_engine(openai_api_key)

    # Sample German text
    sample_text = """
    Salesforce ist eine führende CRM-Software, die von vielen Unternehmen weltweit verwendet wird.
    Die Plattform bietet umfassende Funktionen für Vertrieb, Marketing und Kundenservice.
    HubSpot ist ebenfalls eine beliebte Alternative für mittelständische Unternehmen.
    Microsoft Dynamics 365 integriert sich nahtlos in andere Microsoft-Produkte.
    Viele Kunden sind mit Salesforce sehr zufrieden und empfehlen es weiter.
    """

    # Target brands to detect
    target_brands = ["Salesforce", "HubSpot", "Microsoft Dynamics", "SAP"]

    print(f"📝 Analyzing text (length: {len(sample_text)} characters)")
    print(f"🎯 Looking for brands: {', '.join(target_brands)}")
    print("\n" + "=" * 50)

    try:
        # Perform brand detection
        result = await engine.detect_brands(
            text=sample_text,
            target_brands=target_brands,
            config=DetectionConfig(market_code="DE"),
        )

        # Display results
        print(f"✅ Detection completed in {result.processing_time_ms:.1f}ms")
        print(f"🔍 Found {result.total_brands_found} brand mentions")
        print(f"🌍 Market: {result.market_adapter_used}")
        print(f"🗣️  Language: {result.language}")

        if result.mentions:
            print("\n📊 Brand Mentions Found:")
            print("-" * 30)

            for mention in result.mentions:
                print(f"🏢 Brand: {mention.brand}")
                print(f"   📄 Text: '{mention.original_text}'")
                print(f"   🎯 Confidence: {mention.confidence:.2f}")
                print(
                    f"   💭 Sentiment: {mention.sentiment_score:.2f} ({mention.sentiment_polarity.value})"
                )
                print(f"   🔧 Method: {mention.detection_method.value}")
                print(f"   📍 Mentions: {mention.mention_count}")

                if mention.contexts:
                    print(f"   📝 Context: '{mention.contexts[0].sentence[:100]}...'")
                print()
        else:
            print("❌ No brand mentions found")

        # Test competitive analysis
        print("\n" + "=" * 50)
        print("🏆 Running Competitive Analysis...")

        orchestrator = create_orchestrator(openai_api_key)

        competitive_analysis = await orchestrator.detect_competitive_analysis(
            text=sample_text,
            client_brand="Salesforce",
            competitors=["HubSpot", "Microsoft Dynamics"],
            market_code="DE",
        )

        print(f"📈 Client Brand: {competitive_analysis['client_brand']}")
        print(f"🏁 Total Mentions: {competitive_analysis['total_mentions']}")

        if competitive_analysis.get("brand_breakdown"):
            print("\n📊 Brand Breakdown:")
            for brand, data in competitive_analysis["brand_breakdown"].items():
                print(f"   {brand}: {data['mention_count']} mentions")

        if competitive_analysis.get("competitive_insights"):
            print("\n💡 Insights:")
            for insight in competitive_analysis["competitive_insights"]:
                print(f"   • {insight}")

        # Display engine statistics
        print("\n" + "=" * 50)
        print("📈 Engine Statistics:")
        stats = engine.get_detection_statistics()
        print(f"   Success Rate: {stats['success_rate_percent']:.1f}%")
        print(f"   Cache Hit Rate: {stats['cache_hit_rate_percent']:.1f}%")
        print(f"   Total Detections: {stats['detection_stats']['total_detections']}")

    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        engine.cleanup()
        print("\n🧹 Cleanup completed")


if __name__ == "__main__":
    print("Brand Detection Engine - Example Usage")
    print("=" * 40)
    asyncio.run(main())

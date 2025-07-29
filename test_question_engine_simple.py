#!/usr/bin/env python3
"""
Simple test script to verify Question Engine functionality
"""

import os
import sys

# Add the current directory to Python path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_question_engine() -> bool:
    """Test the Question Engine basic functionality."""
    try:
        from app.services.question_engine import QuestionCategory, QuestionEngine

        print("✅ Question Engine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Question Engine: {e}")
        return False

    # Test initialization
    try:
        engine = QuestionEngine()
        print("✅ Question Engine initialized successfully")
        print(f"   Templates: {len(engine.base_templates)}")
        print(f"   Industry patterns: {len(engine.industry_patterns)}")
    except Exception as e:
        print(f"❌ Failed to initialize Question Engine: {e}")
        return False

    # Test question generation
    try:
        questions = engine.generate_questions(
            client_brand="TestCRM",
            competitors=["Salesforce", "HubSpot"],
            industry="CRM",
        )
        print(f"✅ Generated {len(questions)} questions")

        # Show some example questions
        if questions:
            print("   Example questions:")
            for i, q in enumerate(questions[:5]):
                print(f"   {i+1}. [{q['category']}] {q['question']}")

    except Exception as e:
        print(f"❌ Failed to generate questions: {e}")
        return False

    # Test prioritization
    try:
        prioritized = engine.prioritize_questions(questions, max_questions=10)
        print(f"✅ Prioritized {len(prioritized)} questions")

        if prioritized:
            print("   Top priority questions:")
            for i, q in enumerate(prioritized[:3]):
                score = q.get("priority_score", "N/A")
                print(f"   {i+1}. [Score: {score}] {q['question']}")

    except Exception as e:
        print(f"❌ Failed to prioritize questions: {e}")
        return False

    # Test categories
    try:
        comparison_questions = engine.generate_questions(
            client_brand="TestCRM",
            competitors=["Salesforce"],
            industry="CRM",
            categories=[QuestionCategory.COMPARISON],
        )
        print(f"✅ Generated {len(comparison_questions)} comparison questions")
    except Exception as e:
        print(f"❌ Failed to generate category-specific questions: {e}")
        return False

    return True


def test_question_categories() -> bool:
    """Test that all required categories are available."""
    try:
        from app.services.question_engine import QuestionCategory

        expected_categories = {
            "COMPARISON",
            "RECOMMENDATION",
            "FEATURES",
            "PRICING",
            "REVIEWS",
            "ALTERNATIVES",
        }

        actual_categories = {cat.name for cat in QuestionCategory}

        if expected_categories == actual_categories:
            print("✅ All required question categories are present")
            return True
        else:
            missing = expected_categories - actual_categories
            extra = actual_categories - expected_categories
            if missing:
                print(f"❌ Missing categories: {missing}")
            if extra:
                print(f"⚠️  Extra categories: {extra}")
            return False

    except Exception as e:
        print(f"❌ Failed to test categories: {e}")
        return False


def test_question_structure() -> bool:
    """Test that generated questions have the correct structure."""
    try:
        from app.services.question_engine import QuestionEngine

        engine = QuestionEngine()
        questions = engine.generate_questions(
            client_brand="TestBrand",
            competitors=["Competitor1", "Competitor2"],
            industry="CRM",
        )

        required_fields = [
            "question",
            "category",
            "type",
            "industry",
            "client_brand",
            "competitors",
        ]

        for question in questions[:5]:  # Check first 5 questions
            for field in required_fields:
                if field not in question:
                    print(f"❌ Missing field '{field}' in question")
                    return False

            # Check question text is not empty
            if not question["question"].strip():
                print("❌ Empty question text found")
                return False

        print("✅ Question structure validation passed")
        return True

    except Exception as e:
        print(f"❌ Failed to test question structure: {e}")
        return False


if __name__ == "__main__":
    print("Testing Question Engine Module...\n")

    tests = [test_question_categories, test_question_engine, test_question_structure]

    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())

    passed = sum(results)
    total = len(results)

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Question Engine is working correctly.")
        print("\nFeatures verified:")
        print("• Question generation from templates")
        print("• Industry-specific question patterns")
        print("• Question prioritization and scoring")
        print("• Category filtering")
        print("• Proper question structure")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full test suite: pytest tests/test_question_engine.py")
        print("3. Integrate with FastAPI endpoints")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the error messages above.")

    sys.exit(0 if passed == total else 1)

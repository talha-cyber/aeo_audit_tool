"""
Tests for the Question Engine module.

Validates question generation, prioritization, and scoring functionality.
"""

from typing import Any, Dict, List

import pytest

from app.services.question_engine import QuestionCategory, QuestionEngine


class TestQuestionEngine:
    """Test suite for QuestionEngine class."""

    @pytest.fixture
    def engine(self) -> QuestionEngine:
        """Create a QuestionEngine instance for testing."""
        return QuestionEngine()

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Sample data for testing."""
        return {
            "client_brand": "TestCRM",
            "competitors": ["Salesforce", "HubSpot"],
            "industry": "CRM",
        }

    def test_question_engine_initialization(self, engine: QuestionEngine) -> None:
        """Test that QuestionEngine initializes correctly."""
        assert engine.base_templates is not None
        assert len(engine.base_templates) > 0
        assert engine.industry_patterns is not None
        assert len(engine.industry_patterns) > 0

        # Verify all required categories are covered
        template_categories = {template.category for template in engine.base_templates}
        expected_categories = {
            QuestionCategory.COMPARISON,
            QuestionCategory.RECOMMENDATION,
            QuestionCategory.ALTERNATIVES,
            QuestionCategory.FEATURES,
            QuestionCategory.PRICING,
            QuestionCategory.REVIEWS,
        }
        assert template_categories == expected_categories

    @pytest.mark.parametrize(
        "industry,competitors",
        [
            ("CRM", ["Salesforce", "HubSpot"]),
            ("Marketing Automation", ["Marketo", "Pardot", "ActiveCampaign"]),
            ("Project Management", ["Asana", "Monday.com"]),
            ("Analytics", ["Google Analytics"]),
        ],
    )
    def test_generate_questions_basic(
        self, engine: QuestionEngine, industry: str, competitors: List[str]
    ) -> None:
        """Test basic question generation for different industries."""
        client_brand = "TestBrand"
        questions = engine.generate_questions(
            client_brand=client_brand, competitors=competitors, industry=industry
        )

        # Verify questions were generated
        assert len(questions) > 0

        # Verify all questions have required fields
        for question in questions:
            assert "question" in question
            assert "category" in question
            assert "type" in question
            assert "industry" in question
            assert "client_brand" in question
            assert "competitors" in question

        # Verify client brand appears in some questions
        client_questions = [q for q in questions if client_brand in q["question"]]
        assert len(client_questions) > 0

        # Verify competitors appear in some questions
        for competitor in competitors:
            competitor_questions = [q for q in questions if competitor in q["question"]]
            assert len(competitor_questions) > 0

        # Verify industry appears in some questions
        industry_questions = [q for q in questions if industry in q["question"]]
        assert len(industry_questions) > 0

    def test_generate_questions_with_categories(
        self, engine: QuestionEngine, sample_data: Dict[str, Any]
    ) -> None:
        """Test question generation with specific categories."""
        categories = [QuestionCategory.COMPARISON, QuestionCategory.PRICING]
        questions = engine.generate_questions(
            client_brand=sample_data["client_brand"],
            competitors=sample_data["competitors"],
            industry=sample_data["industry"],
            categories=categories,
        )

        # Verify only specified categories are present
        question_categories = {
            q["category"] for q in questions if q["category"] != "industry_specific"
        }
        expected_categories = {cat.value for cat in categories}
        assert question_categories.issubset(expected_categories)

    def test_generate_questions_question_types(
        self, engine: QuestionEngine, sample_data: Dict[str, Any]
    ) -> None:
        """Test that different question types are generated."""
        questions = engine.generate_questions(**sample_data)

        question_types = {q["type"] for q in questions}
        expected_types = {
            "industry_general",
            "brand_specific",
            "competitor_specific",
            "alternative_seeking",
            "industry_specific",
        }

        # At least some of these types should be present
        assert len(question_types.intersection(expected_types)) > 0

        # Verify target_brand is set for applicable types
        brand_specific = [q for q in questions if q["type"] == "brand_specific"]
        for question in brand_specific:
            assert "target_brand" in question
            assert question["target_brand"] == sample_data["client_brand"]

        competitor_specific = [
            q for q in questions if q["type"] == "competitor_specific"
        ]
        for question in competitor_specific:
            assert "target_brand" in question
            assert question["target_brand"] in sample_data["competitors"]

    def test_generate_questions_industry_specific(
        self, engine: QuestionEngine, sample_data: Dict[str, Any]
    ) -> None:
        """Test generation of industry-specific questions."""
        questions = engine.generate_questions(**sample_data)

        # Find industry-specific questions
        industry_questions = [
            q for q in questions if q["category"] == "industry_specific"
        ]

        # Should have industry-specific questions for CRM
        assert len(industry_questions) > 0

        # Verify structure of industry-specific questions
        for question in industry_questions:
            assert question["type"] == "industry_specific"
            assert question["industry"] == sample_data["industry"]

    def test_prioritize_questions_basic(
        self, engine: QuestionEngine, sample_data: Dict[str, Any]
    ) -> None:
        """Test basic question prioritization."""
        questions = engine.generate_questions(**sample_data)
        max_questions = 50

        prioritized = engine.prioritize_questions(questions, max_questions)

        # Verify correct number of questions returned
        assert len(prioritized) <= max_questions
        assert len(prioritized) <= len(questions)

        # Verify all questions have priority scores
        for question in prioritized:
            assert "priority_score" in question
            assert isinstance(question["priority_score"], (int, float))

        # Verify questions are sorted by priority (descending)
        scores = [q["priority_score"] for q in prioritized]
        assert scores == sorted(scores, reverse=True)

    def test_prioritize_questions_scoring(self, engine: QuestionEngine) -> None:
        """Test that priority scoring works correctly."""
        # Create test questions with different categories
        test_questions = [
            {"category": "comparison", "type": "industry_general"},
            {"category": "pricing", "type": "brand_specific"},
            {"category": "alternatives", "type": "alternative_seeking"},
            {"category": "features", "type": "competitor_specific"},
        ]

        prioritized = engine.prioritize_questions(test_questions)

        # Verify scoring logic
        comparison_q = next(q for q in prioritized if q["category"] == "comparison")
        pricing_q = next(q for q in prioritized if q["category"] == "pricing")
        alternatives_q = next(q for q in prioritized if q["category"] == "alternatives")

        # Comparison should score higher than pricing
        assert comparison_q["priority_score"] > pricing_q["priority_score"]

        # Industry general should get boost
        assert comparison_q["priority_score"] == 12  # 10 + 2 boost

        # Alternative seeking should get boost
        assert alternatives_q["priority_score"] == 9  # 8 + 1 boost

    def test_prioritize_questions_max_limit(
        self, engine: QuestionEngine, sample_data: Dict[str, Any]
    ) -> None:
        """Test that max_questions limit is respected."""
        questions = engine.generate_questions(**sample_data)

        # Test with different limits
        for max_questions in [5, 10, 25]:
            prioritized = engine.prioritize_questions(questions, max_questions)
            assert len(prioritized) <= max_questions

    def test_empty_competitors_list(self, engine: QuestionEngine) -> None:
        """Test question generation with empty competitors list."""
        questions = engine.generate_questions(
            client_brand="SoloBrand", competitors=[], industry="CRM"
        )

        # Should still generate questions
        assert len(questions) > 0

        # Should not have competitor-specific questions
        competitor_questions = [
            q for q in questions if q["type"] == "competitor_specific"
        ]
        assert len(competitor_questions) == 0

        alternative_questions = [
            q for q in questions if q["type"] == "alternative_seeking"
        ]
        assert len(alternative_questions) == 0

    def test_unknown_industry(self, engine: QuestionEngine) -> None:
        """Test question generation with unknown industry."""
        questions = engine.generate_questions(
            client_brand="TestBrand",
            competitors=["Competitor1"],
            industry="UnknownIndustry",
        )

        # Should still generate basic questions
        assert len(questions) > 0

        # Should not have industry-specific questions for unknown industry
        industry_questions = [
            q for q in questions if q["category"] == "industry_specific"
        ]
        assert len(industry_questions) == 0

    @pytest.mark.parametrize(
        "category",
        [
            QuestionCategory.COMPARISON,
            QuestionCategory.RECOMMENDATION,
            QuestionCategory.FEATURES,
            QuestionCategory.PRICING,
            QuestionCategory.REVIEWS,
            QuestionCategory.ALTERNATIVES,
        ],
    )
    def test_single_category_generation(
        self,
        engine: QuestionEngine,
        sample_data: Dict[str, Any],
        category: QuestionCategory,
    ) -> None:
        """Test question generation for individual categories."""
        questions = engine.generate_questions(categories=[category], **sample_data)

        # Verify questions were generated
        assert len(questions) > 0

        # Verify only specified category (plus industry_specific) is present
        categories_found = {q["category"] for q in questions}
        expected_categories = {category.value, "industry_specific"}
        assert categories_found.issubset(expected_categories)

    def test_question_structure_validation(
        self, engine: QuestionEngine, sample_data: Dict[str, Any]
    ) -> None:
        """Test that generated questions have correct structure."""
        questions = engine.generate_questions(**sample_data)

        required_fields = [
            "question",
            "category",
            "type",
            "industry",
            "client_brand",
            "competitors",
        ]

        for question in questions:
            # Verify all required fields are present
            for field in required_fields:
                assert (
                    field in question
                ), f"Missing field {field} in question: {question}"

            # Verify question text is not empty
            assert len(question["question"].strip()) > 0

            # Verify category is valid
            valid_categories = [cat.value for cat in QuestionCategory] + [
                "industry_specific"
            ]
            assert question["category"] in valid_categories

            # Verify type is valid
            valid_types = [
                "industry_general",
                "brand_specific",
                "competitor_specific",
                "alternative_seeking",
                "industry_specific",
            ]
            assert question["type"] in valid_types
